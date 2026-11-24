"""Frame-level evaluation on the UCF-Crime test set.

Scores every test video with the trained anomaly classifier (or, with
``--baseline motion``, a simple optical-flow baseline), spreads each clip
score over its frames, and computes frame-level ROC-AUC and related metrics
against the official temporal annotations.

There is no synthetic fallback: if the dataset or the trained head is missing
the script exits with an error instead of writing made-up numbers.

Usage:
    python -m evaluation.evaluate_ucf_crime \
        --dataset-path data/UCF-Crime/ --output-dir results/

The annotation file (``Temporal_Anomaly_Annotation_for_Testing_Videos.txt``)
has lines like::

    Abuse028_x264.mp4  Abuse  165  240  -1  -1
    Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.utils import load_config

ClipScorer = Callable[[list[np.ndarray]], float]


def load_ucf_crime_annotations(annotation_path: str) -> dict[str, list[tuple[int, int]]]:
    """Parse the UCF-Crime temporal annotation file.

    Returns:
        Video file name → list of (start_frame, end_frame) anomaly intervals.
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    annotations: dict[str, list[tuple[int, int]]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        # Skip the category column (and anything else non-numeric).
        nums = [int(x) for x in parts[1:] if x.lstrip("-").isdigit()]
        intervals = [
            (nums[i], nums[i + 1])
            for i in range(0, len(nums) - 1, 2)
            if nums[i] != -1 and nums[i + 1] != -1
        ]
        annotations[parts[0]] = intervals

    logger.info(f"Loaded annotations for {len(annotations)} videos")
    return annotations


def generate_frame_labels(total_frames: int, anomaly_intervals: list[tuple[int, int]]) -> np.ndarray:
    """Binary per-frame labels (1 = anomalous). Interval ends are inclusive."""
    labels = np.zeros(total_frames, dtype=np.int32)
    for start, end in anomaly_intervals:
        labels[max(0, start) : min(total_frames, end + 1)] = 1
    return labels


def score_video(
    video_path: str,
    scorer: ClipScorer,
    clip_length: int = 16,
    stride: int = 16,
) -> np.ndarray | None:
    """Per-frame anomaly scores for a video.

    A clip of ``clip_length`` consecutive frames is scored every ``stride``
    frames; each frame takes the max score of the clips covering it. Frames
    before the first full clip take the first clip's score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    scores: list[float] = []
    buffer: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(frame)
        buffer = buffer[-clip_length:]
        scores.append(np.nan)
        if len(buffer) == clip_length and (frame_idx + 1 - clip_length) % stride == 0:
            s = scorer(buffer)
            lo = frame_idx + 1 - clip_length
            window = np.array(scores[lo:], dtype=float)
            scores[lo:] = np.fmax(window, s).tolist()
        frame_idx += 1
    cap.release()

    if not scores:
        return None
    arr = np.array(scores, dtype=np.float32)
    # Fill trailing / leading frames not covered by any clip with the nearest score.
    valid = np.flatnonzero(~np.isnan(arr))
    if valid.size == 0:
        return None
    nearest = valid[np.clip(np.searchsorted(valid, np.arange(arr.size)), 0, valid.size - 1)]
    arr[np.isnan(arr)] = arr[nearest[np.isnan(arr)]]
    return arr


def motion_score(frames: list[np.ndarray]) -> float:
    """Optical-flow magnitude baseline (not a learned model)."""
    gray = [cv2.cvtColor(cv2.resize(f, (320, 240)), cv2.COLOR_BGR2GRAY) for f in frames]
    mags = []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(flow, axis=2).mean()))
    avg = float(np.mean(mags)) if mags else 0.0
    return float(1.0 / (1.0 + np.exp(-0.5 * (avg - 5.0))))


def find_video(dataset_path: Path, video_name: str) -> Path | None:
    direct = [
        dataset_path / video_name,
        dataset_path / "Videos" / video_name,
        dataset_path / "Test" / video_name,
    ]
    for candidate in direct:
        if candidate.exists():
            return candidate
    matches = list(dataset_path.rglob(video_name))
    return matches[0] if matches else None


def evaluate_dataset(
    dataset_path: str,
    scorer: ClipScorer,
    scorer_name: str,
    annotation_file: str = "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    output_dir: str = "results/",
    max_videos: int | None = None,
    clip_length: int = 16,
    stride: int = 16,
) -> dict:
    from evaluation.compute_metrics import compute_all_metrics

    dataset = Path(dataset_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_ucf_crime_annotations(str(dataset / annotation_file))
    items = list(annotations.items())[:max_videos] if max_videos else list(annotations.items())

    all_gt, all_pred, used, missing = [], [], [], []
    for name, intervals in tqdm(items, desc="Evaluating videos"):
        path = find_video(dataset, name)
        if path is None:
            missing.append(name)
            continue
        scores = score_video(str(path), scorer, clip_length=clip_length, stride=stride)
        if scores is None:
            missing.append(name)
            continue
        all_gt.append(generate_frame_labels(len(scores), intervals))
        all_pred.append(scores)
        used.append(name)

    if missing:
        logger.warning(f"{len(missing)} annotated videos missing or unreadable (first: {missing[:3]})")
    if not used:
        raise SystemExit("No test videos could be evaluated — check --dataset-path.")

    gt = np.concatenate(all_gt)
    pred = np.concatenate(all_pred)
    if gt.min() == gt.max():
        raise SystemExit("Evaluated videos contain only one class; ROC-AUC is undefined.")

    metrics = compute_all_metrics(gt, pred)
    results = {
        "dataset": "UCF-Crime test set",
        "scorer": scorer_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_videos": len(used),
        "num_missing_videos": len(missing),
        "total_frames": int(gt.size),
        "clip_length": clip_length,
        "stride": stride,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.success(f"Evaluation complete | {scorer_name} | ROC-AUC: {metrics['roc_auc']:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="UCF-Crime frame-level evaluation")
    parser.add_argument("--dataset-path", default="data/UCF-Crime/")
    parser.add_argument("--annotation-file", default="Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--stride", type=int, default=16, help="Frames between scored clips")
    parser.add_argument(
        "--baseline",
        choices=["motion"],
        default=None,
        help="Evaluate the optical-flow baseline instead of the trained classifier",
    )
    args = parser.parse_args()

    anomaly_cfg = load_config(args.config).get("anomaly", {})
    clip_length = anomaly_cfg.get("clip_length", 16)

    if args.baseline == "motion":
        scorer, name = motion_score, "optical-flow baseline"
    else:
        from src.anomaly_classifier import AnomalyClassifier

        classifier = AnomalyClassifier(anomaly_cfg)
        scorer = lambda clip: classifier.classify_clip(clip).anomaly_score  # noqa: E731
        name = f"VideoMAE ({classifier.backbone}) + MIL head ({anomaly_cfg.get('head_weights')})"

    evaluate_dataset(
        dataset_path=args.dataset_path,
        scorer=scorer,
        scorer_name=name,
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        clip_length=clip_length,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
