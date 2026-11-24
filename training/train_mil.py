"""Train the anomaly head on UCF-Crime with the MIL ranking loss.

Two steps:

1. **extract** — split each training video into ``--segments`` (32) equal
   temporal segments, take a contiguous ``clip_length``-frame clip from the
   centre of each segment, and save its frozen VideoMAE embedding.
   Output: one ``<video>.npy`` of shape (segments, D) per video, plus
   ``index.json`` with the video-level labels.

2. **train** — train ``AnomalyHead`` with the Sultani et al. MIL ranking loss
   (anomalous bag max score > normal bag max score) and save it to
   ``weights/anomaly_head.pt``.

Usage:
    python -m training.train_mil extract \
        --dataset-path data/UCF-Crime --list Anomaly_Train.txt \
        --out data/features/train

    python -m training.train_mil train \
        --features data/features/train --out weights/anomaly_head.pt

``Anomaly_Train.txt`` is the official UCF-Crime training split: one relative
video path per line, e.g. ``Abuse/Abuse001_x264.mp4`` or
``Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.anomaly_classifier import (
    DEFAULT_BACKBONE,
    DEFAULT_HEAD_WEIGHTS,
    AnomalyHead,
    ClipFeatureExtractor,
    mil_ranking_loss,
    resolve_device,
    save_head,
)

# ── Feature extraction ─────────────────────────────────────


def segment_clip_indices(total_frames: int, num_segments: int, clip_length: int) -> list[list[int]]:
    """Frame indices of a contiguous clip centred in each of ``num_segments`` segments."""
    if total_frames <= 0:
        return []
    bounds = np.linspace(0, total_frames, num_segments + 1)
    clips = []
    for s in range(num_segments):
        centre = int((bounds[s] + bounds[s + 1]) / 2)
        start = min(max(0, centre - clip_length // 2), max(0, total_frames - clip_length))
        clips.append([min(start + i, total_frames - 1) for i in range(clip_length)])
    return clips


def read_frames(video_path: str, wanted: set[int], max_side: int = 320) -> dict[int, np.ndarray]:
    """Read only the wanted frame indices, sequentially (seeking is unreliable)."""
    cap = cv2.VideoCapture(video_path)
    frames: dict[int, np.ndarray] = {}
    last = max(wanted) if wanted else -1
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames[idx] = frame
        idx += 1
    cap.release()
    return frames


def nearest_frame(frames: dict[int, np.ndarray], index: int) -> np.ndarray:
    """Frame at ``index``, or the nearest earlier decoded one (or the first)."""
    decoded = sorted(frames)
    pos = int(np.searchsorted(decoded, index, side="right")) - 1
    return frames[decoded[max(pos, 0)]]


def video_label(rel_path: str) -> int:
    """1 = anomalous video, 0 = normal (UCF-Crime naming convention)."""
    return 0 if "normal" in rel_path.lower() else 1


def extract(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = Path(args.list)
    if not list_path.is_absolute():
        list_path = dataset / list_path
    videos = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if args.max_videos:
        videos = videos[: args.max_videos]

    extractor = ClipFeatureExtractor(
        backbone=args.backbone, device=resolve_device(args.device), clip_length=args.clip_length
    )

    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for rel in tqdm(videos, desc="Extracting features"):
        key = rel.replace("/", "__").replace("\\", "__")
        feat_path = out_dir / f"{key}.npy"
        if feat_path.exists():
            index[key] = {"video": rel, "label": video_label(rel)}
            continue

        video_path = dataset / rel
        if not video_path.exists():
            logger.warning(f"Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        clips_idx = segment_clip_indices(total, args.segments, args.clip_length)
        if not clips_idx:
            logger.warning(f"Unreadable video: {video_path}")
            continue

        frames = read_frames(str(video_path), {i for c in clips_idx for i in c})
        if not frames:
            logger.warning(f"No frames decoded: {video_path}")
            continue
        # Header frame counts can overstate the real length: substitute the
        # nearest earlier decoded frame for any index that was not decoded.
        clips = [[nearest_frame(frames, i) for i in c] for c in clips_idx]

        feats = []
        for b in range(0, len(clips), args.batch_size):
            feats.append(extractor.extract(clips[b : b + args.batch_size]))
        np.save(feat_path, np.concatenate(feats).astype(np.float32))
        index[key] = {"video": rel, "label": video_label(rel)}
        index_path.write_text(json.dumps(index, indent=1))

    index_path.write_text(json.dumps(index, indent=1))
    meta = {"backbone": args.backbone, "segments": args.segments, "clip_length": args.clip_length}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    n_anom = sum(v["label"] for v in index.values())
    logger.success(f"Features for {len(index)} videos ({n_anom} anomalous) in {out_dir}")


# ── Training ───────────────────────────────────────────────


def load_bags(features_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    index = json.loads((features_dir / "index.json").read_text())
    meta_path = features_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    anomalous, normal = [], []
    for key, entry in index.items():
        bag = np.load(features_dir / f"{key}.npy")
        (anomalous if entry["label"] == 1 else normal).append(bag)
    return anomalous, normal, meta


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    features_dir = Path(args.features)
    anomalous, normal, meta = load_bags(features_dir)
    if not anomalous or not normal:
        raise SystemExit(
            f"Need both anomalous and normal bags; found {len(anomalous)} / {len(normal)}"
        )
    backbone = meta.get("backbone", args.backbone)
    feature_dim = anomalous[0].shape[1]
    device = resolve_device(args.device)

    head = AnomalyHead(feature_dim=feature_dim).to(device)
    optim = torch.optim.Adagrad(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(
        f"Training on {len(anomalous)} anomalous / {len(normal)} normal bags | "
        f"dim={feature_dim} | device={device}"
    )
    head.train()
    for it in range(1, args.iterations + 1):
        a = torch.as_tensor(np.stack(random.choices(anomalous, k=args.batch_size)), device=device)
        n = torch.as_tensor(np.stack(random.choices(normal, k=args.batch_size)), device=device)
        loss = mil_ranking_loss(head(a), head(n))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % args.log_every == 0 or it == 1:
            logger.info(f"iter {it:>6} | loss {loss.item():.4f}")

    head.eval()
    save_head(
        head,
        args.out,
        backbone=backbone,
        extra={
            "iterations": args.iterations,
            "num_anomalous": len(anomalous),
            "num_normal": len(normal),
            "segments": meta.get("segments"),
            "clip_length": meta.get("clip_length"),
        },
    )
    logger.success(f"Saved trained head to {args.out}")


# ── CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MIL anomaly head on UCF-Crime")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="Extract per-segment VideoMAE features")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--list", default="Anomaly_Train.txt", help="Video list (relative to dataset path)")
    p.add_argument("--out", required=True, help="Output feature directory")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--segments", type=int, default=32)
    p.add_argument("--clip-length", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-videos", type=int, default=None)
    p.set_defaults(func=extract)

    t = sub.add_parser("train", help="Train the anomaly head")
    t.add_argument("--features", required=True)
    t.add_argument("--out", default=DEFAULT_HEAD_WEIGHTS)
    t.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Used only if meta.json is missing")
    t.add_argument("--iterations", type=int, default=20000)
    t.add_argument("--batch-size", type=int, default=30)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--weight-decay", type=float, default=1e-3)
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
