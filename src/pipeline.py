"""End-to-end video anomaly detection pipeline.

    VideoSource → YOLOv8 + ByteTrack (detect & track) → anomaly scoring → overlays → display/save

Anomaly scoring uses a frozen VideoMAE backbone plus a MIL head that you train
on UCF-Crime (``training/train_mil.py``). Two scoring modes:

* ``frame`` (default) — score clips of whole frames. This matches how the head
  is trained (UCF-Crime labels are per video, not per object).
* ``track`` — score clips cropped around each tracked object. Experimental:
  the head never saw object crops during training.

If no trained head is available the pipeline still runs detection and
tracking, and the overlay says that anomaly scoring is off.

Usage:
    python -m src.pipeline --source video.mp4 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.anomaly_classifier import AnomalyClassifier, AnomalyHeadNotFoundError
from src.heatmap import (
    draw_alert_banner,
    draw_detections,
    draw_fps_overlay,
    draw_status_text,
    generate_anomaly_heatmap,
)
from src.tracker import MultiObjectTracker
from src.utils import FPSCounter, LatencyTracker, load_config, redact_url, setup_logging
from src.video_io import VideoSource, VideoWriter

SCENE_ID = -1  # key used for the whole-frame score in frame mode


class AnomalyDetectionPipeline:
    """Detection + tracking + (optional) anomaly scoring.

    Args:
        config:     Full pipeline configuration dictionary.
        tracker:    Optional pre-built tracker (for tests).
        classifier: Optional pre-built classifier (for tests). Pass ``False``
                    to force anomaly scoring off.
    """

    def __init__(self, config: dict, tracker=None, classifier=None):
        self.config = config
        setup_logging(level=config.get("pipeline", {}).get("log_level", "INFO"))

        anomaly_cfg = config.get("anomaly", {})
        self.mode = anomaly_cfg.get("mode", "frame")
        if self.mode not in ("frame", "track"):
            raise ValueError(f"anomaly.mode must be 'frame' or 'track', got {self.mode!r}")
        self.clip_length = anomaly_cfg.get("clip_length", 16)
        self.stride = max(1, anomaly_cfg.get("stride", 8))
        self.anomaly_threshold = anomaly_cfg.get("anomaly_threshold", 0.5)

        self.tracker = tracker or MultiObjectTracker(
            config=config.get("tracker", {}),
            detector_config=config.get("detector", {}),
        )
        self.track_buffer = self.tracker.track_buffer

        self.classifier: AnomalyClassifier | None
        self.anomaly_status: str
        if classifier is False:
            self.classifier, self.anomaly_status = None, "disabled"
        elif classifier is not None:
            self.classifier, self.anomaly_status = classifier, "loaded"
        else:
            try:
                self.classifier = AnomalyClassifier(anomaly_cfg)
                self.anomaly_status = "loaded"
            except AnomalyHeadNotFoundError as e:
                logger.warning(f"Anomaly scoring disabled: {e}")
                self.classifier, self.anomaly_status = None, "no trained head"

        # Per-key state; key is a track id, or SCENE_ID in frame mode.
        self._clips: dict[int, deque[np.ndarray]] = {}
        self._scores: dict[int, float] = {}
        self._last_seen: dict[int, int] = {}
        self._in_alert: set[int] = set()

        self._fps_counter = FPSCounter(smoothing=0.05)
        self._latency_tracker = LatencyTracker()
        self._alert_log: list[dict] = []
        self._classification_errors = 0
        self._frame_idx = 0
        self._wall_start: float | None = None

        pipe_cfg = config.get("pipeline", {})
        self.show_display = pipe_cfg.get("show_display", True)
        self.save_output = pipe_cfg.get("save_output", True)
        self.output_path = pipe_cfg.get("output_path", "outputs/")
        self.viz_config = config.get("visualization", {})

        logger.success(
            f"Pipeline initialized | anomaly scoring: {self.anomaly_status} | mode={self.mode}"
        )

    # ── Public API ─────────────────────────────────────────

    def run(self, source: int | str | None = None) -> dict:
        """Run on a webcam index, stream URL or file path; returns run statistics."""
        if source is None:
            source = self.config.get("video", {}).get("source", 0)

        video_cfg = self.config.get("video", {})
        video_source = VideoSource(
            source=source,
            width=video_cfg.get("width", 1280),
            height=video_cfg.get("height", 720),
            fps=video_cfg.get("fps", 30),
            buffer_size=video_cfg.get("buffer_size", 64),
        )

        writer = None
        logger.info(f"Starting pipeline | source={redact_url(source)}")
        self._wall_start = time.perf_counter()

        try:
            video_source.start()

            if self.save_output:
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
                w, h = video_source.resolution
                writer = VideoWriter(
                    output_path=str(Path(self.output_path) / "output.mp4"),
                    fps=video_source.actual_fps,
                    width=w,
                    height=h,
                )

            while video_source.is_running:
                frame = video_source.read()
                if frame is None:
                    time.sleep(0.001)
                    continue

                if self._frame_idx % 100 == 0:
                    logger.info(f"Processing frame {self._frame_idx}")

                annotated = self.process_frame(frame)

                if self.show_display:
                    cv2.imshow("Anomaly Detection Pipeline", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logger.info("User requested stop")
                        break

                if writer is not None:
                    writer.write(annotated)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            video_source.stop()
            if writer is not None:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

        stats = self._generate_stats()
        logger.info(f"Pipeline finished | {stats['total_frames']} frames processed")
        return stats

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect, track, score and annotate one BGR frame."""
        self._frame_idx += 1
        self._fps_counter.tick()

        t0 = time.perf_counter()
        tracks = self.tracker.update(frame)
        track_ms = (time.perf_counter() - t0) * 1000
        self._latency_tracker.record("detection+tracking", track_ms)

        classifier_ms = 0.0
        if self.classifier is not None:
            self._buffer_clips(frame, tracks)
            if self._frame_idx % self.stride == 0:
                t1 = time.perf_counter()
                self._score_clips(tracks)
                classifier_ms = (time.perf_counter() - t1) * 1000
                self._latency_tracker.record("classification", classifier_ms)
            self._prune_stale()

        return self._annotate(frame, tracks, track_ms, classifier_ms)

    # ── Scoring ────────────────────────────────────────────

    def _keys_for(self, tracks: list) -> list[int]:
        return [SCENE_ID] if self.mode == "frame" else [t.track_id for t in tracks]

    def _buffer_clips(self, frame: np.ndarray, tracks: list) -> None:
        if self.mode == "frame":
            self._push(SCENE_ID, frame)
            return

        h, w = frame.shape[:2]
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                self._push(track.track_id, frame[y1:y2, x1:x2].copy())

    def _push(self, key: int, image: np.ndarray) -> None:
        if key not in self._clips:
            self._clips[key] = deque(maxlen=self.clip_length)
        self._clips[key].append(image)
        self._last_seen[key] = self._frame_idx

    def _score_clips(self, tracks: list) -> None:
        mature = {t.track_id for t in tracks if t.frame_count >= self.tracker.min_track_length}
        for key in self._keys_for(tracks):
            clip = self._clips.get(key)
            if clip is None or len(clip) < self.clip_length:
                continue
            if key != SCENE_ID and key not in mature:
                continue
            try:
                result = self.classifier.classify_clip(list(clip))
            except Exception:
                self._classification_errors += 1
                if self._classification_errors <= 3:
                    logger.exception(f"Classification failed for key {key}")
                continue
            self._scores[key] = result.anomaly_score
            self._update_alert_state(key, result.anomaly_score)

    def _update_alert_state(self, key: int, score: float) -> None:
        """Log one alert when a key crosses the threshold, not on every scoring step."""
        if score >= self.anomaly_threshold and key not in self._in_alert:
            self._in_alert.add(key)
            alert = {
                "frame_idx": self._frame_idx,
                "target": "scene" if key == SCENE_ID else f"track {key}",
                "score": round(score, 4),
                "timestamp": time.time(),
            }
            self._alert_log.append(alert)
            logger.warning(f"ANOMALY | {alert['target']} | score={score:.3f}")
        elif score < self.anomaly_threshold:
            self._in_alert.discard(key)

    def _prune_stale(self) -> None:
        """Drop buffers and scores of tracks unseen for longer than the track buffer."""
        cutoff = self._frame_idx - self.track_buffer
        for key in [k for k, seen in self._last_seen.items() if seen < cutoff]:
            self._clips.pop(key, None)
            self._scores.pop(key, None)
            self._last_seen.pop(key, None)
            self._in_alert.discard(key)

    # ── Visualization ──────────────────────────────────────

    def _annotate(self, frame, tracks, track_ms: float, classifier_ms: float) -> np.ndarray:
        annotated = frame
        visible = {t.track_id for t in tracks}
        track_scores = {k: v for k, v in self._scores.items() if k in visible}

        if self.viz_config.get("show_heatmap", True) and track_scores:
            annotated = generate_anomaly_heatmap(
                annotated, tracks, track_scores, alpha=self.viz_config.get("heatmap_alpha", 0.4)
            )

        annotated = draw_detections(
            annotated,
            tracks,
            anomaly_scores=track_scores,
            config={**self.viz_config, "anomaly_threshold": self.anomaly_threshold},
        )

        if self.viz_config.get("show_fps", True):
            annotated = draw_fps_overlay(
                annotated,
                fps=self._fps_counter.fps,
                detector_ms=track_ms,
                tracker_ms=0,  # included in detector_ms (YOLO .track())
                classifier_ms=classifier_ms,
                num_tracks=self.tracker.num_active_tracks,
            )

        if self.classifier is None:
            annotated = draw_status_text(annotated, f"Anomaly scoring OFF ({self.anomaly_status})")
        elif self.mode == "frame" and SCENE_ID in self._scores:
            annotated = draw_status_text(annotated, f"Scene anomaly score: {self._scores[SCENE_ID]:.2f}")

        # Banner only for things currently in view.
        active = [(k, s) for k, s in self._scores.items() if s >= self.anomaly_threshold and (k == SCENE_ID or k in visible)]
        if active:
            key, score = max(active, key=lambda x: x[1])
            target = "Scene" if key == SCENE_ID else f"Track #{key}"
            annotated = draw_alert_banner(annotated, f"{target} - Score: {score:.2f}", severity="critical")

        return annotated

    # ── Stats ──────────────────────────────────────────────

    def _generate_stats(self) -> dict:
        wall = time.perf_counter() - self._wall_start if self._wall_start else 0.0
        stats = {
            "total_frames": self._frame_idx,
            "avg_fps": self._frame_idx / wall if wall > 0 else 0.0,
            "total_tracks": self.tracker.total_tracks,
            "anomaly_scoring": self.anomaly_status,
            "anomaly_mode": self.mode,
            "total_alerts": len(self._alert_log),
            "classification_errors": self._classification_errors,
            "latency": self._latency_tracker.summary(),
            "alerts": self._alert_log[-50:],
        }
        stats_path = Path(self.output_path) / "pipeline_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, default=str))
        logger.info(f"Stats saved to {stats_path}")
        return stats


# ── CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Video anomaly detection pipeline")
    parser.add_argument("--source", default="0", help="Webcam index, RTSP URL, or file path")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (no window)")
    parser.add_argument("--no-save", action="store_true", help="Do not write the annotated video")
    parser.add_argument("--output-path", default="outputs/")
    parser.add_argument("--hide-fps", action="store_true", help="Hide the FPS overlay")
    parser.add_argument("--mode", choices=["frame", "track"], help="Override anomaly.mode")
    args = parser.parse_args()

    config = load_config(args.config)
    pipe = config.setdefault("pipeline", {})
    if args.no_display:
        pipe["show_display"] = False
    pipe["save_output"] = not args.no_save
    pipe["output_path"] = args.output_path
    config.setdefault("visualization", {})["show_fps"] = not args.hide_fps
    if args.mode:
        config.setdefault("anomaly", {})["mode"] = args.mode

    source: int | str = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    stats = AnomalyDetectionPipeline(config).run(source=source)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total Frames:     {stats['total_frames']}")
    print(f"  Average FPS:      {stats['avg_fps']:.1f}")
    print(f"  Total Tracks:     {stats['total_tracks']}")
    print(f"  Anomaly scoring:  {stats['anomaly_scoring']}")
    print(f"  Total Alerts:     {stats['total_alerts']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
