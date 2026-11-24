"""Tests for anomaly scoring, MIL training, evaluation and pipeline logic.

Uses a tiny randomly initialised VideoMAE built locally, so no model download
is needed.
"""

import argparse
import json

import cv2
import numpy as np
import pytest
import torch

from src.anomaly_classifier import (
    AnomalyClassifier,
    AnomalyHead,
    AnomalyHeadNotFoundError,
    ClipFeatureExtractor,
    load_head,
    mil_ranking_loss,
    normalize_clip_length,
    save_head,
)


@pytest.fixture(scope="module")
def tiny_extractor():
    from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel

    cfg = VideoMAEConfig(
        image_size=32, patch_size=16, num_frames=4, tubelet_size=2,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
    )
    processor = VideoMAEImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    return ClipFeatureExtractor(backbone="tiny", clip_length=4, model=VideoMAEModel(cfg), processor=processor)


def _frames(n, h=48, w=64, value=None):
    rng = np.random.default_rng(0)
    return [
        np.full((h, w, 3), value, np.uint8) if value is not None else rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(n)
    ]


# ── Classifier ─────────────────────────────────────────────


class TestClassifier:
    def test_normalize_clip_length(self):
        assert len(normalize_clip_length(_frames(3), 4)) == 4
        assert len(normalize_clip_length(_frames(10), 4)) == 4
        with pytest.raises(ValueError):
            normalize_clip_length([], 4)

    def test_missing_head_raises(self, tmp_path):
        with pytest.raises(AnomalyHeadNotFoundError):
            AnomalyClassifier({"head_weights": str(tmp_path / "missing.pt"), "device": "cpu"})

    def test_head_roundtrip(self, tmp_path):
        head = AnomalyHead(feature_dim=32)
        save_head(head, tmp_path / "h.pt", backbone="tiny")
        loaded, ckpt = load_head(tmp_path / "h.pt")
        assert ckpt["backbone"] == "tiny"
        x = torch.randn(5, 32)
        head.eval()
        assert torch.allclose(head(x), loaded(x))

    def test_classify_clip(self, tmp_path, tiny_extractor):
        save_head(AnomalyHead(feature_dim=32), tmp_path / "h.pt", backbone="tiny")
        clf = AnomalyClassifier(
            {"head_weights": str(tmp_path / "h.pt"), "backbone": "tiny", "clip_length": 4,
             "device": "cpu", "anomaly_threshold": 0.5},
            feature_extractor=tiny_extractor,
        )
        result = clf.classify_clip(_frames(6))
        assert 0.0 <= result.anomaly_score <= 1.0
        assert result.is_anomaly == (result.anomaly_score >= 0.5)

    def test_feature_dim_mismatch(self, tmp_path, tiny_extractor):
        save_head(AnomalyHead(feature_dim=64), tmp_path / "h.pt", backbone="tiny")
        with pytest.raises(ValueError):
            AnomalyClassifier({"head_weights": str(tmp_path / "h.pt"), "backbone": "tiny", "device": "cpu"},
                              feature_extractor=tiny_extractor)


# ── MIL training ───────────────────────────────────────────


class TestMIL:
    def test_loss_prefers_ranked_bags(self):
        good = mil_ranking_loss(torch.tensor([[0.1, 0.9]]), torch.tensor([[0.1, 0.1]]))
        bad = mil_ranking_loss(torch.tensor([[0.1, 0.1]]), torch.tensor([[0.1, 0.9]]))
        assert good < bad

    def test_segment_clip_indices(self):
        from training.train_mil import segment_clip_indices

        clips = segment_clip_indices(total_frames=100, num_segments=4, clip_length=16)
        assert len(clips) == 4
        assert all(len(c) == 16 and c == list(range(c[0], c[0] + 16)) for c in clips)
        assert all(0 <= i < 100 for c in clips for i in c)
        short = segment_clip_indices(total_frames=5, num_segments=2, clip_length=16)
        assert all(0 <= i < 5 for c in short for i in c)

    def test_video_label(self):
        from training.train_mil import video_label

        assert video_label("Abuse/Abuse001_x264.mp4") == 1
        assert video_label("Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4") == 0

    def test_train_learns_separable_bags(self, tmp_path):
        """On separable synthetic features the trained head must rank anomalies higher."""
        from training.train_mil import train

        rng = np.random.default_rng(0)
        dim, segs = 16, 8
        index = {}
        for i in range(20):
            label = i % 2
            bag = rng.normal(0, 1, (segs, dim)).astype(np.float32)
            if label:
                bag[3, 0] += 6.0  # one anomalous segment per anomalous video
            np.save(tmp_path / f"v{i}.npy", bag)
            index[f"v{i}"] = {"video": f"v{i}", "label": label}
        (tmp_path / "index.json").write_text(json.dumps(index))
        (tmp_path / "meta.json").write_text(json.dumps({"backbone": "tiny"}))

        out = tmp_path / "head.pt"
        train(argparse.Namespace(
            features=str(tmp_path), out=str(out), backbone="tiny", iterations=300, batch_size=8,
            lr=0.05, weight_decay=0.0, log_every=1000, device="cpu", seed=0,
        ))
        head, ckpt = load_head(out)
        assert ckpt["backbone"] == "tiny"
        anomalous = torch.zeros(1, dim)
        anomalous[0, 0] = 6.0
        assert head(anomalous).item() > head(torch.zeros(1, dim)).item()


# ── Evaluation ─────────────────────────────────────────────


class TestEvaluation:
    def test_annotations_with_category_column(self, tmp_path):
        from evaluation.evaluate_ucf_crime import load_ucf_crime_annotations

        f = tmp_path / "ann.txt"
        f.write_text(
            "Abuse028_x264.mp4  Abuse  165  240  -1  -1\n"
            "Arson011_x264.mp4  Arson  150  420  680  1267\n"
            "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n"
        )
        ann = load_ucf_crime_annotations(str(f))
        assert ann["Abuse028_x264.mp4"] == [(165, 240)]
        assert ann["Arson011_x264.mp4"] == [(150, 420), (680, 1267)]
        assert ann["Normal_Videos_003_x264.mp4"] == []

    def test_missing_annotations_fail_loudly(self, tmp_path):
        from evaluation.evaluate_ucf_crime import load_ucf_crime_annotations

        with pytest.raises(FileNotFoundError):
            load_ucf_crime_annotations(str(tmp_path / "nope.txt"))

    def test_frame_labels_inclusive(self):
        from evaluation.evaluate_ucf_crime import generate_frame_labels

        labels = generate_frame_labels(10, [(2, 4)])
        assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

    def test_score_video_covers_every_frame(self, tmp_path):
        from evaluation.evaluate_ucf_crime import score_video

        path = str(tmp_path / "v.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
        for i in range(40):
            writer.write(np.full((32, 32, 3), i * 5, np.uint8))
        writer.release()

        calls = []

        def scorer(clip):
            calls.append(len(clip))
            return float(clip[-1].mean() / 255)

        scores = score_video(path, scorer, clip_length=8, stride=8)
        assert scores is not None and len(scores) == 40
        assert not np.isnan(scores).any()
        assert all(n == 8 for n in calls) and len(calls) == 5


# ── Pipeline logic (fake tracker / classifier) ─────────────


class _FakeTrack:
    def __init__(self, tid, bbox):
        self.track_id = tid
        self.bbox = np.array(bbox, dtype=float)
        self.class_name = "person"
        self.frame_count = 100
        self.trajectory = []


class _FakeTracker:
    track_buffer = 5
    min_track_length = 1

    def __init__(self):
        self.next_tracks = []

    def update(self, frame):
        return self.next_tracks

    @property
    def num_active_tracks(self):
        return len(self.next_tracks)

    @property
    def total_tracks(self):
        return 2


class _FakeClassifier:
    def __init__(self, scores):
        self.scores = scores  # key -> score; scene uses -1
        self.calls = 0

    def classify_clip(self, frames):
        from src.anomaly_classifier import AnomalyResult

        self.calls += 1
        s = self.scores
        return AnomalyResult(anomaly_score=s, is_anomaly=s >= 0.5, latency_ms=0.0)


def _pipeline(tmp_path, mode, classifier):
    from src.pipeline import AnomalyDetectionPipeline

    cfg = {
        "anomaly": {"mode": mode, "clip_length": 4, "stride": 2, "anomaly_threshold": 0.5},
        "pipeline": {"show_display": False, "save_output": False, "output_path": str(tmp_path)},
    }
    return AnomalyDetectionPipeline(cfg, tracker=_FakeTracker(), classifier=classifier)


class TestPipeline:
    def test_disabled_scoring_still_annotates(self, tmp_path):
        pipe = _pipeline(tmp_path, "frame", classifier=False)
        out = pipe.process_frame(np.zeros((120, 160, 3), np.uint8))
        assert out.shape == (120, 160, 3)
        assert pipe._generate_stats()["anomaly_scoring"] == "disabled"

    def test_frame_mode_alerts_once_per_event(self, tmp_path):
        clf = _FakeClassifier(0.9)
        pipe = _pipeline(tmp_path, "frame", clf)
        for _ in range(20):
            pipe.process_frame(np.zeros((120, 160, 3), np.uint8))
        assert clf.calls > 1
        assert len(pipe._alert_log) == 1  # sustained anomaly → one alert
        clf.scores = 0.1
        for _ in range(4):
            pipe.process_frame(np.zeros((120, 160, 3), np.uint8))
        clf.scores = 0.9
        for _ in range(4):
            pipe.process_frame(np.zeros((120, 160, 3), np.uint8))
        assert len(pipe._alert_log) == 2  # new event after dropping below threshold

    def test_track_state_pruned_after_track_buffer(self, tmp_path):
        pipe = _pipeline(tmp_path, "track", _FakeClassifier(0.9))
        frame = np.zeros((120, 160, 3), np.uint8)
        pipe.tracker.next_tracks = [_FakeTrack(7, [10, 10, 60, 80])]
        for _ in range(6):
            pipe.process_frame(frame)
        assert 7 in pipe._clips and 7 in pipe._scores

        pipe.tracker.next_tracks = []
        for _ in range(pipe.track_buffer + 2):
            pipe.process_frame(frame)
        assert 7 not in pipe._clips and 7 not in pipe._scores

    def test_invalid_mode(self, tmp_path):
        with pytest.raises(ValueError):
            _pipeline(tmp_path, "bogus", classifier=False)


# ── Utilities ──────────────────────────────────────────────


def test_redact_url():
    from src.utils import redact_url

    assert redact_url("rtsp://admin:secret@10.0.0.2:554/s") == "rtsp://***@10.0.0.2:554/s"
    assert redact_url("rtsp://camera.local/stream") == "rtsp://camera.local/stream"
    assert redact_url(0) == "0"


def test_live_read_returns_each_frame_once():
    from src.video_io import VideoSource

    src = VideoSource(source="rtsp://camera.local/stream")
    src._buffer.extend([np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)])
    first = src.read()
    assert first is not None and first.max() == 1  # newest frame
    assert src.read() is None  # not returned again
