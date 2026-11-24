"""VideoMAE features + trained MIL head for clip-level anomaly scoring.

The classifier has two parts:

1. ``ClipFeatureExtractor`` — a *frozen* VideoMAE backbone (by default the
   Kinetics-400 fine-tuned checkpoint) that turns a 16-frame clip into a
   768-d embedding (mean-pooled patch tokens).
2. ``AnomalyHead`` — a small MLP that maps an embedding to an anomaly score
   in [0, 1]. It is trained with the MIL ranking loss of Sultani et al.
   (CVPR 2018) on UCF-Crime video-level labels; see ``training/train_mil.py``.

The head **must be trained** before anomaly scores mean anything. If no head
checkpoint is found, ``AnomalyClassifier`` raises ``AnomalyHeadNotFoundError``
and the pipeline runs detection + tracking only, rather than reporting scores
from randomly initialised weights.

Typical usage:
    classifier = AnomalyClassifier(config)          # loads backbone + head
    result = classifier.classify_clip(clip_frames)  # list of BGR frames
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

DEFAULT_BACKBONE = "MCG-NJU/videomae-base-finetuned-kinetics"
DEFAULT_HEAD_WEIGHTS = "weights/anomaly_head.pt"

# UCF-Crime categories (used for dataset parsing and reporting).
UCF_CRIME_CLASSES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


class AnomalyHeadNotFoundError(FileNotFoundError):
    """Raised when no trained anomaly head checkpoint is available."""


@dataclass
class AnomalyResult:
    """Output of a single clip classification.

    Attributes:
        anomaly_score: Score in [0, 1] from the trained head.
        is_anomaly:    ``anomaly_score >= threshold``.
        latency_ms:    Feature extraction + head inference time.
    """

    anomaly_score: float
    is_anomaly: bool
    latency_ms: float


# ── Model components ───────────────────────────────────────


class AnomalyHead(nn.Module):
    """MLP scoring head (Sultani et al. 2018 layout: D → 512 → 32 → 1)."""

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.6):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(..., D) → (...) scores in [0, 1]."""
        return self.net(features).squeeze(-1)


def mil_ranking_loss(
    anomalous_scores: torch.Tensor,
    normal_scores: torch.Tensor,
    lambda_smooth: float = 8e-5,
    lambda_sparse: float = 8e-5,
) -> torch.Tensor:
    """MIL ranking loss from Sultani et al., "Real-world Anomaly Detection in
    Surveillance Videos" (CVPR 2018).

    Args:
        anomalous_scores: (B, S) segment scores of anomalous videos (bags).
        normal_scores:    (B, S) segment scores of normal videos.

    Returns:
        Scalar loss.
    """
    max_anomalous = anomalous_scores.max(dim=1).values
    max_normal = normal_scores.max(dim=1).values
    hinge = torch.relu(1.0 - max_anomalous + max_normal)

    smoothness = ((anomalous_scores[:, 1:] - anomalous_scores[:, :-1]) ** 2).sum(dim=1)
    sparsity = anomalous_scores.sum(dim=1)

    return (hinge + lambda_smooth * smoothness + lambda_sparse * sparsity).mean()


def save_head(head: AnomalyHead, path: str | Path, backbone: str, extra: dict | None = None) -> None:
    """Save a trained head with the metadata needed to reload it safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "feature_dim": head.feature_dim,
            "hidden_dim": head.hidden_dim,
            "backbone": backbone,
            "extra": extra or {},
        },
        path,
    )


def load_head(path: str | Path, device: torch.device | str = "cpu") -> tuple[AnomalyHead, dict]:
    """Load a head checkpoint written by ``save_head``."""
    path = Path(path)
    if not path.exists():
        raise AnomalyHeadNotFoundError(
            f"No trained anomaly head at '{path}'. Train one with "
            "`python -m training.train_mil extract ...` then "
            "`python -m training.train_mil train ...` (see README)."
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    head = AnomalyHead(feature_dim=ckpt["feature_dim"], hidden_dim=ckpt["hidden_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.to(device).eval()
    return head, ckpt


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def normalize_clip_length(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Pad (repeat last frame) or uniformly subsample to exactly ``target`` frames."""
    if not frames:
        raise ValueError("Cannot classify an empty clip")
    n = len(frames)
    if n == target:
        return list(frames)
    if n < target:
        return list(frames) + [frames[-1]] * (target - n)
    indices = np.linspace(0, n - 1, target, dtype=int)
    return [frames[i] for i in indices]


class ClipFeatureExtractor:
    """Frozen VideoMAE backbone producing one embedding per clip.

    Args:
        backbone:    HuggingFace model ID or local path.
        device:      torch device.
        clip_length: Frames per clip expected by the backbone.
        model / processor: Optional pre-built objects (used by tests).
    """

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        device: torch.device | str = "cpu",
        clip_length: int = 16,
        model=None,
        processor=None,
    ):
        self.backbone = backbone
        self.device = torch.device(device)
        self.clip_length = clip_length

        if model is None or processor is None:
            from transformers import VideoMAEImageProcessor, VideoMAEModel

            logger.info(f"Loading VideoMAE backbone: {backbone}")
            processor = VideoMAEImageProcessor.from_pretrained(backbone)
            model = VideoMAEModel.from_pretrained(backbone)

        self.processor = processor
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def preprocess(self, clips: list[list[np.ndarray]]) -> torch.Tensor:
        """BGR clips → (B, T, C, H, W) pixel tensor."""
        videos = []
        for frames in clips:
            frames = normalize_clip_length(frames, self.clip_length)
            videos.append([np.ascontiguousarray(f[:, :, ::-1]) if f.ndim == 3 else f for f in frames])
        return self.processor(videos, return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def extract(self, clips: list[list[np.ndarray]]) -> np.ndarray:
        """Return (B, D) embeddings for a batch of clips."""
        pixel_values = self.preprocess(clips).to(self.device)
        hidden = self.model(pixel_values).last_hidden_state  # (B, tokens, D)
        return hidden.mean(dim=1).cpu().numpy()


class AnomalyClassifier:
    """Clip anomaly scorer = frozen VideoMAE features + trained MIL head.

    Config keys (``anomaly:`` section):
        backbone           HuggingFace ID of the VideoMAE backbone.
        head_weights       Path to the trained head checkpoint.
        clip_length        Frames per clip (16).
        anomaly_threshold  Score threshold for ``is_anomaly``.
        device             "auto" | "cpu" | "cuda:0".

    Raises:
        AnomalyHeadNotFoundError: if ``head_weights`` does not exist.
    """

    def __init__(self, config: dict, feature_extractor: ClipFeatureExtractor | None = None):
        self.config = config
        self.clip_length = config.get("clip_length", 16)
        self.anomaly_threshold = config.get("anomaly_threshold", 0.5)
        self.device = resolve_device(config.get("device", "auto"))
        backbone = config.get("backbone", DEFAULT_BACKBONE)

        # Load the head first: it is cheap and fails fast if missing.
        self.head, ckpt = load_head(config.get("head_weights", DEFAULT_HEAD_WEIGHTS), self.device)
        if ckpt.get("backbone") and ckpt["backbone"] != backbone:
            logger.warning(
                f"Head was trained on features from '{ckpt['backbone']}' but backbone "
                f"is '{backbone}'. Using the head's backbone."
            )
            backbone = ckpt["backbone"]
        self.backbone = backbone

        self.features = feature_extractor or ClipFeatureExtractor(
            backbone=backbone, device=self.device, clip_length=self.clip_length
        )
        if self.features.feature_dim != self.head.feature_dim:
            raise ValueError(
                f"Head expects {self.head.feature_dim}-d features, backbone gives "
                f"{self.features.feature_dim}-d"
            )

        self._latencies: list[float] = []
        logger.success(
            f"AnomalyClassifier ready | backbone={self.backbone} | device={self.device}"
        )

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Score pre-computed (N, D) embeddings."""
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            return self.head(x).cpu().numpy()

    def classify_clip(self, frames: list[np.ndarray]) -> AnomalyResult:
        """Score one clip of BGR frames."""
        t0 = time.perf_counter()
        score = float(self.score_features(self.features.extract([frames]))[0])
        elapsed = time.perf_counter() - t0
        self._latencies = (self._latencies + [elapsed])[-100:]
        return AnomalyResult(
            anomaly_score=score,
            is_anomaly=score >= self.anomaly_threshold,
            latency_ms=elapsed * 1000,
        )

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self._latencies)) * 1000 if self._latencies else 0.0
