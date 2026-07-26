"""Runtime configuration for the soccer-analysis pipeline.

Replaces the hardcoded paths/thresholds scattered across the original
tutorial modules with a single, overridable settings object: defaults live
in ``configs/default.yaml``, can be overridden by ``SOCCER_ANALYSIS_*``
environment variables, and are overridden again by explicit CLI flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CalibrationConfig(BaseModel):
    """Pixel-to-pitch homography inputs for one camera angle / video.

    The four pixel corner points are specific to a given broadcast camera
    setup and must be recalibrated per video (see
    ``configs/calibration/example_pitch.json`` for the tutorial's sample
    footage). This is a per-video static homography — see the project
    README for the known limitation on panning/zooming footage and the
    planned per-frame calibration upgrade.
    """

    pixel_vertices: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    court_length_m: float = 23.32
    court_width_m: float = 68.0

    @classmethod
    def from_json_file(cls, path: Path) -> CalibrationConfig:
        return cls.model_validate(json.loads(Path(path).read_text()))


class PipelineConfig(BaseModel):
    """Tunable thresholds for the detection/tracking/analytics stages."""

    detection_confidence: float = 0.1
    detection_batch_size: int = 20
    ball_max_assignment_distance_px: float = 70.0
    speed_frame_window: int = 5
    frame_rate: float = 24.0
    team_kmeans_random_state: int = 0
    device: str | None = None
    """Inference device: 'cuda', 'mps', 'cpu', or None to auto-detect."""
    calibration_mode: Literal["static", "dynamic"] = "static"
    """'static' = ViewTransformer (one homography from CalibrationConfig).
    'dynamic' = PitchKeypointCalibrator (per-frame center-circle detection,
    experimental classical-CV first cut -- see that class's docstring)."""
    team_classifier: Literal["pixel", "embedding"] = "pixel"
    """'pixel' = TeamAssigner (raw jersey-crop pixel KMeans).
    'embedding' = EmbeddingTeamClassifier (small-CNN-embedding KMeans;
    needs models/team_embedding.onnx -- see scripts/export_team_embedding_model.py)."""
    team_embedding_model_path: str = "models/team_embedding.onnx"


class Settings(BaseSettings):
    """Top-level, overridable settings for a pipeline run."""

    model_config = SettingsConfigDict(env_prefix="SOCCER_ANALYSIS_", env_nested_delimiter="__")

    pipeline: PipelineConfig = PipelineConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)


def resolve_device(preferred: str | None) -> str:
    """Pick a torch device for the UltralyticsDetector backend, auto-detecting
    CUDA/MPS/CPU when unset.

    Only relevant to the torch-backed backend (needs the ``[train]`` extra) —
    the default OnnxDetector backend doesn't use this. Degrades to "cpu" if
    torch isn't installed at all, rather than raising, since callers may
    invoke this speculatively before knowing which backend will be used.
    """
    if preferred is not None:
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
