"""Runtime configuration for the agon pipeline.

Replaces the hardcoded paths/thresholds scattered across the original
tutorial modules with a single, overridable settings object: defaults live
in ``configs/default.yaml``, can be overridden by ``AGON_*``
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


class ClockCalibrationConfig(BaseModel):
    """Pixel region of the broadcast's on-screen match-clock graphic, for
    ``agon.broadcast.clock_reader.ClockReader``.

    Like ``CalibrationConfig``, this is specific to one broadcast's graphics
    layout (position, size, font) and needs recalibrating per broadcast
    source -- there's no universal "the clock is always here" answer across
    different broadcasters' scorebug designs.
    """

    clock_region_px: tuple[int, int, int, int] = Field(
        description="x1, y1, x2, y2 pixel box tightly cropping just the "
        "MM:SS digits (not the surrounding scorebug chrome/logos), from a "
        "frame where the clock is clearly visible."
    )

    @classmethod
    def from_json_file(cls, path: Path) -> ClockCalibrationConfig:
        return cls.model_validate(json.loads(Path(path).read_text()))


class PipelineConfig(BaseModel):
    """Tunable thresholds for the detection/tracking/analytics stages."""

    detection_confidence: float = 0.1
    detection_batch_size: int = 20
    detection_imgsz: int = 640
    """Inference resolution (square) fed to the detector. Must match (or be
    a deliberate choice relative to) whatever resolution the loaded
    checkpoint was actually trained/exported at -- a checkpoint trained at
    1280 but run through the pipeline at the default 640 silently loses
    the entire point of training at higher resolution (this was a real
    bug: OnnxDetector always supported a configurable input_size, but
    nothing above it in the pipeline ever passed one through). For a
    ``.onnx`` checkpoint specifically, a mismatch isn't silent -- an ONNX
    model exported with fixed (non-dynamic) input dims raises a clear
    onnxruntime shape error rather than misbehaving quietly (confirmed:
    running the default 640-trained models/yolo11n.onnx at
    detection_imgsz=960 fails loudly, not silently) -- but the checkpoint
    still needs re-exporting at the new resolution for this setting to do
    anything, it won't magically upscale a 640-trained model's accuracy."""
    ball_max_assignment_distance_px: float = 70.0
    speed_frame_window: int = 5
    frame_rate: float | None = None
    """Frames per second, used for speed-km/h conversion, export
    timestamps, and the annotated video's playback fps. None (default) =
    auto-detect from the input video's own reported fps -- set explicitly
    only to override (e.g. a source that misreports its own fps as 0)."""
    team_kmeans_random_state: int = 0
    device: str | None = None
    """Inference device: 'cuda', 'mps', 'cpu', or None to auto-detect."""
    calibration_mode: Literal["static", "dynamic", "hybrid"] = "static"
    """'static' = ViewTransformer (one homography from CalibrationConfig).
    'dynamic' = PitchKeypointCalibrator (per-frame center-circle detection,
    experimental classical-CV first cut -- see that class's docstring).
    'hybrid' = HybridPitchCalibrator (tries dynamic first, falls back to
    static per point) -- measured at 73.5% position coverage on a real
    clip vs. 34.4%/56.9% for static/dynamic alone, since the two mostly
    fail on different frames rather than the same ones. Recommended over
    either alone when a calibration file is available (still needs one,
    for the fallback)."""
    team_classifier: Literal["pixel", "embedding"] = "pixel"
    """'pixel' = TeamAssigner (raw jersey-crop pixel KMeans).
    'embedding' = EmbeddingTeamClassifier (small-CNN-embedding KMeans;
    needs models/team_embedding.onnx -- see scripts/export_team_embedding_model.py)."""
    team_embedding_model_path: str = "models/team_embedding.onnx"
    tracker_backend: Literal["bytetrack", "botsort"] = "bytetrack"
    """'bytetrack' = supervision.ByteTrack (default, no extra deps).
    'botsort' = BoTSORTTracker (Kalman motion model + camera-motion
    compensation; needs the [train] extra + boxmot -- see that module's
    docstring)."""
    frame_filter_mode: Literal["off", "tag", "strip"] = "off"
    """'off' (default): no change in behavior. 'tag': classify every frame
    (see agon.broadcast.frame_filter) and attach
    frame_classification/game_clock_s to the export, but still process
    every frame. 'strip': additionally drop non-live-play frames (ads,
    replays, graphics) before detection/tracking/video output -- see
    run_pipeline's docstring for what "dropped" means for frame numbering."""
    min_grass_fraction: float = 0.35
    """Below this fraction of pitch-green pixels, a frame is classified as
    a graphic/ad/interstitial rather than live pitch footage. See
    agon.broadcast.frame_filter's docstring for why this
    threshold alone can't distinguish live play from a replay (both show
    the pitch) -- that distinction needs clock_calibration below."""
    clock_calibration_path: str | None = None
    """Path to a ClockCalibrationConfig JSON (see that class). Required for
    game_clock_s tagging and for telling replays apart from live play in
    frame_filter_mode; without it, frame_filter can only distinguish
    graphics/ads (no pitch visible) from everything else."""
    jersey_backend: Literal["off", "ocr", "onnx"] = "off"
    """Which per-frame jersey-number reader to use, if any -- output
    always goes through agon.jersey.aggregator before being trusted (see
    that module's docstring). 'off' (default): jersey classification is
    skipped entirely and every ObjectRecord.jersey_number stays null.
    'ocr' (recommended): agon.jersey.ocr_reader.EasyOcrJerseyReader --
    pretrained scene-text OCR, no jersey_model_path needed, needs the
    [train] extra (pulls in torch via easyocr). 'onnx':
    agon.jersey.onnx_classifier.OnnxJerseyClassifier, a from-scratch
    classifier trained on this project's own crops -- requires
    jersey_model_path. Not recommended: empirically scored *worse* than a
    trivial majority-class baseline (2.4% vs 16.4% top1 accuracy) because
    SN-GSR-2025's jersey label is assigned per track, not per frame,
    poisoning a large fraction of the training crops with a confident
    label the image doesn't actually show -- see agon.jersey.ocr_reader's
    docstring for the full diagnosis. Kept available for anyone with an
    existing checkpoint, not as a recommended default."""
    jersey_model_path: str | None = None
    """Path to an ONNX jersey-number classifier exported by
    scripts/train_jersey_classifier.py --export-onnx (a classes.json
    sidecar must sit alongside it). Only used when jersey_backend='onnx';
    required in that case."""
    jersey_min_confidence: float = 0.5
    """Per-frame confidence threshold applied during track-level
    aggregation (see agon.jersey.aggregator) -- frames below this are
    dropped before voting, not just down-weighted. Single-frame jersey
    reading is unreliable by the underlying task's own nature (see that
    module's docstring), so this default is deliberately not permissive."""
    jersey_min_votes: int = 2
    """Minimum number of separate frames that must agree on a track's
    winning jersey number before it's trusted (agon.jersey.aggregator's
    min_votes) -- a real, observed failure mode motivated this: a
    93%-confidence single-frame misread during validation, where
    min_confidence alone wouldn't have caught it. Requiring corroboration
    from a second frame is cheap for any track that appears in more than
    a couple of frames (the normal case)."""


class Settings(BaseSettings):
    """Top-level, overridable settings for a pipeline run."""

    model_config = SettingsConfigDict(env_prefix="AGON_", env_nested_delimiter="__")

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
