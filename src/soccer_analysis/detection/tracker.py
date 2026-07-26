"""Torch-backed detector: Ultralytics YOLO + supervision.ByteTrack.

Satisfies ``soccer_analysis.interfaces.Detector``. This backend needs the
optional ``train`` extra (``pip install soccer-analysis[train]``, which pulls
in ``torch``/``ultralytics``) — it's meant for training/fine-tuning workflows
and for anyone who already has a working torch install and wants the full
Ultralytics feature set. For a lighter runtime-only dependency footprint, see
``soccer_analysis.detection.onnx_tracker.OnnxDetector``, the default backend.

The ``ultralytics`` import is deliberately deferred to ``__init__`` (not
module level) so that importing ``soccer_analysis.detection`` doesn't require
torch to be installed at all unless this specific class is instantiated.

The ball is intentionally *not* run through the multi-object tracker: broadcast
footage typically shows at most one ball, so each frame's ball detection (if
any) is stored under a fixed track id of 1 rather than an identity-tracked id.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import supervision as sv
from tqdm import tqdm

from soccer_analysis.detection.base import ByteTrackAdapter, FrameTracker, Tracks, run_detection_and_tracking
from soccer_analysis.io.video import Frame

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAME_TO_OBJECT_TYPE = {
    "player": "players",
    "goalkeeper": "players",
    "referee": "referees",
    "ball": "ball",
}


class UltralyticsDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.1,
        class_name_to_object_type: dict[str, str] | None = None,
        tracker: FrameTracker | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "UltralyticsDetector needs the 'train' extra: "
                "pip install 'soccer-analysis[train]'"
            ) from e

        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.tracker = tracker or ByteTrackAdapter()
        self.class_name_to_object_type = (
            class_name_to_object_type or DEFAULT_CLASS_NAME_TO_OBJECT_TYPE
        )

    def _detect_frames(self, frames: list[Frame], batch_size: int = 20) -> list[Any]:
        detections = []
        for i in tqdm(range(0, len(frames), batch_size), desc="Detecting objects"):
            batch = self.model.predict(frames[i : i + batch_size], conf=self.confidence)
            detections += batch
        return detections

    def get_object_tracks(
        self,
        frames: list[Frame],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> Tracks:
        def detect_all_frames() -> list[sv.Detections]:
            raw_detections = self._detect_frames(frames)
            return [sv.Detections.from_ultralytics(d) for d in raw_detections]

        return run_detection_and_tracking(
            detect_all_frames,
            frames=frames,
            class_names=self.model.names,
            class_name_to_object_type=self.class_name_to_object_type,
            tracker=self.tracker,
            read_from_stub=read_from_stub,
            stub_path=stub_path,
        )
