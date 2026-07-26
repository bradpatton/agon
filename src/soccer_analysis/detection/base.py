"""Shared detect -> track assembly logic used by every ``Detector`` backend.

Each backend (torch-backed ``UltralyticsDetector``, onnxruntime-backed
``OnnxDetector``, ...) only needs to produce per-frame ``supervision.Detections``
plus a class-id -> class-name map and a class-name -> object-type map.
Tracking, the players/referees/ball bucketing, and the optional pickle stub
cache are implemented once, here.

The tracker itself is swappable -- anything satisfying
``update_with_detections(detections, frame) -> Detections`` works, which is
what makes ``BoTSORTTracker`` (needs the actual frame image for camera-motion
compensation) a drop-in alongside plain ``supervision.ByteTrack`` (wrapped in
``ByteTrackAdapter`` below, which just ignores the frame). See
``soccer_analysis.detection.botsort_tracker`` for that alternative.

``class_name_to_object_type`` is what makes this generalize across
checkpoints with different label sets — a soccer-tuned model might expose
``player``/``goalkeeper``/``referee``/``ball``, while a generic COCO model
only has ``person``/``sports ball``. Either maps cleanly onto our fixed
``players``/``referees``/``ball`` buckets; COCO-backed runs just won't ever
populate ``referees`` (see the ``OnnxDetector`` docstring).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from soccer_analysis.geometry.bbox import BBox, get_center_of_bbox, get_foot_position
from soccer_analysis.io.video import Frame

Tracks = dict[str, list[dict[int, dict]]]

# Shared by both Detector implementations so a given checkpoint's classes map
# the same way regardless of backend. A soccer-fine-tuned checkpoint exposes
# player/goalkeeper/referee/ball; a generic COCO checkpoint (the only kind
# available without training one) only has person/sports ball. Keeping both
# conventions in one place matters in practice, not just in theory: this is
# exactly the gap that made UltralyticsDetector silently track zero players
# against a COCO checkpoint until this default was unified with
# OnnxDetector's (which already had to handle the COCO case) -- caught by
# validating both backends end-to-end in Docker against the same checkpoint.
DEFAULT_CLASS_NAME_TO_OBJECT_TYPE = {
    "player": "players",
    "goalkeeper": "players",
    "referee": "referees",
    "ball": "ball",
    "person": "players",
    "sports ball": "ball",
}


class FrameTracker(Protocol):
    def update_with_detections(
        self, detections: sv.Detections, frame: Frame
    ) -> sv.Detections: ...


class ByteTrackAdapter:
    """Wraps supervision.ByteTrack to match FrameTracker's signature (it
    doesn't use the frame image at all -- pure IoU/Kalman tracking)."""

    def __init__(self, byte_track: sv.ByteTrack | None = None):
        self._tracker = byte_track or sv.ByteTrack()

    def update_with_detections(self, detections: sv.Detections, frame: Frame) -> sv.Detections:
        return self._tracker.update_with_detections(detections)


def add_position_to_tracks(tracks: Tracks) -> None:
    for object_type, object_tracks in tracks.items():
        for frame_track in object_tracks:
            for track_info in frame_track.values():
                bbox: BBox = track_info["bbox"]
                if object_type == "ball":
                    track_info["position"] = get_center_of_bbox(bbox)
                else:
                    track_info["position"] = get_foot_position(bbox)


def interpolate_ball_positions(
    ball_positions: list[dict[int, dict[str, Any]]],
) -> list[dict[int, dict[str, Any]]]:
    # A missing detection defaults to 4 NaNs, not []: pandas can't infer a
    # 4-column frame from empty rows, and reliably does so when the ball
    # goes completely undetected for the whole clip (weak ball-class
    # detection is common, e.g. the tiny/distant ball against a generic
    # COCO checkpoint) -- confirmed by hitting this exact crash validating
    # UltralyticsDetector end-to-end for the first time in Docker.
    boxes = [entry.get(1, {}).get("bbox", [np.nan] * 4) for entry in ball_positions]
    df_ball_positions = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])

    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()

    return [{1: {"bbox": row}} for row in df_ball_positions.to_numpy().tolist()]


def run_detection_and_tracking(
    detect_all_frames: Callable[[], list[sv.Detections]],
    frames: list[Frame],
    class_names: dict[int, str],
    class_name_to_object_type: dict[str, str],
    tracker: FrameTracker,
    read_from_stub: bool = False,
    stub_path: str | Path | None = None,
) -> Tracks:
    if read_from_stub and stub_path is not None and Path(stub_path).exists():
        with open(stub_path, "rb") as f:
            return pickle.load(f)

    detections_per_frame = detect_all_frames()
    tracks = _assemble_tracks(
        detections_per_frame, frames, class_names, class_name_to_object_type, tracker
    )

    if stub_path is not None:
        Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump(tracks, f)

    return tracks


def _assemble_tracks(
    detections_per_frame: list[sv.Detections],
    frames: list[Frame],
    class_names: dict[int, str],
    class_name_to_object_type: dict[str, str],
    tracker: FrameTracker,
) -> Tracks:
    tracks: Tracks = {"players": [], "referees": [], "ball": []}

    for frame, detection_supervision in tqdm(
        zip(frames, detections_per_frame), total=len(frames), desc="Tracking objects"
    ):
        detection_with_tracks = tracker.update_with_detections(detection_supervision, frame)

        tracks["players"].append({})
        tracks["referees"].append({})
        tracks["ball"].append({})

        # Ball is not identity-tracked (see module docstring in onnx_tracker.py /
        # tracker.py): always stored under a fixed track id of 1.
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]
            object_type = class_name_to_object_type.get(class_names[cls_id])
            if object_type in ("players", "referees"):
                tracks[object_type][-1][track_id] = {"bbox": bbox}

        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            object_type = class_name_to_object_type.get(class_names[cls_id])
            if object_type == "ball":
                tracks["ball"][-1][1] = {"bbox": bbox}

    return tracks
