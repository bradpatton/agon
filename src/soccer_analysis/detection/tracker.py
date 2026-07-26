"""Object detection + multi-object tracking for players/referees/ball.

Wraps an Ultralytics YOLO model for detection and ``supervision.ByteTrack``
for tracking, satisfying ``soccer_analysis.interfaces.Detector``. See that
protocol's docstring for newer detector/tracker alternatives worth
benchmarking (a current YOLO/RT-DETR checkpoint, BoT-SORT).

The ball is intentionally *not* run through the multi-object tracker: broadcast
footage typically shows at most one ball, so each frame's ball detection (if
any) is stored under a fixed track id of 1 rather than an identity-tracked id.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from soccer_analysis.geometry.bbox import BBox, get_center_of_bbox, get_foot_position
from soccer_analysis.io.video import Frame

logger = logging.getLogger(__name__)

Tracks = dict[str, list[dict[int, dict[str, Any]]]]


class Tracker:
    def __init__(self, model_path: str, device: str = "cpu", confidence: float = 0.1):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: Tracks) -> None:
        for object_type, object_tracks in tracks.items():
            for frame_track in object_tracks:
                for track_info in frame_track.values():
                    bbox: BBox = track_info["bbox"]
                    if object_type == "ball":
                        track_info["position"] = get_center_of_bbox(bbox)
                    else:
                        track_info["position"] = get_foot_position(bbox)

    def interpolate_ball_positions(
        self, ball_positions: list[dict[int, dict[str, Any]]]
    ) -> list[dict[int, dict[str, Any]]]:
        boxes = [entry.get(1, {}).get("bbox", []) for entry in ball_positions]
        df_ball_positions = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        return [{1: {"bbox": row}} for row in df_ball_positions.to_numpy().tolist()]

    def detect_frames(self, frames: list[Frame], batch_size: int = 20) -> list[Any]:
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
        if read_from_stub and stub_path is not None and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks: Tracks = {"players": [], "referees": [], "ball": []}

        for detection in tqdm(detections, desc="Tracking objects"):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Treat goalkeepers as players for tracking/team-assignment purposes.
            for i, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][-1][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][-1][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][-1][1] = {"bbox": bbox}

        if stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
