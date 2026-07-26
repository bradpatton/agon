"""Computes rolling speed (km/h) and cumulative distance (m) per tracked player."""

from __future__ import annotations

import logging

import cv2

from soccer_analysis.geometry.bbox import get_foot_position, measure_distance
from soccer_analysis.io.video import Frame

logger = logging.getLogger(__name__)

_EXCLUDED_OBJECT_TYPES = {"ball", "referees"}


class SpeedDistanceEstimator:
    def __init__(self, frame_window: int = 5, frame_rate: float = 24.0):
        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def add_speed_and_distance_to_tracks(self, tracks: dict) -> None:
        total_distance: dict[str, dict[int, float]] = {}

        for object_type, object_tracks in tracks.items():
            if object_type in _EXCLUDED_OBJECT_TYPES:
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]["position_transformed"]
                    end_position = object_tracks[last_frame][track_id]["position_transformed"]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_kmh = (distance_covered / time_elapsed) * 3.6

                    total_distance.setdefault(object_type, {})
                    total_distance[object_type][track_id] = (
                        total_distance[object_type].get(track_id, 0.0) + distance_covered
                    )

                    for batch_frame_num in range(frame_num, last_frame):
                        if track_id not in tracks[object_type][batch_frame_num]:
                            continue
                        tracks[object_type][batch_frame_num][track_id]["speed"] = speed_kmh
                        tracks[object_type][batch_frame_num][track_id]["distance"] = (
                            total_distance[object_type][track_id]
                        )

    def draw_speed_and_distance(self, frames: list[Frame], tracks: dict) -> None:
        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type in _EXCLUDED_OBJECT_TYPES:
                    continue

                for track_info in object_tracks[frame_num].values():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None or distance is None:
                        continue

                    x, y = get_foot_position(track_info["bbox"])
                    position = (int(x), int(y) + 40)

                    cv2.putText(
                        frame, f"{speed:.2f} km/h", position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                    )
                    cv2.putText(
                        frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                    )
