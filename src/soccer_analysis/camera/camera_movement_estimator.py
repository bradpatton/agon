"""Optical-flow-based camera movement (pan/translation) estimation.

Tracks a handful of background feature points frame-to-frame with
Lucas-Kanade optical flow and reports the dominant translation, so player
positions can be adjusted to compensate for camera pan. This only corrects
for translation, not zoom/rotation — see the README for that limitation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from soccer_analysis.geometry.bbox import Point, measure_distance, measure_xy_distance
from soccer_analysis.io.video import Frame

logger = logging.getLogger(__name__)


class CameraMovementEstimator:
    def __init__(self, first_frame: Frame, pitch_pixel_vertices: npt.NDArray | None = None):
        """
        Args:
            first_frame: first frame of the clip, used to size the feature mask
                and seed the initial set of tracked points.
            pitch_pixel_vertices: optional calibrated pitch polygon (4 points,
                pixel space). When given, feature points are restricted to the
                background *outside* the pitch, since features moving with
                players on the pitch would otherwise be mistaken for camera
                movement. Falls back to generic left/right border strips
                (sized relative to frame width, not hardcoded pixels) when no
                calibration is available.
        """
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        height, width = first_frame.shape[:2]
        mask_features = np.zeros((height, width), dtype=np.uint8)

        if pitch_pixel_vertices is not None:
            mask_features[:] = 1
            cv2.fillPoly(mask_features, [pitch_pixel_vertices.astype(np.int32)], 0)
        else:
            border = max(1, int(width * 0.02))
            mask_features[:, :border] = 1
            mask_features[:, -border:] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(
        self, tracks: dict, camera_movement_per_frame: list[Point]
    ) -> None:
        for object_tracks in tracks.values():
            for frame_num, frame_track in enumerate(object_tracks):
                for track_info in frame_track.values():
                    position = track_info["position"]
                    dx, dy = camera_movement_per_frame[frame_num]
                    track_info["position_adjusted"] = (position[0] - dx, position[1] - dy)

    def get_camera_movement(
        self,
        frames: list[Frame],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> list[Point]:
        if read_from_stub and stub_path is not None and Path(stub_path).exists():
            data = json.loads(Path(stub_path).read_text())
            return [tuple(pair) for pair in data]

        camera_movement: list[Point] = [(0.0, 0.0)] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # mypy can't resolve cv2's overloads through a **dict[str, object]
        # unpack below; the dict's actual values match the expected kwarg
        # types at runtime.
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)  # type: ignore[call-overload]

        for frame_num in tqdm(range(1, len(frames)), desc="Estimating camera movement"):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )  # type: ignore[call-overload]

            max_distance = 0.0
            camera_movement_x, camera_movement_y = 0.0, 0.0

            for new, old in zip(new_features, old_features, strict=True):
                new_point = tuple(new.ravel())
                old_point = tuple(old.ravel())

                distance = measure_distance(new_point, old_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = (camera_movement_x, camera_movement_y)
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)  # type: ignore[call-overload]

            old_gray = frame_gray.copy()

        if stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            Path(stub_path).write_text(json.dumps(camera_movement))

        return camera_movement

    def draw_camera_movement(
        self, frames: list[Frame], camera_movement_per_frame: list[Point]
    ) -> list[Frame]:
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            output_frames.append(frame)

        return output_frames
