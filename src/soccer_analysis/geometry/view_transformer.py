"""Pixel-space -> pitch-space (meters) perspective transform.

Uses a single static homography computed from four calibrated pixel corner
points (see ``CalibrationConfig`` / ``configs/calibration/*.json``). This
still assumes the camera doesn't pan, tilt, or zoom during the clip beyond
what the optical-flow camera-movement compensation corrects for (translation
only) — see the README for the known limitation and the planned upgrade to
per-frame pitch-keypoint calibration.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import cv2

from soccer_analysis.config import CalibrationConfig
from soccer_analysis.geometry.bbox import Point


class ViewTransformer:
    def __init__(self, calibration: CalibrationConfig):
        self.calibration = calibration

        pixel_vertices = np.array(calibration.pixel_vertices, dtype=np.float32)
        target_vertices = np.array(
            [
                [0, calibration.court_width_m],
                [0, 0],
                [calibration.court_length_m, 0],
                [calibration.court_length_m, calibration.court_width_m],
            ],
            dtype=np.float32,
        )

        self.pixel_vertices = pixel_vertices
        self.target_vertices = target_vertices
        self.perspective_transform = cv2.getPerspectiveTransform(pixel_vertices, target_vertices)

    def transform_point(self, point: Point) -> Point | None:
        """Project a pixel-space point into pitch-space meters.

        Returns None if the point falls outside the calibrated pitch
        boundary (e.g. a player standing on the touchline sideline area, or
        a tracking artifact off the visible pitch).
        """
        pixel_point = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, pixel_point, False) >= 0
        if not is_inside:
            return None

        reshaped_point: npt.NDArray[np.float32] = np.array(
            [[point]], dtype=np.float32
        )
        transformed = cv2.perspectiveTransform(reshaped_point, self.perspective_transform)
        x, y = transformed.reshape(-1, 2)[0]
        return float(x), float(y)

    def add_transformed_position_to_tracks(self, tracks: dict) -> None:
        for object_tracks in tracks.values():
            for frame_track in object_tracks:
                for track_info in frame_track.values():
                    position = track_info["position_adjusted"]
                    track_info["position_transformed"] = self.transform_point(position)
