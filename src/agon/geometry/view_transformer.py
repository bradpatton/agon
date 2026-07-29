"""Pixel-space -> pitch-space (meters) perspective transform.

Uses a single static homography computed from four calibrated pixel corner
points (see ``CalibrationConfig`` / ``configs/calibration/*.json``). This
still assumes the camera doesn't pan, tilt, or zoom during the clip beyond
what the optical-flow camera-movement compensation corrects for (translation
only) — see ``agon.geometry.pitch_keypoint_calibrator`` for a
per-frame alternative and its own, different limitations.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt

from agon.config import CalibrationConfig
from agon.geometry.bbox import Point
from agon.interfaces import PitchCalibrator
from agon.io.video import Frame


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

    def calibrate(self, frames: list[Frame], frame_offset: int = 0) -> None:
        """No-op: this calibrator's transform is fixed at construction time."""

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None:
        """Project a pixel-space point into pitch-space meters.

        ``frame_idx`` is unused (the transform is the same for every frame)
        -- it only exists so this satisfies the same ``PitchCalibrator``
        protocol as the dynamic per-frame calibrator.

        Returns None if the point falls outside the calibrated pitch
        boundary (e.g. a player standing on the touchline sideline area, or
        a tracking artifact off the visible pitch), or if the point itself
        is NaN (the ball's position when it was never detected anywhere in
        this chunk -- interpolate_ball_positions can't fill a gap it has no
        real detection on either side of; see that function's docstring).
        """
        if math.isnan(point[0]) or math.isnan(point[1]):
            return None

        pixel_point = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, pixel_point, False) >= 0
        if not is_inside:
            return None

        reshaped_point: npt.NDArray[np.float32] = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(reshaped_point, self.perspective_transform)
        x, y = transformed.reshape(-1, 2)[0]
        return float(x), float(y)


def add_transformed_position_to_tracks(
    tracks: dict, calibrator: PitchCalibrator, frame_offset: int = 0
) -> None:
    """``frame_offset``: ``tracks``' first frame's global (match-relative)
    index -- must match whatever offset ``calibrator.calibrate()`` was
    called with for a dynamic calibrator's per-frame transforms to line up
    (see ``PitchKeypointCalibrator``). 0 (the default) is correct for a
    single whole-clip call."""
    for object_tracks in tracks.values():
        for local_idx, frame_track in enumerate(object_tracks):
            for track_info in frame_track.values():
                position = track_info["position_adjusted"]
                track_info["position_transformed"] = calibrator.transform_point(
                    position, frame_offset + local_idx
                )
