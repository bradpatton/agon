import math

import cv2
import numpy as np
import pytest

from agon.geometry.pitch_keypoint_calibrator import (
    PitchKeypointCalibrator,
    _closest_equivalent_angle,
    _detect_center_circle,
)


class TestClosestEquivalentAngle:
    def test_no_previous_returns_unchanged(self):
        assert _closest_equivalent_angle(0.5, None) == 0.5

    def test_keeps_angle_close_to_previous(self):
        assert _closest_equivalent_angle(0.5, previous=0.6) == pytest.approx(0.5)

    def test_flips_to_the_closer_equivalent(self):
        # 0.1 and (0.1 + pi) represent the same undirected line; if the
        # previous frame resolved near pi, prefer the +pi branch so the
        # transform doesn't jump 180 degrees between frames.
        result = _closest_equivalent_angle(0.1, previous=math.pi - 0.05)
        assert result == pytest.approx(0.1 + math.pi)


class TestDetectCenterCircle:
    def _mask_with_ellipse(self, center, axes, size=(600, 800)) -> np.ndarray:
        mask = np.zeros(size, dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, 2)
        return mask

    def test_finds_a_plausible_broadcast_ellipse(self):
        mask = self._mask_with_ellipse(center=(400, 300), axes=(200, 60))
        result = _detect_center_circle(mask)

        assert result is not None
        cx, cy, semi_major, semi_minor = result
        assert cx == pytest.approx(400, abs=5)
        assert cy == pytest.approx(300, abs=5)
        assert semi_major == pytest.approx(200, abs=10)
        assert semi_minor == pytest.approx(60, abs=10)

    def test_rejects_near_circular_blob(self):
        # Aspect ratio too close to 1 -- a broadcast center circle is
        # always elongated by perspective; a near-perfect circle is more
        # likely a logo or other artifact (see module docstring).
        mask = self._mask_with_ellipse(center=(400, 300), axes=(150, 145))
        assert _detect_center_circle(mask) is None

    def test_rejects_too_small(self):
        mask = self._mask_with_ellipse(center=(400, 300), axes=(20, 8))
        assert _detect_center_circle(mask) is None

    def test_rejects_graphics_banner_region(self):
        # Near the top of the frame -- where broadcast scoreboard/graphics
        # overlays live, not the pitch.
        mask = self._mask_with_ellipse(center=(400, 40), axes=(150, 50))
        assert _detect_center_circle(mask) is None

    def test_no_ellipse_returns_none(self):
        mask = np.zeros((600, 800), dtype=np.uint8)
        assert _detect_center_circle(mask) is None


class TestPitchKeypointCalibrator:
    def test_transform_point_none_when_uncalibrated(self):
        calibrator = PitchKeypointCalibrator()
        assert calibrator.transform_point((100, 100), frame_idx=0) is None

    def test_calibrate_on_no_frames_leaves_everything_unresolved(self):
        calibrator = PitchKeypointCalibrator()
        calibrator.calibrate([])
        assert calibrator.transform_point((100, 100), frame_idx=0) is None

    def test_inverse_transform_point_none_when_uncalibrated(self):
        calibrator = PitchKeypointCalibrator()
        assert calibrator.inverse_transform_point((0, 0), frame_idx=0) is None


class TestInverseTransformPoint:
    """Uses a real calibrate() call against a synthetic pitch frame (grass
    green background, a white ellipse standing in for the center circle,
    a horizontal white line through it standing in for the halfway line)
    so these exercise the actual resolved rotation/scale, not a hand-built
    transform -- inverse_transform_point exists specifically to support
    self-consistency checks against a real resolved calibration (see
    scripts/validate_pitch_calibration_self_consistency.py)."""

    def _pitch_frame(self, center=(400, 300), axes=(180, 60), size=(600, 800)) -> np.ndarray:
        frame = np.zeros((*size, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)  # pure green grass, BGR
        cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 255), 3)
        cx, cy = center
        cv2.line(frame, (cx - 300, cy), (cx + 300, cy), (255, 255, 255), 2)
        return frame

    def test_round_trips_the_circle_center(self):
        calibrator = PitchKeypointCalibrator()
        calibrator.calibrate([self._pitch_frame(center=(400, 300))])

        pitch_point = calibrator.transform_point((400, 300), frame_idx=0)
        assert pitch_point is not None
        # Not exactly (0, 0): cv2.fitEllipse on a real rasterized ellipse
        # resolves a center a fraction of a pixel off (400, 300) -- real
        # detection noise, not a bug in the transform math.
        assert pitch_point == pytest.approx((0.0, 0.0), abs=1e-2)

        pixel_point = calibrator.inverse_transform_point(pitch_point, frame_idx=0)
        assert pixel_point == pytest.approx((400, 300), abs=1e-6)

    def test_round_trips_arbitrary_points(self):
        calibrator = PitchKeypointCalibrator()
        calibrator.calibrate([self._pitch_frame(center=(400, 300))])

        for original in [(400, 200), (250, 350), (600, 450)]:
            pitch_point = calibrator.transform_point(original, frame_idx=0)
            assert pitch_point is not None
            recovered = calibrator.inverse_transform_point(pitch_point, frame_idx=0)
            assert recovered == pytest.approx(original, abs=1e-6)
