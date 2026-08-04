import numpy as np
import pytest

from agon.analytics.ball_height import BALL_DIAMETER_M, estimate_ball_position_3d
from agon.geometry.camera_pose import CameraPose, project_point


def _look_at_rotation(
    position: np.ndarray, target: np.ndarray, world_up: np.ndarray | None = None
) -> np.ndarray:
    """Same construction as tests/test_camera_pose.py's own helper -- kept
    as an independent copy rather than a shared import, matching this
    project's existing precedent for small test-only geometry helpers."""
    if world_up is None:
        world_up = np.array([0.0, 0.0, -1.0])
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.vstack([right, -true_up, forward])


def _synthetic_bbox_for_ball(
    pose: CameraPose, ball_position_m: tuple[float, float, float], ball_diameter_m: float
) -> tuple[float, float, float, float]:
    """Builds the bbox a perfectly-detected ball of the given real diameter,
    at the given real 3D position, would produce under ``pose`` -- the
    exact inverse of estimate_ball_position_3d's own math, used to
    construct a known-correct round-trip fixture."""
    pixel = project_point(pose, ball_position_m)
    assert pixel is not None
    position = np.array(pose.position)
    depth_m = (pose.rotation @ (np.array(ball_position_m) - position))[2]
    focal_length_px = (pose.x_focal_length + pose.y_focal_length) / 2
    apparent_diameter_px = ball_diameter_m * focal_length_px / depth_m
    half = apparent_diameter_px / 2
    return (pixel[0] - half, pixel[1] - half, pixel[0] + half, pixel[1] + half)


class TestEstimateBallPosition3d:
    def _pose(self) -> CameraPose:
        position = np.array([0.0, -60.0, -20.0])
        rotation = _look_at_rotation(position, np.array([0.0, 0.0, 0.0]))
        return CameraPose(
            position=position,
            rotation=rotation,
            x_focal_length=1400.0,
            y_focal_length=1400.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080,
        )

    def test_recovers_a_ball_resting_on_the_ground(self):
        pose = self._pose()
        true_position = (5.0, 2.0, 0.0)
        bbox = _synthetic_bbox_for_ball(pose, true_position, BALL_DIAMETER_M)

        recovered = estimate_ball_position_3d(pose, bbox)

        assert recovered is not None
        assert recovered == pytest.approx(true_position, abs=1e-3)

    def test_recovers_an_airborne_ball(self):
        pose = self._pose()
        true_position = (8.0, -5.0, -2.5)  # 2.5m up (z negative = up)
        bbox = _synthetic_bbox_for_ball(pose, true_position, BALL_DIAMETER_M)

        recovered = estimate_ball_position_3d(pose, bbox)

        assert recovered is not None
        assert recovered == pytest.approx(true_position, abs=1e-3)

    def test_a_farther_ball_produces_a_smaller_apparent_bbox(self):
        # Sanity check on the fixture/round-trip itself, not just the
        # estimator: a real, physically-expected relationship.
        pose = self._pose()
        near_bbox = _synthetic_bbox_for_ball(pose, (0.0, 0.0, 0.0), BALL_DIAMETER_M)
        far_bbox = _synthetic_bbox_for_ball(pose, (0.0, 40.0, 0.0), BALL_DIAMETER_M)

        near_size = near_bbox[2] - near_bbox[0]
        far_size = far_bbox[2] - far_bbox[0]
        assert far_size < near_size

    def test_uses_the_smaller_bbox_dimension_not_an_average(self):
        # A motion-blurred detection: width inflated by blur along the
        # ball's direction of travel, height still reflects the true
        # apparent diameter. The estimate should match a *clean* bbox
        # using that same (smaller) diameter, not something in between.
        pose = self._pose()
        true_position = (3.0, 0.0, -1.0)
        clean_bbox = _synthetic_bbox_for_ball(pose, true_position, BALL_DIAMETER_M)
        x1, y1, x2, y2 = clean_bbox
        blurred_bbox = (x1 - 15.0, y1, x2 + 15.0, y2)  # width inflated, height untouched

        clean_result = estimate_ball_position_3d(pose, clean_bbox)
        blurred_result = estimate_ball_position_3d(pose, blurred_bbox)

        assert clean_result is not None
        assert blurred_result is not None
        assert blurred_result == pytest.approx(clean_result, abs=1e-6)

    def test_degenerate_bbox_returns_none(self):
        pose = self._pose()
        assert estimate_ball_position_3d(pose, (100.0, 100.0, 100.0, 100.0)) is None
