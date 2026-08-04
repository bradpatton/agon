import cv2
import numpy as np
import pytest

from agon.geometry.camera_pose import (
    CameraPose,
    camera_pose_from_homography,
    estimate_focal_length_from_plane_homography,
    homography_from_point_transform,
    pan_tilt_roll_to_rotation,
    pixel_to_ray,
    project_point,
    rotation_matrix_to_pan_tilt_roll,
)
from agon.geometry.pitch_keypoint_calibrator import PitchKeypointCalibrator
from agon.geometry.pitch_keypoints import LINE_ENDPOINTS_M


def _look_at_rotation(
    position: np.ndarray, target: np.ndarray, world_up: np.ndarray | None = None
) -> np.ndarray:
    """Builds a world-to-camera rotation matrix for a camera at ``position``
    looking at ``target`` -- used to construct synthetic test cameras
    directly from a physically-meaningful position/target, sidestepping the
    pan/tilt/roll Euler-angle functions' own non-uniqueness (two distinct
    angle triples can represent the same rotation matrix -- see
    ``TestPanTiltRollRoundTrip``) for test-input construction."""
    if world_up is None:
        world_up = np.array([0.0, 0.0, -1.0])
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.vstack([right, -true_up, forward])


def _project_visible_pitch_keypoints(
    pose: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects every canonical pitch keypoint (z=0) through ``pose``,
    keeping only those that land in front of the camera and inside the
    image frame -- the same real-world constraint a detector would face."""
    world_pts = []
    pixel_pts = []
    for endpoints in LINE_ENDPOINTS_M.values():
        for x, y in endpoints:
            pixel = project_point(pose, (x, y, 0.0))
            if pixel is None:
                continue
            if not (0 <= pixel[0] <= pose.image_width and 0 <= pixel[1] <= pose.image_height):
                continue
            world_pts.append((x, y))
            pixel_pts.append(pixel)
    return (
        np.array(world_pts, dtype=np.float64).reshape(-1, 1, 2),
        np.array(pixel_pts, dtype=np.float64).reshape(-1, 1, 2),
    )


class TestPanTiltRollRoundTrip:
    def test_rotation_to_angles_to_rotation(self):
        # Regression test for a real subtlety found while porting this
        # module: sn-calibration's own rotation_matrix_to_pan_tilt_roll
        # internally transposes its input before extracting angles, so
        # pan_tilt_roll_to_orientation(...) is *not* its inverse --
        # pan_tilt_roll_to_rotation(...) (== .T of that) is. Confirmed
        # directly, independent of any homography decomposition.
        rotation = pan_tilt_roll_to_rotation(np.radians(12.0), np.radians(103.0), np.radians(-4.0))
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(rotation)
        reconstructed = pan_tilt_roll_to_rotation(pan, tilt, roll)
        assert reconstructed == pytest.approx(rotation, abs=1e-9)


class TestEstimateFocalLengthFromPlaneHomography:
    def test_degenerate_homography_returns_none(self):
        assert estimate_focal_length_from_plane_homography(np.eye(3), (960.0, 540.0)) is None


class TestCameraPoseFromHomography:
    """Synthetic round-trip: build a camera from a known position/target/
    focal length, project real canonical pitch keypoints through it, solve
    the same cv2.findHomography this project already uses elsewhere, then
    decompose and check the recovered camera reproduces the same image --
    the functionally-relevant property, robust to the Euler-angle
    non-uniqueness noted above (rather than comparing raw pan/tilt/roll,
    which can differ between two equally-valid representations of the same
    rotation)."""

    def test_degenerate_homography_returns_none(self):
        assert camera_pose_from_homography(np.eye(3), 1920, 1080) is None

    def test_real_homography_resolves_above_the_pitch_not_below(self):
        # Regression test for a real sign-ambiguity bug found validating
        # ball-height estimation against real footage: this exact
        # homography (fit from 7 RANSAC-inlier keypoints,
        # match_10min_sample.mp4 frame 25575, <5px reprojection residual --
        # a well-conditioned, real, non-degenerate fit) decomposed to a
        # camera position 14.8m *below* the pitch before the fix. See
        # camera_pose_from_homography's own docstring for the full
        # diagnosis (both sign candidates give identical focal
        # length/in-plane rotation, differing only in the physically
        # impossible vs. correct camera height).
        homography = np.array(
            [
                [5.20014307e01, 4.21490446e01, 2.87065624e03],
                [2.07200801e00, -3.99233709e00, 7.61784161e02],
                [-7.65883754e-03, 1.79469253e-02, 1.00000000e00],
            ]
        )
        pose = camera_pose_from_homography(homography, 1920, 1080)
        assert pose is not None
        assert pose.position[2] == pytest.approx(-14.82, abs=0.1)

    @pytest.mark.parametrize(
        "position,target,focal_length",
        [
            (np.array([0.0, -60.0, -20.0]), np.array([0.0, 0.0, 0.0]), 1400.0),
            (np.array([25.0, -70.0, -25.0]), np.array([10.0, 0.0, 0.0]), 1600.0),
            # z > 0 (below the pitch) was here originally -- an arbitrary
            # choice at the time, not a real camera configuration this
            # module claims to support. camera_pose_from_homography now
            # enforces the real physical constraint that a broadcast/
            # tactical soccer camera is always above the pitch (see its
            # docstring's sign-ambiguity note) -- z < 0 here reflects that.
            (np.array([-15.0, 55.0, -15.0]), np.array([-10.0, 0.0, 0.0]), 1200.0),
        ],
    )
    def test_recovers_a_camera_that_reprojects_correctly(self, position, target, focal_length):
        rotation = _look_at_rotation(position, target)
        true_pose = CameraPose(
            position=position,
            rotation=rotation,
            x_focal_length=focal_length,
            y_focal_length=focal_length,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080,
        )
        world_pts, pixel_pts = _project_visible_pitch_keypoints(true_pose)
        assert len(world_pts) >= 4  # cv2.findHomography's own minimum

        homography, _ = cv2.findHomography(world_pts, pixel_pts, method=0)
        recovered = camera_pose_from_homography(homography, 1920, 1080)
        assert recovered is not None

        assert recovered.position == pytest.approx(position, abs=1.0)
        assert recovered.x_focal_length == pytest.approx(focal_length, rel=0.02)

        # The property that actually matters: every canonical keypoint
        # reprojects to (near enough) the same pixel under the recovered
        # camera as under the true one, including points *not* used to fit
        # the homography (findHomography used only the in-frame subset).
        max_reprojection_diff = 0.0
        for endpoints in LINE_ENDPOINTS_M.values():
            for x, y in endpoints:
                true_pixel = project_point(true_pose, (x, y, 0.0))
                recovered_pixel = project_point(recovered, (x, y, 0.0))
                if true_pixel is None or recovered_pixel is None:
                    continue
                diff = np.hypot(
                    true_pixel[0] - recovered_pixel[0], true_pixel[1] - recovered_pixel[1]
                )
                max_reprojection_diff = max(max_reprojection_diff, diff)
        assert max_reprojection_diff < 0.5


class TestHomographyFromPointTransform:
    """Validates the affine-homography helper against a real, resolved
    PitchKeypointCalibrator -- the calibrator this decomposition needs to
    work with today, ahead of the trained keypoint model (see project plan
    Phase 14)."""

    def _resolved_calibrator(self) -> PitchKeypointCalibrator:
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)
        cv2.ellipse(frame, (400, 300), (180, 60), 0, 0, 360, (255, 255, 255), 3)
        cv2.line(frame, (100, 300), (700, 300), (255, 255, 255), 2)
        calibrator = PitchKeypointCalibrator()
        calibrator.calibrate([frame])
        return calibrator

    def test_none_when_uncalibrated(self):
        calibrator = PitchKeypointCalibrator()
        result = homography_from_point_transform(
            lambda p: calibrator.inverse_transform_point(p, frame_idx=0)
        )
        assert result is None

    def test_matches_the_calibrator_own_inverse_transform(self):
        calibrator = self._resolved_calibrator()
        homography = homography_from_point_transform(
            lambda p: calibrator.inverse_transform_point(p, frame_idx=0)
        )
        assert homography is not None

        for pitch_point in [(0.0, 0.0), (5.0, -3.0), (-9.15, 0.0), (12.0, 20.0)]:
            expected = calibrator.inverse_transform_point(pitch_point, frame_idx=0)
            homogeneous = homography @ np.array([pitch_point[0], pitch_point[1], 1.0])
            actual = (homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2])
            assert actual == pytest.approx(expected, abs=1e-6)


class TestPixelToRay:
    def test_ray_at_the_correct_depth_recovers_the_original_point(self):
        position = np.array([0.0, -60.0, -20.0])
        rotation = _look_at_rotation(position, np.array([0.0, 0.0, 0.0]))
        pose = CameraPose(
            position=position,
            rotation=rotation,
            x_focal_length=1400.0,
            y_focal_length=1400.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080,
        )

        true_point = np.array([5.0, 3.0, -1.0])
        pixel = project_point(pose, tuple(true_point))
        assert pixel is not None

        origin, direction = pixel_to_ray(pose, pixel)
        assert origin == pytest.approx(position)
        assert np.linalg.norm(direction) == pytest.approx(1.0)

        true_depth = (rotation @ (true_point - position))[2]
        direction_cam_z = (rotation @ direction)[2]
        t = true_depth / direction_cam_z
        recovered = origin + t * direction

        assert recovered == pytest.approx(true_point, abs=1e-6)

    def test_every_point_on_the_ray_projects_to_the_same_pixel(self):
        position = np.array([10.0, -50.0, -15.0])
        rotation = _look_at_rotation(position, np.array([0.0, 0.0, 0.0]))
        pose = CameraPose(
            position=position,
            rotation=rotation,
            x_focal_length=1200.0,
            y_focal_length=1200.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080,
        )

        pixel = (1000.0, 600.0)
        origin, direction = pixel_to_ray(pose, pixel)

        for t in (10.0, 25.0, 40.0):
            point = origin + t * direction
            reprojected = project_point(pose, tuple(point))
            assert reprojected == pytest.approx(pixel, abs=1e-6)
