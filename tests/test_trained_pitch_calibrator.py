from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agon.geometry.pitch_keypoints import CANONICAL_KEYPOINTS, canonical_keypoint_real_xy
from agon.geometry.trained_pitch_calibrator import (
    NUM_KEYPOINTS,
    TrainedPitchCalibrator,
    decode_best_pitch_keypoints,
    fit_homography_from_keypoints,
)


def _fake_output(num_anchors: int, best_idx: int, best_keypoints_row: np.ndarray) -> np.ndarray:
    """Builds a (1, 119, num_anchors) array where every anchor has a low
    class score except ``best_idx``, which carries ``best_keypoints_row``
    (114 values: 38 keypoints x (x, y, confidence))."""
    output = np.zeros((1, 4 + 1 + NUM_KEYPOINTS * 3, num_anchors), dtype=np.float32)
    output[0, 4, :] = 0.01  # every anchor's class score starts low
    output[0, 4, best_idx] = 0.9
    output[0, 5:, best_idx] = best_keypoints_row
    return output


class TestDecodeBestPitchKeypoints:
    def test_picks_the_highest_confidence_anchor(self):
        row = np.zeros(NUM_KEYPOINTS * 3, dtype=np.float32)
        row[0:3] = [100.0, 200.0, 0.8]  # first keypoint
        output = _fake_output(num_anchors=5, best_idx=2, best_keypoints_row=row)

        decoded = decode_best_pitch_keypoints(output, scale=1.0, pad=(0.0, 0.0))

        assert decoded.shape == (NUM_KEYPOINTS, 3)
        assert decoded[0] == pytest.approx([100.0, 200.0, 0.8])

    def test_unletterboxes_pixel_coordinates(self):
        row = np.zeros(NUM_KEYPOINTS * 3, dtype=np.float32)
        row[0:3] = [510.0, 310.0, 0.9]  # network-input-space coords
        output = _fake_output(num_anchors=3, best_idx=0, best_keypoints_row=row)

        # A 2x downscale with (10, 10) padding -- e.g. a 480x480 region
        # resized+padded into a 960x960 network input.
        decoded = decode_best_pitch_keypoints(output, scale=2.0, pad=(10.0, 10.0))

        expected_x = (510.0 - 10.0) / 2.0
        expected_y = (310.0 - 10.0) / 2.0
        assert decoded[0][:2] == pytest.approx([expected_x, expected_y])

    def test_confidence_is_not_rescaled(self):
        row = np.zeros(NUM_KEYPOINTS * 3, dtype=np.float32)
        row[0:3] = [0.0, 0.0, 0.73]
        output = _fake_output(num_anchors=2, best_idx=1, best_keypoints_row=row)

        decoded = decode_best_pitch_keypoints(output, scale=3.0, pad=(5.0, 5.0))

        assert decoded[0][2] == pytest.approx(0.73)


class TestFitHomographyFromKeypoints:
    def _confident_keypoints_from_real_correspondences(
        self, homography: np.ndarray, num_points: int
    ) -> np.ndarray:
        """Builds a (38, 3) keypoints array where the first ``num_points``
        canonical keypoints are placed at their true projection under a
        known homography (high confidence), and the rest are absent (zero
        confidence) -- simulates a frame where only some pitch features are
        visible, same as real footage."""
        keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        for i in range(num_points):
            name, endpoint_idx = CANONICAL_KEYPOINTS[i]
            x, y = canonical_keypoint_real_xy(name, endpoint_idx)
            homogeneous = homography @ np.array([x, y, 1.0])
            px, py = homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]
            keypoints[i] = [px, py, 0.9]
        return keypoints

    def test_returns_none_below_min_keypoints(self):
        homography = np.array([[50.0, 0.0, 960.0], [0.0, 50.0, 540.0], [0.0, 0.0, 1.0]])
        keypoints = self._confident_keypoints_from_real_correspondences(homography, num_points=3)

        result = fit_homography_from_keypoints(
            keypoints, keypoint_confidence=0.3, min_keypoints=4, ransac_reproj_threshold=10.0
        )

        assert result is None

    def test_fits_a_homography_that_reprojects_correctly(self):
        true_homography = np.array([[50.0, 5.0, 960.0], [-3.0, 45.0, 540.0], [0.0001, 0.0, 1.0]])
        keypoints = self._confident_keypoints_from_real_correspondences(
            true_homography, num_points=10
        )

        fitted = fit_homography_from_keypoints(
            keypoints, keypoint_confidence=0.3, min_keypoints=4, ransac_reproj_threshold=10.0
        )

        assert fitted is not None
        name, endpoint_idx = CANONICAL_KEYPOINTS[0]
        x, y = canonical_keypoint_real_xy(name, endpoint_idx)
        true_pixel = true_homography @ np.array([x, y, 1.0])
        fitted_pixel = fitted @ np.array([x, y, 1.0])
        true_pixel /= true_pixel[2]
        fitted_pixel /= fitted_pixel[2]
        assert fitted_pixel[:2] == pytest.approx(true_pixel[:2], abs=1.0)

    def test_ignores_points_below_confidence_threshold(self):
        homography = np.array([[50.0, 0.0, 960.0], [0.0, 50.0, 540.0], [0.0, 0.0, 1.0]])
        keypoints = self._confident_keypoints_from_real_correspondences(homography, num_points=10)
        keypoints[:, 2] = 0.1  # drop every point's confidence below the threshold

        result = fit_homography_from_keypoints(
            keypoints, keypoint_confidence=0.3, min_keypoints=4, ransac_reproj_threshold=10.0
        )

        assert result is None


def _make_calibrator_with_mocked_session(
    keypoints_by_frame: dict[int, np.ndarray],
) -> TrainedPitchCalibrator:
    """Constructs a TrainedPitchCalibrator with onnxruntime mocked out --
    this class's ONNX-session-dependent parts follow this project's
    existing precedent for ONNX-backed classes (validated against real
    footage/scripts, not unit tests), but the frame-indexing/homography
    bookkeeping around it is real, testable logic that deserves coverage
    without needing an actual model file."""
    fake_input = MagicMock()
    fake_input.name = "images"
    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [fake_input]

    with patch(
        "agon.geometry.trained_pitch_calibrator.ort.InferenceSession", return_value=fake_session
    ):
        calibrator = TrainedPitchCalibrator(model_path="fake.onnx")

    call_count = {"n": 0}

    def fake_detect_keypoints(frame):
        idx = call_count["n"]
        call_count["n"] += 1
        return keypoints_by_frame[idx]

    calibrator._detect_keypoints = fake_detect_keypoints  # type: ignore[method-assign]
    return calibrator


class TestTrainedPitchCalibrator:
    def _keypoints_for_homography(self, homography: np.ndarray, num_points: int = 10) -> np.ndarray:
        keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        for i in range(num_points):
            name, endpoint_idx = CANONICAL_KEYPOINTS[i]
            x, y = canonical_keypoint_real_xy(name, endpoint_idx)
            homogeneous = homography @ np.array([x, y, 1.0])
            px, py = homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]
            keypoints[i] = [px, py, 0.9]
        return keypoints

    def test_transform_point_none_when_uncalibrated(self):
        calibrator = _make_calibrator_with_mocked_session({})
        assert calibrator.transform_point((100, 100), frame_idx=0) is None

    def test_transform_point_none_for_nan_point(self):
        calibrator = _make_calibrator_with_mocked_session({})
        assert calibrator.transform_point((float("nan"), 5.0), frame_idx=0) is None

    def test_calibrate_resolves_frames_with_enough_confident_keypoints(self):
        homography = np.array([[50.0, 0.0, 960.0], [0.0, 50.0, 540.0], [0.0, 0.0, 1.0]])
        frames = {
            0: self._keypoints_for_homography(homography, num_points=10),
            1: np.zeros((NUM_KEYPOINTS, 3)),  # nothing confident -- unresolved
        }
        calibrator = _make_calibrator_with_mocked_session(frames)

        calibrator.calibrate([object(), object()])

        assert calibrator.homography(frame_idx=0) is not None
        assert calibrator.homography(frame_idx=1) is None

    def test_transform_point_round_trips_through_the_fitted_homography(self):
        homography = np.array([[50.0, 5.0, 960.0], [-3.0, 45.0, 540.0], [0.0, 0.0, 1.0]])
        frames = {0: self._keypoints_for_homography(homography, num_points=10)}
        calibrator = _make_calibrator_with_mocked_session(frames)
        calibrator.calibrate([object()])

        name, endpoint_idx = CANONICAL_KEYPOINTS[0]
        world_point = canonical_keypoint_real_xy(name, endpoint_idx)
        homogeneous = homography @ np.array([world_point[0], world_point[1], 1.0])
        pixel_point = (homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2])

        recovered = calibrator.transform_point(pixel_point, frame_idx=0)

        assert recovered is not None
        assert recovered == pytest.approx(world_point, abs=1.0)

    def test_calibrate_respects_frame_offset(self):
        homography = np.array([[50.0, 0.0, 960.0], [0.0, 50.0, 540.0], [0.0, 0.0, 1.0]])
        frames = {0: self._keypoints_for_homography(homography, num_points=10)}
        calibrator = _make_calibrator_with_mocked_session(frames)

        calibrator.calibrate([object()], frame_offset=500)

        assert calibrator.homography(frame_idx=500) is not None
        assert calibrator.homography(frame_idx=0) is None

    def test_inverse_transform_point_none_when_unresolved(self):
        calibrator = _make_calibrator_with_mocked_session({})
        assert calibrator.inverse_transform_point((0.0, 0.0), frame_idx=0) is None

    def test_inverse_transform_point_round_trips_with_transform_point(self):
        homography = np.array([[50.0, 5.0, 960.0], [-3.0, 45.0, 540.0], [0.0, 0.0, 1.0]])
        frames = {0: self._keypoints_for_homography(homography, num_points=10)}
        calibrator = _make_calibrator_with_mocked_session(frames)
        calibrator.calibrate([object()])

        for pitch_point in [(0.0, 0.0), (10.0, -5.0), (-30.0, 20.0)]:
            pixel_point = calibrator.inverse_transform_point(pitch_point, frame_idx=0)
            assert pixel_point is not None
            recovered = calibrator.transform_point(pixel_point, frame_idx=0)
            assert recovered == pytest.approx(pitch_point, abs=1e-6)
