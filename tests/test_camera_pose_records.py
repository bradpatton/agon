import numpy as np
import pytest

from agon.pipeline import _camera_pose_records

# A real homography (pitch-plane meters -> pixels) known to decompose
# successfully -- generated the same way tests/test_camera_pose.py's own
# synthetic-camera tests do (project real canonical pitch keypoints through
# a known look-at camera, fit via cv2.findHomography), pinned here as a
# literal so this test doesn't depend on cv2/camera_pose's own fitting path
# to construct its fixture.
_REAL_HOMOGRAPHY = np.array(
    [
        [-2.21359429e01, 1.43999976e01, 9.60000001e02],
        [-2.05358927e-07, 1.09999823e00, 5.40000006e02],
        [-4.68732112e-10, 1.49999974e-02, 1.00000000e00],
    ]
)


class _CalibratorWithHomography:
    def __init__(self, homography_by_frame: dict[int, np.ndarray]):
        self.homography_by_frame = homography_by_frame

    def homography(self, frame_idx: int = 0):
        return self.homography_by_frame.get(frame_idx)


class _CalibratorWithoutHomography:
    """Like ViewTransformer/PitchKeypointCalibrator -- satisfies
    PitchCalibrator but exposes no .homography() at all."""


class TestCameraPoseRecords:
    def test_none_for_every_frame_when_calibrator_has_no_homography_method(self):
        calibrator = _CalibratorWithoutHomography()
        records = _camera_pose_records(
            calibrator, num_frames=3, image_width=1920, image_height=1080
        )
        assert records == [None, None, None]

    def test_none_for_frames_the_calibrator_did_not_resolve(self):
        calibrator = _CalibratorWithHomography({0: _REAL_HOMOGRAPHY})
        records = _camera_pose_records(
            calibrator, num_frames=2, image_width=1920, image_height=1080
        )
        assert records[0] is not None
        assert records[1] is None

    def test_populates_real_pose_fields_from_a_real_homography(self):
        calibrator = _CalibratorWithHomography({0: _REAL_HOMOGRAPHY})
        [record] = _camera_pose_records(
            calibrator, num_frames=1, image_width=1920, image_height=1080
        )

        assert record is not None
        assert record.x_focal_length_px == pytest.approx(1400.0, abs=1.0)
        assert record.y_focal_length_px == pytest.approx(1400.0, abs=1.0)
        assert record.position_m == pytest.approx((0.0, -60.0, -20.0), abs=1e-2)

    def test_respects_frame_offset(self):
        calibrator = _CalibratorWithHomography({500: _REAL_HOMOGRAPHY})
        records = _camera_pose_records(
            calibrator, num_frames=1, image_width=1920, image_height=1080, frame_offset=500
        )
        assert records[0] is not None
