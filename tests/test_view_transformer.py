import pytest

from agon.config import CalibrationConfig
from agon.geometry.view_transformer import ViewTransformer


def _identity_calibration() -> CalibrationConfig:
    # Pixel rectangle numerically equal to the target pitch rectangle, so
    # transform_point should be near-identity inside the quad -- easy to
    # assert on without a separate "compute the expected homography" step.
    return CalibrationConfig(
        pixel_vertices=[(0, 10), (0, 0), (20, 0), (20, 10)],
        court_length_m=20,
        court_width_m=10,
    )


def test_transform_point_inside_polygon():
    transformer = ViewTransformer(_identity_calibration())
    x, y = transformer.transform_point((10, 5))
    assert x == pytest.approx(10, abs=1e-3)
    assert y == pytest.approx(5, abs=1e-3)


def test_transform_point_outside_polygon_returns_none():
    transformer = ViewTransformer(_identity_calibration())
    assert transformer.transform_point((-50, -50)) is None


def test_calibrate_is_a_noop_and_frame_idx_is_ignored():
    transformer = ViewTransformer(_identity_calibration())
    transformer.calibrate(frames=[])  # must not raise
    a = transformer.transform_point((10, 5), frame_idx=0)
    b = transformer.transform_point((10, 5), frame_idx=99)
    assert a == b
