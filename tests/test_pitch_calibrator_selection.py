import pytest

from agon.config import CalibrationConfig
from agon.geometry.hybrid_pitch_calibrator import HybridPitchCalibrator
from agon.geometry.pitch_keypoint_calibrator import PitchKeypointCalibrator
from agon.geometry.view_transformer import ViewTransformer
from agon.pipeline import _make_pitch_calibrator

_CALIBRATION = CalibrationConfig(
    pixel_vertices=[(0, 0), (100, 0), (100, 100), (0, 100)],
    court_length_m=105.0,
    court_width_m=68.0,
)


def test_static_returns_view_transformer():
    calibrator = _make_pitch_calibrator(_CALIBRATION, mode="static")
    assert isinstance(calibrator, ViewTransformer)


def test_dynamic_returns_pitch_keypoint_calibrator():
    calibrator = _make_pitch_calibrator(_CALIBRATION, mode="dynamic")
    assert isinstance(calibrator, PitchKeypointCalibrator)


def test_hybrid_returns_hybrid_pitch_calibrator():
    calibrator = _make_pitch_calibrator(_CALIBRATION, mode="hybrid")
    assert isinstance(calibrator, HybridPitchCalibrator)


def test_trained_without_model_path_raises_clear_error():
    with pytest.raises(ValueError, match="pitch_calibration_model_path"):
        _make_pitch_calibrator(_CALIBRATION, mode="trained", pitch_calibration_model_path=None)


def test_trained_with_missing_checkpoint_file_raises():
    with pytest.raises(Exception):  # noqa: B017 -- onnxruntime's own error type, not ours to pin
        _make_pitch_calibrator(
            _CALIBRATION, mode="trained", pitch_calibration_model_path="/nonexistent/best.onnx"
        )
