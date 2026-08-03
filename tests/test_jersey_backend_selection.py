import pytest

from agon.config import PipelineConfig
from agon.pipeline import _build_jersey_classifier


def test_off_returns_none():
    config = PipelineConfig(jersey_backend="off")
    assert _build_jersey_classifier(config) is None


def test_onnx_without_model_path_raises_clear_error():
    config = PipelineConfig(jersey_backend="onnx", jersey_model_path=None)
    with pytest.raises(ValueError, match="jersey_model_path"):
        _build_jersey_classifier(config)


def test_onnx_with_missing_checkpoint_file_raises():
    config = PipelineConfig(jersey_backend="onnx", jersey_model_path="/nonexistent/best.onnx")
    with pytest.raises(Exception):  # noqa: B017 -- onnxruntime's own error type, not ours to pin
        _build_jersey_classifier(config)


def test_ocr_backend_attempts_to_build_easy_ocr_reader():
    pytest.importorskip("easyocr")
    config = PipelineConfig(jersey_backend="ocr")
    classifier = _build_jersey_classifier(config)
    assert type(classifier).__name__ == "EasyOcrJerseyReader"
