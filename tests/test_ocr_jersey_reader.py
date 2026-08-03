import sys

import cv2
import numpy as np
import pytest

from agon.jersey.ocr_reader import EasyOcrJerseyReader


def test_raises_friendly_error_without_easyocr_installed(monkeypatch):
    # Simulates a user without the [train] extra, regardless of whether
    # easyocr actually happens to be installed in this environment --
    # matches the friendly-error pattern OnnxJerseyClassifier/ClockReader
    # use for their own optional dependencies.
    monkeypatch.setitem(sys.modules, "easyocr", None)

    with pytest.raises(ImportError, match=r"agon\[train\]"):
        EasyOcrJerseyReader()


class TestClassify:
    def _frame_with_digits(self, text: str, size: tuple[int, int] = (200, 120)) -> np.ndarray:
        frame = np.zeros((*size, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        cv2.putText(
            frame, text, (10, size[0] // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4
        )
        return frame

    def test_reads_a_clearly_rendered_number(self):
        pytest.importorskip("easyocr")
        frame = self._frame_with_digits("7")
        reader = EasyOcrJerseyReader()

        number, confidence = reader.classify(frame, (0, 0, frame.shape[1], frame.shape[0]))

        assert number == 7
        assert confidence > 0.0

    def test_blank_crop_returns_none(self):
        pytest.importorskip("easyocr")
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        reader = EasyOcrJerseyReader()

        number, confidence = reader.classify(frame, (0, 0, frame.shape[1], frame.shape[0]))

        assert number is None
        assert confidence == 0.0

    def test_empty_bbox_returns_none_without_calling_ocr(self):
        pytest.importorskip("easyocr")
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        reader = EasyOcrJerseyReader()

        # Degenerate box (x1 == x2) crops to zero width -- should short-circuit
        # before ever calling into easyocr.
        number, confidence = reader.classify(frame, (50, 0, 50, 120))

        assert number is None
        assert confidence == 0.0

    def test_min_confidence_gates_low_confidence_reads(self):
        pytest.importorskip("easyocr")
        frame = self._frame_with_digits("7")
        reader = EasyOcrJerseyReader(min_confidence=1.1)  # impossible to clear

        number, confidence = reader.classify(frame, (0, 0, frame.shape[1], frame.shape[0]))

        assert number is None
        assert confidence > 0.0  # still reports the real confidence, just below the gate
