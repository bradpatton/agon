import cv2
import numpy as np
import pytest

from soccer_analysis.broadcast.clock_reader import ClockReader, _parse_clock_text
from soccer_analysis.config import ClockCalibrationConfig


class TestParseClockText:
    def test_parses_minutes_seconds(self):
        assert _parse_clock_text("12:34") == 754.0

    def test_parses_stoppage_time_minutes(self):
        assert _parse_clock_text("103:07") == 6187.0

    def test_strips_surrounding_whitespace_and_newlines(self):
        assert _parse_clock_text(" 12:34\n") == 754.0

    def test_blank_text_is_unreadable(self):
        assert _parse_clock_text("") is None

    def test_stoppage_time_annotation_is_unreadable(self):
        # "45+2" style stoppage-time formats aren't attempted -- see module
        # docstring -- a safe "no answer" rather than a wrong guess.
        assert _parse_clock_text("45+2:10") is None

    def test_invalid_seconds_is_unreadable(self):
        assert _parse_clock_text("12:75") is None

    def test_garbled_ocr_output_is_unreadable(self):
        assert _parse_clock_text("l2:3S") is None


class TestClockReader:
    def _frame_with_clock_text(self, text: str, size=(80, 200)) -> np.ndarray:
        frame = np.zeros((*size, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)
        cv2.putText(frame, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        return frame

    def test_reads_a_clearly_rendered_clock(self):
        pytest.importorskip("pytesseract")
        frame = self._frame_with_clock_text("67:23")
        reader = ClockReader(ClockCalibrationConfig(clock_region_px=(0, 0, 200, 80)))

        assert reader.read(frame) == pytest.approx(4043.0)

    def test_blank_region_is_unreadable(self):
        pytest.importorskip("pytesseract")
        frame = np.zeros((80, 200, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)
        reader = ClockReader(ClockCalibrationConfig(clock_region_px=(0, 0, 200, 80)))

        assert reader.read(frame) is None

    def test_empty_crop_region_is_unreadable(self):
        pytest.importorskip("pytesseract")
        frame = self._frame_with_clock_text("12:00")
        # A degenerate region (x1 == x2) that would crop to zero width.
        reader = ClockReader(ClockCalibrationConfig(clock_region_px=(50, 0, 50, 80)))

        assert reader.read(frame) is None
