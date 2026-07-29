import numpy as np
import pytest

from soccer_analysis.broadcast.frame_filter import (
    FrameClassification,
    classify_frame,
    grass_fraction,
)


def _solid_bgr_frame(bgr: tuple[int, int, int], size=(100, 100)) -> np.ndarray:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def _pitch_green_frame(size=(100, 100)) -> np.ndarray:
    # Mid-range pitch green in BGR, lands inside frame_filter's grass HSV range.
    return _solid_bgr_frame((40, 140, 40), size)


def _graphic_frame(size=(100, 100)) -> np.ndarray:
    # A saturated blue broadcast-graphic-like color, well outside grass HSV.
    return _solid_bgr_frame((200, 40, 20), size)


class _StubClockReader:
    """Fixed-answer stand-in for ClockReader, so frame_filter's combination
    logic can be tested without pytesseract/Tesseract installed."""

    def __init__(self, answer: float | None):
        self.answer = answer
        self.calls = 0

    def read(self, frame: np.ndarray) -> float | None:
        self.calls += 1
        return self.answer


class TestGrassFraction:
    def test_solid_pitch_green_is_near_one(self):
        assert grass_fraction(_pitch_green_frame()) == pytest.approx(1.0, abs=1e-6)

    def test_solid_non_green_is_near_zero(self):
        assert grass_fraction(_graphic_frame()) < 0.01


class TestClassifyFrameWithoutClockReader:
    def test_pitch_frame_is_live_play(self):
        classification, game_clock_s = classify_frame(_pitch_green_frame())
        assert classification == FrameClassification.LIVE_PLAY
        assert game_clock_s is None

    def test_graphic_frame_is_graphic(self):
        classification, game_clock_s = classify_frame(_graphic_frame())
        assert classification == FrameClassification.GRAPHIC
        assert game_clock_s is None

    def test_cannot_distinguish_replay_without_a_clock_reader(self):
        # Documented limitation (see module docstring): a replay is still
        # pitch footage, so without a clock signal it's indistinguishable
        # from live play and classified as LIVE_PLAY.
        classification, _ = classify_frame(_pitch_green_frame(), clock_reader=None)
        assert classification == FrameClassification.LIVE_PLAY


class TestClassifyFrameWithClockReader:
    def test_readable_clock_on_pitch_is_live_play_with_game_clock(self):
        classification, game_clock_s = classify_frame(
            _pitch_green_frame(), clock_reader=_StubClockReader(754.0)
        )
        assert classification == FrameClassification.LIVE_PLAY
        assert game_clock_s == 754.0

    def test_unreadable_clock_on_pitch_is_replay(self):
        classification, game_clock_s = classify_frame(
            _pitch_green_frame(), clock_reader=_StubClockReader(None)
        )
        assert classification == FrameClassification.REPLAY
        assert game_clock_s is None

    def test_off_pitch_is_graphic_without_calling_clock_reader(self):
        # Deliberate: OCR is ~150x more expensive than the grass check
        # (measured), so classify_frame short-circuits to GRAPHIC on a
        # frame with no pitch visible rather than spending an OCR call on
        # it -- see classify_frame's docstring for the real run this was
        # found on (~6 hours for 30,000 frames before this short-circuit).
        # Trade-off: a tight in-play close-up with little grass visible
        # but a genuinely readable clock would now also be misclassified
        # as GRAPHIC -- accepted as a minor, documented cost for a large,
        # measured performance win.
        stub = _StubClockReader(100.0)
        classification, game_clock_s = classify_frame(_graphic_frame(), clock_reader=stub)

        assert classification == FrameClassification.GRAPHIC
        assert game_clock_s is None
        assert stub.calls == 0
