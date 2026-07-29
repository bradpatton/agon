"""OCR-based match-clock reader: crops a calibrated broadcast scorebug
region and parses the MM:SS digits into elapsed game-time seconds.

Backed by Tesseract (via pytesseract) -- optional ``[ocr]`` extra, see
pyproject.toml. The import is deferred to ``__init__`` (not module level),
same pattern as ``soccer_analysis.detection.tracker``'s torch import, so
importing this module doesn't require pytesseract/Tesseract unless a
``ClockReader`` is actually constructed.

Deliberately simple: assumes a clean, high-contrast "MM:SS" render (typical
of broadcast scorebug fonts) and a fixed calibrated region. Doesn't attempt
stoppage-time formats ("45+2"), half/period indicators, or a clock that
moves or resizes mid-broadcast -- any of those just come back as an
unreadable (None) frame rather than a wrong answer, consistent with how
the rest of this project's classical-CV pieces (see
``soccer_analysis.geometry.pitch_keypoint_calibrator``) prefer "no answer"
over a silently-wrong one.
"""

from __future__ import annotations

import re

import cv2

from soccer_analysis.config import ClockCalibrationConfig
from soccer_analysis.io.video import Frame

_CLOCK_PATTERN = re.compile(r"^(\d{1,3}):(\d{2})$")


def _parse_clock_text(text: str) -> float | None:
    """ "67:23" -> 4043.0 seconds. Anything that doesn't match exactly --
    blank, garbled OCR output, a stoppage-time "45+2" annotation this
    doesn't attempt to parse -- returns None rather than guessing."""
    match = _CLOCK_PATTERN.match(text.strip())
    if match is None:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    if seconds >= 60:
        return None
    return float(minutes * 60 + seconds)


class ClockReader:
    def __init__(self, calibration: ClockCalibrationConfig):
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "ClockReader needs pytesseract and a system Tesseract install "
                "-- pip install 'soccer-analysis[ocr]' for the Python wrapper, "
                "plus `brew install tesseract` (macOS) or `apt install "
                "tesseract-ocr` (Debian/Ubuntu) for the OCR engine itself."
            ) from exc

        self._pytesseract = pytesseract
        self.calibration = calibration

    def read(self, frame: Frame) -> float | None:
        """Returns elapsed game-time seconds, or None if the clock region
        isn't a readable "MM:SS" in this frame (not currently on screen,
        obscured, or a format this doesn't parse) -- see module docstring.
        """
        x1, y1, x2, y2 = self.calibration.clock_region_px
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = self._pytesseract.image_to_string(
            thresholded, config="--psm 7 -c tessedit_char_whitelist=0123456789:"
        )
        return _parse_clock_text(text)
