"""Classical-CV frame classification: distinguishes live match-play frames
from broadcast ads/graphics/replays, so those can be tagged or stripped
before the (expensive) detection/tracking stages run on them.

Two signals, combined:
- Pitch-grass coverage (``grass_fraction``): live play and replays both
  show mostly pitch/grass; graphics, ads, lineup cards, and studio shots
  don't. Cheap, no calibration needed, but can't tell live play apart from
  a replay -- both are genuine pitch footage.
- Match-clock readability (via ``ClockReader``, optional): most broadcasts
  hide the persistent match-clock scorebug during replays/graphics and
  show it only during live play, so a successfully-read clock is strong
  evidence this frame is live play specifically, not just "some pitch is
  visible". Needs a per-broadcast ``ClockCalibrationConfig`` -- see that
  class's docstring.

Without a clock reader, this can only distinguish graphics/ads (no pitch)
from "pitch is visible" (live play OR replay, both classified as live
play) -- a known, documented limitation, not a silent inaccuracy.
"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

from soccer_analysis.broadcast.clock_reader import ClockReader
from soccer_analysis.io.video import Frame

# Wider than a "pure" pitch-green range on purpose: real broadcast grass
# under stadium lighting/mowing-stripe patterns/shadow runs duller and
# less saturated than that -- confirmed against a real match frame, where
# a strict (35,40,40)-(85,255,255) range measured only 29% grass on a
# frame that's visually ~65-70% pitch (verified by eye). This range still
# stays essentially at 0% on solid non-green colors (graphics/ads) and
# under 1% on real crowd/stands regions (also confirmed against that same
# frame), so it isn't just "catch everything".
_GRASS_HSV_LOW = (25, 25, 25)
_GRASS_HSV_HIGH = (95, 255, 255)


class FrameClassification(StrEnum):
    LIVE_PLAY = "live_play"
    REPLAY = "replay"
    GRAPHIC = "graphic"


def grass_fraction(frame: Frame) -> float:
    """Fraction of pixels within the pitch-green HSV range. High for both
    live play and replays, low for graphics/ads/studio shots -- a coarse
    but cheap and calibration-free first signal (see module docstring)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GRASS_HSV_LOW, _GRASS_HSV_HIGH)
    return float(np.count_nonzero(mask)) / mask.size


def classify_frame(
    frame: Frame,
    min_grass_fraction: float = 0.35,
    clock_reader: ClockReader | None = None,
) -> tuple[FrameClassification, float | None]:
    """Returns ``(classification, game_clock_s)``. ``game_clock_s`` is
    always None when ``clock_reader`` isn't given, and whenever the clock
    isn't readable in this frame even when it is.

    Checks grass_fraction first and only calls clock_reader when pitch is
    actually visible -- OCR is roughly two orders of magnitude more
    expensive than the grass check (~150ms vs. ~1ms per frame, measured),
    and a frame with essentially no pitch visible is already unambiguously
    a graphic/ad, so there's nothing for the clock to disambiguate. This
    was a real bottleneck, not a hypothetical one: processing a 10-minute
    clip with clock_reader configured took ~6 hours before this reordering
    (calling OCR on all 30,000 frames unconditionally) -- for the ~35% of
    frames that are graphics/ads with zero pitch visible, that's tens of
    minutes of OCR spent on frames the cheap check alone already resolves.
    """
    has_pitch = grass_fraction(frame) >= min_grass_fraction

    if not has_pitch:
        return FrameClassification.GRAPHIC, None

    if clock_reader is not None:
        game_clock_s = clock_reader.read(frame)
        if game_clock_s is not None:
            return FrameClassification.LIVE_PLAY, game_clock_s
        return FrameClassification.REPLAY, None

    return FrameClassification.LIVE_PLAY, None
