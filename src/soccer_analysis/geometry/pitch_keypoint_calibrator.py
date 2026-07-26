"""Per-frame pitch calibration via classical-CV center-circle detection.

NOT a trained keypoint model (see the README's modernization notes for why:
that needs training data/weights this project doesn't have). This is a
classical-CV first cut: per frame, segment pitch-line pixels in HSV, find the
center circle's contour and fit an ellipse to it (its real-world radius,
9.15m, gives scale + position), and the halfway line's angle via Hough line
detection (gives rotation). Frames where the circle isn't visible (tight
shots, corners, replays -- most of a real broadcast match) return None, same
as "outside the calibrated pitch polygon" in ``ViewTransformer``.

Ellipse fitting, not ``cv2.HoughCircles``, is deliberate: viewed from a
broadcast camera angle, the center circle projects as a significantly
elongated ellipse (3-4x aspect ratio is typical), not a true circle, so
circle-only Hough detection cannot find it -- it was tried first and reliably
locked onto unrelated circular-ish clutter instead. Fitting an ellipse to the
largest plausible closed white contour (with sanity filters on area, aspect
ratio, and vertical position to reject broadcast graphics overlays) finds it
correctly.

Known, real limitations -- this is a bounded improvement, not a general
solution:
- Scale is estimated from the ellipse's semi-major axis, which is less
  foreshortened than the semi-minor axis under a typical shallow broadcast
  tilt angle -- a reasonable approximation for wide "tactical" camera shots,
  not an exact solve. Recovering true metric scale from a single
  perspective-projected ellipse in general needs the full camera
  projection, which is a harder problem than this module attempts (that's
  what a trained keypoint model, using multiple pitch features at once,
  is actually for).
- Produces a similarity transform (rotation + uniform scale + translation),
  not a full projective homography, so it doesn't correct
  perspective/keystoning the way a 4-point (or full keypoint-model) solve
  does.
- The halfway line's direction is only recoverable mod-pi (an undirected
  line, and ``cv2.HoughLinesP``'s endpoint order is arbitrary), and there's
  no way to tell *which* half of the pitch is which from the circle + line
  alone. Both ambiguities are resolved by picking whichever choice is
  closest to the previous successfully-calibrated frame, which prevents
  180-degree flip artifacts within one continuous shot but does not anchor
  to a globally consistent "attacking direction". Speed/distance
  calculations (which only use deltas between two points already expressed
  in the same frame's convention) are unaffected by this; comparing
  absolute positions across a camera cut is not reliable.
- The shape/size/position filters reduce but do not eliminate false
  positives on *other* pitch arcs (penalty box, corners) that happen to
  pass the same aspect-ratio and size checks -- confirmed empirically
  against a real broadcast clip, where a corner arc was occasionally
  matched once the camera panned away from the center circle. A
  frame-to-frame size-consistency check (see ``calibrate()``) catches the
  obvious jumps but not every case, since a wrong arc can coincidentally be
  a similar apparent size to the last genuine detection.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import numpy.typing as npt

from soccer_analysis.geometry.bbox import Point
from soccer_analysis.io.video import Frame

logger = logging.getLogger(__name__)

CENTER_CIRCLE_RADIUS_M = 9.15


def _segment_pitch_lines(frame: Frame) -> npt.NDArray[np.uint8]:
    """Binary mask of likely pitch-line pixels: white, and near grass."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass_mask = cv2.dilate(grass_mask, np.ones((25, 25), np.uint8))
    line_mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    return cv2.bitwise_and(line_mask, grass_mask)


def _detect_center_circle(
    line_mask: npt.NDArray[np.uint8],
) -> tuple[float, float, float, float] | None:
    """Finds the center-circle contour and fits an ellipse to it.

    Returns (cx, cy, semi_major_px, semi_minor_px), or None. See module
    docstring for why this is ellipse fitting rather than Hough circle
    detection, and for the sanity filters below (area, aspect ratio,
    vertical position) that exist specifically to reject broadcast graphics
    overlays and player-number blobs, which are also small white regions in
    the same mask.
    """
    height, width = line_mask.shape[:2]

    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    best: tuple[float, float, float, float] | None = None
    best_semi_major = -1.0
    for contour in contours:
        if len(contour) < 30:  # cv2.fitEllipse needs enough points to be stable
            continue

        # NOTE: cv2.contourArea is a poor size filter here -- the mask holds
        # thin (1-3px) line outlines, not filled blobs, so a genuine ellipse
        # contour and a tiny self-intersection artifact (e.g. where the
        # halfway line crosses the circle) can both have near-zero enclosed
        # area despite wildly different real extents. Filter on the fitted
        # ellipse's own semi-major axis instead, which does discriminate.
        (cx, cy), (minor_axis, major_axis), _angle = cv2.fitEllipse(contour)
        semi_major, semi_minor = major_axis / 2, minor_axis / 2
        if semi_minor < 1:
            continue

        if not (width * 0.15 <= semi_major <= width * 0.9):
            continue
        aspect = semi_major / semi_minor
        if not (1.5 <= aspect <= 8.0):  # broadcast center circles are elongated
            continue
        if cy < height * 0.15:  # reject the broadcast graphics/scoreboard band
            continue

        if semi_major > best_semi_major:
            best_semi_major = semi_major
            best = (float(cx), float(cy), float(semi_major), float(semi_minor))

    return best


def _detect_halfway_line_angle(
    line_mask: npt.NDArray[np.uint8], center: Point, radius: float
) -> float | None:
    """Angle (radians, mod pi) of the line through the circle, if found."""
    lines = cv2.HoughLinesP(
        line_mask,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(10, int(radius * 1.3)),
        maxLineGap=20,
    )
    if lines is None:
        return None

    cx, cy = center
    best_angle: float | None = None
    best_dist = radius * 0.6
    # HoughLinesP's output shape varies by OpenCV build ((N,1,4) vs (N,4)).
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        num = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
        den = math.hypot(y2 - y1, x2 - x1)
        if den == 0:
            continue
        dist = num / den
        if dist < best_dist:
            best_dist = dist
            best_angle = math.atan2(y2 - y1, x2 - x1) % math.pi
    return best_angle


def _closest_equivalent_angle(angle_mod_pi: float, previous: float | None) -> float:
    """Picks whichever of the two mod-pi-equivalent angles is closest to the
    previous frame's resolved angle, so the transform doesn't flip 180
    degrees frame to frame from noise alone (see module docstring)."""
    if previous is None:
        return angle_mod_pi
    candidates = (angle_mod_pi, angle_mod_pi + math.pi)
    return min(candidates, key=lambda a: abs(((a - previous + math.pi) % (2 * math.pi)) - math.pi))


class PitchKeypointCalibrator:
    def __init__(self, court_width_m: float = 68.0):
        self.court_width_m = court_width_m
        self._transforms: dict[int, tuple[Point, npt.NDArray[np.float64], float]] = {}

    def calibrate(self, frames: list[Frame]) -> None:
        previous_angle: float | None = None
        previous_semi_major: float | None = None
        rejected_as_inconsistent = 0
        for frame_idx, frame in enumerate(frames):
            line_mask = _segment_pitch_lines(frame)
            circle = _detect_center_circle(line_mask)
            if circle is None:
                continue

            cx, cy, semi_major, _semi_minor = circle

            # A real center circle's apparent size changes gradually (camera
            # zoom/distance), not in sudden jumps -- a >1.6x frame-to-frame
            # ratio usually means a *different* pitch arc (penalty box,
            # corner) of similar apparent size was matched instead. This
            # doesn't catch every false positive (a wrong arc can coincidentally
            # be a similar size to the last real detection) but it does catch
            # the clear outliers cheaply.
            if previous_semi_major is not None:
                ratio = semi_major / previous_semi_major
                if not (0.625 <= ratio <= 1.6):
                    rejected_as_inconsistent += 1
                    continue
            previous_semi_major = semi_major

            raw_angle = _detect_halfway_line_angle(line_mask, (cx, cy), semi_major)
            if raw_angle is not None:
                angle = _closest_equivalent_angle(raw_angle, previous_angle)
            else:
                angle = previous_angle if previous_angle is not None else 0.0
            previous_angle = angle

            psi = math.pi / 2 - angle
            rotation = np.array(
                [[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]]
            )
            # Semi-major axis as the scale reference -- see module docstring.
            scale = CENTER_CIRCLE_RADIUS_M / semi_major
            self._transforms[frame_idx] = ((cx, cy), rotation, scale)

        logger.info(
            "Dynamic pitch calibration: found the center circle in %d/%d frames "
            "(%d more rejected as size-inconsistent with the previous detection)",
            len(self._transforms),
            len(frames),
            rejected_as_inconsistent,
        )

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None:
        transform = self._transforms.get(frame_idx)
        if transform is None:
            return None

        (cx, cy), rotation, scale = transform
        delta = np.array([point[0] - cx, point[1] - cy])
        rotated = rotation @ delta
        return float(rotated[0] * scale), float(rotated[1] * scale)
