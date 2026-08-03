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
- **Fixed, but worth recording**: a rendered pitch-marking overlay tool
  (``scripts/render_pitch_markings_overlay.py``) visually surfaced that
  this calibrator's circle position/scale were reasonably accurate on
  real footage but the rotation was consistently off by ~90 degrees.
  Root cause, confirmed by direct inspection: ``_detect_halfway_line_angle``'s
  ``minLineLength`` (a fraction of the circle's radius) was too strict --
  the real halfway line's longest unbroken Hough-detected segment near the
  circle was consistently shorter than required (players standing on the
  line, motion blur, and Hough's own segmentation all fragment a long
  line), so detection silently failed on every tested frame and fell back
  to a hardcoded 0.0-radian default with no warning. Fixed on two fronts:
  a looser, empirically-retested ``minLineLength``, and no longer falling
  back to an arbitrary guess when there's truly no signal (skips the frame
  instead, consistent with this project's "don't guess when unsure" rule
  elsewhere).
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import numpy.typing as npt

from agon.broadcast.frame_filter import grass_fraction
from agon.geometry.bbox import Point
from agon.io.video import Frame

logger = logging.getLogger(__name__)

CENTER_CIRCLE_RADIUS_M = 9.15


def _segment_pitch_lines(frame: Frame) -> npt.NDArray[np.uint8]:
    """Binary mask of likely pitch-line pixels: white, and near grass."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass_mask = cv2.dilate(grass_mask, np.ones((25, 25), np.uint8))
    line_mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    return cv2.bitwise_and(line_mask, grass_mask)  # type: ignore[return-value]  # cv2 stubs: dtype imprecise


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
    """Angle (radians, mod pi) of the line through the circle, if found.

    ``minLineLength=radius*0.7``, not ``radius*1.3`` as an earlier version
    had it -- found via a real, confirmed bug: on real footage, the actual
    halfway line's longest unbroken Hough-detected segment near the circle
    was ~0.89x the circle's semi-major radius (players standing on the
    line, motion blur, and Hough's own probabilistic segmentation all
    fragment a long line into shorter aligned pieces rather than one
    continuous one) -- so the stricter 1.3x threshold rejected every real
    candidate outright on every tested frame, silently returning None
    every time. Confirmed by lowering the threshold and checking that a
    strong, consistent, real-line-passing-through-the-circle candidate
    was recovered on all 5 frames tested (this is empirically retested,
    not just theoretically justified).
    """
    lines = cv2.HoughLinesP(
        line_mask,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(10, int(radius * 0.7)),
        maxLineGap=20,
    )
    if lines is None:
        return None

    cx, cy = center
    best_angle: float | None = None
    # Tightened from 0.6x after loosening minLineLength above (see that
    # change's note) surfaced a real confound: a wide, flattened ellipse's
    # own boundary (the near-straight arc along its top/bottom, typical for
    # a broadcast-projected center circle) can itself register as a
    # Hough candidate passing within 0.6x of the center. Real halfway-line
    # detections across 5 real frames all measured well under 0.1x
    # (6.7-39.3px against a ~530px radius); the ellipse-boundary artifact
    # measured ~0.33x on a synthetic test -- a wide margin between the two,
    # not a knife-edge tuning.
    best_dist = radius * 0.15
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
    """Local pitch-space coordinate system (per resolved frame): origin at
    the detected circle center, x = perpendicular to the halfway line
    (toward/away from the goals), y = along the halfway line (toward the
    touchlines). This isn't just an implementation detail -- see
    ``inverse_transform_point`` for why the convention matters to callers
    beyond ``transform_point`` itself."""

    def __init__(self, court_width_m: float = 68.0, min_grass_fraction: float = 0.35):
        """``min_grass_fraction``: skip circle detection entirely on frames
        with less pitch-green coverage than this (see
        ``agon.broadcast.frame_filter.grass_fraction``) -- a real, found
        false positive motivated this: a sustained broadcast bumper/
        transition graphic got confidently matched as the center circle on
        real match footage (its glowing near-white outline over a
        greenish background passed the existing aspect-ratio/size
        filters). Cheap (~1ms/frame, no OCR) and catches genuine
        no-pitch-visible frames (graphics, ads, studio shots) reliably.

        Real, stated limitation, found empirically, not assumed: this does
        NOT catch every such false positive -- a colorful graphic
        overlaid on a still-visible, still-green pitch background (as in
        the case that motivated this) can keep grass_fraction comfortably
        above this default threshold. Full coverage of that specific case
        needs either a live-play-vs-replay signal (this project's own
        clock-based one is documented as currently unreliable -- see the
        project plan's Phase 8) or a more targeted visual heuristic, not
        yet built. Matching PipelineConfig.min_grass_fraction's own
        default (0.35) for consistency, not because that specific value
        was independently tuned for this use.
        """
        self.court_width_m = court_width_m
        self.min_grass_fraction = min_grass_fraction
        self._transforms: dict[int, tuple[Point, npt.NDArray[np.float64], float]] = {}
        # Persistent (not reset per calibrate() call) so the temporal
        # disambiguation described in the module docstring carries across
        # chunk boundaries in streaming/windowed processing, rather than
        # forgetting the previous chunk's last resolved angle/size and
        # re-seeding blind at the start of every chunk.
        self._previous_angle: float | None = None
        self._previous_semi_major: float | None = None

    def calibrate(self, frames: list[Frame], frame_offset: int = 0) -> None:
        """``frame_offset``: this call's first frame's global (match-relative)
        index. Only matters for chunked/streaming processing, where
        ``frames`` covers one chunk, not the whole clip -- 0 (the default)
        is correct for a single whole-clip call, and also the value used by
        ``transform_point``'s default ``frame_idx=0`` for a non-chunked
        caller.
        """
        rejected_as_inconsistent = 0
        rejected_as_not_pitch = 0
        rejected_as_no_angle = 0
        resolved_this_call = 0
        for local_idx, frame in enumerate(frames):
            if grass_fraction(frame) < self.min_grass_fraction:
                rejected_as_not_pitch += 1
                continue

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
            if self._previous_semi_major is not None:
                ratio = semi_major / self._previous_semi_major
                if not (0.625 <= ratio <= 1.6):
                    rejected_as_inconsistent += 1
                    continue
            self._previous_semi_major = semi_major

            raw_angle = _detect_halfway_line_angle(line_mask, (cx, cy), semi_major)
            if raw_angle is not None:
                angle = _closest_equivalent_angle(raw_angle, self._previous_angle)
            elif self._previous_angle is not None:
                angle = self._previous_angle
            else:
                # No line detected on this frame, and no previous frame's angle to
                # fall back on -- a real bug used to default to 0.0 here (an
                # arbitrary, unverified guess), confirmed to actively produce a
                # ~90-degree-wrong transform on real footage (see module docstring's
                # halfway-line-angle-detection note). Skipping the frame entirely is
                # consistent with this project's "don't guess when unsure" rule
                # applied everywhere else (jersey OCR, aggregator min_votes).
                rejected_as_no_angle += 1
                continue
            self._previous_angle = angle

            psi = math.pi / 2 - angle
            rotation = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
            # Semi-major axis as the scale reference -- see module docstring.
            scale = CENTER_CIRCLE_RADIUS_M / semi_major
            self._transforms[frame_offset + local_idx] = ((cx, cy), rotation, scale)
            resolved_this_call += 1

        logger.info(
            "Dynamic pitch calibration: found the center circle in %d/%d frames "
            "(%d rejected outright as not-enough-pitch-visible, %d more rejected as "
            "size-inconsistent with the previous detection, %d more rejected for no "
            "resolvable halfway-line angle and no previous frame to fall back on)",
            resolved_this_call,
            len(frames),
            rejected_as_not_pitch,
            rejected_as_inconsistent,
            rejected_as_no_angle,
        )

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None:
        # NaN point: the ball's position when it was never detected anywhere
        # in this chunk (see interpolate_ball_positions) -- null out rather
        # than propagate NaN into the export (invalid JSON per spec, and
        # meaningless as a "position").
        if math.isnan(point[0]) or math.isnan(point[1]):
            return None

        transform = self._transforms.get(frame_idx)
        if transform is None:
            return None

        (cx, cy), rotation, scale = transform
        delta = np.array([point[0] - cx, point[1] - cy])
        rotated = rotation @ delta
        return float(rotated[0] * scale), float(rotated[1] * scale)

    def inverse_transform_point(self, pitch_point: Point, frame_idx: int = 0) -> Point | None:
        """The inverse of ``transform_point``: given a pitch-space point
        (meters, in this calibrator's own local coordinate system -- origin
        at the circle center, x = along the pitch length/toward the goals,
        y = along the halfway line/toward the touchlines), returns its
        predicted pixel-space location for the given frame, or None if that
        frame has no resolved transform.

        Exists for self-consistency checking, not normal pipeline use: since
        this calibrator's scale/rotation are derived *only* from the center
        circle and halfway-line angle, predicting where some *other* known
        pitch feature should be (e.g. a touchline, at y = +/-
        court_width_m/2, x = 0) and checking whether the frame actually has
        a pitch-line pixel there is a real, buildable check of whether the
        resolved transform is trustworthy -- unlike comparing this
        calibrator's own output to itself, which is tautological. See
        ``scripts/validate_pitch_calibration_self_consistency.py``.
        """
        transform = self._transforms.get(frame_idx)
        if transform is None:
            return None

        (cx, cy), rotation, scale = transform
        rotated = np.array([pitch_point[0] / scale, pitch_point[1] / scale])
        # rotation is orthogonal (built from cos/sin of a single angle), so
        # its inverse is its transpose -- no separate matrix inversion needed.
        delta = rotation.T @ rotated
        return float(cx + delta[0]), float(cy + delta[1])
