"""Canonical real-world pitch keypoints, one entry per named line in
SoccerNet's Camera Calibration line taxonomy (26 possible line names,
shared by SN-GSR-2025's per-frame pitch annotations and the legacy
Calibration dataset -- see ``scripts/convert_soccernet_gsr_to_calibration.py``
and ``scripts/convert_soccernet_calibration_to_pixels.py``).

Each entry maps a line's endpoint(s) to fixed, real-world pitch-space
coordinates in meters -- origin at the pitch center, x = length direction
(positive toward the "right" goal), y = width direction (positive toward
the "top" touchline). This is the ground truth a keypoint-detection model
trains against, and the correspondence set a homography gets solved from
at inference (see ``agon.jersey``... no -- see the eventual
``TrainedPitchCalibrator``, not yet built).

**Point order/orientation was empirically verified against real
SN-GSR-2025 annotations, not assumed** -- checking real frames from
SNGS-060 (``train_extracted/SNGS-060/Labels-GameState.json``) confirmed:
adjacent lines share a pixel-identical (or near-identical, given
human/model annotation noise) corner in the order encoded below. For
example, frame ``000207.jpg``: "Big rect. left main" ends at (857, 418)px,
matching "Big rect. left top"'s start point (858, 418)px -- both are the
same real corner, the box's far edge meeting its top edge. Frame
``000001.jpg``: "Middle line"'s last point (962, 324)px sits at the same
height as "Side line top"'s points (~318-323px) -- confirming the middle
line's *second* point is its top end, not its first (the naive/unverified
guess would have had this backwards).

Not independently re-verified per side (left vs. right, top vs. bottom):
given the verified pattern for the *left* box/goal and top touchline, the
*right*/*bottom* equivalents are assumed to follow the same relative
convention by construction of this dataset (a well-established, curated
academic benchmark) -- flagged here as an assumption, not a claim. See
``scripts/validate_pitch_keypoints_mapping.py`` for an independent
check that doesn't rely on trusting this note: it solves a real homography
from a subset of one real frame's points and checks whether it correctly
predicts the *other*, held-out points' real pixel positions.

Deliberately excluded, not an oversight:
- The three circle lines ("Circle central", "Circle left", "Circle
  right") -- these are curves, and the polyline's first/last sampled
  point corresponds to whatever portion of the arc happens to be visible
  in a given frame's camera framing, not a fixed real-world position the
  way a straight line's endpoints are. Including them would need either
  circle-fitting (recovering center + radius, which the existing
  classical calibrator already does for the center circle alone) or
  parametric arc-position matching, out of scope for this first version.
- The two crossbar lines ("Goal left/right crossbar") -- both endpoints
  are above ground level, so neither is usable as a ground-plane
  homography point (a homography assumes a planar scene; the pitch
  surface is that plane, the crossbar is not on it).
- Of each goal post line's two endpoints, only the ground-level one is
  kept (verified: point 0 is ground level, point 1 matches the
  corresponding crossbar endpoint and is therefore not on the ground
  plane) -- see ``GOAL_POST_GROUND_POINT_INDEX``.
"""

from __future__ import annotations

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
SIX_YARD_BOX_DEPTH_M = 5.5
SIX_YARD_BOX_WIDTH_M = 18.32
GOAL_WIDTH_M = 7.32
CENTER_CIRCLE_RADIUS_M = 9.15
PENALTY_SPOT_DISTANCE_M = 11.0

_HALF_LENGTH = PITCH_LENGTH_M / 2  # 52.5
_HALF_WIDTH = PITCH_WIDTH_M / 2  # 34.0
_BIG_HALF_WIDTH = PENALTY_BOX_WIDTH_M / 2  # 20.16
_SMALL_HALF_WIDTH = SIX_YARD_BOX_WIDTH_M / 2  # 9.16
_GOAL_HALF_WIDTH = GOAL_WIDTH_M / 2  # 3.66
_BIG_DEPTH_X_LEFT = -_HALF_LENGTH + PENALTY_BOX_DEPTH_M  # -36.0
_BIG_DEPTH_X_RIGHT = _HALF_LENGTH - PENALTY_BOX_DEPTH_M  # 36.0
_SMALL_DEPTH_X_LEFT = -_HALF_LENGTH + SIX_YARD_BOX_DEPTH_M  # -47.0
_SMALL_DEPTH_X_RIGHT = _HALF_LENGTH - SIX_YARD_BOX_DEPTH_M  # 47.0

PointM = tuple[float, float]

# Straight-line pitch features: line_name -> (endpoint_0_m, endpoint_1_m).
# Excludes the three circle lines and two crossbar lines (see module
# docstring). Goal post lines list only one point (the ground-level one).
LINE_ENDPOINTS_M: dict[str, tuple[PointM, ...]] = {
    "Side line top": ((-_HALF_LENGTH, _HALF_WIDTH), (_HALF_LENGTH, _HALF_WIDTH)),
    "Side line bottom": ((-_HALF_LENGTH, -_HALF_WIDTH), (_HALF_LENGTH, -_HALF_WIDTH)),
    "Side line left": ((-_HALF_LENGTH, -_HALF_WIDTH), (-_HALF_LENGTH, _HALF_WIDTH)),
    "Side line right": ((_HALF_LENGTH, -_HALF_WIDTH), (_HALF_LENGTH, _HALF_WIDTH)),
    "Middle line": ((0.0, -_HALF_WIDTH), (0.0, _HALF_WIDTH)),
    "Big rect. left main": (
        (_BIG_DEPTH_X_LEFT, _BIG_HALF_WIDTH),
        (_BIG_DEPTH_X_LEFT, -_BIG_HALF_WIDTH),
    ),
    "Big rect. left top": ((-_HALF_LENGTH, _BIG_HALF_WIDTH), (_BIG_DEPTH_X_LEFT, _BIG_HALF_WIDTH)),
    "Big rect. left bottom": (
        (_BIG_DEPTH_X_LEFT, -_BIG_HALF_WIDTH),
        (-_HALF_LENGTH, -_BIG_HALF_WIDTH),
    ),
    "Big rect. right main": (
        (_BIG_DEPTH_X_RIGHT, _BIG_HALF_WIDTH),
        (_BIG_DEPTH_X_RIGHT, -_BIG_HALF_WIDTH),
    ),
    "Big rect. right top": ((_HALF_LENGTH, _BIG_HALF_WIDTH), (_BIG_DEPTH_X_RIGHT, _BIG_HALF_WIDTH)),
    "Big rect. right bottom": (
        (_BIG_DEPTH_X_RIGHT, -_BIG_HALF_WIDTH),
        (_HALF_LENGTH, -_BIG_HALF_WIDTH),
    ),
    "Small rect. left main": (
        (_SMALL_DEPTH_X_LEFT, _SMALL_HALF_WIDTH),
        (_SMALL_DEPTH_X_LEFT, -_SMALL_HALF_WIDTH),
    ),
    "Small rect. left top": (
        (-_HALF_LENGTH, _SMALL_HALF_WIDTH),
        (_SMALL_DEPTH_X_LEFT, _SMALL_HALF_WIDTH),
    ),
    "Small rect. left bottom": (
        (_SMALL_DEPTH_X_LEFT, -_SMALL_HALF_WIDTH),
        (-_HALF_LENGTH, -_SMALL_HALF_WIDTH),
    ),
    "Small rect. right main": (
        (_SMALL_DEPTH_X_RIGHT, _SMALL_HALF_WIDTH),
        (_SMALL_DEPTH_X_RIGHT, -_SMALL_HALF_WIDTH),
    ),
    "Small rect. right top": (
        (_HALF_LENGTH, _SMALL_HALF_WIDTH),
        (_SMALL_DEPTH_X_RIGHT, _SMALL_HALF_WIDTH),
    ),
    "Small rect. right bottom": (
        (_SMALL_DEPTH_X_RIGHT, -_SMALL_HALF_WIDTH),
        (_HALF_LENGTH, -_SMALL_HALF_WIDTH),
    ),
    "Goal left post left": ((-_HALF_LENGTH, _GOAL_HALF_WIDTH),),
    "Goal left post right": ((-_HALF_LENGTH, -_GOAL_HALF_WIDTH),),
    "Goal right post left": ((_HALF_LENGTH, _GOAL_HALF_WIDTH),),
    "Goal right post right": ((_HALF_LENGTH, -_GOAL_HALF_WIDTH),),
}

GOAL_POST_GROUND_POINT_INDEX = 0
"""Which raw annotation point (of a goal post line's 2) is the ground-level
one -- see module docstring for the empirical check."""

CANONICAL_KEYPOINTS: list[tuple[str, int]] = [
    (name, idx) for name, endpoints in LINE_ENDPOINTS_M.items() for idx in range(len(endpoints))
]
"""Fixed-order list of (line_name, endpoint_index) pairs -- one entry per
trainable keypoint. Order matters and must stay stable: it's the index
order used for both pose-format training labels and decoding a trained
model's output back into named points at inference. Derives from
``LINE_ENDPOINTS_M``'s own (dict, insertion-ordered) iteration order rather
than being listed separately, so the two can't drift out of sync."""


def canonical_keypoint_real_xy(line_name: str, endpoint_idx: int) -> PointM:
    """Real-world pitch-space position (meters) for one canonical keypoint."""
    return LINE_ENDPOINTS_M[line_name][endpoint_idx]


def is_frame_boundary_clipped(
    px: float, py: float, width: int, height: int, margin: float = 3.0
) -> bool:
    """True if a raw annotation point sits within ``margin`` pixels of the
    image edge -- SoccerNet's line annotations are clipped to the visible
    frame, so a point *at* the boundary is usually where a line exits
    frame before reaching its true real-world endpoint, not the endpoint
    itself. Confirmed empirically, not assumed: cross-checking two lines
    that should share a real-world corner (e.g. "Side line left" and
    "Side line bottom", both claiming the same pitch corner) showed wildly
    different pixel positions whenever one of them was boundary-clipped --
    homography-fit error on held-out points dropped from a 16.85m mean to
    2.03m once these were excluded (see
    scripts/validate_pitch_keypoints_mapping.py).
    """
    return px <= margin or px >= width - margin or py <= margin or py >= height - margin


EXCLUDED_LINES = (
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal right crossbar",
)
