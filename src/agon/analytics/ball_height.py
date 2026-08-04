"""Estimates the ball's real 3D position (pitch-space meters, including
height off the ground) from its detected bounding box and a resolved
camera pose -- project plan Phase 7 item 3 / the "ball height (Z)" item
tabled since this project had no real camera model to project a 3D ball
trajectory through. That block no longer holds: ``calibration_mode:
"trained"`` (directly, or via ``"hybrid"``) resolves a full
``agon.geometry.camera_pose.CameraPose`` for a real fraction of frames,
which is exactly what this needs.

**The core problem this solves**: every existing ``PitchCalibrator`` in
this project (``ViewTransformer``, ``PitchKeypointCalibrator``,
``TrainedPitchCalibrator``) maps a pixel to pitch-space *assuming the point
lies on the ground* (``z=0``) -- correct for a player's feet, wrong for an
airborne ball. A single 2D detection plus a camera pose defines a full 3D
ray (``agon.geometry.camera_pose.pixel_to_ray``) the ball's true position
lies somewhere along -- but pixel position alone can't say where. This
module breaks that ambiguity using the ball's *known real-world size*: a
detector's bounding box measures the ball's *apparent* size, and
apparent size shrinks with distance in a way that directly encodes depth
(the same "hold a coin at arm's length" cue human depth perception uses
for a known-size object) -- classical similar-triangles monocular depth
estimation, not a new technique.

**Real, inherent limitation, measured against real footage, not
guessed**: this module's math is validated to sub-millimeter accuracy
synthetically (``tests/test_ball_height.py``) -- but a real-footage
validation (``scripts/validate_ball_height.py``, real ball detections +
real ``TrainedPitchCalibrator`` poses on ``match_10min_sample.mp4``)
found estimated heights of 15-26m, not the near-zero values open play
should show. That run also surfaced and confirmed a fix for a real,
independent bug (a sign ambiguity in ``camera_pose_from_homography`` --
see that function's docstring), but heights stayed implausible even
after the fix. Checked, not assumed: the error does **not** correlate
with detection confidence (high-confidence 0.7+ ball detections were
just as wrong as low-confidence ones, correlation ~0.11) and the
recovered camera height itself is stable frame-to-frame (~15m,
consistently, not noisy) -- ruling out simple detection noise or random
pose jitter as the explanation. The likely real cause: on-plane (``z=0``)
accuracy only requires the fitted homography to be correct *at* the
pitch surface, which `TrainedPitchCalibrator`'s RANSAC-filtered,
low-reprojection-error points do achieve -- but this module's ray-casting
needs the *entire* 3D decomposition (rotation, position, and focal length,
independently) to be accurate, and small per-quantity errors that barely
affect on-plane behavior can compound severely when extrapolated tens of
meters along a ray at the shallow viewing angles typical of a broadcast
camera -- consistent with, and a concrete real-world consequence of, the
"sensitive to noise" limitation `camera_pose_from_homography`'s own
docstring already flags, now shown to matter far more for off-plane
(ball height) use than for the on-plane position estimates this project
already ships. This project has no independently-annotated ground-truth
camera calibration to isolate camera-pose absolute accuracy from
detection noise further (the same standing gap Phase 12 already
surfaced) -- until one exists, or multi-frame trajectory smoothing
across a flight segment is built (a real, separate, larger follow-on,
not attempted here), **do not trust this module's real-footage output.**
It is shipped as a validated building block (the math is right), not a
production-ready feature -- not wired into the pipeline or export schema.
"""

from __future__ import annotations

from agon.geometry.bbox import BBox
from agon.geometry.camera_pose import CameraPose, Point3D, pixel_to_ray

BALL_DIAMETER_M = 0.22
"""FIFA Law 2: circumference 68-70cm -> diameter ~21.6-22.3cm. 0.22m is a
reasonable midpoint, not a per-ball measurement -- real match balls do
vary slightly within that legal range, a source of small systematic error
this module doesn't attempt to correct for."""


def estimate_ball_position_3d(
    pose: CameraPose, bbox: BBox, ball_diameter_m: float = BALL_DIAMETER_M
) -> Point3D | None:
    """Estimates the ball's 3D pitch-space position (meters) from one
    frame's detected bounding box and resolved camera pose.

    Uses the *smaller* of the bbox's width/height as the apparent diameter,
    not an average of the two -- deliberate, not arbitrary: motion blur
    along the ball's direction of travel elongates the apparent shape
    along one axis specifically (the axis the ball moved along during the
    exposure), so the smaller axis is the more reliable measurement of the
    ball's true apparent diameter, and using it (rather than the blurred,
    inflated larger axis) avoids systematically underestimating depth
    (overestimating apparent size) on fast-moving/blurred detections.

    Returns ``None`` if the bbox is degenerate (zero width and height) --
    not a real detection.
    """
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    apparent_diameter_px = min(width, height)
    if apparent_diameter_px <= 0:
        return None

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    origin, direction = pixel_to_ray(pose, center)

    # Similar triangles: apparent_size_px / focal_length_px = real_size_m / depth_m.
    focal_length_px = (pose.x_focal_length + pose.y_focal_length) / 2
    depth_m = (ball_diameter_m * focal_length_px) / apparent_diameter_px

    # `direction`'s own camera-space Z-component says how much of a unit
    # step along `direction` corresponds to one unit of optical-axis depth
    # -- see pixel_to_ray's docstring for why depth alone doesn't pin down
    # a point on the ray without this.
    direction_camera_z = (pose.rotation @ direction)[2]
    if direction_camera_z <= 1e-6:
        return None

    t = depth_m / direction_camera_z
    point = origin + t * direction
    return float(point[0]), float(point[1]), float(point[2])
