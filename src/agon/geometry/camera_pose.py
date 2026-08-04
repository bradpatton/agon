"""Full 3D camera pose (position, orientation, focal length) recovered from
a pitch-plane homography -- project plan Phase 14, the answer to "can the
goal's off-plane geometry help resolve camera angle": it turns out a single
planar homography (the same ``cv2.findHomography`` this project already
computes/will compute from pitch keypoints) is *already* enough to recover a
full 6DOF-plus-focal-length camera pose, via a well-known single-view
self-calibration result -- no off-plane goal points required for this first
step (see module docstring in ``agon.geometry.pitch_keypoints`` and the
project plan's Phase 14 entry for how goal points fit in as a later,
additive refinement).

**This is a direct, cited port, not an original derivation.** The core
algorithm is Algorithm 8.2 and Example 8.1 of Hartley & Zisserman, *Multiple
View Geometry in Computer Vision* (2nd ed.), and this module's
implementation follows the SoccerNet project's own official baseline,
``github.com/SoccerNet/sn-calibration`` (``src/camera.py``), adapted to this
project's typing/style conventions and ``agon.geometry.pitch_keypoints``'s
coordinate convention (origin at pitch center, x = length direction, y =
width direction, meters, z = 0 for ground-plane points) -- confirmed to
match ``sn-calibration``'s own ``soccerpitch.py`` world-point definitions
directly (e.g. ``top_left_corner = (-length/2, -width/2, 0)``), so no
coordinate-system translation was needed between the two.

The self-calibration idea, briefly: a homography's first two columns are
the camera rotation matrix's first two columns, scaled -- and rotation
matrices have orthonormal columns. That orthonormality constrains the
"image of the absolute conic" enough to solve for focal length from a
*single* homography, under a zero-skew/centered-principal-point assumption
(``estimate_focal_length_from_plane_homography``). Once focal length is
known, the homography can be un-calibrated and decomposed directly into
rotation + translation (``camera_pose_from_homography``) -- giving pan,
tilt, roll, and 3D position, not just a flat pixel<->pitch-meters mapping.

Known limitation, inherited from the algorithm itself, not this port: the
single-homography self-calibration is sensitive to noise in the input
homography, and assumes zero skew + a centered principal point (a
reasonable approximation for broadcast cameras, not an exact solve) -- see
``camera_pose_from_homography``'s docstring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from agon.geometry.bbox import Point

Point3D = tuple[float, float, float]


@dataclass
class CameraPose:
    """A recovered camera pose: position/orientation in the same
    pitch-space meters convention as ``agon.geometry.pitch_keypoints``, plus
    the focal length/principal point needed to project further 3D points."""

    position: npt.NDArray[np.float64]  # (3,) meters, pitch-space convention
    rotation: npt.NDArray[np.float64]  # (3, 3) world -> camera
    x_focal_length: float
    y_focal_length: float
    principal_point: Point
    image_width: int
    image_height: int


def pan_tilt_roll_to_orientation(pan: float, tilt: float, roll: float) -> npt.NDArray[np.float64]:
    """Euler angles (radians) -> rotation matrix. Ported verbatim from
    ``sn-calibration``'s ``camera.py`` -- see module docstring."""
    pan_r = np.array([[np.cos(pan), -np.sin(pan), 0], [np.sin(pan), np.cos(pan), 0], [0, 0, 1]])
    roll_r = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )
    tilt_r = np.array(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
    )
    return pan_r @ tilt_r @ roll_r


def pan_tilt_roll_to_rotation(pan: float, tilt: float, roll: float) -> npt.NDArray[np.float64]:
    """Euler angles (radians) -> the world-to-camera rotation matrix in the
    same convention ``CameraPose.rotation``/``project_point`` use (and that
    ``rotation_matrix_to_pan_tilt_roll`` inverts). This is
    ``pan_tilt_roll_to_orientation(...).T``, *not*
    ``pan_tilt_roll_to_orientation(...)`` directly -- a real gotcha inherited
    from ``sn-calibration``'s own source, where the transpose is applied
    inline at its one call site (``Camera.from_json_parameters``) rather
    than folded into the base function. Confirmed by round-tripping a
    synthetic pose through ``camera_pose_from_homography`` during this
    port: using ``pan_tilt_roll_to_orientation`` untransposed to build a
    test camera made the recovered rotation come back as this matrix's
    transpose, not itself -- this function exists so callers who want to
    *construct* a ``CameraPose`` from angles (as opposed to reproducing the
    cited paper's internal derivation step-for-step) have one non-footgun
    entry point."""
    return pan_tilt_roll_to_orientation(pan, tilt, roll).T


def rotation_matrix_to_pan_tilt_roll(
    rotation: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Inverse of ``pan_tilt_roll_to_rotation``. Two solutions exist;
    picks the one with the smaller roll, since broadcast camera operators
    minimize roll in practice (same tie-break ``sn-calibration`` uses)."""
    orientation = rotation.T
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = -first_tilt

    sign_first = 1.0 if np.sin(first_tilt) > 0.0 else -1.0
    sign_second = 1.0 if np.sin(second_tilt) > 0.0 else -1.0

    first_pan = np.arctan2(sign_first * orientation[0, 2], sign_first * -orientation[1, 2])
    second_pan = np.arctan2(sign_second * orientation[0, 2], sign_second * -orientation[1, 2])
    first_roll = np.arctan2(sign_first * orientation[2, 0], sign_first * orientation[2, 1])
    second_roll = np.arctan2(sign_second * orientation[2, 0], sign_second * orientation[2, 1])

    if abs(first_roll) < abs(second_roll):
        return float(first_pan), float(first_tilt), float(first_roll)
    return float(second_pan), float(second_tilt), float(second_roll)


def estimate_focal_length_from_plane_homography(
    homography: npt.NDArray[np.float64], principal_point: Point
) -> tuple[float, float] | None:
    """Recovers (x_focal_length, y_focal_length) from a single pitch-plane
    homography, assuming zero skew and the given (fixed) principal point --
    Hartley & Zisserman Algorithm 8.2. Returns None if the homography
    doesn't yield a valid (positive-definite) solution -- a real, expected
    outcome for a noisy or near-degenerate homography, not a bug to guess
    past.

    The principal point is taken as a fixed input (normally the image
    center) rather than solved for, matching ``sn-calibration``'s own
    documented choice: the principal point *can* be extracted from this same
    algorithm, but doing so is noticeably noisier in practice than just
    fixing it at the image center.
    """
    h = homography.reshape(9)
    a = np.zeros((5, 6))
    a[0, 1] = 1.0
    a[1, 0] = 1.0
    a[1, 2] = -1.0
    a[2, 3] = principal_point[1] / principal_point[0]
    a[2, 4] = -1.0
    a[3, 0] = h[0] * h[1]
    a[3, 1] = h[0] * h[4] + h[1] * h[3]
    a[3, 2] = h[3] * h[4]
    a[3, 3] = h[0] * h[7] + h[1] * h[6]
    a[3, 4] = h[3] * h[7] + h[4] * h[6]
    a[3, 5] = h[6] * h[7]
    a[4, 0] = h[0] * h[0] - h[1] * h[1]
    a[4, 1] = 2 * h[0] * h[3] - 2 * h[1] * h[4]
    a[4, 2] = h[3] * h[3] - h[4] * h[4]
    a[4, 3] = 2 * h[0] * h[6] - 2 * h[1] * h[7]
    a[4, 4] = 2 * h[3] * h[6] - 2 * h[4] * h[7]
    a[4, 5] = h[6] * h[6] - h[7] * h[7]

    _u, _s, vh = np.linalg.svd(a)
    w = vh[-1]
    if w[5] == 0:
        return None
    omega = np.array(
        [
            [w[0] / w[5], w[1] / w[5], w[3] / w[5]],
            [w[1] / w[5], w[2] / w[5], w[4] / w[5]],
            [w[3] / w[5], w[4] / w[5], 1.0],
        ]
    )

    try:
        k_t_inv = np.linalg.cholesky(omega)
    except np.linalg.LinAlgError:
        return None

    k = np.linalg.inv(k_t_inv.T)
    k /= k[2, 2]
    x_focal_length, y_focal_length = float(k[0, 0]), float(k[1, 1])
    if x_focal_length <= 0 or y_focal_length <= 0:
        return None
    return x_focal_length, y_focal_length


def camera_pose_from_homography(
    homography: npt.NDArray[np.float64], image_width: int, image_height: int
) -> CameraPose | None:
    """Decomposes a pitch-plane homography (pitch-space meters -> pixels,
    same convention/orientation ``cv2.findHomography`` would be called with
    given ``agon.geometry.pitch_keypoints``' coordinates as source points)
    into a full camera pose. Returns None if the homography doesn't yield a
    valid focal-length solution (see
    ``estimate_focal_length_from_plane_homography``) -- a frame this can't
    resolve should be skipped, not guessed at, consistent with this
    project's calibrators elsewhere (``PitchKeypointCalibrator``'s own
    "don't guess when unsure" rule).

    Sensitive to noise in the input homography, same as the cited algorithm
    in general -- this is a known, inherent property of single-view
    self-calibration (it has far less constraint to work with than a
    multi-view calibration would), not specific to this port.
    """
    principal_point = (image_width / 2, image_height / 2)
    focal_lengths = estimate_focal_length_from_plane_homography(homography, principal_point)
    if focal_lengths is None:
        return None
    x_focal_length, y_focal_length = focal_lengths

    calibration = np.array(
        [
            [x_focal_length, 0, principal_point[0]],
            [0, y_focal_length, principal_point[1]],
            [0, 0, 1],
        ]
    )

    h_prime = np.linalg.inv(calibration) @ homography
    norm0 = np.linalg.norm(h_prime[:, 0])
    norm1 = np.linalg.norm(h_prime[:, 1])
    if norm0 == 0 or norm1 == 0:
        return None
    lambda1 = 1 / norm0
    lambda2 = 1 / norm1
    lambda3 = np.sqrt(lambda1 * lambda2)

    r0 = h_prime[:, 0] * lambda1
    r1 = h_prime[:, 1] * lambda2
    r2 = np.cross(r0, r1)

    rotation = np.column_stack((r0, r1, r2))
    u, _s, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, 2] *= -1
        rotation = u @ vh

    translation = h_prime[:, 2] * lambda3
    position = -rotation.T @ translation

    return CameraPose(
        position=position,
        rotation=rotation,
        x_focal_length=x_focal_length,
        y_focal_length=y_focal_length,
        principal_point=principal_point,
        image_width=image_width,
        image_height=image_height,
    )


def pan_tilt_roll_degrees(pose: CameraPose) -> tuple[float, float, float]:
    """Convenience wrapper: ``pose``'s orientation as (pan, tilt, roll) in
    degrees, the human-readable form used for logging/reporting."""
    pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(pose.rotation)
    return np.degrees(pan), np.degrees(tilt), np.degrees(roll)


def homography_from_point_transform(
    inverse_transform_fn: Callable[[Point], Point | None],
) -> npt.NDArray[np.float64] | None:
    """Derives the equivalent 3x3 homography (pitch-space meters -> pixels)
    from any pitch-space-to-pixel point-transform function -- e.g. a bound
    ``PitchKeypointCalibrator.inverse_transform_point`` for one resolved
    frame -- by evaluating it at the origin and the two unit basis points
    and solving for the implied affine map (translation + 2x2 linear part).

    Exists specifically so this module's homography decomposition can run
    against ``PitchKeypointCalibrator`` today: that calibrator only ever
    exposes point-transform functions, since internally it's a similarity
    transform (rotation + uniform scale + translation), not a stored 3x3
    matrix -- see its module docstring. This helper is exact for a
    calibrator whose transform genuinely is affine (true for
    ``PitchKeypointCalibrator``'s similarity transform, a special case of
    affine); a calibrator with real projective/perspective terms (the
    eventual ``TrainedPitchCalibrator``, once it fits keypoints via
    ``cv2.findHomography`` directly) should pass its already-projective
    homography straight to ``camera_pose_from_homography`` instead of going
    through this approximation.

    **Real limitation, found running this against real footage, not
    theoretical**: the homography this produces is exact for what
    ``PitchKeypointCalibrator`` computes, but that's *still* the wrong kind
    of input for ``camera_pose_from_homography`` -- a pure similarity
    transform's two columns are always equal-magnitude and exactly
    orthogonal by construction, which is precisely the degenerate case the
    self-calibration algorithm can't solve (it needs the differential
    foreshortening between axes that only comes from genuine camera
    perspective/tilt). Confirmed on ``benchmark_clip.mp4``:
    ``camera_pose_from_homography`` returned None for all 113/113 frames
    ``PitchKeypointCalibrator`` itself resolved (see
    ``scripts/decompose_pitch_camera_pose.py`` and the project plan's
    Phase 14 status note). A genuinely projective homography (e.g. a
    real 4+ point ``cv2.findHomography``/``cv2.getPerspectiveTransform``
    fit) does not have this problem -- confirmed by feeding the same
    decomposition ``ViewTransformer``'s fit from real hand-picked pixel
    corners instead, which succeeded numerically. So this bridge is
    correctness-preserving but not sufficient on its own to make
    ``PitchKeypointCalibrator`` a useful *input* to camera-pose
    decomposition; that needs either a real multi-feature homography for
    the classical calibrator (project plan Phase 12 item 2, not yet built)
    or the trained-keypoint-model calibrator's own real projective fit.

    Returns None if the transform is undefined (e.g. no resolved
    calibration) at any of the three probe points.
    """
    origin = inverse_transform_fn((0.0, 0.0))
    unit_x = inverse_transform_fn((1.0, 0.0))
    unit_y = inverse_transform_fn((0.0, 1.0))
    if origin is None or unit_x is None or unit_y is None:
        return None

    return np.array(
        [
            [unit_x[0] - origin[0], unit_y[0] - origin[0], origin[0]],
            [unit_x[1] - origin[1], unit_y[1] - origin[1], origin[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def project_point(pose: CameraPose, point_3d: Point3D) -> Point | None:
    """Projects a 3D pitch-space point (meters, z=0 for ground-plane points)
    into this pose's pixel space. No lens-distortion model applied (this
    project doesn't have distortion-calibration data) -- returns None for a
    point behind the camera, same convention as ``sn-calibration``'s own
    ``project_point``."""
    relative = np.array(point_3d) - pose.position
    rotated = pose.rotation @ relative
    if rotated[2] <= 1e-3:
        return None
    rotated = rotated / rotated[2]
    x = rotated[0] * pose.x_focal_length + pose.principal_point[0]
    y = rotated[1] * pose.y_focal_length + pose.principal_point[1]
    return float(x), float(y)
