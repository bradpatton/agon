"""Per-frame pitch calibration via a trained keypoint-detection model --
the real fix `PitchKeypointCalibrator`'s own docstring flags as blocked on
"training data/weights this project doesn't have" (project plan Phase 7
item 2 / Phase 12 item 5 / Phase 13). That block no longer holds: a
YOLO11n-pose checkpoint trained on 82,945 frames (SN-GSR-2025 + the legacy
Calibration dataset, see ``scripts/train_pitch_calibration.py``) predicts
all 38 canonical pitch keypoints (``agon.geometry.pitch_keypoints``) per
frame, each with its own pixel position and confidence.

Given at least ``min_keypoints`` confident keypoints in a frame, this class
solves a real projective homography (``cv2.findHomography`` with RANSAC)
from the canonical real-world (meters) <-> predicted pixel correspondences
-- unlike ``PitchKeypointCalibrator``'s similarity transform (rotation +
uniform scale only), this is a genuine perspective-correcting homography,
using as many independently-detected features as a frame happens to show
rather than just the center circle. It's also, not incidentally, the first
calibrator in this project whose homography is real/projective enough to
feed ``agon.geometry.camera_pose.camera_pose_from_homography`` (see that
module's docstring for why ``PitchKeypointCalibrator``'s output can't).

**Keypoint decoding was verified against real ONNX output before writing
any of the code below, not assumed** -- confirmed empirically (not from
Ultralytics' docs) that both the predicted bounding box and the 38
keypoints share one coordinate convention: absolute pixel positions in the
model's own (letterboxed, e.g. 960x960) input space, needing the same
unletterbox transform (subtract pad, divide by scale) already used for
detection boxes elsewhere in this project (``agon.detection.onnx_tracker``).
Also verified visually on two real frames: a tight center-circle shot
correctly placed "Middle line" point 0 exactly on the halfway-line/touchline
intersection; a wider goal-mouth shot (0.95 class confidence) correctly
placed 13 keypoints (goalpost base, both six-yard-box corners, a penalty-box
corner, the top touchline) each precisely on their real markings. A
close-up shot with no pitch visible at all correctly scored 0.04 class
confidence -- exactly the "no pitch here" signal this calibrator's
``min_keypoints`` gate depends on.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from agon.geometry.bbox import Point
from agon.geometry.pitch_keypoints import CANONICAL_KEYPOINTS, canonical_keypoint_real_xy
from agon.io.video import Frame

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = len(CANONICAL_KEYPOINTS)


def _letterbox(
    frame: Frame, new_shape: tuple[int, int]
) -> tuple[Frame, float, tuple[float, float]]:
    """Same square-letterbox scheme as ``agon.detection.onnx_tracker``
    (independent copy, not a shared import -- this project's other ONNX
    consumers, e.g. ``agon.jersey.onnx_classifier``, each own their
    preprocessing rather than reaching into another module's private
    helpers)."""
    h, w = frame.shape[:2]
    new_h, new_w = new_shape
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = new_w - resized_w, new_h - resized_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (float(left), float(top))  # type: ignore[return-value]  # cv2 stubs: dtype imprecise


def _preprocess(
    frame: Frame, input_size: tuple[int, int]
) -> tuple[npt.NDArray[np.float32], float, tuple[float, float]]:
    padded, scale, pad = _letterbox(frame, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...], scale, pad


def decode_best_pitch_keypoints(
    output: npt.NDArray[np.float32], scale: float, pad: tuple[float, float]
) -> npt.NDArray[np.float64]:
    """Decodes a raw ``(1, 4 + 1 + 38*3, num_anchors)`` YOLO11-pose ONNX
    output (4 bbox + 1 "pitch" class score + 38 keypoints x (x, y,
    confidence)) into one ``(38, 3)`` array of (pixel_x, pixel_y,
    confidence) per canonical keypoint, in the *original* (pre-letterbox)
    frame's pixel space.

    Picks the single highest-class-confidence anchor and decodes only its
    keypoints -- there is exactly one "pitch" instance per frame by
    construction (see ``scripts/convert_soccernet_calibration_to_pose.py``,
    which only ever writes one "pitch" object per training label), so
    unlike multi-instance object detection this needs no NMS across
    anchors, just picking the best one.

    Pure function, no ONNX Runtime session needed -- deliberately separated
    from ``TrainedPitchCalibrator._detect_keypoints`` so decode logic can be
    unit-tested against a hand-built array, without needing the real model
    file (matching this project's existing test-suite discipline of no
    video/model file required).
    """
    predictions = output[0].T  # (num_anchors, 119)
    class_scores = predictions[:, 4]
    best_idx = int(np.argmax(class_scores))

    keypoints = predictions[best_idx, 5:].reshape(NUM_KEYPOINTS, 3).astype(np.float64)
    pad_x, pad_y = pad
    keypoints[:, 0] = (keypoints[:, 0] - pad_x) / scale
    keypoints[:, 1] = (keypoints[:, 1] - pad_y) / scale
    return keypoints


def fit_homography_from_keypoints(
    keypoints: npt.NDArray[np.float64],
    keypoint_confidence: float,
    min_keypoints: int,
    ransac_reproj_threshold: float,
) -> npt.NDArray[np.float64] | None:
    """Given one frame's decoded ``(38, 3)`` keypoint array (see
    ``decode_best_pitch_keypoints``), filters to confident points and solves
    a pitch(meters)->pixel homography via RANSAC, or returns None if there
    aren't enough confident points to fit one.

    Pure function (no ONNX session), separated out from
    ``TrainedPitchCalibrator.calibrate`` specifically so this project's
    geometric fitting logic gets the same synthetic-input test coverage as
    every other calibrator here, without needing a real model file --
    ``TrainedPitchCalibrator``'s own ONNX-session-dependent parts
    (``_detect_keypoints``) follow this project's existing precedent for
    ONNX-backed classes (validated against real footage/scripts, not unit
    tests -- see ``OnnxDetector``/``OnnxJerseyClassifier``, neither of which
    has unit tests over its actual model-running code either).
    """
    world_points = []
    pixel_points = []
    for i, (name, endpoint_idx) in enumerate(CANONICAL_KEYPOINTS):
        px, py, confidence = keypoints[i]
        if confidence < keypoint_confidence:
            continue
        world_points.append(canonical_keypoint_real_xy(name, endpoint_idx))
        pixel_points.append((px, py))

    if len(world_points) < min_keypoints:
        return None

    homography, _inliers = cv2.findHomography(
        np.array(world_points, dtype=np.float64),
        np.array(pixel_points, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
    return homography  # type: ignore[return-value]  # cv2 stubs: dtype imprecise


class TrainedPitchCalibrator:
    """See module docstring. Satisfies ``agon.interfaces.PitchCalibrator``."""

    def __init__(
        self,
        model_path: str,
        keypoint_confidence: float = 0.3,
        min_keypoints: int = 4,
        input_size: tuple[int, int] = (960, 960),
        ransac_reproj_threshold: float = 10.0,
    ):
        """``min_keypoints=4``: ``cv2.findHomography`` needs at least 4
        non-collinear correspondences -- matches
        ``convert_soccernet_calibration_to_pose.py --min-keypoints``'s own
        default for the same reason, applied here at inference instead of
        training-data-filtering time. ``ransac_reproj_threshold`` is in
        destination (pixel) units -- a real, single confident-but-slightly-
        wrong keypoint (e.g. a partially-occluded box corner) shouldn't be
        able to distort the whole homography; RANSAC lets
        ``cv2.findHomography`` down-weight it as an outlier instead of
        fitting to it directly, given enough other correspondences to fit
        against.
        """
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.keypoint_confidence = keypoint_confidence
        self.min_keypoints = min_keypoints
        self.ransac_reproj_threshold = ransac_reproj_threshold
        # frame_idx -> homography, pitch-space meters -> pixels (matches
        # agon.geometry.camera_pose's convention, so self.homography()'s
        # output can be passed to camera_pose_from_homography directly).
        self._homographies: dict[int, npt.NDArray[np.float64]] = {}
        self._inverse_homographies: dict[int, npt.NDArray[np.float64]] = {}

    def _detect_keypoints(self, frame: Frame) -> npt.NDArray[np.float64]:
        input_tensor, scale, pad = _preprocess(frame, self.input_size)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        return decode_best_pitch_keypoints(output, scale, pad)

    def calibrate(self, frames: list[Frame], frame_offset: int = 0) -> None:
        resolved = 0
        for local_idx, frame in enumerate(frames):
            keypoints = self._detect_keypoints(frame)
            homography = fit_homography_from_keypoints(
                keypoints,
                self.keypoint_confidence,
                self.min_keypoints,
                self.ransac_reproj_threshold,
            )
            if homography is None:
                continue

            frame_idx = frame_offset + local_idx
            self._homographies[frame_idx] = homography
            self._inverse_homographies[frame_idx] = np.linalg.inv(homography)  # type: ignore[assignment]  # cv2 stubs: dtype imprecise
            resolved += 1

        logger.info(
            "Trained pitch calibration: resolved %d/%d frames (>= %d keypoints "
            "at >= %.2f confidence, homography fit succeeded)",
            resolved,
            len(frames),
            self.min_keypoints,
            self.keypoint_confidence,
        )

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None:
        if math.isnan(point[0]) or math.isnan(point[1]):
            return None

        inverse = self._inverse_homographies.get(frame_idx)
        if inverse is None:
            return None

        homogeneous = inverse @ np.array([point[0], point[1], 1.0])
        if homogeneous[2] == 0:
            return None
        return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])

    def inverse_transform_point(self, pitch_point: Point, frame_idx: int = 0) -> Point | None:
        """The inverse of ``transform_point``: pitch-space meters -> pixel
        space, for one resolved frame. Same convention/purpose as
        ``PitchKeypointCalibrator.inverse_transform_point`` (self-consistency
        checks, visual overlay tools -- see that method's docstring) --
        both use ``agon.geometry.pitch_keypoints``' coordinate convention
        (origin at pitch center, x = length, y = width), so a caller like
        ``scripts/render_pitch_markings_overlay.py`` works against either
        calibrator unchanged."""
        homography = self._homographies.get(frame_idx)
        if homography is None:
            return None

        homogeneous = homography @ np.array([pitch_point[0], pitch_point[1], 1.0])
        if homogeneous[2] == 0:
            return None
        return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])

    def homography(self, frame_idx: int = 0) -> npt.NDArray[np.float64] | None:
        """The resolved pitch(meters)->pixel homography for one frame, or
        None if unresolved. Not part of the ``PitchCalibrator`` protocol --
        exposed specifically so callers (e.g.
        ``agon.geometry.camera_pose.camera_pose_from_homography``) can get
        the real projective homography directly, rather than reconstructing
        an approximation of it via ``homography_from_point_transform``
        (which exists for calibrators that *don't* expose one)."""
        return self._homographies.get(frame_idx)
