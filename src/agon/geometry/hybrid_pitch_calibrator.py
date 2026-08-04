"""Falls back between two ``PitchCalibrator``s per point, instead of
committing to one calibration strategy for a whole clip.

Motivated by a real, measured gap: on a real broadcast clip, the static
``ViewTransformer`` (one fixed homography for the whole clip) resolved a
pitch position for only 34.4% of player detections, and the dynamic
``PitchKeypointCalibrator`` (per-frame center-circle detection) resolved
56.9% -- but checking *which* detections each one covered showed they
mostly fail on different frames, not the same ones. Trying the primary
calibrator first and falling back to the secondary when it returns None
covered 73.5% of the same detections -- using two calibrators that
already exist, with zero new CV/ML work. See the project plan's Phase 12
for the full measurement.

The obvious default pairing is dynamic-primary/static-fallback (prefer
the per-frame calibration when it's available, since it tracks real
camera movement instead of assuming a fixed shot), but this class doesn't
hardcode that -- it just tries whichever calibrator is given first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from agon.geometry.bbox import Point
from agon.interfaces import PitchCalibrator
from agon.io.video import Frame


class HybridPitchCalibrator:
    def __init__(self, primary: PitchCalibrator, fallback: PitchCalibrator):
        self.primary = primary
        self.fallback = fallback

    def calibrate(self, frames: list[Frame], frame_offset: int = 0) -> None:
        self.primary.calibrate(frames, frame_offset)
        self.fallback.calibrate(frames, frame_offset)

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None:
        result = self.primary.transform_point(point, frame_idx)
        if result is not None:
            return result
        return self.fallback.transform_point(point, frame_idx)

    def homography(self, frame_idx: int = 0) -> npt.NDArray[np.float64] | None:
        """Passthrough to whichever wrapped calibrator resolved this frame
        *and* exposes a real homography (today: only
        ``TrainedPitchCalibrator``, or another ``HybridPitchCalibrator``
        wrapping one -- this method recurses correctly since it has the
        same name/signature). Not part of the ``PitchCalibrator`` protocol,
        same as ``TrainedPitchCalibrator.homography`` -- exists so wrapping
        a trained calibrator in a hybrid chain doesn't silently lose its
        camera-pose-decomposition capability (see
        ``agon.geometry.camera_pose``). Tries primary then fallback, same
        order as ``transform_point`` -- returns None if neither resolved
        this frame with a real homography (e.g. both wrapped calibrators
        are similarity-transform-only, or neither resolved this frame at
        all), not a guess.
        """
        for calibrator in (self.primary, self.fallback):
            homography_fn = getattr(calibrator, "homography", None)
            if homography_fn is None:
                continue
            result = homography_fn(frame_idx)
            if result is not None:
                return result
        return None
