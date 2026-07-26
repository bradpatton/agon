"""BoT-SORT tracking (via the ``boxmot`` package) behind the same
``FrameTracker`` interface as ``ByteTrackAdapter``, so it's a drop-in
``tracker=`` argument to ``UltralyticsDetector``/``OnnxDetector`` -- no
changes needed to ``detection/base.py``'s assembly logic.

Needs the ``[train]`` extra: ``boxmot`` imports ``torch`` unconditionally at
import time, even in motion-only mode with ``with_reid=False`` -- confirmed
empirically (not assumed) by tracing its import chain, which pulls in
``boxmot.trackers.common.geometry`` -> ``torch`` regardless of which tracker
class you actually use. So there's no way to get real BoT-SORT without
torch; it lives in the same torch-requiring family as
``UltralyticsDetector``, not alongside the torch-free ``OnnxDetector``
default.

``with_reid=False`` by default: boxmot's appearance/ReID matching needs its
own separate pretrained weights (not shipped by this project), and the part
of BoT-SORT most relevant here is its Kalman-filter/IoU motion model plus
``use_cmc=True`` camera-motion compensation (ECC-based, estimated from
consecutive frame images) -- which is why ``FrameTracker`` threads the
actual frame through, unlike plain ByteTrack. That CMC output covers
similar ground to ``soccer_analysis.camera.camera_movement_estimator``
and ``soccer_analysis.geometry.pitch_keypoint_calibrator``; reconciling
those three into one camera-motion source is future work, not done here.
Pass ``with_reid=True`` and a ``reid_model`` path to enable appearance
matching once a ReID checkpoint is available.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from soccer_analysis.io.video import Frame


class BoTSORTTracker:
    def __init__(
        self,
        with_reid: bool = False,
        use_cmc: bool = True,
        frame_rate: int = 24,
        reid_model: str | None = None,
        **kwargs,
    ):
        try:
            from boxmot.trackers.bbox.botsort import BotSort
        except ImportError as e:
            raise ImportError(
                "BoTSORTTracker needs the 'train' extra (torch + boxmot): "
                "pip install 'soccer-analysis[train]' boxmot"
            ) from e

        self._tracker = BotSort(
            reid_model=reid_model,
            with_reid=with_reid,
            use_cmc=use_cmc,
            frame_rate=frame_rate,
            **kwargs,
        )

    def update_with_detections(self, detections: sv.Detections, frame: Frame) -> sv.Detections:
        if len(detections) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.concatenate(
                [
                    detections.xyxy.astype(np.float32),
                    detections.confidence.reshape(-1, 1).astype(np.float32),
                    detections.class_id.reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            )

        result = np.asarray(self._tracker.update(dets, frame))
        if result.size == 0:
            return sv.Detections.empty()

        # boxmot's TrackResults columns: x1, y1, x2, y2, track_id, conf, cls, det_ind
        return sv.Detections(
            xyxy=result[:, 0:4].astype(np.float32),
            tracker_id=result[:, 4].astype(int),
            confidence=result[:, 5].astype(np.float32),
            class_id=result[:, 6].astype(int),
        )
