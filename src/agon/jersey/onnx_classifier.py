"""ONNX-backed jersey number classifier.

Satisfies ``agon.interfaces.JerseyClassifier``. Loads a model exported by
``scripts/train_jersey_classifier.py --export-onnx`` (an Ultralytics
classification checkpoint), plus the ``classes.json`` sidecar that script
writes alongside it -- Ultralytics' output-index -> label-string order
isn't reliably recoverable from the ONNX file alone, so it's written out
explicitly at export time rather than re-derived here.

Empirically validated against a real exported model (2026-08-03):
32 real jersey crops run through both this class and
``YOLO(best.pt).predict()`` on the ML training machine. The output-format
assumption (ONNX graph output is already post-softmax, not raw logits) was
correct -- confirmed by reading Ultralytics' own ``Classify.forward()``
(``ultralytics/nn/modules/head.py``), which applies ``x.softmax(1)``
whenever ``export=True``. The preprocessing assumption was wrong and has
been fixed: Ultralytics' classification preprocessing (``classify_transforms``
in ``ultralytics/data/augment.py``) uses ``DEFAULT_MEAN=(0,0,0)``,
``DEFAULT_STD=(1,1,1)`` -- i.e. plain [0, 1] scaling, no ImageNet
normalization -- plus a shortest-edge resize + center-crop for a square
target size, not a direct stretch-resize. The original version of this
class used ImageNet mean/std and a direct resize, which produced
essentially random predictions (3/32 matches, mean confidence gap 0.536)
against the real reference model; both are fixed below and re-validated
(32/32 label matches, mean confidence gap 0.0087, max 0.114 -- the small
remaining gap is float32 rounding-order differences between torch and
onnxruntime, not a preprocessing mismatch).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from agon.geometry.bbox import BBox
from agon.io.video import Frame

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "unknown"


class OnnxJerseyClassifier:
    def __init__(self, model_path: str, confidence_threshold: float = 0.0):
        """``confidence_threshold``: below this, ``classify()`` returns
        ``(None, confidence)`` even for a numeric top class, so a caller
        doesn't have to separately re-check confidence itself. 0.0
        (default) never abstains here -- ``agon.jersey.aggregator``
        applies its own, track-level threshold regardless, so a low
        per-call value is fine; set this higher only if you also want
        single-frame calls used outside that aggregator to self-filter.
        """
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (int(input_shape[-1]), int(input_shape[-2]))  # (W, H)
        self.confidence_threshold = confidence_threshold

        classes_path = Path(model_path).with_name("classes.json")
        if not classes_path.exists():
            raise FileNotFoundError(
                f"{classes_path} not found -- expected alongside {model_path}, written by "
                f"scripts/train_jersey_classifier.py --export-onnx."
            )
        raw_names: dict[str, str] = json.loads(classes_path.read_text())
        self.index_to_label: dict[int, str] = {int(k): v for k, v in raw_names.items()}

    def _preprocess(self, crop: Frame) -> npt.NDArray[np.float32]:
        """Mirrors Ultralytics' ``classify_transforms`` (see module
        docstring): for a square target size, that's a shortest-edge resize
        (preserving aspect ratio) followed by a center-crop to the target
        size -- not a direct stretch-resize, which would distort the crop
        and was the original, wrong version of this method."""
        target_w, target_h = self.input_size
        h, w = crop.shape[:2]
        if target_w == target_h:
            scale = target_w / min(h, w)
            new_w, new_h = round(w * scale), round(h * scale)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top = (new_h - target_h) // 2
            left = (new_w - target_w) // 2
            square = resized[top : top + target_h, left : left + target_w]
        else:
            square = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = rgb.transpose(2, 0, 1)
        return chw[np.newaxis, ...].astype(np.float32)

    def classify(self, frame: Frame, bbox: BBox) -> tuple[int | None, float]:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        input_tensor = self._preprocess(crop)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        probs = output[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        label = self.index_to_label.get(top_idx, UNKNOWN_LABEL)

        if label == UNKNOWN_LABEL or confidence < self.confidence_threshold:
            return None, confidence
        try:
            return int(label), confidence
        except ValueError:
            # Defensive: a label that isn't a digit string and isn't literally
            # "unknown" would mean classes.json doesn't match the expected
            # training format (see convert_soccernet_gsr_to_jersey_crops.py).
            logger.warning("Unexpected jersey class label %r, treating as unknown", label)
            return None, confidence
