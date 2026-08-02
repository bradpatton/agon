"""ONNX-backed jersey number classifier.

Satisfies ``agon.interfaces.JerseyClassifier``. Loads a model exported by
``scripts/train_jersey_classifier.py --export-onnx`` (an Ultralytics
classification checkpoint), plus the ``classes.json`` sidecar that script
writes alongside it -- Ultralytics' output-index -> label-string order
isn't reliably recoverable from the ONNX file alone, so it's written out
explicitly at export time rather than re-derived here.

**Not yet empirically verified against a real exported model** -- written
before this project's first jersey-classifier ONNX export finished
training. Preprocessing (ImageNet mean/std normalization) and the
assumption that the ONNX graph's output is already post-softmax
probabilities (not raw logits) both match Ultralytics' documented
classification convention, but neither has been cross-checked yet against
``YOLO(onnx_path).predict()``'s own output on the same crops. Do that
check before trusting this in production -- a preprocessing mismatch
degrades accuracy silently, unlike a shape mismatch, which fails loudly
(see OnnxDetector's docstring for the equivalent, already-validated case).
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

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = normalized.transpose(2, 0, 1)
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
