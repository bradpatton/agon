"""ONNX Runtime-backed detector -- the default, torch-free Detector.

Satisfies ``soccer_analysis.interfaces.Detector`` without depending on torch
or the ``ultralytics`` package at runtime: only ``onnxruntime`` (a much
lighter, more broadly-wheeled dependency -- see the README's modernization
notes on why torch is pushed to an optional ``[train]`` extra instead).

Export any Ultralytics checkpoint to a compatible ``.onnx`` file with
``model.export(format="onnx", imgsz=640, dynamic=False)`` (requires the
``[train]`` extra for that one-time conversion; the resulting ``.onnx`` file
itself needs nothing but this module + onnxruntime to run).

A generic COCO checkpoint (e.g. Ultralytics' pretrained ``yolo11n.onnx``) only
has ``person`` and ``sports ball`` classes -- there's no way to distinguish
players from referees, so ``tracks["referees"]`` will always be empty when
run against one. Use a soccer-fine-tuned checkpoint (with
player/goalkeeper/referee/ball classes) for that distinction.

The ball is intentionally *not* run through the multi-object tracker: see
``soccer_analysis.detection.tracker`` for why (same reasoning applies here).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import supervision as sv
from tqdm import tqdm

from soccer_analysis.detection.base import (
    ByteTrackAdapter,
    FrameTracker,
    Tracks,
    run_detection_and_tracking,
)
from soccer_analysis.io.video import Frame

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAME_TO_OBJECT_TYPE = {
    "player": "players",
    "goalkeeper": "players",
    "referee": "referees",
    "ball": "ball",
    # Generic COCO checkpoints have no soccer-specific classes: map the
    # closest COCO labels so the pipeline still runs (see module docstring
    # for the "referees" limitation this implies).
    "person": "players",
    "sports ball": "ball",
}


def _letterbox(
    frame: Frame, new_shape: tuple[int, int]
) -> tuple[Frame, float, tuple[float, float]]:
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
    return padded, scale, (float(left), float(top))


def _preprocess(
    frame: Frame, input_size: tuple[int, int]
) -> tuple[npt.NDArray[np.float32], float, tuple[float, float]]:
    padded, scale, pad = _letterbox(frame, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...], scale, pad


def _postprocess(
    output: npt.NDArray[np.float32],
    scale: float,
    pad: tuple[float, float],
    original_shape: tuple[int, int],
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> sv.Detections:
    """Decodes a raw (1, 4+num_classes, num_anchors) YOLOv8/11-style output."""
    predictions = output[0].T  # (num_anchors, 4 + num_classes)
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    keep = confidences >= confidence_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    if len(boxes_cxcywh) == 0:
        return sv.Detections.empty()

    cx, cy, w, h = (boxes_cxcywh[:, i] for i in range(4))
    x1 = cx - w / 2
    y1 = cy - h / 2

    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    w = w / scale
    h = h / scale

    height, width = original_shape
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    w = np.clip(w, 0, width - x1)
    h = np.clip(h, 0, height - y1)

    indices = cv2.dnn.NMSBoxes(
        bboxes=np.stack([x1, y1, w, h], axis=1).tolist(),
        scores=confidences.tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=nms_iou_threshold,
    )
    if len(indices) == 0:
        return sv.Detections.empty()
    indices = np.array(indices).reshape(-1)

    xyxy = np.stack([x1, y1, x1 + w, y1 + h], axis=1)[indices]
    return sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=confidences[indices].astype(np.float32),
        class_id=class_ids[indices].astype(int),
    )


class OnnxDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.1,
        nms_iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        class_name_to_object_type: dict[str, str] | None = None,
        tracker: FrameTracker | None = None,
    ):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.confidence = confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.class_names = self._read_class_names()
        self.tracker = tracker or ByteTrackAdapter()
        self.class_name_to_object_type = (
            class_name_to_object_type or DEFAULT_CLASS_NAME_TO_OBJECT_TYPE
        )

    def _read_class_names(self) -> dict[int, str]:
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" not in meta:
            raise ValueError(
                "ONNX model has no 'names' metadata -- was it exported via "
                "Ultralytics' model.export(format='onnx')?"
            )
        return ast.literal_eval(meta["names"])

    def _detect_one(self, frame: Frame) -> sv.Detections:
        input_tensor, scale, pad = _preprocess(frame, self.input_size)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        return _postprocess(
            output, scale, pad, frame.shape[:2], self.confidence, self.nms_iou_threshold
        )

    def get_object_tracks(
        self,
        frames: list[Frame],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> Tracks:
        def detect_all_frames() -> list[sv.Detections]:
            return [
                self._detect_one(frame)
                for frame in tqdm(frames, desc="Detecting objects (onnxruntime)")
            ]

        return run_detection_and_tracking(
            detect_all_frames,
            frames=frames,
            class_names=self.class_names,
            class_name_to_object_type=self.class_name_to_object_type,
            tracker=self.tracker,
            read_from_stub=read_from_stub,
            stub_path=stub_path,
        )
