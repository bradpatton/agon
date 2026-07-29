"""Embedding-based team classification via a small ONNX CNN backbone + KMeans.

Satisfies ``agon.interfaces.TeamClassifier``. Crops each player
detection, embeds it with a small pretrained MobileNetV3-Small backbone
(exported to ONNX via ``scripts/export_team_embedding_model.py`` -- torch is
only needed for that one-time export, same pattern as the ONNX detector),
and KMeans-clusters the embeddings instead of raw jersey-crop pixel colors
the way ``agon.team.team_assigner.TeamAssigner`` does. General
visual features (a CNN backbone) hold up better than raw pixels under
lighting variation, motion blur, and similar kit colors.

Still can't distinguish "team" from "referee" on its own -- that requires a
soccer-specific detector with a dedicated referee class (see
``agon.detection.onnx_tracker`` for where that limitation
actually originates; this classifier just clusters whatever player
detections it's given).

The clustering decision is embedding-based, but ``team_colors`` (used only
for drawing ellipses in the annotated video) is a separate, simple average
pixel color per cluster -- embeddings don't have a natural "color", and
annotation doesn't need clustering-quality features, just something visually
distinguishable.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from sklearn.cluster import KMeans

from agon.geometry.bbox import BBox
from agon.io.video import Frame

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = (224, 224)


def _preprocess_crop(crop: Frame) -> npt.NDArray[np.float32]:
    resized = cv2.resize(crop, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = normalized.transpose(2, 0, 1)
    return chw[np.newaxis, ...].astype(np.float32)


def _representative_color(frame: Frame, bbox: BBox) -> npt.NDArray[np.float64]:
    """Mean pixel color of the crop's top half, for drawing only (see module docstring)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = frame[y1:y2, x1:x2]
    top_half = crop[: crop.shape[0] // 2, :]
    if top_half.size == 0:
        return np.array([128.0, 128.0, 128.0])
    return top_half.reshape(-1, 3).mean(axis=0)


class EmbeddingTeamClassifier:
    def __init__(self, model_path: str, random_state: int = 0):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.team_colors: dict[int, npt.NDArray[np.float64]] = {}
        self.player_team_dict: dict[int, int] = {}

    def _embed(self, frame: Frame, bbox: BBox) -> npt.NDArray[np.float32]:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            n_features = self.session.get_outputs()[0].shape[-1]
            return np.zeros(n_features, dtype=np.float32)
        input_tensor = _preprocess_crop(crop)
        embedding = self.session.run(None, {self.input_name: input_tensor})[0]
        return embedding[0]

    def assign_team_color(self, frame: Frame, player_detections: dict[int, dict[str, Any]]) -> None:
        player_ids = list(player_detections.keys())
        embeddings = np.stack(
            [self._embed(frame, player_detections[pid]["bbox"]) for pid in player_ids]
        )
        colors = [
            _representative_color(frame, player_detections[pid]["bbox"]) for pid in player_ids
        ]

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(embeddings)
        self.kmeans = kmeans

        for cluster_id in (0, 1):
            members = [c for c, label in zip(colors, labels, strict=True) if label == cluster_id]
            mean_color = np.mean(members, axis=0) if members else np.array([128.0, 128.0, 128.0])
            self.team_colors[cluster_id + 1] = mean_color

        for player_id, label in zip(player_ids, labels, strict=True):
            self.player_team_dict[player_id] = int(label) + 1

    def get_player_team(self, frame: Frame, player_bbox: BBox, player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            raise RuntimeError("assign_team_color() must be called before get_player_team()")

        embedding = self._embed(frame, player_bbox)
        team_id = int(self.kmeans.predict(embedding.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
