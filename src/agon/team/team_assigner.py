"""Assigns each tracked player to one of two teams via jersey-color clustering.

KMeans over raw pixel colors, satisfying ``agon.interfaces.TeamClassifier``.
See that protocol's docstring for a more robust embedding-based alternative
worth evaluating.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from agon.geometry.bbox import BBox
from agon.io.video import Frame

logger = logging.getLogger(__name__)


class TeamAssigner:
    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.team_colors: dict[int, npt.NDArray[np.float64]] = {}
        self.player_team_dict: dict[int, int] = {}
        self.kmeans: KMeans | None = None

    def _get_clustering_model(self, image: npt.NDArray[np.uint8]) -> KMeans:
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=self.random_state)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: Frame, bbox: BBox) -> npt.NDArray[np.float64]:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        image = frame[y1:y2, x1:x2]
        top_half_image = image[: image.shape[0] // 2, :]

        kmeans = self._get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame: Frame, player_detections: dict[int, dict[str, Any]]) -> None:
        player_colors = [
            self.get_player_color(frame, detection["bbox"])
            for detection in player_detections.values()
        ]

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=self.random_state)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: Frame, player_bbox: BBox, player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            raise RuntimeError("assign_team_color() must be called before get_player_team()")

        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
