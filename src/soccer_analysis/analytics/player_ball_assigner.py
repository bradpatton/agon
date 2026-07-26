"""Assigns ball possession to the nearest player within a distance threshold."""

from __future__ import annotations

from typing import Any

from soccer_analysis.geometry.bbox import BBox, get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance_px: float = 70.0):
        self.max_player_ball_distance_px = max_player_ball_distance_px

    def assign_ball_to_player(
        self, players: dict[int, dict[str, Any]], ball_bbox: BBox
    ) -> int | None:
        ball_position = get_center_of_bbox(ball_bbox)

        best_player_id: int | None = None
        best_distance = float("inf")

        for player_id, player in players.items():
            x1, _, x2, y2 = player["bbox"]
            distance = min(
                measure_distance((x1, y2), ball_position),
                measure_distance((x2, y2), ball_position),
            )

            if distance < self.max_player_ball_distance_px and distance < best_distance:
                best_distance = distance
                best_player_id = player_id

        return best_player_id
