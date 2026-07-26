"""Versioned schema for the ML-ready tracking data export -- the actual
point of this project (see the top-level README): structured per-frame
player/ball/team/position/speed data that downstream ML tooling can load
directly, not just an annotated video.

``schema_version`` is on every top-level record so a consumer can detect a
future breaking change instead of silently misreading fields. Bump it
whenever a field is renamed, removed, or changes meaning (adding an
optional field is not a breaking change and doesn't need a bump).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class ObjectClass(StrEnum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


def object_class_for(object_type: str, raw_class_name: str) -> ObjectClass:
    """Maps a tracks-dict bucket ("players"/"referees"/"ball") plus the
    detector's raw class name to one of the four canonical classes above.

    The raw class name is what disambiguates player vs. goalkeeper -- both
    land in the "players" bucket during tracking (see
    ``soccer_analysis.detection.base``), so the bucket alone can't tell them
    apart. Works the same whether the checkpoint used soccer-specific labels
    (player/goalkeeper/referee/ball) or a generic COCO model
    (person/sports ball, which the bucket already normalized to
    "players"/"ball" -- there's no way to recover "goalkeeper" from a COCO
    checkpoint, so those always export as PLAYER).
    """
    if object_type == "referees":
        return ObjectClass.REFEREE
    if object_type == "ball":
        return ObjectClass.BALL
    return ObjectClass.GOALKEEPER if raw_class_name == "goalkeeper" else ObjectClass.PLAYER


class ObjectRecord(BaseModel):
    track_id: int
    object_class: ObjectClass = Field(alias="class")
    team: int | None = Field(default=None, description="1 or 2; null for ball/referee/unassigned.")
    bbox_px: tuple[float, float, float, float] = Field(description="x1, y1, x2, y2 in pixel space.")
    position_px: tuple[float, float] = Field(
        description="Foot position for players/referees, center for the ball; pixel space."
    )
    position_pitch_m: tuple[float, float] | None = Field(
        default=None,
        description="Position in pitch-space meters, per the active PitchCalibrator. "
        "Null when outside the calibrated pitch area, or when the calibrator "
        "found no usable reference in this frame (see PitchKeypointCalibrator).",
    )
    speed_kmh: float | None = None
    distance_m: float | None = Field(
        default=None, description="Cumulative distance covered so far."
    )
    has_ball: bool = False

    model_config = {"populate_by_name": True}


class FrameRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video_id: str
    frame_id: int
    timestamp_s: float
    camera_movement_px: tuple[float, float]
    team_ball_control: int = Field(description="1 or 2; 0 if no team has had the ball yet.")
    objects: list[ObjectRecord]


class PlayerSummary(BaseModel):
    track_id: int
    team: int | None
    total_distance_m: float
    avg_speed_kmh: float
    max_speed_kmh: float


class MatchSummary(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video_id: str
    frame_count: int
    frame_rate: float
    team_1_possession_pct: float
    team_2_possession_pct: float
    players: list[PlayerSummary]


def build_frame_records(
    tracks: dict[str, list[dict[int, dict[str, Any]]]],
    team_ball_control: list[int],
    camera_movement_per_frame: list[tuple[float, float]],
    video_id: str,
    frame_rate: float,
) -> list[FrameRecord]:
    num_frames = len(tracks["players"])
    records = []

    for frame_idx in range(num_frames):
        objects: list[ObjectRecord] = []
        for object_type in ("players", "referees", "ball"):
            for track_id, info in tracks[object_type][frame_idx].items():
                position_transformed = info.get("position_transformed")
                objects.append(
                    ObjectRecord(
                        track_id=int(track_id),
                        **{"class": object_class_for(object_type, info.get("class_name", ""))},
                        team=info.get("team"),
                        bbox_px=tuple(info["bbox"]),
                        position_px=tuple(info["position"]),
                        position_pitch_m=(
                            tuple(position_transformed)
                            if position_transformed is not None
                            else None
                        ),
                        speed_kmh=info.get("speed"),
                        distance_m=info.get("distance"),
                        has_ball=bool(info.get("has_ball", False)),
                    )
                )

        records.append(
            FrameRecord(
                video_id=video_id,
                frame_id=frame_idx,
                timestamp_s=frame_idx / frame_rate,
                camera_movement_px=camera_movement_per_frame[frame_idx],
                team_ball_control=int(team_ball_control[frame_idx]),
                objects=objects,
            )
        )

    return records


def build_match_summary(
    tracks: dict[str, list[dict[int, dict[str, Any]]]],
    team_ball_control: list[int],
    video_id: str,
    frame_rate: float,
) -> MatchSummary:
    control = list(team_ball_control)
    team_1_frames = sum(1 for c in control if c == 1)
    team_2_frames = sum(1 for c in control if c == 2)
    denom = team_1_frames + team_2_frames
    team_1_pct = (team_1_frames / denom * 100) if denom else 0.0
    team_2_pct = (team_2_frames / denom * 100) if denom else 0.0

    player_stats: dict[int, dict[str, Any]] = {}
    for frame_track in tracks["players"]:
        for track_id, info in frame_track.items():
            stats = player_stats.setdefault(
                int(track_id), {"team": None, "distances": [], "speeds": []}
            )
            if info.get("team") is not None:
                stats["team"] = info["team"]
            if info.get("distance") is not None:
                stats["distances"].append(info["distance"])
            if info.get("speed") is not None:
                stats["speeds"].append(info["speed"])

    players = [
        PlayerSummary(
            track_id=track_id,
            team=stats["team"],
            total_distance_m=max(stats["distances"]) if stats["distances"] else 0.0,
            avg_speed_kmh=(sum(stats["speeds"]) / len(stats["speeds"])) if stats["speeds"] else 0.0,
            max_speed_kmh=max(stats["speeds"]) if stats["speeds"] else 0.0,
        )
        for track_id, stats in sorted(player_stats.items())
    ]

    return MatchSummary(
        video_id=video_id,
        frame_count=len(control),
        frame_rate=frame_rate,
        team_1_possession_pct=team_1_pct,
        team_2_possession_pct=team_2_pct,
        players=players,
    )
