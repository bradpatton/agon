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

from dataclasses import dataclass, field
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
    ``agon.detection.base``), so the bucket alone can't tell them
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
    jersey_number: int | None = Field(
        default=None,
        description="0-99, or null for ball/referee/unassigned/illegible. Always null "
        "until a jersey-number recognizer is wired into the pipeline (project plan "
        "Phase 7, item 3) -- the export field exists now so the schema doesn't need a "
        "breaking change once one is. See scripts/convert_soccernet_gsr_to_jersey_crops.py "
        "and scripts/train_jersey_classifier.py for the training side of that model, "
        "already built and validated end-to-end.",
    )

    model_config = {"populate_by_name": True}


class CameraPoseRecord(BaseModel):
    """Full 3D camera pose for one frame, from
    ``agon.geometry.camera_pose.camera_pose_from_homography`` -- decoded
    from a real projective pitch-plane homography, not just a flat
    pixel<->pitch mapping. Deliberately holds pan/tilt/roll (degrees) and
    position (meters) rather than the raw 3x3 rotation matrix
    ``CameraPose.rotation`` carries internally -- easier for downstream
    consumers to use directly, and still fully reconstructable (see
    ``agon.geometry.camera_pose.pan_tilt_roll_to_rotation``) for anyone who
    needs the matrix back.

    Only ever populated for frames resolved by a calibrator that exposes a
    real per-frame homography (today: ``TrainedPitchCalibrator``, or a
    ``HybridPitchCalibrator`` wrapping one) -- see
    ``agon.geometry.camera_pose``'s own module docstring for why
    ``ViewTransformer``/``PitchKeypointCalibrator`` structurally can't
    produce one. Most frames in most clips will have this null; that's
    real "no per-frame homography available here," not a bug.
    """

    pan_degrees: float
    tilt_degrees: float
    roll_degrees: float
    position_m: tuple[float, float, float] = Field(
        description="Camera position in pitch-space meters (same convention as "
        "agon.geometry.pitch_keypoints: origin at pitch center, x = length, y = width)."
    )
    x_focal_length_px: float
    y_focal_length_px: float


class FrameRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video_id: str
    frame_id: int
    timestamp_s: float
    camera_movement_px: tuple[float, float]
    team_ball_control: int = Field(description="1 or 2; 0 if no team has had the ball yet.")
    objects: list[ObjectRecord]
    frame_classification: str | None = Field(
        default=None,
        description="'live_play'/'replay'/'graphic' (see "
        "agon.broadcast.frame_filter), or null when "
        "frame_filter_mode='off' (the default).",
    )
    game_clock_s: float | None = Field(
        default=None,
        description="Elapsed game-time seconds read from the broadcast's "
        "on-screen match clock (see agon.broadcast.clock_reader). "
        "Null unless a clock_calibration was configured and the clock was "
        "readable in this frame -- distinct from timestamp_s, which is the "
        "video file's own time axis.",
    )
    camera_pose: CameraPoseRecord | None = Field(
        default=None,
        description="Full 3D camera pose for this frame, if a calibrator that exposes a "
        "real per-frame homography resolved it (see CameraPoseRecord). Null otherwise "
        "-- most frames, with today's calibrators.",
    )


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
    frame_offset: int = 0,
    frame_ids: list[int] | None = None,
    frame_classifications: list[str] | None = None,
    game_clock_s_per_frame: list[float | None] | None = None,
    camera_poses: list[CameraPoseRecord | None] | None = None,
) -> list[FrameRecord]:
    """``frame_offset``: global (match-relative) index of ``tracks``' first
    frame. Only matters for chunked/streaming processing, where ``tracks``
    covers one chunk, not the whole clip -- 0 (the default) is correct for
    a single whole-clip call.

    ``frame_ids``: explicit, parallel-to-``tracks["players"]`` source-video
    frame index per record, overriding the ``frame_offset + local_idx``
    computation. Needed when ``agon.broadcast``'s frame filter
    has dropped some frames from this call's ``tracks`` (frame_filter_mode=
    'strip'), so the surviving frames are no longer contiguously numbered
    and a single offset can't reconstruct their true indices. None (the
    default) keeps the contiguous frame_offset + local_idx behavior.

    ``frame_classifications``/``game_clock_s_per_frame``/``camera_poses``:
    parallel to ``tracks["players"]`` (one entry per frame in this call).
    All default to None (not just an empty list) when the corresponding
    feature isn't in use, leaving every record's field null.
    """
    num_frames = len(tracks["players"])
    records = []

    for local_idx in range(num_frames):
        frame_idx = frame_ids[local_idx] if frame_ids is not None else frame_offset + local_idx
        objects: list[ObjectRecord] = []
        for object_type in ("players", "referees", "ball"):
            for track_id, info in tracks[object_type][local_idx].items():
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
                        jersey_number=info.get("jersey_number"),
                    )
                )

        records.append(
            FrameRecord(
                video_id=video_id,
                frame_id=frame_idx,
                timestamp_s=frame_idx / frame_rate,
                camera_movement_px=camera_movement_per_frame[local_idx],
                team_ball_control=int(team_ball_control[local_idx]),
                objects=objects,
                frame_classification=(
                    frame_classifications[local_idx] if frame_classifications is not None else None
                ),
                game_clock_s=(
                    game_clock_s_per_frame[local_idx]
                    if game_clock_s_per_frame is not None
                    else None
                ),
                camera_pose=(camera_poses[local_idx] if camera_poses is not None else None),
            )
        )

    return records


@dataclass
class MatchStats:
    """Running accumulator for build_match_summary's inputs, so streaming
    processing can fold in one chunk at a time (accumulate_match_stats)
    instead of needing the whole match's tracks in memory to compute a
    summary (finalize_match_summary)."""

    team_1_frames: int = 0
    team_2_frames: int = 0
    player_stats: dict[int, dict[str, Any]] = field(default_factory=dict)


def accumulate_match_stats(
    stats: MatchStats,
    tracks: dict[str, list[dict[int, dict[str, Any]]]],
    team_ball_control: list[int],
) -> None:
    """Mutates ``stats`` in place, folding in one chunk's worth of tracks."""
    for c in team_ball_control:
        if c == 1:
            stats.team_1_frames += 1
        elif c == 2:
            stats.team_2_frames += 1

    for frame_track in tracks["players"]:
        for track_id, info in frame_track.items():
            player = stats.player_stats.setdefault(
                int(track_id), {"team": None, "distances": [], "speeds": []}
            )
            if info.get("team") is not None:
                player["team"] = info["team"]
            if info.get("distance") is not None:
                player["distances"].append(info["distance"])
            if info.get("speed") is not None:
                player["speeds"].append(info["speed"])


def finalize_match_summary(
    stats: MatchStats, video_id: str, frame_count: int, frame_rate: float
) -> MatchSummary:
    denom = stats.team_1_frames + stats.team_2_frames
    team_1_pct = (stats.team_1_frames / denom * 100) if denom else 0.0
    team_2_pct = (stats.team_2_frames / denom * 100) if denom else 0.0

    players = [
        PlayerSummary(
            track_id=track_id,
            team=player["team"],
            total_distance_m=max(player["distances"]) if player["distances"] else 0.0,
            avg_speed_kmh=(
                sum(player["speeds"]) / len(player["speeds"]) if player["speeds"] else 0.0
            ),
            max_speed_kmh=max(player["speeds"]) if player["speeds"] else 0.0,
        )
        for track_id, player in sorted(stats.player_stats.items())
    ]

    return MatchSummary(
        video_id=video_id,
        frame_count=frame_count,
        frame_rate=frame_rate,
        team_1_possession_pct=team_1_pct,
        team_2_possession_pct=team_2_pct,
        players=players,
    )


def build_match_summary(
    tracks: dict[str, list[dict[int, dict[str, Any]]]],
    team_ball_control: list[int],
    video_id: str,
    frame_rate: float,
) -> MatchSummary:
    """Whole-clip convenience wrapper around accumulate_match_stats +
    finalize_match_summary, for the non-streaming pipeline."""
    stats = MatchStats()
    accumulate_match_stats(stats, tracks, team_ball_control)
    return finalize_match_summary(stats, video_id, len(team_ball_control), frame_rate)
