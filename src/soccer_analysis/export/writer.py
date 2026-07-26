"""Writers for the ML-ready tracking data export: JSONL (streaming,
human-inspectable) and Parquet (columnar, for ML loading via pandas/polars).
Both consume the same ``soccer_analysis.export.schema`` records, so they
always agree on content -- only the container format differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from soccer_analysis.export.schema import FrameRecord, MatchSummary


def write_jsonl(records: list[FrameRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(record.model_dump_json(by_alias=True))
            f.write("\n")


def _flatten_rows(records: list[FrameRecord]) -> list[dict[str, Any]]:
    """One row per object per frame; frames with zero objects still get one
    row (all object fields null) so every frame is represented for
    continuous time-series joins."""
    empty_object_fields: dict[str, Any] = {
        "track_id": None,
        "class": None,
        "team": None,
        "bbox_x1_px": None,
        "bbox_y1_px": None,
        "bbox_x2_px": None,
        "bbox_y2_px": None,
        "position_x_px": None,
        "position_y_px": None,
        "position_x_pitch_m": None,
        "position_y_pitch_m": None,
        "speed_kmh": None,
        "distance_m": None,
        "has_ball": None,
    }

    rows = []
    for record in records:
        base = {
            "schema_version": record.schema_version,
            "video_id": record.video_id,
            "frame_id": record.frame_id,
            "timestamp_s": record.timestamp_s,
            "camera_movement_x_px": record.camera_movement_px[0],
            "camera_movement_y_px": record.camera_movement_px[1],
            "team_ball_control": record.team_ball_control,
        }
        if not record.objects:
            rows.append({**base, **empty_object_fields})
            continue

        for obj in record.objects:
            pitch = obj.position_pitch_m
            rows.append(
                {
                    **base,
                    "track_id": obj.track_id,
                    "class": obj.object_class.value,
                    "team": obj.team,
                    "bbox_x1_px": obj.bbox_px[0],
                    "bbox_y1_px": obj.bbox_px[1],
                    "bbox_x2_px": obj.bbox_px[2],
                    "bbox_y2_px": obj.bbox_px[3],
                    "position_x_px": obj.position_px[0],
                    "position_y_px": obj.position_px[1],
                    "position_x_pitch_m": pitch[0] if pitch is not None else None,
                    "position_y_pitch_m": pitch[1] if pitch is not None else None,
                    "speed_kmh": obj.speed_kmh,
                    "distance_m": obj.distance_m,
                    "has_ball": obj.has_ball,
                }
            )
    return rows


def write_parquet(records: list[FrameRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(_flatten_rows(records))
    pq.write_table(table, path)


def write_match_summary(summary: MatchSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2))


def write_schema_json(path: str | Path) -> None:
    """Publishes FrameRecord's JSON Schema so downstream consumers can
    validate against the export format without importing this package."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(FrameRecord.model_json_schema(by_alias=True), indent=2))
