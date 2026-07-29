"""Writers for the ML-ready tracking data export: JSONL (streaming,
human-inspectable) and Parquet (columnar, for ML loading via pandas/polars).
Both consume the same ``agon.export.schema`` records, so they
always agree on content -- only the container format differs.

``JsonlWriter``/``ParquetChunkWriter`` are the incremental (open-once,
write-many, close) variants used by streaming/chunked processing
(``agon.pipeline.run_pipeline_streaming``), so a multi-hour run
never has to hold every record for the whole clip in memory at once.
``write_jsonl``/``write_parquet`` (whole-list-at-once) are kept for the
non-streaming pipeline and are now thin wrappers around the same
incremental writers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from agon.export.schema import FrameRecord, MatchSummary

# Explicit, fixed schema (rather than letting pyarrow infer one per chunk):
# a chunk that happens to have no objects at all (e.g. a short chunk with
# zero detections) would otherwise infer null-typed columns, which then
# conflicts with a later chunk's concrete-typed data for the same column --
# ParquetWriter requires every chunk written through it to match the schema
# it was opened with.
PARQUET_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_id", pa.int64()),
        ("timestamp_s", pa.float64()),
        ("camera_movement_x_px", pa.float64()),
        ("camera_movement_y_px", pa.float64()),
        ("team_ball_control", pa.int64()),
        ("track_id", pa.int64()),
        ("class", pa.string()),
        ("team", pa.int64()),
        ("bbox_x1_px", pa.float64()),
        ("bbox_y1_px", pa.float64()),
        ("bbox_x2_px", pa.float64()),
        ("bbox_y2_px", pa.float64()),
        ("position_x_px", pa.float64()),
        ("position_y_px", pa.float64()),
        ("position_x_pitch_m", pa.float64()),
        ("position_y_pitch_m", pa.float64()),
        ("speed_kmh", pa.float64()),
        ("distance_m", pa.float64()),
        ("has_ball", pa.bool_()),
    ]
)


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


class JsonlWriter:
    """Open-once, write-many-chunks, close context manager for JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")  # noqa: SIM115 -- closed in close()/__exit__
        self._records_written = 0

    def write_chunk(self, records: list[FrameRecord]) -> None:
        for record in records:
            self._file.write(record.model_dump_json(by_alias=True))
            self._file.write("\n")
        self._records_written += len(records)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class ParquetChunkWriter:
    """Open-once, write-many-chunks, close context manager for Parquet,
    via pyarrow's row-group-per-chunk incremental writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(str(self.path), PARQUET_SCHEMA)

    def write_chunk(self, records: list[FrameRecord]) -> None:
        rows = _flatten_rows(records)
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        self._writer.write_table(table)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> ParquetChunkWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def write_jsonl(records: list[FrameRecord], path: str | Path) -> None:
    with JsonlWriter(path) as writer:
        writer.write_chunk(records)


def write_parquet(records: list[FrameRecord], path: str | Path) -> None:
    with ParquetChunkWriter(path) as writer:
        writer.write_chunk(records)


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
