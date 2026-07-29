"""Converts SoccerNet Game State Reconstruction (SN-GSR-2025) sequences'
pitch-line annotations into a clean, pixel-space training format for a
learned pitch-calibration model (project plan Phase 7, item 2 -- the
eventual replacement for ``soccer_analysis.geometry.pitch_keypoint_calibrator
.PitchKeypointCalibrator``'s classical-CV first cut).

SN-GSR's ``category_id=5`` ("pitch") annotations give one entry per frame:
a dict of named pitch lines (e.g. "Circle left", "Side line top" -- the
standard SoccerNet Camera Calibration line taxonomy, 26 possible names) to
a polyline of points, normalized to [0, 1] in image space. Real broadcast
footage only shows part of the pitch at once, so most frames have a subset
of the 26 lines (observed on one real sequence: 3-13 lines visible per
frame, averaging ~10) -- absence of a line is a real "not visible", not a
gap to fill in.

This script does NOT train a model -- it only produces the intermediate
per-sequence JSON (pixel-space points, image dims, one file per SNGS-*
sequence) that a future heatmap/keypoint-regression training script would
load. Deliberately scoped this way: picking the actual model architecture
(backbone, heatmap resolution, loss) is a separate design decision, not
just a data-format conversion.

Usage:
    python scripts/convert_soccernet_gsr_to_calibration.py <sngs_root> <output_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PITCH_CATEGORY_ID = 5


def convert_sequence(seq_dir: Path) -> dict[str, Any]:
    labels_path = seq_dir / "Labels-GameState.json"
    data = json.loads(labels_path.read_text())

    im_dir = data["info"]["im_dir"]
    images_by_id = {img["image_id"]: img for img in data["images"]}

    frames = {}
    for ann in data["annotations"]:
        if ann["category_id"] != PITCH_CATEGORY_ID:
            continue
        img = images_by_id.get(ann["image_id"])
        if img is None:
            continue

        width, height = img["width"], img["height"]
        lines_px = {
            line_name: [[pt["x"] * width, pt["y"] * height] for pt in points]
            for line_name, points in ann["lines"].items()
        }
        frames[img["file_name"]] = {
            "image_path": str((seq_dir / im_dir / img["file_name"]).resolve()),
            "width": width,
            "height": height,
            "lines_px": lines_px,
        }

    return {
        "sequence": seq_dir.name,
        "num_frames_with_pitch_annotation": len(frames),
        "frames": frames,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/convert_soccernet_gsr_to_calibration.py "
            "<sngs_root> <output_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    sngs_root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p for p in sngs_root.glob("SNGS-*") if p.is_dir())
    if not seq_dirs:
        print(f"No SNGS-* sequence directories found under {sngs_root}", file=sys.stderr)
        sys.exit(1)

    total_frames = 0
    for seq_dir in seq_dirs:
        converted = convert_sequence(seq_dir)
        out_path = output_dir / f"{seq_dir.name}.json"
        out_path.write_text(json.dumps(converted))
        print(
            f"{seq_dir.name}: {converted['num_frames_with_pitch_annotation']} frames -> {out_path}"
        )
        total_frames += converted["num_frames_with_pitch_annotation"]

    print(
        f"\nTotal: {total_frames} frames with pitch-line annotations "
        f"across {len(seq_dirs)} sequences"
    )


if __name__ == "__main__":
    main()
