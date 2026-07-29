"""Converts SoccerNet's legacy Camera Calibration dataset (standalone
action/replay images + per-image line-segment JSON, downloaded via
``download_soccernet_legacy.py --task calibration``) into the same
pixel-space training format ``convert_soccernet_gsr_to_calibration.py``
produces from SN-GSR-2025 -- both feed the same future pitch-calibration
keypoint model (project plan Phase 7, item 2), which doesn't exist yet.

Unlike SN-GSR (short video sequences, high frame-to-frame redundancy),
this dataset is standalone images curated from distinct action-spotting
moments across many different games/broadcasts/camera angles -- much
higher scene diversity per image, genuinely complementary training data,
not just more of the same. Same line-name taxonomy as SN-GSR's pitch
annotations (confirmed: e.g. "Goal right crossbar", "Circle right"),
same [0, 1]-normalized point format, just a flat ``{line_name:
[{x,y},...]}`` dict per image rather than nested under a frame/category
structure.

Usage:
    python scripts/convert_soccernet_calibration_to_pixels.py \
        <extracted_split_dir> <output_dir> <split_name>

``<extracted_split_dir>``: the directory containing ``*.jpg``/``*.json``
pairs (e.g. the extracted ``test/`` folder from ``calibration/test.zip``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python scripts/convert_soccernet_calibration_to_pixels.py "
            "<extracted_split_dir> <output_dir> <split_name>",
            file=sys.stderr,
        )
        sys.exit(1)

    split_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    split_name = sys.argv[3]
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(split_dir.glob("*.json"))
    if not json_paths:
        print(f"No .json annotation files found under {split_dir}", file=sys.stderr)
        sys.exit(1)

    frames = {}
    skipped = 0
    for json_path in json_paths:
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]

        lines = json.loads(json_path.read_text())
        lines_px = {
            line_name: [[pt["x"] * width, pt["y"] * height] for pt in points]
            for line_name, points in lines.items()
        }
        frames[image_path.name] = {
            "image_path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "lines_px": lines_px,
        }

    converted = {
        "sequence": f"calibration_{split_name}",
        "num_frames_with_pitch_annotation": len(frames),
        "frames": frames,
    }
    out_path = output_dir / f"calibration_{split_name}.json"
    out_path.write_text(json.dumps(converted))

    print(f"{split_name}: {len(frames)} images -> {out_path} ({skipped} skipped)")


if __name__ == "__main__":
    main()
