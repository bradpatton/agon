"""Converts SoccerNet Game State Reconstruction (SN-GSR-2025) player/
goalkeeper crops + jersey-number labels into an image-classification
training set (project plan Phase 7, item 3 -- jersey number recognition,
which doesn't exist anywhere in this codebase today).

SN-GSR's ``attributes.jersey`` is a 0-99 digit string when legible, or
``null`` when the annotator couldn't read it (observed on one real
sequence: over half of player/goalkeeper instances are null -- a player
with their back turned, motion blur, or too far from camera is common in
broadcast footage, not an edge case). Null crops go in an ``unknown``
class rather than being dropped, so the trained classifier can express
"can't tell" instead of being forced to guess -- matching the standard
formulation of the academic SoccerNet Jersey Number Recognition task.

Output layout is ``<output_dir>/<split>/<label>/<crop>.jpg`` (label =
jersey digit string or "unknown") -- Ultralytics' own image-classification
training format (``YOLO("yolo11n-cls.pt").train(data=<output_dir>, ...)``),
chosen deliberately so jersey-number training reuses the same
library/pattern as detection training (see train_detector.py) instead of
introducing a separate torchvision training loop.

Crops are the full player/goalkeeper bounding box (there's no finer
"just the number" annotation available in this data -- academic baselines
for this task crop the same way, letting the classifier learn to locate
the number within the crop itself, usually on the back of the shirt).

Usage:
    python scripts/convert_soccernet_gsr_to_jersey_crops.py <sngs_root> <output_dir> <split_name>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2

PLAYER_CATEGORY_IDS = {1, 2}  # player, goalkeeper
UNKNOWN_LABEL = "unknown"
_JERSEY_PATTERN = re.compile(r"^\d{1,2}$")


def _label_for(jersey: str | None) -> str:
    if jersey is not None and _JERSEY_PATTERN.match(jersey):
        return jersey
    return UNKNOWN_LABEL


def convert_sequence(seq_dir: Path, output_split_dir: Path) -> tuple[int, dict[str, int]]:
    """Returns (num_crops_written, counts_per_label)."""
    labels_path = seq_dir / "Labels-GameState.json"
    data = json.loads(labels_path.read_text())

    im_dir = data["info"]["im_dir"]
    images_by_id = {img["image_id"]: img for img in data["images"]}
    seq_name = seq_dir.name

    frame_cache: dict[str, object] = {}
    num_written = 0
    counts: dict[str, int] = {}

    for ann in data["annotations"]:
        if ann["category_id"] not in PLAYER_CATEGORY_IDS:
            continue
        img = images_by_id.get(ann["image_id"])
        if img is None:
            continue

        image_path = seq_dir / im_dir / img["file_name"]
        if image_path not in frame_cache:
            frame_cache[image_path] = cv2.imread(str(image_path))
        frame = frame_cache[image_path]
        if frame is None:
            continue

        bbox = ann["bbox_image"]
        x1, y1 = int(bbox["x"]), int(bbox["y"])
        x2, y2 = x1 + int(bbox["w"]), y1 + int(bbox["h"])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        label = _label_for(ann["attributes"].get("jersey"))

        label_dir = output_split_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        crop_path = label_dir / f"{seq_name}_{ann['id']}.jpg"
        cv2.imwrite(str(crop_path), crop)

        num_written += 1
        counts[label] = counts.get(label, 0) + 1

    return num_written, counts


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python scripts/convert_soccernet_gsr_to_jersey_crops.py "
            "<sngs_root> <output_dir> <split_name>",
            file=sys.stderr,
        )
        sys.exit(1)

    sngs_root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    split_name = sys.argv[3]
    output_split_dir = output_dir / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p for p in sngs_root.glob("SNGS-*") if p.is_dir())
    if not seq_dirs:
        print(f"No SNGS-* sequence directories found under {sngs_root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    total_counts: dict[str, int] = {}
    for seq_dir in seq_dirs:
        num_written, counts = convert_sequence(seq_dir, output_split_dir)
        print(f"{seq_dir.name}: {num_written} crops")
        total += num_written
        for label, count in counts.items():
            total_counts[label] = total_counts.get(label, 0) + count

    print(f"\nTotal ({split_name}): {total} crops across {len(total_counts)} classes")
    for label, count in sorted(total_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
