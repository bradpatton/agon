"""Converts SoccerNet Game State Reconstruction (SN-GSR-2025) sequences into
YOLO-format detection training data, for fine-tuning
``agon.detection.tracker.UltralyticsDetector``'s checkpoint (see
the project plan's Phase 7, item 1).

Each SNGS-* sequence directory (``Labels-GameState.json`` + ``img1/*.jpg``,
one folder per downloaded split -- see ``download_soccernet_gsr.py``) becomes
YOLO label ``.txt`` files (one per frame, symlinked alongside the source
image) plus a ``dataset.yaml`` ultralytics can train against directly.

Class mapping matches ``agon.export.schema.ObjectClass`` exactly
-- SN-GSR's ``category_id`` 1-4 (1-indexed, COCO-style) map to YOLO's
0-indexed classes by subtracting 1:
    0 player, 1 goalkeeper, 2 referee, 3 ball
``category_id`` 5 (pitch line annotations) and above are for the separate
calibration-keypoint conversion (see convert_soccernet_gsr_to_calibration.py),
not detection -- skipped here.

Images are symlinked into the output layout rather than copied, so this
doesn't double the on-disk size of a dataset that's already tens of GB.

Usage:
    python scripts/convert_soccernet_gsr_to_yolo.py <sngs_root> <output_dir> <split_name>

``<sngs_root>``: a directory containing one or more SNGS-* sequence folders
(e.g. one extracted SoccerNet split). ``<split_name>``: 'train'/'val'/'test'
-- run once per downloaded split, all pointed at the same ``<output_dir>``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
DETECTION_CATEGORY_IDS = {1, 2, 3, 4}  # matches CLASS_NAMES order, 1-indexed


def _yolo_line(class_id: int, bbox_image: dict, img_width: int, img_height: int) -> str | None:
    x_center = bbox_image["x_center"] / img_width
    y_center = bbox_image["y_center"] / img_height
    width = bbox_image["w"] / img_width
    height = bbox_image["h"] / img_height

    # Clamp rather than drop: a box that straddles the image edge is still a
    # real, usable training example once clipped to the visible region --
    # only a genuinely degenerate (zero-area) box after clamping is useless.
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    if width <= 0.0 or height <= 0.0:
        return None

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_sequence(seq_dir: Path, images_out: Path, labels_out: Path) -> tuple[int, int]:
    """Returns (num_images_written, num_boxes_written)."""
    labels_path = seq_dir / "Labels-GameState.json"
    data = json.loads(labels_path.read_text())

    im_dir = data["info"]["im_dir"]
    images_by_id = {img["image_id"]: img for img in data["images"]}
    lines_by_image: dict[str, list[str]] = {image_id: [] for image_id in images_by_id}

    for ann in data["annotations"]:
        if ann["category_id"] not in DETECTION_CATEGORY_IDS:
            continue
        image_id = ann["image_id"]
        img = images_by_id.get(image_id)
        if img is None:
            continue
        class_id = ann["category_id"] - 1
        line = _yolo_line(class_id, ann["bbox_image"], img["width"], img["height"])
        if line is not None:
            lines_by_image[image_id].append(line)

    num_images = 0
    num_boxes = 0
    seq_name = seq_dir.name
    for image_id, img in images_by_id.items():
        src_image = seq_dir / im_dir / img["file_name"]
        if not src_image.exists():
            continue

        stem = f"{seq_name}_{Path(img['file_name']).stem}"
        dst_image = images_out / f"{stem}{src_image.suffix}"
        dst_label = labels_out / f"{stem}.txt"

        if not dst_image.exists():
            # Relative, not src_image.resolve() (absolute): an absolute
            # host-filesystem symlink target breaks the moment this output
            # directory is bind-mounted somewhere else (e.g. into a
            # training container) -- the same portability bug as
            # dataset.yaml's `path:` key above, confirmed the same way.
            relative_target = os.path.relpath(src_image.resolve(), dst_image.parent.resolve())
            dst_image.symlink_to(relative_target)
        dst_label.write_text("\n".join(lines_by_image[image_id]) + "\n")

        num_images += 1
        num_boxes += len(lines_by_image[image_id])

    return num_images, num_boxes


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python scripts/convert_soccernet_gsr_to_yolo.py "
            "<sngs_root> <output_dir> <split_name>",
            file=sys.stderr,
        )
        sys.exit(1)

    sngs_root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    split_name = sys.argv[3]

    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p for p in sngs_root.glob("SNGS-*") if p.is_dir())
    if not seq_dirs:
        print(f"No SNGS-* sequence directories found under {sngs_root}", file=sys.stderr)
        sys.exit(1)

    total_images = 0
    total_boxes = 0
    for seq_dir in seq_dirs:
        num_images, num_boxes = convert_sequence(seq_dir, images_out, labels_out)
        print(f"{seq_dir.name}: {num_images} frames, {num_boxes} boxes")
        total_images += num_images
        total_boxes += num_boxes

    print(f"\nTotal ({split_name}): {total_images} frames, {total_boxes} boxes")

    dataset_yaml = output_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        # No `path:` key, deliberately: ultralytics resolves train/val/test
        # relative to the yaml file's own directory when path is omitted,
        # which keeps this portable across machines/mounts (an absolute
        # host path baked in here broke the moment this same output_dir
        # got bind-mounted at a different path inside a training
        # container -- confirmed by hitting exactly that FileNotFoundError
        # smoke-testing this script end-to-end).
        dataset_yaml.write_text(
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)) + "\n"
        )
        print(f"\nWrote {dataset_yaml}")


if __name__ == "__main__":
    main()
