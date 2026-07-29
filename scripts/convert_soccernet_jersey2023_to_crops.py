"""Converts SoccerNet's legacy Jersey Number Recognition dataset
(downloaded via ``download_soccernet_legacy.py --task jersey-2023``) into
the same ``<split>/<label>/*.jpg`` classification format
``convert_soccernet_gsr_to_jersey_crops.py`` produces from SN-GSR-2025 --
both feed the same ``train_jersey_classifier.py``, unchanged.

This dataset is organized by *tracklet*, not by single-frame instance:
``images/<tracklet_id>/<tracklet_id>_<frame>.jpg``, with one label per
tracklet in ``<split>_gt.json`` (``{tracklet_id: jersey_number}``, ``-1``
for illegible -- mapped to the same "unknown" class GSR uses). Real,
significant volume: one real split had 1,211 tracklets averaging ~467
crops each -- almost entirely redundant for training (a tracklet's own
frames are highly correlated, the player barely moves frame to frame at
video rate), so this samples evenly across each tracklet's frames up to
``--max-per-tracklet`` rather than taking every single one, keeping
per-class volume reasonable instead of massively oversampling whichever
tracklets happen to have the most frames.

Usage:
    python scripts/convert_soccernet_jersey2023_to_crops.py \\
        <extracted_split_dir> <output_dir> <split_name> [--max-per-tracklet N]

``<extracted_split_dir>``: the extracted split folder containing
``images/`` and ``<split_name>_gt.json`` (e.g. the extracted
``jersey-2023/test.zip`` -> ``test/``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

UNKNOWN_LABEL = "unknown"


def _label_for(jersey_number: int) -> str:
    if 0 <= jersey_number <= 99:
        return str(jersey_number)
    return UNKNOWN_LABEL


def _evenly_sampled(paths: list[Path], max_count: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    step = len(paths) / max_count
    return [paths[int(i * step)] for i in range(max_count)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("extracted_split_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("split_name")
    parser.add_argument(
        "--max-per-tracklet",
        type=int,
        default=15,
        help="Max crops sampled (evenly, not just the first N) per tracklet (default: 15).",
    )
    args = parser.parse_args()

    gt_path = args.extracted_split_dir / f"{args.split_name}_gt.json"
    images_dir = args.extracted_split_dir / "images"
    if not gt_path.exists() or not images_dir.exists():
        print(f"Expected {gt_path} and {images_dir} -- not found.", file=sys.stderr)
        raise SystemExit(1)

    ground_truth: dict[str, int] = json.loads(gt_path.read_text())

    output_split_dir = args.output_dir / args.split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    counts: dict[str, int] = {}
    for tracklet_id, jersey_number in ground_truth.items():
        tracklet_dir = images_dir / tracklet_id
        if not tracklet_dir.is_dir():
            continue

        crop_paths = sorted(tracklet_dir.glob("*.jpg"))
        label = _label_for(jersey_number)
        label_dir = output_split_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for crop_path in _evenly_sampled(crop_paths, args.max_per_tracklet):
            dst = label_dir / f"jersey2023_{crop_path.name}"
            if not dst.exists():
                relative_target = os.path.relpath(crop_path.resolve(), dst.parent.resolve())
                dst.symlink_to(relative_target)
            total += 1
            counts[label] = counts.get(label, 0) + 1

    print(
        f"{args.split_name}: {total} crops across {len(counts)} classes, "
        f"{len(ground_truth)} tracklets"
    )
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
