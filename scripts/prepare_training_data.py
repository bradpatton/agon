"""One-command data pipeline for the training machine: download SoccerNet
SN-GSR-2025 splits, extract them, and run both conversion scripts (YOLO
detection format + jersey-number crops) -- the "best training data" path
established in the project plan (SN-GSR-2025's train+valid splits are the
highest-leverage, actually-labeled data this project can train on today;
see docs/TRAINING.md for why the raw broadcast videos aren't a substitute --
no bounding-box labels ship with them).

Meant to run *inside* the training container (or natively on the training
machine) against a persistent volume -- see docs/TRAINING.md Step 4-5,
which this script replaces with one command. It shells out to the existing
standalone scripts (download_soccernet_gsr.py, convert_soccernet_gsr_to_yolo.py,
convert_soccernet_gsr_to_jersey_crops.py) rather than reimplementing them,
so each remains independently runnable/testable exactly as before.

Idempotent: re-running skips whatever's already downloaded/extracted
(download_soccernet_gsr.py's underlying huggingface_hub call, and this
script's own extraction step, both check for existing output first).

Usage:
    python scripts/prepare_training_data.py --split train valid
    python scripts/prepare_training_data.py --split test   # small, for a quick validation pass
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_DIR = REPO_ROOT / "data" / "soccernet" / "gamestate-2025"
YOLO_OUT = REPO_ROOT / "data" / "soccernet_yolo"
JERSEY_OUT = REPO_ROOT / "data" / "soccernet_jersey"


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _split_to_yolo_subset(split: str) -> str:
    """SN-GSR splits are named train/valid/test; the conversion scripts'
    (and Ultralytics' own) convention is train/val/test -- 'valid' is the
    one mismatch."""
    return "val" if split == "valid" else split


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "valid", "test", "challenge"],
        default=["train", "valid"],
        help="SN-GSR-2025 splits to prepare (default: train valid -- the real training set).",
    )
    parser.add_argument(
        "--skip-jersey",
        action="store_true",
        help="Skip the jersey-number crop conversion (detection data only).",
    )
    args = parser.parse_args()

    python = sys.executable

    print(f"=== Step 1/3: downloading {', '.join(args.split)} ===")
    _run([python, str(REPO_ROOT / "scripts" / "download_soccernet_gsr.py"), "--split", *args.split])

    for split in args.split:
        zip_path = GSR_DIR / f"{split}.zip"
        extracted_dir = GSR_DIR / f"{split}_extracted"
        subset = _split_to_yolo_subset(split)

        if not extracted_dir.exists():
            print(f"=== Step 2/3: extracting {split}.zip ===")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extracted_dir)
        else:
            print(f"=== Step 2/3: {extracted_dir} already exists, skipping extraction ===")

        print(f"=== Step 3/3: converting {split} -> {subset} ===")
        _run(
            [
                python,
                str(REPO_ROOT / "scripts" / "convert_soccernet_gsr_to_yolo.py"),
                str(extracted_dir),
                str(YOLO_OUT),
                subset,
            ]
        )
        if not args.skip_jersey:
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts" / "convert_soccernet_gsr_to_jersey_crops.py"),
                    str(extracted_dir),
                    str(JERSEY_OUT),
                    subset,
                ]
            )

    print("\nDone.")
    print(f"Detection dataset: {YOLO_OUT / 'dataset.yaml'}")
    if not args.skip_jersey:
        print(f"Jersey dataset:    {JERSEY_OUT}")
    print("\nNext: scripts/train_detector.py --data " + str(YOLO_OUT / "dataset.yaml"))


if __name__ == "__main__":
    main()
