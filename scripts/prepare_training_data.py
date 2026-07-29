"""One-command data pipeline for the training machine: download, extract,
and convert SoccerNet data from every source this project can currently
train from -- SN-GSR-2025 (detection + jersey numbers) and the legacy
Calibration + Jersey-2023 datasets (confirmed working with the generic
public password after being wrongly assumed broken earlier in this
project's development -- see download_soccernet_legacy.py's docstring).

Meant to run *inside* the training container (or natively on the training
machine) against a persistent volume -- see docs/TRAINING.md, which this
script replaces several manual steps with one command. Shells out to the
existing standalone download/convert scripts rather than reimplementing
them, so each remains independently runnable/testable exactly as before.

Idempotent: re-running skips whatever's already downloaded/extracted.

Output layout (unchanged regardless of source, so downstream training
scripts don't need to know or care where a given example came from):
  data/soccernet_yolo/         -- detection training set (GSR only; the
                                   legacy Tracking dataset could extend
                                   this too but isn't wired in yet, see
                                   the project plan)
  data/soccernet_jersey/       -- jersey-number crops, GSR + legacy-2023
                                   combined into the same <split>/<label>/
                                   tree (filenames are prefixed by source,
                                   so nothing collides)
  data/soccernet_calibration/  -- pitch-line pixel-space JSON, GSR +
                                   legacy combined (one file per source/
                                   split/sequence; no model trains on this
                                   yet, see the project plan for why)

Usage:
    python scripts/prepare_training_data.py                # everything, default splits
    python scripts/prepare_training_data.py --sources gsr  # GSR-2025 only
    python scripts/prepare_training_data.py --split test   # small, for a quick validation pass
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
GSR_DIR = REPO_ROOT / "data" / "soccernet" / "gamestate-2025"
LEGACY_DIR = REPO_ROOT / "data" / "soccernet"
YOLO_OUT = REPO_ROOT / "data" / "soccernet_yolo"
JERSEY_OUT = REPO_ROOT / "data" / "soccernet_jersey"
CALIBRATION_OUT = REPO_ROOT / "data" / "soccernet_calibration"

SOURCES = ("gsr", "legacy-calibration", "legacy-jersey")


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _split_to_yolo_subset(split: str) -> str:
    """SoccerNet splits are named train/valid/test; the conversion
    scripts' (and Ultralytics' own) convention is train/val/test --
    'valid' is the one mismatch."""
    return "val" if split == "valid" else split


def _prepare_gsr(splits: list[str], skip_jersey: bool) -> None:
    print(f"=== GSR-2025: downloading {', '.join(splits)} ===")
    _run([sys.executable, str(SCRIPTS / "download_soccernet_gsr.py"), "--split", *splits])

    for split in splits:
        zip_path = GSR_DIR / f"{split}.zip"
        extracted_dir = GSR_DIR / f"{split}_extracted"
        subset = _split_to_yolo_subset(split)

        if not extracted_dir.exists():
            print(f"=== GSR-2025: extracting {split}.zip ===")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extracted_dir)
        else:
            print(f"=== GSR-2025: {extracted_dir} already exists, skipping extraction ===")

        print(f"=== GSR-2025: converting {split} -> {subset} (detection) ===")
        _run(
            [
                sys.executable,
                str(SCRIPTS / "convert_soccernet_gsr_to_yolo.py"),
                str(extracted_dir),
                str(YOLO_OUT),
                subset,
            ]
        )
        if not skip_jersey:
            print(f"=== GSR-2025: converting {split} -> {subset} (jersey crops) ===")
            _run(
                [
                    sys.executable,
                    str(SCRIPTS / "convert_soccernet_gsr_to_jersey_crops.py"),
                    str(extracted_dir),
                    str(JERSEY_OUT),
                    subset,
                ]
            )


def _prepare_legacy_calibration(splits: list[str]) -> None:
    task_dir = LEGACY_DIR / "calibration"
    print(f"=== Legacy Calibration: downloading {', '.join(splits)} ===")
    _run(
        [
            sys.executable,
            str(SCRIPTS / "download_soccernet_legacy.py"),
            "--task",
            "calibration",
            "--split",
            *splits,
        ]
    )

    for split in splits:
        zip_path = task_dir / f"{split}.zip"
        extracted_dir = task_dir / f"{split}_extracted"
        if not zip_path.exists():
            print(f"=== Legacy Calibration: no {split}.zip (not every split exists), skipping ===")
            continue

        if not extracted_dir.exists():
            print(f"=== Legacy Calibration: extracting {split}.zip ===")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extracted_dir)
        else:
            print(
                f"=== Legacy Calibration: {extracted_dir} already exists, skipping extraction ==="
            )

        print(f"=== Legacy Calibration: converting {split} ===")
        _run(
            [
                sys.executable,
                str(SCRIPTS / "convert_soccernet_calibration_to_pixels.py"),
                str(extracted_dir / split),
                str(CALIBRATION_OUT),
                split,
            ]
        )


def _prepare_legacy_jersey(splits: list[str], max_per_tracklet: int) -> None:
    task_dir = LEGACY_DIR / "jersey-2023"
    print(f"=== Legacy Jersey-2023: downloading {', '.join(splits)} ===")
    _run(
        [
            sys.executable,
            str(SCRIPTS / "download_soccernet_legacy.py"),
            "--task",
            "jersey-2023",
            "--split",
            *splits,
        ]
    )

    for split in splits:
        zip_path = task_dir / f"{split}.zip"
        extracted_dir = task_dir / f"{split}_extracted"
        if not zip_path.exists():
            print(f"=== Legacy Jersey-2023: no {split}.zip (not every split exists), skipping ===")
            continue

        if not extracted_dir.exists():
            print(f"=== Legacy Jersey-2023: extracting {split}.zip (many small files, slow) ===")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extracted_dir)
        else:
            print(
                f"=== Legacy Jersey-2023: {extracted_dir} already exists, skipping extraction ==="
            )

        subset = _split_to_yolo_subset(split)
        print(f"=== Legacy Jersey-2023: converting {split} -> {subset} ===")
        _run(
            [
                sys.executable,
                str(SCRIPTS / "convert_soccernet_jersey2023_to_crops.py"),
                str(extracted_dir / split),
                str(JERSEY_OUT),
                subset,
                "--max-per-tracklet",
                str(max_per_tracklet),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCES,
        default=list(SOURCES),
        help="Which data sources to pull (default: all three).",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "valid"],
        help="Splits to prepare, applied to every selected source (default: train valid). "
        "Not every source has every split; missing ones are skipped, not an error.",
    )
    parser.add_argument("--skip-jersey", action="store_true", help="Skip GSR's jersey conversion.")
    parser.add_argument(
        "--max-per-tracklet",
        type=int,
        default=15,
        help="Legacy Jersey-2023 only: crops sampled per tracklet (default: 15).",
    )
    args = parser.parse_args()

    if "gsr" in args.sources:
        _prepare_gsr(args.split, args.skip_jersey)
    if "legacy-calibration" in args.sources:
        _prepare_legacy_calibration(args.split)
    if "legacy-jersey" in args.sources:
        _prepare_legacy_jersey(args.split, args.max_per_tracklet)

    print("\nDone.")
    if (YOLO_OUT / "dataset.yaml").exists():
        print(f"Detection dataset:   {YOLO_OUT / 'dataset.yaml'}")
    if JERSEY_OUT.exists():
        print(f"Jersey dataset:      {JERSEY_OUT}")
    if CALIBRATION_OUT.exists():
        print(f"Calibration dataset: {CALIBRATION_OUT} (no trainer yet -- see project plan)")
    print("\nNext: scripts/train_detector.py --data " + str(YOLO_OUT / "dataset.yaml"))


if __name__ == "__main__":
    main()
