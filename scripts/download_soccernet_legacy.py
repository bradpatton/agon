"""Downloads SoccerNet's legacy (pre-2025) labeled datasets -- Tracking,
Calibration, Re-Identification, and the full tracklet-video Jersey Number
set -- into ``data/soccernet/<task>/``. These are separate, broader
datasets than SN-GSR-2025 (which only covers a curated subset of games);
see the project plan for the full comparison.

Uses the ``SoccerNet`` pip package's own ``SoccerNetDownloader``, unlike
``download_soccernet_gsr.py`` (which uses ``huggingface_hub`` directly for
the newer, HuggingFace-hosted datasets). These four datasets are still
OwnCloud-hosted and were wrongly assumed broken earlier in this project's
development -- they returned 401 when tested with the personal NDA
password issued for raw broadcast video downloads. They actually download
fine with SoccerNet's **generic public password** (the literal string
``"SoccerNet"``, also this package's own default) -- confirmed by
downloading real, correctly-sized data from all four
(tracking/calibration/reid/jersey-2023), not just checking HTTP status.
The raw broadcast videos are the only asset that genuinely needs the
personal NDA password; everything here does not.

Needs the ``SoccerNet`` pip package (``pip install SoccerNet``) -- not a
project dependency (this is a one-off data-acquisition script, same
reasoning as ``download_soccernet_gsr.py`` not depending on anything
beyond ``huggingface_hub``).

Usage:
    python scripts/download_soccernet_legacy.py --task calibration --split test
    python scripts/download_soccernet_legacy.py --task tracking --split train valid test
    python scripts/download_soccernet_legacy.py --task all --split test   # one of each
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soccernet"
TASKS = ("tracking", "calibration", "reid", "jersey-2023")
GENERIC_PASSWORD = "SoccerNet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=[*TASKS, "all"],
        default=["all"],
        help="Which dataset(s) to fetch. 'all' = one of each (default).",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        help="train/valid/test/challenge, per task (default: test, the smallest, for "
        "validation). Not every task has every split; unavailable ones print a message "
        "rather than erroring.",
    )
    args = parser.parse_args()

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print(
            "download_soccernet_legacy.py needs the SoccerNet pip package: pip install SoccerNet",
        )
        raise SystemExit(1) from None

    tasks = TASKS if "all" in args.task else tuple(args.task)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(DATA_DIR))

    for task in tasks:
        print(f"=== {task} ({', '.join(args.split)}) ===")
        downloader.downloadDataTask(task=task, split=list(args.split), password=GENERIC_PASSWORD)

    print("\nDone. Downloaded under:", DATA_DIR)


if __name__ == "__main__":
    main()
