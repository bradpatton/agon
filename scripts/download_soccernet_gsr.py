"""Downloads SoccerNet Game State Reconstruction (SN-GSR-2025) splits from
HuggingFace into ``data/soccernet/gamestate-2025/`` -- the source data for
every Phase 7 training script in this project (detection fine-tuning,
pitch-calibration keypoints, jersey number recognition; see
``docs/TRAINING.md``).

Uses ``SoccerNet/SN-GSR-2025`` directly via ``huggingface_hub``, not the
``SoccerNet`` pip package's own downloader: that package's
``downloadDataTask(task="tracking"/"calibration", ...)`` NDA-password
OwnCloud path is broken (confirmed via a direct ``curl`` against the
ownCloud server itself -- a genuine 401, not a client bug). SN-GSR-2025 is
the actively-maintained replacement, hosted on HuggingFace, publicly
accessible (``gated: False``, confirmed) -- no password, no account, no
NDA needed for this specific dataset.

Split sizes (whole-split zips, confirmed via the HuggingFace API): challenge
5.31GB, test 8.85GB, train 9.76GB, valid 11.17GB. ``challenge`` has no
ground-truth labels (it's the competition submission split) -- not useful
for training, skip it. For a real training run you need at least
``train`` (and ideally ``valid`` too, for during-training validation
metrics) -- budget ~20GB combined plus room to extract.

Usage:
    python scripts/download_soccernet_gsr.py --split test          # smallest labeled split
    python scripts/download_soccernet_gsr.py --split train valid   # for a real training run
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soccernet" / "gamestate-2025"
VALID_SPLITS = ("train", "valid", "test", "challenge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=VALID_SPLITS,
        default=["test"],
        help="One or more splits to fetch (default: test, the smallest labeled split).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "download_soccernet_gsr.py needs huggingface_hub: pip install huggingface_hub",
        )
        raise SystemExit(1) from None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    patterns = [f"{split}.zip" for split in args.split] + ["README.md"]
    print(f"Downloading {', '.join(args.split)} from SoccerNet/SN-GSR-2025 to {DATA_DIR} ...")
    snapshot_download(
        repo_id="SoccerNet/SN-GSR-2025",
        repo_type="dataset",
        revision="main",
        local_dir=str(DATA_DIR),
        allow_patterns=patterns,
    )
    print("Done. Extract with e.g.:")
    for split in args.split:
        print(f"  unzip {DATA_DIR}/{split}.zip -d {DATA_DIR}/{split}_extracted")


if __name__ == "__main__":
    main()
