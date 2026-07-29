"""Fetches the demo YOLO checkpoint and sample match footage referenced by
the README's quickstart, so a fresh clone is runnable without hunting down
assets manually. Both are hosted on the original football_analysis
tutorial's Google Drive (see README credits) -- this script doesn't fetch
anything not already linked from this project's own docs.

Needs the ``assets`` extra: ``pip install 'agon[assets]'``
(just ``gdown``, for handling Google Drive's large-file download flow).

Usage:
    python scripts/download_assets.py [--model-only | --video-only] [--force]

Checksum verification: a sha256 is computed and printed after every
download, but only *checked* against a known-good value if one has been
filled in below. None are filled in yet -- this script hasn't had a
verified-trustworthy download recorded against it yet. If you download
these assets and confirm they're what you expect, consider filling in
KNOWN_SHA256 and opening a PR so future users get real verification instead
of just a printed checksum to eyeball.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MODEL_URL = "https://drive.google.com/uc?id=1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK"
VIDEO_URL = "https://drive.google.com/uc?id=1t6agoqggZKx6thamUuPAIdN_1zR9v9S_"

MODEL_PATH = Path("models/best.pt")
VIDEO_PATH = Path("input_videos/08fd33_4.mp4")

KNOWN_SHA256: dict[Path, str | None] = {
    MODEL_PATH: None,
    VIDEO_PATH: None,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"{dest} already exists, skipping (use --force to re-download)")
        return

    try:
        import gdown
    except ImportError:
        print(
            "This script needs gdown: pip install 'agon[assets]'",
            file=sys.stderr,
        )
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(dest), quiet=False)

    checksum = _sha256(dest)
    expected = KNOWN_SHA256.get(dest)
    if expected is None:
        print(f"{dest}: sha256={checksum} (no known-good checksum on file to verify against yet)")
    elif checksum != expected:
        dest.unlink()
        raise RuntimeError(
            f"{dest}: sha256 mismatch (got {checksum}, expected {expected}) "
            "-- deleted, do not trust this download"
        )
    else:
        print(f"{dest}: sha256 verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already exists."
    )
    args = parser.parse_args()

    if args.model_only and args.video_only:
        parser.error("--model-only and --video-only are mutually exclusive")

    if not args.video_only:
        _download(MODEL_URL, MODEL_PATH, args.force)
    if not args.model_only:
        _download(VIDEO_URL, VIDEO_PATH, args.force)

    print(
        "\nNote: this downloads the original tutorial's torch/.pt checkpoint. "
        "The default pipeline backend is onnxruntime -- export it once with "
        "the [train] extra installed:\n"
        '  python -c "from ultralytics import YOLO; '
        "YOLO('models/best.pt').export(format='onnx', imgsz=640, dynamic=False)\""
    )


if __name__ == "__main__":
    main()
