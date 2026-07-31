"""Trains a jersey-number classifier on a converted SoccerNet crop dataset
(see ``convert_soccernet_gsr_to_jersey_crops.py``) -- project plan Phase 7,
item 3. Jersey number recognition doesn't exist anywhere in this codebase
yet; this is the training half, mirroring ``train_detector.py``'s pattern.

Needs the ``[train]`` extra: ``pip install 'agon[train]'``.

Uses Ultralytics' image-classification training mode (``YOLO("*-cls.pt")``)
rather than a separate torchvision training loop, since the conversion
script already writes ``<split>/<label>/*.jpg`` -- Ultralytics' own
classification data format -- so this reuses the same
library/checkpoint-management/export pattern as detection training
end-to-end. Unlike detection training, there's no existing local base
checkpoint in ``models/`` to start from (this project has never had a
classification model before); ``--base-model yolo11n-cls.pt`` (the
default) is a bare Ultralytics checkpoint *name*, which it auto-downloads
on first use, not a local path.

The trained model outputs one of the training classes per crop -- the
jersey digit string, or "unknown" (SN-GSR's own label for a crop the
human annotators themselves couldn't read -- see the conversion script's
docstring for why that's a real, common class here, not an edge case).

Usage:
    python scripts/train_jersey_classifier.py --data data/soccernet_jersey --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing train/<label>/*.jpg (and optionally val/<label>/*.jpg), "
        "written by convert_soccernet_gsr_to_jersey_crops.py.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n-cls.pt",
        help="Ultralytics classification checkpoint name (auto-downloaded) or local path "
        "to fine-tune from. Default: yolo11n-cls.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=128, help="Crop input resolution, square.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Fixed batch size, or -1 (default) for Ultralytics' AutoBatch: picks the "
        "largest batch that fits in whatever VRAM is actually available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', '0,1,2' for specific GPUs, 'mps', or 'cpu'. Default: auto-detect and "
        "use every visible CUDA GPU (see _auto_device) -- pass this explicitly to override, "
        "e.g. to pin to a subset.",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", type=str, default="jersey")
    return parser.parse_args()


def _auto_device() -> str | None:
    """Comma-separated indices of every visible CUDA GPU (e.g. "0,1,2"), so
    a training command doesn't need editing every time a GPU is added or
    removed from the machine. Returns None (ultralytics' own auto-detect)
    when there's zero or one GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    count = torch.cuda.device_count()
    if count <= 1:
        return None
    return ",".join(str(i) for i in range(count))


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        print(f"Data directory not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not (args.data / "train").exists():
        print(f"Expected {args.data}/train/<label>/*.jpg -- not found.", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "train_jersey_classifier.py needs the 'train' extra: pip install 'agon[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    device = args.device if args.device is not None else _auto_device()
    if args.device is None and device is not None:
        print(f"Auto-detected {device.count(',') + 1} GPUs, using device={device}")

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
    )
    print(f"\nTraining done. Results/weights under: {results.save_dir}")


if __name__ == "__main__":
    main()
