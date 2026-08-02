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
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Directory containing train/<label>/*.jpg (and optionally val/<label>/*.jpg), "
        "written by convert_soccernet_gsr_to_jersey_crops.py. Required unless --resume is "
        "given.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a last.pt checkpoint to resume interrupted training from (e.g. after "
        "an OOM kill or crash) -- picks up at the next epoch using that checkpoint's own "
        "saved training args, ignoring every other flag below. Ultralytics' own resume "
        "mechanism, not a custom one.",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader worker processes (ultralytics default: 8). Lower on a small-VRAM "
        "GPU that OOMs partway through training -- see train_detector.py's docstring.",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", type=str, default="jersey")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export the best checkpoint to ONNX at the same imgsz after training (default: "
        "on), plus a classes.json sidecar mapping output index -> label string (agon.jersey."
        "OnnxJerseyClassifier needs this; the ONNX file alone doesn't reliably expose "
        "Ultralytics' internal class-index order).",
    )
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
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


def _resolve_batch(batch: int, device: str | None) -> int:
    """Ultralytics' AutoBatch (batch=-1) doesn't support multi-GPU training
    at all -- see train_detector.py's docstring for the real error this
    was confirmed against. Falls back to a conservative explicit batch (8
    per GPU) only when --batch was left at its auto default and multiple
    GPUs are in play."""
    num_gpus = (device.count(",") + 1) if device and "," in device else 1
    if batch == -1 and num_gpus > 1:
        fallback = 8 * num_gpus
        print(
            f"AutoBatch isn't supported for multi-GPU training -- using a conservative "
            f"explicit batch={fallback} (8 per GPU x {num_gpus} GPUs). Pass --batch "
            f"explicitly for a different value."
        )
        return fallback
    return batch


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "train_jersey_classifier.py needs the 'train' extra: pip install 'agon[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.resume is not None:
        if not args.resume.exists():
            print(f"Resume checkpoint not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        print(f"Resuming from {args.resume} (ignoring --data/--imgsz/--batch/etc.)")
        model = YOLO(str(args.resume))
        # --workers is a pure DataLoader setting (unlike --data/--imgsz, which must match
        # the checkpoint's architecture) -- safe, and sometimes necessary, to override even
        # on resume: this project hit a real host-RAM OOM kill caused by the checkpoint's
        # own saved workers=8, which resume=True alone would otherwise silently repeat.
        resume_kwargs = {"resume": True}
        if args.workers is not None:
            resume_kwargs["workers"] = args.workers
        results = model.train(**resume_kwargs)
        _finish(model, results, args.export_onnx)
        return

    if args.data is None:
        print("--data is required unless --resume is given.", file=sys.stderr)
        sys.exit(1)
    if not args.data.exists():
        print(f"Data directory not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not (args.data / "train").exists():
        print(f"Expected {args.data}/train/<label>/*.jpg -- not found.", file=sys.stderr)
        sys.exit(1)

    device = args.device if args.device is not None else _auto_device()
    if args.device is None and device is not None:
        print(f"Auto-detected {device.count(',') + 1} GPUs, using device={device}")

    workers = args.workers if args.workers is not None else 8  # ultralytics' own default
    batch = _resolve_batch(args.batch, device)

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=device,
        workers=workers,
        project=str(args.project),
        name=args.name,
    )
    _finish(model, results, args.export_onnx)


def _finish(model: YOLO, results, export_onnx: bool) -> None:
    """Shared by both the fresh-training and --resume paths, mirroring
    train_detector.py's _finish(). Reads the actual trained imgsz back off
    the model's own trainer args rather than trusting --imgsz on the CLI,
    which is meaningless in --resume mode."""
    print(f"\nTraining done. Results/weights under: {results.save_dir}")

    imgsz = model.trainer.args.imgsz
    if export_onnx:
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            print(f"Expected best.pt at {best_weights}, skipping ONNX export.", file=sys.stderr)
            return

        from ultralytics import YOLO as _YOLO

        trained_model = _YOLO(str(best_weights))
        onnx_path = trained_model.export(format="onnx", imgsz=imgsz, dynamic=False)

        # agon.jersey.OnnxJerseyClassifier needs output-index -> label mapping, and
        # Ultralytics' own class order (trained_model.names, index -> label string,
        # sorted by the training data's label folder names) isn't reliably recoverable
        # from the ONNX file alone -- write it explicitly rather than parse ONNX
        # metadata format, which is undocumented and could change between versions.
        classes_path = Path(onnx_path).with_name("classes.json")
        classes_path.write_text(json.dumps(trained_model.names))

        print(f"Exported ONNX ({imgsz}x{imgsz}): {onnx_path}")
        print(f"Wrote class mapping: {classes_path}")
        print(
            "Set PipelineConfig.jersey_model_path to this .onnx path (classes.json must "
            "sit alongside it) to use this checkpoint in the pipeline."
        )


if __name__ == "__main__":
    main()
