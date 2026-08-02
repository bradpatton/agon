"""Fine-tunes the YOLO detector on a converted SoccerNet dataset (see
``convert_soccernet_gsr_to_yolo.py``) -- project plan Phase 7, item 1.

Needs the ``[train]`` extra: ``pip install 'agon[train]'``
(torch + ultralytics). This is a one-time/occasional training step, not
part of the runtime package -- same reasoning as
``export_team_embedding_model.py``.

Resolution matters here specifically because the ball is a tiny object in
broadcast-resolution frames -- see ``agon.config.PipelineConfig
.detection_imgsz``'s docstring for the real bug this project had (inference
silently stuck at 640 regardless of what a checkpoint was trained at) that
made this worth being deliberate about. Whatever ``--imgsz`` you train
with, export ONNX at the *same* resolution (``--export-onnx``, on by
default) and set ``detection_imgsz`` to match when running the pipeline --
mismatches fail loudly (a clear onnxruntime shape error), not silently, but
still won't do what you want without the matching config.

Usage:
    python scripts/train_detector.py --data data/soccernet_yolo/dataset.yaml \\
        --base-model models/yolo11n.pt --imgsz 960 --epochs 50

Runs (weights, logs) land under ``--project``/``--name`` (ultralytics'
own convention) -- default ``runs/train/soccernet``.
"""

from __future__ import annotations

import argparse
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
        help="Path to dataset.yaml (written by convert_soccernet_gsr_to_yolo.py). "
        "Required unless --resume is given.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a last.pt checkpoint to resume interrupted training from (e.g. after "
        "a crash or power loss) -- picks up at the next epoch using that checkpoint's own "
        "saved training args (data/imgsz/epochs/batch/device/etc.), ignoring every other "
        "flag below. Ultralytics' own resume mechanism, not a custom one.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/yolo11n.pt"),
        help="Checkpoint to fine-tune from (default: models/yolo11n.pt, the same "
        "COCO-pretrained checkpoint already used elsewhere in this project).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training (and, if --export-onnx, export) resolution, square. Higher "
        "helps small-object recall (the ball) at the cost of ~quadratic compute "
        "(1280 is ~4x the compute of 640). Default 640 matches the current model.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Fixed batch size, or -1 (default) for Ultralytics' AutoBatch: a few trial "
        "passes pick the largest batch that fits in whatever VRAM is actually available, "
        "rather than a fixed guess that OOMs on smaller GPUs or wastes memory on larger "
        "ones. Set a fixed value if you want reproducible batch sizes across runs.",
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
        help="DataLoader worker processes (ultralytics default: 8). Each worker holds its "
        "own prefetched/pinned-memory batch, so on a small-VRAM GPU that OOMs partway "
        "through an epoch (not on the first batch -- a fragmentation/prefetch-pressure "
        "symptom, not a 'too big to ever fit' one), lowering this is a real lever "
        "alongside --batch. None = ultralytics' own default.",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", type=str, default="soccernet")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export the best checkpoint to ONNX at the same --imgsz after training (default: on).",
    )
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    return parser.parse_args()


def _auto_device() -> str | None:
    """Comma-separated indices of every visible CUDA GPU (e.g. "0,1,2"), so
    a training command doesn't need editing every time a GPU is added or
    removed from the machine -- a real, hit-in-practice annoyance with a
    hardcoded --device list. Returns None (ultralytics' own auto-detect,
    picks a single GPU or CPU) when there's zero or one GPU, since a
    comma-joined single index isn't meaningfully different from letting
    ultralytics choose itself."""
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
    at all -- confirmed by hitting its own ValueError on real 2-GPU
    hardware ("AutoBatch with batch<1 not supported for Multi-GPU
    training"), not just inferred. Falls back to a conservative explicit
    batch (8 per GPU, a valid multiple of the GPU count as ultralytics
    requires) only when the caller left --batch at its auto default and
    multiple GPUs are actually in play; a single GPU (or an explicit
    --batch override) passes through unchanged."""
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
            "train_detector.py needs the 'train' extra: pip install 'agon[train]'",
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
        # on resume, e.g. if the original run's saved workers value caused an OOM.
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
        print(f"Dataset config not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not args.base_model.exists():
        # Not a hard failure: a recognized Ultralytics checkpoint name (e.g.
        # yolo11n.pt) auto-downloads to this exact path on first use,
        # regardless of the directory prefix -- that's the documented
        # first-run behavior (see docs/TRAINING.md), and exiting here would
        # silently prevent it from ever happening.
        print(
            f"{args.base_model} not found locally -- if this is a recognized "
            f"Ultralytics checkpoint name (e.g. yolo11n.pt), it will be "
            f"downloaded automatically (needs internet access). If it's meant "
            f"to be a specific fine-tuned checkpoint you already have, double "
            f"check --base-model.",
            file=sys.stderr,
        )

    device = args.device if args.device is not None else _auto_device()
    if args.device is None and device is not None:
        print(f"Auto-detected {device.count(',') + 1} GPUs, using device={device}")
    workers = args.workers if args.workers is not None else 8  # ultralytics' own default
    batch = _resolve_batch(args.batch, device)

    model = YOLO(str(args.base_model))
    results = model.train(
        data=str(args.data),
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
    """Shared by both the fresh-training and --resume paths. Reads the actual
    trained imgsz back off the model's own trainer args rather than trusting
    the CLI's --imgsz, which is meaningless in --resume mode (it's still
    sitting at its argparse default, not whatever the checkpoint actually
    trained at)."""
    print(f"\nTraining done. Results/weights under: {results.save_dir}")

    imgsz = model.trainer.args.imgsz
    if export_onnx:
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            print(f"Expected best.pt at {best_weights}, skipping ONNX export.", file=sys.stderr)
            return

        from ultralytics import YOLO

        trained_model = YOLO(str(best_weights))
        onnx_path = trained_model.export(format="onnx", imgsz=imgsz, dynamic=False)
        print(f"Exported ONNX ({imgsz}x{imgsz}): {onnx_path}")
        print(
            f"Set PipelineConfig.detection_imgsz={imgsz} (or configs/*.yaml's "
            f"detection_imgsz) to match when running the pipeline with this checkpoint."
        )


if __name__ == "__main__":
    main()
