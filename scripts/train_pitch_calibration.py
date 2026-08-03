"""Trains a pitch-calibration keypoint model on the dataset produced by
``convert_soccernet_calibration_to_pose.py`` -- project plan Phase 7 item 2
/ Phase 12 item 5, the eventual replacement for
``agon.geometry.pitch_keypoint_calibrator.PitchKeypointCalibrator``'s
classical-CV, center-circle-only first cut.

Needs the ``[train]`` extra: ``pip install 'agon[train]'`` (torch +
ultralytics). Uses Ultralytics' pose-estimation training mode
(``YOLO("*-pose.pt")``) rather than a custom heatmap-regression loop --
the dataset already writes Ultralytics' own pose-label format (one
"pitch" object per frame, a fixed 46-keypoint vector -- see
``agon.geometry.pitch_keypoints.CANONICAL_KEYPOINTS``), so this reuses the
same library/checkpoint-management/export pattern as detection and jersey
training (``train_detector.py``, ``train_jersey_classifier.py``) rather
than introducing a separate training loop.

At inference, a trained checkpoint's predicted keypoints (each either
present with a pixel position, or absent) feed ``cv2.findHomography`` --
see the eventual ``TrainedPitchCalibrator`` (not yet built) for how that
gets wired into ``agon.interfaces.PitchCalibrator``.

Usage:
    python scripts/train_pitch_calibration.py \\
        --data data/soccernet_pose/dataset.yaml \\
        --base-model yolo11n-pose.pt --imgsz 960 --epochs 50
"""

from __future__ import annotations

import argparse
import os
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
        help="Path to dataset.yaml (written by convert_soccernet_calibration_to_pose.py). "
        "Required unless --resume is given.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a last.pt checkpoint to resume interrupted training from. Ultralytics' "
        "own resume mechanism -- ignores every other flag below, same as train_detector.py.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n-pose.pt",
        help="Ultralytics pose checkpoint name (auto-downloaded) or local path to fine-tune "
        "from. Default: yolo11n-pose.pt.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Training (and, if --export-onnx, export) resolution, square. Default 960, "
        "not Ultralytics' own 640 default -- pitch lines are thin, and this project's "
        "detection training already found real accuracy benefit from a higher resolution "
        "for small/thin real features on broadcast-resolution frames.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Fixed batch size, or -1 (default) for Ultralytics' AutoBatch. See "
        "train_detector.py's docstring -- AutoBatch doesn't support multi-GPU training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', '0,1,2' for specific GPUs, 'mps', or 'cpu'. Default: auto-detect and "
        "use every visible CUDA GPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader worker processes. Default: auto-detect from available system RAM -- "
        "see train_detector.py's _auto_workers docstring for why this matters (a real host-"
        "RAM OOM kill on this project's own jersey-classifier training).",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", type=str, default="pitch_calibration")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export the best checkpoint to ONNX at the same --imgsz after training (default: on).",
    )
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    return parser.parse_args()


def _auto_device() -> str | None:
    """See train_detector.py's identical helper for the full reasoning."""
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


def _auto_workers(gb_per_worker: float = 2.0, reserve_gb: float = 3.0) -> int | None:
    """See train_detector.py's identical helper for the full reasoning
    (a real host-RAM OOM kill on this project's own jersey-classifier
    training is what motivated this, not a hypothetical concern)."""
    try:
        import psutil
    except ImportError:
        return None

    available_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = os.cpu_count() or 1
    budget_gb = max(0.0, available_gb - reserve_gb)
    memory_based = int(budget_gb // gb_per_worker)
    return max(0, min(cpu_count, memory_based))


def _resolve_workers(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    auto = _auto_workers()
    return auto if auto is not None else 8


def _resolve_batch(batch: int, device: str | None) -> int:
    """See train_detector.py's identical helper -- AutoBatch (batch=-1)
    doesn't support multi-GPU training, confirmed on real hardware there."""
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
            "train_pitch_calibration.py needs the 'train' extra: pip install 'agon[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.resume is not None:
        if not args.resume.exists():
            print(f"Resume checkpoint not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        print(f"Resuming from {args.resume} (ignoring --data/--imgsz/--batch/etc.)")
        model = YOLO(str(args.resume))
        resolved_workers = _resolve_workers(args.workers)
        if args.workers is None:
            print(f"Auto-detected workers={resolved_workers} based on available system RAM")
        results = model.train(resume=True, workers=resolved_workers)
        _finish(model, results, args.export_onnx)
        return

    if args.data is None:
        print("--data is required unless --resume is given.", file=sys.stderr)
        sys.exit(1)
    if not args.data.exists():
        print(f"Dataset config not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    device = args.device if args.device is not None else _auto_device()
    if args.device is None and device is not None:
        print(f"Auto-detected {device.count(',') + 1} GPUs, using device={device}")
    workers = _resolve_workers(args.workers)
    if args.workers is None:
        print(f"Auto-detected workers={workers} based on available system RAM")
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
    """See train_detector.py's identical helper -- reads the actual
    trained imgsz back off the model's own trainer args, meaningless to
    trust the CLI's --imgsz in --resume mode."""
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
        print(f"Exported ONNX ({imgsz}x{imgsz}): {onnx_path}")


if __name__ == "__main__":
    main()
