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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset.yaml (written by convert_soccernet_gsr_to_yolo.py).",
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
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', 'cuda:0,1,2' for multi-GPU, 'mps', or 'cpu'. None = ultralytics auto-detect.",
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


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        print(f"Dataset config not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not args.base_model.exists():
        print(f"Base checkpoint not found: {args.base_model}", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "train_detector.py needs the 'train' extra: pip install 'agon[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(str(args.base_model))
    results = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )
    print(f"\nTraining done. Results/weights under: {results.save_dir}")

    if args.export_onnx:
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            print(f"Expected best.pt at {best_weights}, skipping ONNX export.", file=sys.stderr)
            return

        trained_model = YOLO(str(best_weights))
        onnx_path = trained_model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
        print(f"Exported ONNX ({args.imgsz}x{args.imgsz}): {onnx_path}")
        print(
            f"Set PipelineConfig.detection_imgsz={args.imgsz} (or configs/*.yaml's "
            f"detection_imgsz) to match when running the pipeline with this checkpoint."
        )


if __name__ == "__main__":
    main()
