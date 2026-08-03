"""Cross-checks agon.jersey.onnx_classifier.OnnxJerseyClassifier's real
output against Ultralytics' own YOLO(best.pt).predict() on the same real
crops, class by class.

This exists because OnnxJerseyClassifier reimplements Ultralytics'
classification preprocessing by hand (see that module's docstring) rather
than calling Ultralytics itself -- a silent preprocessing mismatch (wrong
normalization, wrong resize) degrades accuracy without raising any error,
unlike a shape mismatch. Run this after any change to
OnnxJerseyClassifier._preprocess() or any retraining that produces a new
best.pt/best.onnx pair, needs the `[train]` extra (for ultralytics) plus a
directory of `<label>/*.jpg` crops such as the one
convert_soccernet_gsr_to_jersey_crops.py writes.

Usage:
    python scripts/validate_jersey_onnx.py \\
        --pt models/runs/train/jersey-gsr/weights/best.pt \\
        --onnx models/runs/train/jersey-gsr/weights/best.onnx \\
        --data data/soccernet_jersey/train
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agon.jersey.onnx_classifier import OnnxJerseyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pt", type=Path, required=True, help="Reference checkpoint (best.pt).")
    parser.add_argument(
        "--onnx", type=Path, required=True, help="Exported ONNX model (classes.json alongside it)."
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Directory of <label>/*.jpg crops to sample from."
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated label subset to sample (default: every label subdirectory found).",
    )
    parser.add_argument(
        "--per-class", type=int, default=4, help="Crops to sample per class (default: 4)."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "validate_jersey_onnx.py needs the 'train' extra: pip install 'agon[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    random.seed(args.seed)
    reference = YOLO(str(args.pt))
    candidate = OnnxJerseyClassifier(str(args.onnx))

    classes = (
        args.classes.split(",")
        if args.classes
        else sorted(p.name for p in args.data.iterdir() if p.is_dir())
    )
    samples: list[tuple[str, str]] = []
    for cls in classes:
        files = sorted(glob.glob(str(args.data / cls / "*.jpg")))
        if not files:
            continue
        samples.extend((cls, f) for f in random.sample(files, min(args.per_class, len(files))))

    print(f"Testing {len(samples)} real crops across {len(classes)} classes\n")
    header = f"{'true':<8} {'ref_pred':<8} {'ref_conf':<10} {'onnx_pred':<10} {'onnx_conf':<10}"
    print(f"{header} match")

    agree = 0
    total = 0
    conf_diffs = []

    for true_label, path in samples:
        raw = cv2.imread(path)
        if raw is None:
            continue
        img = raw.astype(np.uint8)
        h, w = img.shape[:2]

        ref_result = reference.predict(path, imgsz=candidate.input_size[0], verbose=False)[0]
        ref_top1_idx = int(ref_result.probs.top1)
        ref_conf = float(ref_result.probs.top1conf)
        ref_label = ref_result.names[ref_top1_idx]

        onnx_pred, onnx_conf = candidate.classify(img, (0, 0, w, h))
        onnx_label = "unknown" if onnx_pred is None else str(onnx_pred)

        match = ref_label == onnx_label
        agree += int(match)
        total += 1
        conf_diffs.append(abs(ref_conf - onnx_conf))

        print(
            f"{true_label:<8} {ref_label:<8} {ref_conf:<10.4f} {onnx_label:<10} {onnx_conf:<10.4f} "
            f"{'YES' if match else 'NO'}"
        )

    if total == 0:
        print("No crops found -- check --data.", file=sys.stderr)
        sys.exit(1)

    print(
        f"\n{agree}/{total} predictions matched between reference and ONNX "
        f"({100 * agree / total:.1f}%)"
    )
    print(
        f"Confidence diff: mean={sum(conf_diffs) / len(conf_diffs):.4f}, max={max(conf_diffs):.4f}"
    )


if __name__ == "__main__":
    main()
