"""Compares the trained pitch-calibration model's real predictions against
human-annotated ground truth (``scripts/annotate_ground_truth.py``'s
output), in raw pixels -- no homography fit, no FIFA-dimension assumption,
no self-consistency proxy. This is the most direct possible check of
keypoint localization accuracy: for each keypoint a human marked on a real
frame, how far away (in pixels) did the model actually predict it?

Every prior accuracy check in this project either measured something else
(coverage, cross-calibrator agreement, self-consistency between pairs of
keypoints that should coincide) or depended on the FIFA pitch-dimension
assumption plus a homography fit (``leave_one_out_position_errors``,
median 16.7m error -- real, but conflates keypoint localization error with
homography-fit sensitivity). This script isolates the first half of that
chain directly.

Usage:
    python scripts/measure_ground_truth_accuracy.py \\
        --ground-truth data/ground_truth/match_10min_sample.json \\
        --model models/pitch_calibration/best.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.pitch_keypoints import CANONICAL_KEYPOINTS  # noqa: E402
from agon.geometry.trained_pitch_calibrator import TrainedPitchCalibrator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--keypoint-confidence", type=float, default=0.3)
    return parser.parse_args()


def _read_frame(path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
        return frame.astype(np.uint8)
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    with open(args.ground_truth) as f:
        gt_data = json.load(f)

    video_path = gt_data["video"]
    calibrator = TrainedPitchCalibrator(str(args.model))

    per_keypoint_errors: dict[str, list[float]] = defaultdict(list)
    per_keypoint_missed = 0
    all_errors: list[float] = []
    total_annotated = 0

    for frame_idx_str, points in gt_data["frames"].items():
        frame_idx = int(frame_idx_str)
        frame = _read_frame(video_path, frame_idx)
        predicted = calibrator._detect_keypoints(frame)  # noqa: SLF001 -- measurement-only access

        for i, (name, endpoint_idx) in enumerate(CANONICAL_KEYPOINTS):
            key = f"{name}|{endpoint_idx}"
            gt_point = points.get(key)
            if gt_point is None:
                continue
            total_annotated += 1

            pred_x, pred_y, pred_conf = predicted[i]
            if pred_conf < args.keypoint_confidence:
                per_keypoint_missed += 1
                continue

            error_px = float(np.hypot(pred_x - gt_point[0], pred_y - gt_point[1]))
            per_keypoint_errors[key].append(error_px)
            all_errors.append(error_px)

    print(
        f"{total_annotated} human-annotated keypoints across "
        f"{len(gt_data['frames'])} frames.\n"
        f"{per_keypoint_missed} had no confident model prediction "
        f"(< {args.keypoint_confidence} confidence) to compare against.\n"
        f"{len(all_errors)} direct pixel-error measurements.\n"
    )
    if not all_errors:
        return

    print(f"{'keypoint':<35} {'n':>4} {'median_px':>10} {'mean_px':>9} {'max_px':>8}")
    for key, errors in sorted(per_keypoint_errors.items(), key=lambda kv: -np.median(kv[1])):
        arr = np.array(errors)
        print(
            f"{key:<35} {len(arr):>4} {np.median(arr):>10.1f} "
            f"{np.mean(arr):>9.1f} {np.max(arr):>8.1f}"
        )

    all_arr = np.array(all_errors)
    print(
        f"\nOverall keypoint localization error vs. human ground truth (pixels):\n"
        f"  median = {np.median(all_arr):.1f}px\n"
        f"  mean   = {np.mean(all_arr):.1f}px\n"
        f"  p90    = {np.percentile(all_arr, 90):.1f}px\n"
        f"  max    = {np.max(all_arr):.1f}px"
    )


if __name__ == "__main__":
    main()
