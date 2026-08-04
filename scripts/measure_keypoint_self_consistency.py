"""Measures the trained pitch-calibration keypoint model's own pixel-level
localization precision, independent of ground truth, homography fitting,
or pitch-dimension assumptions entirely -- a direct test of the
"keypoint localization precision, not detection confidence, is the likely
cause" hypothesis from ``leave_one_out_position_errors``'s real 16.7m
median-error finding.

Several canonical keypoints are, by the pitch's own geometry, the exact
same real-world point under two different names (e.g. "Big rect. left
main" endpoint 0 and "Big rect. left top" endpoint 1 are both the box's
far corner -- see ``agon.geometry.pitch_keypoints``' module docstring,
confirmed in Phase 13). If the model detects two keypoints that must be
identical at meaningfully different pixel positions, that gap *is* the
model's own localization noise, measured directly in pixels -- no
assumption about this stadium's exact dimensions, no homography fit, no
ground-truth position needed at all.

Usage:
    python scripts/measure_keypoint_self_consistency.py \\
        --input input_videos/match_10min_sample.mp4 \\
        --model models/pitch_calibration/best.onnx \\
        --start-frame 25500 --max-frames 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.pitch_keypoints import CANONICAL_KEYPOINTS, LINE_ENDPOINTS_M  # noqa: E402
from agon.geometry.trained_pitch_calibrator import TrainedPitchCalibrator  # noqa: E402
from agon.io.video import Frame  # noqa: E402


def _coincident_pairs() -> list[tuple[int, int, str, str]]:
    """Returns (index_a, index_b, label_a, label_b) for every pair of
    canonical keypoints that share the exact same real-world position."""
    by_point: dict[tuple[float, float], list[int]] = {}
    for i, (name, endpoint_idx) in enumerate(CANONICAL_KEYPOINTS):
        point = LINE_ENDPOINTS_M[name][endpoint_idx]
        by_point.setdefault(point, []).append(i)

    pairs = []
    for indices in by_point.values():
        if len(indices) < 2:
            continue
        i, j = indices[0], indices[1]
        name_i, idx_i = CANONICAL_KEYPOINTS[i]
        name_j, idx_j = CANONICAL_KEYPOINTS[j]
        pairs.append((i, j, f"{name_i} pt{idx_i}", f"{name_j} pt{idx_j}"))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keypoint-confidence", type=float, default=0.3)
    return parser.parse_args()


def _read_video_slice(path: Path, start_frame: int, max_frames: int | None) -> list[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[Frame] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame.astype(np.uint8))
    finally:
        cap.release()
    return frames


def main() -> None:
    args = parse_args()
    frames = _read_video_slice(args.input, args.start_frame, args.max_frames)
    print(f"Read {len(frames)} frames from {args.input} (starting at frame {args.start_frame})")
    if not frames:
        return

    pairs = _coincident_pairs()
    print(f"Checking {len(pairs)} coincident-corner keypoint pairs per frame.\n")

    calibrator = TrainedPitchCalibrator(str(args.model))
    per_pair_distances: dict[str, list[float]] = {}
    all_distances: list[float] = []

    for frame in frames:
        keypoints = calibrator._detect_keypoints(frame)  # noqa: SLF001 -- direct access for measurement
        for i, j, label_a, label_b in pairs:
            pa, pb = keypoints[i], keypoints[j]
            if pa[2] < args.keypoint_confidence or pb[2] < args.keypoint_confidence:
                continue
            distance_px = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            key = f"{label_a} == {label_b}"
            per_pair_distances.setdefault(key, []).append(distance_px)
            all_distances.append(distance_px)

    print(f"{len(all_distances)} total pair measurements across {len(frames)} frames.\n")
    if not all_distances:
        return

    print(f"{'pair':<45} {'n':>5} {'median_px':>10} {'mean_px':>9} {'max_px':>8}")
    for key, distances in sorted(per_pair_distances.items(), key=lambda kv: -np.median(kv[1])):
        arr = np.array(distances)
        median, mean, worst = np.median(arr), np.mean(arr), np.max(arr)
        print(f"{key:<45} {len(arr):>5} {median:>10.1f} {mean:>9.1f} {worst:>8.1f}")

    all_arr = np.array(all_distances)
    print(
        f"\nOverall pixel self-consistency (same real point, two different "
        f"detected pixel positions):\n"
        f"  median = {np.median(all_arr):.1f}px\n"
        f"  mean   = {np.mean(all_arr):.1f}px\n"
        f"  p90    = {np.percentile(all_arr, 90):.1f}px\n"
        f"  max    = {np.max(all_arr):.1f}px"
    )


if __name__ == "__main__":
    main()
