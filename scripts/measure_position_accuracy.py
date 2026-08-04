"""The real ground-truth player-position accuracy measurement this project
has never had -- every prior "accuracy" check (coverage percentages,
cross-calibrator agreement, self-consistency touchline hit-rates) measured
something else. See ``agon.geometry.trained_pitch_calibrator.
leave_one_out_position_errors``'s docstring for the method: for each real,
confidently-detected pitch keypoint in a frame, fit a homography from every
*other* confident keypoint (exactly as a real player position estimate
would be -- the player's own position is never part of the fit), then
check how far the held-out keypoint's predicted pitch-space position is
from its independently known true position (FIFA pitch dimensions, not
another calibrator's opinion).

Usage:
    python scripts/measure_position_accuracy.py \\
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

from agon.geometry.trained_pitch_calibrator import (  # noqa: E402
    TrainedPitchCalibrator,
    leave_one_out_position_errors,
)
from agon.io.video import Frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keypoint-confidence", type=float, default=0.3)
    parser.add_argument("--min-keypoints", type=int, default=4)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=10.0)
    parser.add_argument(
        "--exclude-line",
        action="append",
        default=[],
        dest="excluded_line_names",
        help="Canonical line name to exclude entirely (both from fitting and evaluation), "
        "e.g. --exclude-line 'Small rect. left main' --exclude-line 'Small rect. left top' -- "
        "repeatable. See leave_one_out_position_errors's docstring for why this exists "
        "(six-yard-box keypoints were found to be frequently misdetected on real footage).",
    )
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

    calibrator = TrainedPitchCalibrator(str(args.model))
    excluded_line_names = frozenset(args.excluded_line_names)
    if excluded_line_names:
        print(f"Excluding {sorted(excluded_line_names)} from fitting and evaluation.")

    all_errors: list[float] = []
    frames_checked = 0
    for frame in frames:
        keypoints = calibrator._detect_keypoints(frame)  # noqa: SLF001 -- direct access for measurement
        errors = leave_one_out_position_errors(
            keypoints,
            keypoint_confidence=args.keypoint_confidence,
            min_keypoints=args.min_keypoints,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            excluded_line_names=excluded_line_names,
        )
        if errors:
            frames_checked += 1
            all_errors.extend(errors)

    print(
        f"\n{frames_checked}/{len(frames)} frames had enough confident keypoints "
        f"(>= {args.min_keypoints + 1}) to run a leave-one-out check.\n"
        f"{len(all_errors)} total held-out position measurements."
    )
    if not all_errors:
        return

    errors_arr = np.array(all_errors)
    print(
        f"\nPixel->pitch position error (meters), held-out real ground truth:\n"
        f"  median = {np.median(errors_arr):.3f}m\n"
        f"  mean   = {np.mean(errors_arr):.3f}m\n"
        f"  p90    = {np.percentile(errors_arr, 90):.3f}m\n"
        f"  max    = {np.max(errors_arr):.3f}m\n"
        f"  min    = {np.min(errors_arr):.3f}m"
    )


if __name__ == "__main__":
    main()
