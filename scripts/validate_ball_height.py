"""Validates ``agon.analytics.ball_height.estimate_ball_position_3d``
against real footage -- project plan's "ball height (Z)" item, tabled
since Phase 7 for lack of a real camera model, now unblocked by
``TrainedPitchCalibrator``/``agon.geometry.camera_pose``.

Runs real ball detection (a soccer-fine-tuned checkpoint) and the trained
pitch calibrator over the same clip, and for every frame with both a ball
detection *and* a resolved camera pose, estimates 3D ball position and
reports real numbers -- not a unit test (the math is already covered
synthetically in ``tests/test_ball_height.py``), a sanity check on real,
noisy detections.

What "sane" looks like, concretely: a ball resting/rolling on the ground
should estimate a height near -0.11m (the ball's radius -- its *center*
sits half a diameter above the ground, not at z=0; z is negative-for-up in
this project's convention), consistently across consecutive frames of
open play. A real kick/cross/clearance should show a real, physically
plausible rise-and-fall across a short window of frames, not noise.

**Real result from running this against match_10min_sample.mp4, not
sane**: 15-26m estimated heights across 121 checkable frames, regardless
of detection confidence (checked directly -- no correlation). Running
this script surfaced and helped confirm the fix for a real, independent
bug in ``camera_pose_from_homography`` (a sign ambiguity -- see that
function's docstring), but heights stayed implausible even after the fix.
See ``agon.analytics.ball_height``'s module docstring for the full
diagnosis and why this isn't wired into the pipeline. Kept as a real,
reusable diagnostic -- rerun it after any future change to
``TrainedPitchCalibrator``/``camera_pose_from_homography`` to check
whether the underlying accuracy has improved.

Usage:
    python scripts/validate_ball_height.py \\
        --input input_videos/match_10min_sample.mp4 \\
        --detector-model models/finetuned/yolo11n_soccernet_subset.onnx \\
        --pitch-model models/pitch_calibration/best.onnx \\
        --start-frame 25500 --max-frames 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.analytics.ball_height import estimate_ball_position_3d  # noqa: E402
from agon.detection.onnx_tracker import OnnxDetector  # noqa: E402
from agon.geometry.camera_pose import camera_pose_from_homography  # noqa: E402
from agon.geometry.trained_pitch_calibrator import TrainedPitchCalibrator  # noqa: E402
from agon.io.video import Frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--pitch-model", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
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
    height, width = frames[0].shape[:2]

    detector = OnnxDetector(str(args.detector_model))
    tracks = detector.get_object_tracks(frames)
    ball_frames = tracks["ball"]

    calibrator = TrainedPitchCalibrator(str(args.pitch_model))
    calibrator.calibrate(frames)

    results = []
    for frame_idx in range(len(frames)):
        ball = ball_frames[frame_idx]
        if not ball:
            continue
        homography = calibrator.homography(frame_idx)
        if homography is None:
            continue
        pose = camera_pose_from_homography(homography, width, height)
        if pose is None:
            continue
        bbox = tuple(ball[1]["bbox"])
        position = estimate_ball_position_3d(pose, bbox)
        if position is not None:
            results.append((frame_idx, position))

    print(
        f"\n{len(results)}/{len(frames)} frames had both a ball detection and a resolved "
        f"camera pose."
    )
    if not results:
        return

    print(f"\n{'frame':>6} {'x_m':>8} {'y_m':>8} {'z_m':>8}")
    for frame_idx, (x, y, z) in results[:30]:
        print(f"{frame_idx:>6} {x:>8.2f} {y:>8.2f} {z:>8.2f}")
    if len(results) > 30:
        print(f"... ({len(results) - 30} more)")

    z_values = np.array([z for _frame_idx, (_x, _y, z) in results])
    print(
        f"\nz (height, negative=up) over {len(z_values)} frames: "
        f"median={np.median(z_values):.3f}m, min={z_values.min():.3f}m, "
        f"max={z_values.max():.3f}m, std={z_values.std():.3f}m. "
        f"A grounded ball's center should be near -0.11m (its own radius)."
    )


if __name__ == "__main__":
    main()
