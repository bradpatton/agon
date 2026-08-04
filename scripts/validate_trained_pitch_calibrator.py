"""Validates ``TrainedPitchCalibrator`` against real footage -- project plan
Phase 13/14's actual next step now that the keypoint model has trained.

Reports two things, both real measurements, not assertions:
1. Coverage: what fraction of frames get a resolved homography, compared
   against the classical ``PitchKeypointCalibrator`` on the same clip (same
   methodology as Phase 12's static/dynamic/hybrid coverage measurement).
2. Whether this calibrator's homography is *actually* usable by
   ``agon.geometry.camera_pose.camera_pose_from_homography`` -- the specific
   question Phase 14 was left blocked on, since ``PitchKeypointCalibrator``'s
   similarity-transform output structurally can't feed it. Unlike that
   dead end, this calibrator solves a real multi-point projective
   homography, so this is the first real chance to check.

Usage:
    python scripts/validate_trained_pitch_calibrator.py \\
        --input input_videos/match_10min_sample.mp4 \\
        --model models/pitch_calibration/best.onnx \\
        --start-frame 26000 --max-frames 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.camera_pose import (  # noqa: E402
    camera_pose_from_homography,
    pan_tilt_roll_degrees,
)
from agon.geometry.pitch_keypoint_calibrator import PitchKeypointCalibrator  # noqa: E402
from agon.geometry.trained_pitch_calibrator import TrainedPitchCalibrator  # noqa: E402
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

    trained = TrainedPitchCalibrator(
        str(args.model),
        keypoint_confidence=args.keypoint_confidence,
        min_keypoints=args.min_keypoints,
    )
    trained.calibrate(frames)
    trained_resolved = {i for i in range(len(frames)) if trained.homography(i) is not None}

    classical = PitchKeypointCalibrator()
    classical.calibrate(frames)
    classical_resolved = set(classical._transforms.keys())  # noqa: SLF001 -- inspecting resolved frames

    print(
        f"\nCoverage over {len(frames)} frames:\n"
        f"  TrainedPitchCalibrator (keypoint model): {len(trained_resolved)}/{len(frames)} "
        f"({100 * len(trained_resolved) / len(frames):.1f}%)\n"
        f"  PitchKeypointCalibrator (classical, center-circle): "
        f"{len(classical_resolved)}/{len(frames)} "
        f"({100 * len(classical_resolved) / len(frames):.1f}%)\n"
        f"  Resolved by trained but not classical: {len(trained_resolved - classical_resolved)}\n"
        f"  Resolved by classical but not trained: {len(classical_resolved - trained_resolved)}"
    )

    # The real open question from Phase 14: can this calibrator's homography
    # actually feed camera_pose_from_homography, unlike PitchKeypointCalibrator's?
    decomposed = 0
    poses = []
    for frame_idx in sorted(trained_resolved):
        homography = trained.homography(frame_idx)
        assert homography is not None
        pose = camera_pose_from_homography(homography, width, height)
        if pose is not None:
            decomposed += 1
            poses.append((frame_idx, pose))

    print(
        f"\ncamera_pose_from_homography succeeded on {decomposed}/{len(trained_resolved)} "
        f"resolved frames (the specific thing PitchKeypointCalibrator's similarity-transform "
        f"output couldn't do at all -- see Phase 14)."
    )
    if poses:
        print(f"\n{'frame':>6} {'pan':>8} {'tilt':>8} {'roll':>7} {'focal':>8}")
        for frame_idx, pose in poses[:15]:
            pan, tilt, roll = pan_tilt_roll_degrees(pose)
            print(
                f"{frame_idx:>6} {pan:>8.2f} {tilt:>8.2f} {roll:>7.2f} {pose.x_focal_length:>8.1f}"
            )
        if len(poses) > 15:
            print(f"... ({len(poses) - 15} more)")


if __name__ == "__main__":
    main()
