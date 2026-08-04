"""Decomposes PitchKeypointCalibrator's resolved per-frame transform into a
full 3D camera pose (pan, tilt, roll, position, focal length) via
``agon.geometry.camera_pose`` -- project plan Phase 14, validating the
homography-decomposition approach against real footage ahead of the
trained-keypoint-model calibrator it's ultimately meant for.

Since ``PitchKeypointCalibrator`` only exposes point-transform functions
(it's a similarity transform internally, not a stored matrix), each
resolved frame's equivalent homography is derived via
``homography_from_point_transform`` before decomposing -- an approximation
appropriate for today's classical calibrator; the eventual
``TrainedPitchCalibrator`` will produce a real projective homography
directly and skip that step.

This is a real-footage sanity check, not a unit test: a genuinely correct
decomposition should produce (a) a plausible tilt angle for an elevated
broadcast tactical camera, (b) a plausible focal length (hundreds to low
thousands of pixels, not near-zero or absurdly large), and (c) frame-to-frame
*smoothness* across a static or slow-panning shot -- a real camera doesn't
teleport, so wild jumps between consecutive resolved frames indicate either
decomposition noise or an underlying bad calibration, not a real camera
move.

Usage:
    python scripts/decompose_pitch_camera_pose.py \\
        --input input_videos/benchmark_clip.mp4 --max-frames 180
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.camera_pose import (  # noqa: E402
    CameraPose,
    camera_pose_from_homography,
    homography_from_point_transform,
    pan_tilt_roll_degrees,
)
from agon.geometry.pitch_keypoint_calibrator import PitchKeypointCalibrator  # noqa: E402
from agon.io.video import Frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
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


def decompose_resolved_frames(
    calibrator: PitchKeypointCalibrator, num_frames: int, image_width: int, image_height: int
) -> dict[int, CameraPose]:
    """Returns {frame_idx: CameraPose} for every frame where both the
    calibrator resolved a transform *and* the homography decomposition
    itself succeeded -- fewer than the calibrator's own resolved count is
    expected (single-view self-calibration is more sensitive to noise than
    the plain point-transform it's built from), not a bug."""
    poses: dict[int, CameraPose] = {}
    for frame_idx in range(num_frames):
        homography = homography_from_point_transform(
            lambda p, i=frame_idx: calibrator.inverse_transform_point(p, frame_idx=i)
        )
        if homography is None:
            continue
        pose = camera_pose_from_homography(homography, image_width, image_height)
        if pose is not None:
            poses[frame_idx] = pose
    return poses


def main() -> None:
    args = parse_args()
    frames = _read_video_slice(args.input, args.start_frame, args.max_frames)
    print(f"Read {len(frames)} frames from {args.input} (starting at frame {args.start_frame})")
    if not frames:
        return
    height, width = frames[0].shape[:2]

    calibrator = PitchKeypointCalibrator()
    calibrator.calibrate(frames)
    num_resolved = len(calibrator._transforms)  # noqa: SLF001 -- inspecting resolved frames

    poses = decompose_resolved_frames(calibrator, len(frames), width, height)
    print(
        f"{num_resolved}/{len(frames)} frames had a resolved pitch transform; "
        f"{len(poses)}/{num_resolved} of those also yielded a valid camera-pose decomposition."
    )
    if not poses:
        print("No frames decomposed -- nothing further to report.")
        return

    print(
        f"\n{'frame':>6} {'pan':>8} {'tilt':>8} {'roll':>7} {'x':>8} {'y':>8} {'z':>8} {'focal':>8}"
    )
    for frame_idx in sorted(poses):
        pose = poses[frame_idx]
        pan, tilt, roll = pan_tilt_roll_degrees(pose)
        x, y, z = pose.position
        print(
            f"{frame_idx:>6} {pan:>8.2f} {tilt:>8.2f} {roll:>7.2f} "
            f"{x:>8.2f} {y:>8.2f} {z:>8.2f} {pose.x_focal_length:>8.1f}"
        )

    # Frame-to-frame smoothness across consecutive resolved+decomposed
    # frames -- the real sanity check described in the module docstring.
    ordered = sorted(poses)
    consecutive_pairs = [(a, b) for a, b in zip(ordered, ordered[1:], strict=False) if b - a == 1]
    if consecutive_pairs:
        position_deltas = []
        focal_deltas = []
        tilt_deltas = []
        for a, b in consecutive_pairs:
            pa, pb = poses[a], poses[b]
            position_deltas.append(float(np.linalg.norm(pa.position - pb.position)))
            focal_deltas.append(abs(pa.x_focal_length - pb.x_focal_length))
            _pan_a, tilt_a, _roll_a = pan_tilt_roll_degrees(pa)
            _pan_b, tilt_b, _roll_b = pan_tilt_roll_degrees(pb)
            tilt_deltas.append(abs(tilt_a - tilt_b))
        print(
            f"\nOver {len(consecutive_pairs)} consecutive-frame pairs: "
            f"median position jump = {np.median(position_deltas):.2f}m "
            f"(max {np.max(position_deltas):.2f}m), "
            f"median focal-length jump = {np.median(focal_deltas):.1f}px "
            f"(max {np.max(focal_deltas):.1f}px), "
            f"median tilt jump = {np.median(tilt_deltas):.2f} deg "
            f"(max {np.max(tilt_deltas):.2f} deg)."
        )
    else:
        print("\nNo consecutive-frame pairs both decomposed -- can't measure smoothness.")


if __name__ == "__main__":
    main()
