"""Self-consistency check for PitchKeypointCalibrator: predicts where the
touchlines should be in pixel space using *only* the resolved center-circle
+ halfway-line transform plus the pitch's known real width, then checks
whether the frame actually has a real pitch-line pixel there.

Why this is a real check and not circular: the calibrator's transform is
built entirely from the center circle (scale, position) and the halfway
line's angle (rotation) -- it never looks at the touchlines at all. A
touchline sits at a known, fixed offset from the circle center *given only
the pitch's real width* (a touchline is where the halfway line, extended
court_width_m/2 meters from the center circle, meets the pitch boundary --
see PitchKeypointCalibrator's class docstring for the coordinate
convention). If the resolved transform is accurate, projecting that known
point back into pixel space should land on the real touchline visible in
the frame; if it doesn't, the transform is wrong (or the pitch feature
just isn't visible in this frame -- also checked and reported separately).

This only checks touchlines (the one other feature the current 68m-width
parameter gives us a known distance to) -- it cannot check penalty-box or
goal-line features, since PitchKeypointCalibrator has no court_length_m
and doesn't know which way is "toward which goal." See the project plan's
Phase 12 item 2 (full multi-feature homography) for what would unlock that.

Usage:
    python scripts/validate_pitch_calibration_self_consistency.py \\
        --input input_videos/benchmark_clip.mp4 \\
        --court-width-m 68.0 \\
        --draw-sample-frame /tmp/touchline_check.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.bbox import Point  # noqa: E402
from agon.geometry.pitch_keypoint_calibrator import (  # noqa: E402
    PitchKeypointCalibrator,
    _segment_pitch_lines,
)
from agon.io.video import Frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--court-width-m", type=float, default=68.0)
    parser.add_argument(
        "--search-radius-px",
        type=int,
        default=25,
        help="How close (pixels) a real line pixel must be to the predicted "
        "touchline point to count as a match. Default 25.",
    )
    parser.add_argument(
        "--draw-sample-frame",
        type=Path,
        default=None,
        help="If given, saves one frame with the predicted touchline points drawn on "
        "it (green = matched a real line pixel nearby, red = didn't) for a visual check.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip to this frame before reading -- useful for checking a slice of a "
        "long match instead of loading the whole thing into memory.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Read at most this many frames starting at --start-frame. Default: whole video.",
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


def _nearest_line_pixel_distance(
    line_mask: np.ndarray, point: Point, search_radius_px: int
) -> float | None:
    """Distance (px) from ``point`` to the nearest nonzero pixel in
    ``line_mask`` within a ``search_radius_px`` window, or None if there
    isn't one that close."""
    px, py = point
    h, w = line_mask.shape[:2]
    x1, x2 = max(0, px - search_radius_px), min(w, px + search_radius_px + 1)
    y1, y2 = max(0, py - search_radius_px), min(h, py + search_radius_px + 1)
    window = line_mask[y1:y2, x1:x2]

    ys, xs = np.nonzero(window)
    if len(xs) == 0:
        return None

    distances = np.hypot(xs + x1 - px, ys + y1 - py)
    best = float(distances.min())
    return best if best <= search_radius_px else None


def main() -> None:
    args = parse_args()
    frames = _read_video_slice(args.input, args.start_frame, args.max_frames)
    print(f"Read {len(frames)} frames from {args.input} (starting at frame {args.start_frame})")

    calibrator = PitchKeypointCalibrator(court_width_m=args.court_width_m)
    calibrator.calibrate(frames)

    half_width = args.court_width_m / 2
    touchline_points: list[Point] = [(0.0, half_width), (0.0, -half_width)]

    checkable = 0
    hits = 0
    offsets: list[float] = []
    sample_frame: Frame | None = None
    sample_predictions: list[tuple[tuple[int, int], bool]] = []

    for frame_idx, frame in enumerate(frames):
        h, w = frame.shape[:2]
        line_mask: np.ndarray | None = None
        frame_predictions: list[tuple[tuple[int, int], bool]] = []

        for touchline_point in touchline_points:
            pixel = calibrator.inverse_transform_point(touchline_point, frame_idx)
            if pixel is None:
                continue
            px, py = int(pixel[0]), int(pixel[1])
            if not (0 <= px < w and 0 <= py < h):
                continue

            if line_mask is None:
                line_mask = _segment_pitch_lines(frame)
            checkable += 1
            distance = _nearest_line_pixel_distance(line_mask, (px, py), args.search_radius_px)
            matched = distance is not None
            if matched:
                hits += 1
                offsets.append(distance)  # type: ignore[arg-type]
            frame_predictions.append(((px, py), matched))

        # Opportunistically grab the first frame with at least one
        # checkable prediction for the visual sample -- doesn't affect the
        # aggregate stats above, which still run over every frame.
        if args.draw_sample_frame is not None and sample_frame is None and frame_predictions:
            sample_frame = frame.copy()
            sample_predictions = frame_predictions

    print(f"\nCheckable predictions (touchline point landed in-frame): {checkable}")
    if checkable:
        print(
            f"Matched a real pitch-line pixel within {args.search_radius_px}px: "
            f"{hits} ({100 * hits / checkable:.1f}%)"
        )
    if offsets:
        offsets.sort()
        print(f"Mean offset to nearest line pixel: {sum(offsets) / len(offsets):.1f}px")
        print(f"Median offset: {offsets[len(offsets) // 2]:.1f}px")

    if args.draw_sample_frame is not None and sample_frame is not None:
        for (px, py), matched in sample_predictions:
            color = (0, 200, 0) if matched else (0, 0, 220)
            cv2.drawMarker(sample_frame, (px, py), color, cv2.MARKER_CROSS, 30, 3)
        cv2.imwrite(str(args.draw_sample_frame), sample_frame)
        print(f"\nSaved a sample frame with predicted touchline points to {args.draw_sample_frame}")


if __name__ == "__main__":
    main()
