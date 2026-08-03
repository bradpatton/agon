"""Draws every standard soccer pitch marking (touchlines, goal lines,
penalty boxes, six-yard boxes, penalty spots, penalty arcs, corner arcs,
halfway line, center circle) onto real footage, computed purely from
PitchKeypointCalibrator's resolved per-frame transform plus fixed,
real-world pitch dimensions -- not detected, calculated.

This is a visual sibling to
scripts/validate_pitch_calibration_self_consistency.py's touchline check
(same idea -- predict a known real feature's pixel location, see if it's
actually there -- extended to every standard marking and made visual
instead of a single hit/miss percentage), and a direct, look-at-it way to
judge calibration quality: if the drawn lines land on the real pitch
markings, the transform is trustworthy for that frame; if they don't,
it isn't, and *how* they're wrong (offset? rotated? wrong scale?) is
informative in a way a single aggregate number isn't.

Only straight-line features (touchlines, goal lines, box edges) are drawn
exactly; the center circle, penalty arcs, and corner arcs are approximated
as true circles under PitchKeypointCalibrator's transform. This is
deliberate, not an oversight: that transform is a similarity transform
(uniform scale + rotation, no perspective correction -- see that class's
own docstring), so it maps real circles to *circles* in pixel space, not
the elongated ellipses a real broadcast camera's perspective actually
produces. Seeing the drawn (circular) center-circle overlay visibly not
match the real (elliptical) center circle in the footage is itself a
direct, honest visualization of that known, already-documented limitation
-- not a bug in this script.

Usage:
    python scripts/render_pitch_markings_overlay.py \\
        --input input_videos/match_10min_sample.mp4 \\
        --start-frame 27000 --max-frames 500 \\
        --output-video output/pitch_overlay/match_10min_sample_overlay.mp4 \\
        --output-dir output/pitch_overlay/samples
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
    CENTER_CIRCLE_RADIUS_M,
    PitchKeypointCalibrator,
)
from agon.io.video import Frame, get_video_info, save_video  # noqa: E402

# Fixed by the Laws of the Game -- do not vary by pitch, unlike overall
# length/width (which do, within FIFA's allowed range).
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
SIX_YARD_BOX_DEPTH_M = 5.5
SIX_YARD_BOX_WIDTH_M = 18.32
PENALTY_SPOT_DISTANCE_M = 11.0
CORNER_ARC_RADIUS_M = 1.0

_GOAL_LINE_COLOR = (0, 0, 255)
_TOUCHLINE_COLOR = (255, 0, 0)
_BOX_COLOR = (0, 255, 255)
_CIRCLE_COLOR = (0, 255, 0)
_LINE_THICKNESS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--court-width-m", type=float, default=68.0)
    parser.add_argument("--court-length-m", type=float, default=105.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write annotated sample frames into -- one per resolved frame "
        "encountered, up to --max-samples. At least one of --output-dir/--output-video "
        "is required.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Stop writing sample images after this many (default 10) -- avoids "
        "writing thousands of near-identical images for a long clip. Doesn't limit "
        "--output-video, which always covers every frame read.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Path to write a full annotated .mp4 covering every frame read (overlay drawn "
        "on resolved frames, original frame passed through unchanged otherwise) -- unlike "
        "--output-dir's sample images, this is watchable end-to-end, e.g. in a video player.",
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


def _polyline_points(
    calibrator: PitchKeypointCalibrator, points: list[Point], frame_idx: int
) -> list[tuple[int, int]] | None:
    """Transforms every pitch-space point to pixel space; None if any one
    of them has no resolved transform (shouldn't happen if the caller
    already checked the frame is resolved, but keeps this defensive)."""
    pixel_points = []
    for p in points:
        pixel = calibrator.inverse_transform_point(p, frame_idx)
        if pixel is None:
            return None
        pixel_points.append((int(pixel[0]), int(pixel[1])))
    return pixel_points


def _circle_points(radius_m: float, center: Point = (0.0, 0.0), n: int = 72) -> list[Point]:
    cx, cy = center
    return [
        (cx + radius_m * np.cos(t), cy + radius_m * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n)
    ]


def _arc_points(
    radius_m: float, center: Point, start_deg: float, end_deg: float, n: int = 36
) -> list[Point]:
    cx, cy = center
    return [
        (cx + radius_m * np.cos(np.radians(t)), cy + radius_m * np.sin(np.radians(t)))
        for t in np.linspace(start_deg, end_deg, n)
    ]


def _draw_polyline(
    frame: Frame, points: list[tuple[int, int]] | None, color: tuple[int, int, int], closed: bool
) -> None:
    if points is None:
        return
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], closed, color, _LINE_THICKNESS)


def draw_pitch_markings(
    frame: Frame,
    calibrator: PitchKeypointCalibrator,
    frame_idx: int,
    court_width_m: float,
    court_length_m: float,
) -> Frame:
    """Draws every standard pitch marking onto a copy of ``frame`` using
    only the calibrator's resolved transform for this frame plus fixed
    real-world pitch dimensions -- see module docstring."""
    out = frame.copy()
    half_w = court_width_m / 2
    half_l = court_length_m / 2

    # Touchlines: y = +/- half_w, spanning the full length.
    for y in (half_w, -half_w):
        _draw_polyline(
            out,
            _polyline_points(calibrator, [(-half_l, y), (half_l, y)], frame_idx),
            _TOUCHLINE_COLOR,
            closed=False,
        )

    # Goal lines: x = +/- half_l, spanning the full width.
    for x in (half_l, -half_l):
        _draw_polyline(
            out,
            _polyline_points(calibrator, [(x, -half_w), (x, half_w)], frame_idx),
            _GOAL_LINE_COLOR,
            closed=False,
        )

    # Halfway line.
    _draw_polyline(
        out,
        _polyline_points(calibrator, [(0.0, -half_w), (0.0, half_w)], frame_idx),
        _CIRCLE_COLOR,
        closed=False,
    )

    # Penalty boxes + six-yard boxes + penalty spots, one end at a time.
    for goal_x, direction in ((half_l, -1), (-half_l, 1)):
        box_inner_x = goal_x + direction * PENALTY_BOX_DEPTH_M
        box_points = [
            (goal_x, -PENALTY_BOX_WIDTH_M / 2),
            (box_inner_x, -PENALTY_BOX_WIDTH_M / 2),
            (box_inner_x, PENALTY_BOX_WIDTH_M / 2),
            (goal_x, PENALTY_BOX_WIDTH_M / 2),
        ]
        _draw_polyline(
            out, _polyline_points(calibrator, box_points, frame_idx), _BOX_COLOR, closed=False
        )

        six_yard_inner_x = goal_x + direction * SIX_YARD_BOX_DEPTH_M
        six_yard_points = [
            (goal_x, -SIX_YARD_BOX_WIDTH_M / 2),
            (six_yard_inner_x, -SIX_YARD_BOX_WIDTH_M / 2),
            (six_yard_inner_x, SIX_YARD_BOX_WIDTH_M / 2),
            (goal_x, SIX_YARD_BOX_WIDTH_M / 2),
        ]
        _draw_polyline(
            out, _polyline_points(calibrator, six_yard_points, frame_idx), _BOX_COLOR, closed=False
        )

        spot_x = goal_x + direction * PENALTY_SPOT_DISTANCE_M
        spot_pixel = calibrator.inverse_transform_point((spot_x, 0.0), frame_idx)
        if spot_pixel is not None:
            cv2.drawMarker(
                out,
                (int(spot_pixel[0]), int(spot_pixel[1])),
                _BOX_COLOR,
                cv2.MARKER_CROSS,
                12,
                _LINE_THICKNESS,
            )

        # Penalty arc: the 9.15m-radius arc around the spot, but only the
        # part outside the box -- approximated with a wide angular range
        # centered on the arc's outward-facing side; close enough for a
        # visual check without solving the exact box-intersection angles.
        center_angle = 180.0 if direction == -1 else 0.0
        arc = _arc_points(
            CENTER_CIRCLE_RADIUS_M, (spot_x, 0.0), center_angle - 53, center_angle + 53
        )
        _draw_polyline(
            out, _polyline_points(calibrator, arc, frame_idx), _CIRCLE_COLOR, closed=False
        )

    # Center circle -- see module docstring for why this is drawn as a
    # true circle, and why that's an honest (not hidden) approximation.
    _draw_polyline(
        out,
        _polyline_points(calibrator, _circle_points(CENTER_CIRCLE_RADIUS_M), frame_idx),
        _CIRCLE_COLOR,
        closed=True,
    )

    # Corner arcs.
    for x_sign in (1, -1):
        for y_sign in (1, -1):
            corner = (x_sign * half_l, y_sign * half_w)
            start = {(1, 1): 90, (1, -1): 180, (-1, -1): 270, (-1, 1): 0}[(x_sign, y_sign)]
            arc = _arc_points(CORNER_ARC_RADIUS_M, corner, start, start + 90)
            _draw_polyline(
                out, _polyline_points(calibrator, arc, frame_idx), _BOX_COLOR, closed=False
            )

    return out


def main() -> None:
    args = parse_args()
    if args.output_dir is None and args.output_video is None:
        print("At least one of --output-dir or --output-video is required.", file=sys.stderr)
        sys.exit(1)

    frames = _read_video_slice(args.input, args.start_frame, args.max_frames)
    print(f"Read {len(frames)} frames from {args.input} (starting at frame {args.start_frame})")

    calibrator = PitchKeypointCalibrator(court_width_m=args.court_width_m)
    calibrator.calibrate(frames)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    resolved_count = 0
    video_frames: list[Frame] = []

    for frame_idx, frame in enumerate(frames):
        is_resolved = frame_idx in calibrator._transforms  # noqa: SLF001 -- inspecting resolved frames
        if is_resolved:
            resolved_count += 1
            annotated = draw_pitch_markings(
                frame, calibrator, frame_idx, args.court_width_m, args.court_length_m
            )
        else:
            annotated = frame

        if args.output_video is not None:
            video_frames.append(annotated)

        if args.output_dir is not None and is_resolved and written < args.max_samples:
            out_path = args.output_dir / f"frame_{args.start_frame + frame_idx:06d}.png"
            cv2.imwrite(str(out_path), annotated)
            written += 1

    print(f"{resolved_count}/{len(frames)} frames had a resolved calibration to draw.")

    if args.output_dir is not None:
        print(f"Wrote {written} annotated sample image(s) to {args.output_dir}")

    if args.output_video is not None:
        fps = get_video_info(args.input).fps
        save_video(video_frames, args.output_video, fps=fps)
        print(
            f"Wrote a full annotated video ({len(video_frames)} frames, {fps:.1f}fps) "
            f"to {args.output_video}"
        )

    if resolved_count == 0:
        print("No frames had a resolved calibration -- nothing was drawn anywhere.")


if __name__ == "__main__":
    main()
