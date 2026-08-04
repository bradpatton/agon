"""Interactive click-to-annotate ground truth tool -- this project has
never had real, independently-annotated pixel-level ground truth for its
own target footage, only FIFA pitch-dimension ground truth (used by
``leave_one_out_position_errors``) and SoccerNet's own training/validation
labels (which don't reflect this project's actual failure mode: see
``agon/geometry/trained_pitch_calibrator.py``'s docstring and the plan's
Phase 13/15 notes -- a real six-yard-box/penalty-box confusion was
confirmed present at inference time on real broadcast footage, and
confirmed *absent* from the raw SN-GSR-2025 training labels themselves,
meaning this is a genuine model generalization gap to *this* footage's
domain, not a data or labeling bug SoccerNet's own validation split can
reveal).

This tool lets a human click each of the 38 canonical pitch keypoints
(``agon.geometry.pitch_keypoints.CANONICAL_KEYPOINTS``) directly on a real
video frame, producing ground truth that's independent of any model,
dataset, or homography fit -- the most direct possible check of "is the
model's predicted pixel position for this keypoint actually correct."

Requires a real display (run locally, not over SSH to a headless machine).

Usage:
    python scripts/annotate_ground_truth.py \\
        --input input_videos/match_10min_sample.mp4 \\
        --frames 25500,26000,26500 \\
        --output data/ground_truth/match_10min_sample.json \\
        --scale 1.5

Controls:
    left click   record the current keypoint at the clicked pixel
    s            skip the current keypoint (not visible in this frame)
    u            undo the last recorded/skipped keypoint
    n            finish this frame early and move to the next
    q            quit and save everything annotated so far
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.pitch_keypoints import CANONICAL_KEYPOINTS  # noqa: E402

WINDOW_NAME = "annotate_ground_truth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=str,
        required=True,
        help="Comma-separated frame indices to annotate, e.g. 25500,26000,26500",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Display upscale factor for more precise clicking (default 1.5)",
    )
    return parser.parse_args()


def _read_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
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


def _load_existing(output: Path) -> dict:
    if output.exists():
        with open(output) as f:
            return json.load(f)
    return {"video": None, "frames": {}}


class _ClickState:
    def __init__(self) -> None:
        self.last_click: tuple[int, int] | None = None

    def callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)


def _annotate_frame(frame: np.ndarray, scale: float) -> dict[str, list[float] | None]:
    display_base = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    points: dict[str, list[float] | None] = {}
    recorded_keys: list[str] = []

    click_state = _ClickState()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, click_state.callback)

    idx = 0
    while idx < len(CANONICAL_KEYPOINTS):
        name, endpoint_idx = CANONICAL_KEYPOINTS[idx]
        key = f"{name}|{endpoint_idx}"
        click_state.last_click = None

        display = display_base.copy()
        for k, v in points.items():
            if v is not None:
                cv2.circle(display, (int(v[0] * scale), int(v[1] * scale)), 5, (0, 255, 0), -1)
        cv2.putText(
            display,
            f"[{idx + 1}/{len(CANONICAL_KEYPOINTS)}] {key}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display,
            "click=record  s=skip  u=undo  n=finish frame  q=quit+save",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, display)

        while True:
            k = cv2.waitKey(30) & 0xFF
            if click_state.last_click is not None:
                x, y = click_state.last_click
                points[key] = [x / scale, y / scale]
                recorded_keys.append(key)
                idx += 1
                break
            if k == ord("s"):
                points[key] = None
                recorded_keys.append(key)
                idx += 1
                break
            if k == ord("u"):
                if recorded_keys:
                    last_key = recorded_keys.pop()
                    del points[last_key]
                    idx -= 1
                break
            if k == ord("n"):
                idx = len(CANONICAL_KEYPOINTS)
                break
            if k == ord("q"):
                cv2.destroyWindow(WINDOW_NAME)
                return points

    return points


def main() -> None:
    args = parse_args()
    frame_indices = [int(f.strip()) for f in args.frames.split(",") if f.strip()]

    data = _load_existing(args.output)
    data["video"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for frame_idx in frame_indices:
        print(f"\nAnnotating frame {frame_idx} ({len(CANONICAL_KEYPOINTS)} keypoints)...")
        frame = _read_frame(args.input, frame_idx)
        points = _annotate_frame(frame, args.scale)
        recorded = sum(1 for v in points.values() if v is not None)
        data["frames"][str(frame_idx)] = points
        print(f"  Recorded {recorded}/{len(points)} visible keypoints.")

        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved to {args.output}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
