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

**Batch order (``--batch-order``, default ``point``)**: by default, the
outer loop is the keypoint *type*, not the frame -- you place "Middle
line|0" on every frame in the batch before moving on to the next keypoint
type, instead of placing all 38 keypoints on one frame before moving to
the next. Real user feedback: re-finding which feature you're looking for
on every single frame is far more costly than staying locked onto one
feature across many frames (the same visual search, reused, instead of a
fresh one every frame). All 38 keypoints x all frames are precomputed and
held in memory up front (suggestions/reference overlays included), so
switching between frames mid-keypoint is instant. Pass ``--batch-order
frame`` to get the old frame-major order back if you ever want it -- the
output JSON format is identical either way.

Optional model-assisted mode (``--model``): pre-fills each keypoint with
``TrainedPitchCalibrator``'s own prediction (shown as a yellow "suggested"
marker distinct from a confirmed green one) whenever it's confident enough
-- press ``a`` to accept the suggestion outright instead of clicking, which
is faster when the model happens to be right, and costs nothing when it's
wrong (you just click the real position instead, exactly as without a
model). This does *not* make the resulting ground truth dependent on the
model in any way that would bias `measure_ground_truth_accuracy.py`'s
check of that same model -- every accepted point is either a real human
click or an explicit human confirmation that the suggested pixel matches
what's actually visible in the frame, not a value trusted blindly. Needs
``onnxruntime`` (only imported if ``--model`` is actually passed, so the
tool still runs with just ``opencv-python``/``numpy``/``pydantic`` when
used without model assistance).

Optional second reference overlay (``--roboflow-model``): draws every
confident keypoint from Roboflow's independently-trained pitch-keypoint
model (``roboflow/sports``, a YOLOv8-pose checkpoint trained on Roboflow's
own dataset -- a genuinely different distribution than SoccerNet's, see
the plan's Phase 16 notes) as small cyan reference dots, labeled with its
own vertex numbers (``R01``..``R32``, see ``roboflow/sports``'
``SoccerPitchConfiguration``). Real value confirmed before wiring this in:
on several real frames of this project's target footage, this model
correctly separates the six-yard box from the penalty box where our own
trained model confuses them.

**Deliberately NOT offered as an acceptable suggestion (no ``a``-key
shortcut for it, unlike ``--model``)** -- its 32 keypoints don't have a
verified correspondence to specific slots in our 38-keypoint canonical
scheme (which one is "left" vs "right", which corner is which, would need
independent verification this project hasn't done), so auto-accepting one
under a specific canonical keypoint name risks silently mislabeling
ground truth under a name that doesn't actually match what was clicked.
It's shown purely as a visual reference to help you find real pitch
features faster -- you still decide what it corresponds to (if anything)
and click accordingly. Needs ``ultralytics``/``torch`` (the ``[train]``
extra), only imported if ``--roboflow-model`` is passed.

Controls:
    left click   record the current keypoint at the clicked pixel
    a            accept the shown suggestion as-is (only when one exists)
    s            skip the current keypoint (not visible in this frame)
    u / ctrl-z   undo the last recorded/skipped keypoint (crosses frame and
                 keypoint-type boundaries -- always steps back one action)
    n            in --batch-order point (default): finish this keypoint
                 type across the remaining frames, move to the next type.
                 in --batch-order frame: finish this frame early, move to
                 the next frame.
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

from agon.geometry.pitch_keypoints import (  # noqa: E402
    CANONICAL_KEYPOINTS,
    CENTER_CIRCLE_RADIUS_M,
    GOAL_WIDTH_M,
    LINE_ENDPOINTS_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

WINDOW_NAME = "annotate_ground_truth"

# roboflow/sports' SoccerPitchConfiguration.labels -- the fixed order its
# YOLOv8-pose checkpoint's 32 keypoints come out in. Display-only (see
# module docstring for why these aren't mapped to specific canonical
# keypoint slots).
ROBOFLOW_KEYPOINT_LABELS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
    "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
    "14", "19",
]


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
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional pitch-calibration ONNX model path -- pre-fills suggestions from "
        "TrainedPitchCalibrator's own predictions (see module docstring). Needs onnxruntime.",
    )
    parser.add_argument(
        "--suggestion-confidence",
        type=float,
        default=0.3,
        help="Minimum model confidence to show a keypoint suggestion (default 0.3, matches "
        "TrainedPitchCalibrator's own default)",
    )
    parser.add_argument(
        "--roboflow-model",
        type=Path,
        default=None,
        help="Optional path to Roboflow's pretrained football-pitch-detection.pt checkpoint -- "
        "draws its confident keypoints as a reference-only overlay (see module docstring). "
        "Needs ultralytics/torch ([train] extra).",
    )
    parser.add_argument(
        "--roboflow-confidence",
        type=float,
        default=0.4,
        help="Minimum confidence to show a Roboflow reference keypoint (default 0.4)",
    )
    parser.add_argument(
        "--batch-order",
        choices=["point", "frame"],
        default="point",
        help="'point' (default): batch by keypoint type across all frames, then move to the "
        "next type -- lets you stay locked onto one real-world feature instead of re-finding "
        "a new one every frame. 'frame': the old order, all 38 keypoints per frame before "
        "moving to the next frame. See module docstring.",
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


def _annotate_frame(
    frame: np.ndarray,
    scale: float,
    suggestions: np.ndarray | None = None,
    suggestion_confidence: float = 0.3,
    reference_points: list[tuple[str, float, float]] | None = None,
) -> dict[str, list[float] | None]:
    """``suggestions``, if given, is the ``(38, 3)`` array
    ``TrainedPitchCalibrator._detect_keypoints`` returns (x, y, confidence
    per canonical keypoint, in this same frame's pixel space) -- see module
    docstring for why accepting a suggestion still counts as a real human
    confirmation, not a model-trusting shortcut.

    ``reference_points``, if given, is a list of (label, x, y) from
    Roboflow's model -- drawn every iteration regardless of which
    canonical keypoint is active, since it's a general visual aid, not a
    per-keypoint suggestion (see module docstring)."""
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

        suggestion: tuple[float, float] | None = None
        if suggestions is not None:
            sx, sy, sconf = suggestions[idx]
            if sconf >= suggestion_confidence:
                suggestion = (float(sx), float(sy))

        display = display_base.copy()
        if reference_points:
            for label, rx, ry in reference_points:
                rx_disp, ry_disp = int(rx * scale), int(ry * scale)
                cv2.circle(display, (rx_disp, ry_disp), 3, (255, 255, 0), -1)
                cv2.putText(
                    display,
                    f"R{label}",
                    (rx_disp + 6, ry_disp - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )
        for k, v in points.items():
            if v is not None:
                cv2.circle(display, (int(v[0] * scale), int(v[1] * scale)), 5, (0, 255, 0), -1)
        if suggestion is not None:
            sx_disp, sy_disp = int(suggestion[0] * scale), int(suggestion[1] * scale)
            cv2.drawMarker(
                display, (sx_disp, sy_disp), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2
            )
        cv2.putText(
            display,
            f"[{idx + 1}/{len(CANONICAL_KEYPOINTS)}] {key}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        controls = "click=record  s=skip  u/ctrl-z=undo  n=finish frame  q=quit+save"
        if suggestion is not None:
            controls = "a=accept suggestion (yellow x)  " + controls
        cv2.putText(
            display, controls, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        if reference_points:
            cv2.putText(
                display,
                "cyan dots = Roboflow reference (visual aid only, not clickable/acceptable)",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
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
            if k == ord("a") and suggestion is not None:
                points[key] = [suggestion[0], suggestion[1]]
                recorded_keys.append(key)
                idx += 1
                break
            if k == ord("s"):
                points[key] = None
                recorded_keys.append(key)
                idx += 1
                break
            if k == ord("u") or k == 26:  # 26 == Ctrl+Z
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


_CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "Big rect.": (230, 130, 40),  # blue (BGR)
    "Small rect.": (0, 150, 255),  # orange
    "Goal": (60, 60, 230),  # red
}
_STRUCTURE_COLOR = (170, 170, 170)  # gray -- touchlines and halfway line


def _category_color(name: str) -> tuple[int, int, int]:
    for prefix, color in _CATEGORY_COLORS.items():
        if name.startswith(prefix):
            return color
    return _STRUCTURE_COLOR


def _centered_text(
    img: np.ndarray,
    text: str,
    cy: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(
        img,
        text,
        (w // 2 - tw // 2, cy + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _pitch_diagram(
    keypoint_idx: int, width: int, height: int, highlight_color: tuple[int, int, int]
) -> np.ndarray:
    """A small top-down schematic of the real pitch, drawn from
    ``agon.geometry.pitch_keypoints``' own coordinates (not eyeballed --
    the same convention the tool and the trained model both use: origin
    at pitch center, x = length, y = width, positive y = top touchline),
    with the current keypoint's exact real-world position highlighted --
    shows *where on the pitch* to look, not just a text name."""
    pad = 20
    scale = min((width - 2 * pad) / PITCH_LENGTH_M, (height - 2 * pad) / PITCH_WIDTH_M)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 70, 45)  # dark pitch green

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(width / 2 + x * scale), int(height / 2 - y * scale)

    line_color = (170, 190, 175)
    for endpoints in LINE_ENDPOINTS_M.values():
        if len(endpoints) == 2:
            cv2.line(img, to_px(*endpoints[0]), to_px(*endpoints[1]), line_color, 1, cv2.LINE_AA)

    for gx in (-PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2):
        depth = -2.2 if gx < 0 else 2.2
        p1, p2 = to_px(gx, GOAL_WIDTH_M / 2), to_px(gx, -GOAL_WIDTH_M / 2)
        p1d, p2d = to_px(gx + depth, GOAL_WIDTH_M / 2), to_px(gx + depth, -GOAL_WIDTH_M / 2)
        cv2.line(img, p1, p1d, line_color, 1, cv2.LINE_AA)
        cv2.line(img, p1d, p2d, line_color, 1, cv2.LINE_AA)
        cv2.line(img, p2d, p2, line_color, 1, cv2.LINE_AA)

    cv2.circle(img, to_px(0, 0), int(CENTER_CIRCLE_RADIUS_M * scale), line_color, 1, cv2.LINE_AA)

    name, endpoint_idx = CANONICAL_KEYPOINTS[keypoint_idx]
    hp = to_px(*LINE_ENDPOINTS_M[name][endpoint_idx])
    cv2.circle(img, hp, 20, highlight_color, 2, cv2.LINE_AA)
    cv2.circle(img, hp, 11, highlight_color, -1, cv2.LINE_AA)
    cv2.circle(img, hp, 11, (255, 255, 255), 2, cv2.LINE_AA)

    return img


_INTERSTITIAL_LEGEND = [
    "click     record the point here",
    "a         accept the model's suggestion (yellow x), if shown",
    "s         skip -- not visible in this frame",
    "u/ctrl-z  undo the last action",
    "n         finish this point across remaining frames, move to next",
    "q         save everything and quit",
]


def _show_new_point_interstitial(
    shape: tuple[int, ...],
    key: str,
    keypoint_idx: int,
    n_points: int,
    n_frames: int,
    color: tuple[int, int, int],
) -> None:
    """A full, unmissable screen shown once at the start of every new
    keypoint type -- a text label alone is too easy to miss while
    clicking quickly through a batch, this forces a deliberate pause
    (blocks on any real keypress; clicks during it are ignored since the
    caller never checks ``click_state`` here). Includes a to-scale pitch
    diagram (see ``_pitch_diagram``) and the full control legend, so both
    "where do I click" and "what are my options" are answered right here
    at the moment of highest risk for applying the wrong label."""
    h, w = shape[0], shape[1]
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    banner[:] = tuple(int(c * 0.20) for c in color)
    cv2.rectangle(banner, (6, 6), (w - 7, h - 7), color, 10)

    header_top = int(h * 0.06)
    _centered_text(banner, "NEW POINT TYPE", header_top, 1.1, color, 3)
    _centered_text(banner, key, header_top + 62, 1.5, (255, 255, 255), 3)
    _centered_text(
        banner,
        f"point {keypoint_idx + 1}/{n_points}  --  {n_frames} frames in this batch",
        header_top + 118,
        0.8,
        (215, 215, 215),
        2,
    )

    diagram_w, diagram_h = min(560, w - 160), min(360, int(h * 0.38))
    diagram = _pitch_diagram(keypoint_idx, diagram_w, diagram_h, color)
    dx, dy = w // 2 - diagram_w // 2, header_top + 155
    banner[dy : dy + diagram_h, dx : dx + diagram_w] = diagram
    cv2.rectangle(banner, (dx, dy), (dx + diagram_w, dy + diagram_h), color, 2)

    legend_y = dy + diagram_h + 55
    for i, line in enumerate(_INTERSTITIAL_LEGEND):
        _centered_text(banner, line, legend_y + i * 32, 0.6, (225, 225, 225), 1)

    _centered_text(banner, "press any key to begin", h - 36, 0.75, color, 2)
    cv2.imshow(WINDOW_NAME, banner)
    cv2.waitKey(0)


def _text_with_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )


def _annotate_by_point(
    frame_indices: list[int],
    frames: dict[int, np.ndarray],
    scale: float,
    all_suggestions: dict[int, np.ndarray] | None,
    suggestion_confidence: float,
    all_reference_points: dict[int, list[tuple[str, float, float]]] | None,
    data: dict,
    output_path: Path,
) -> None:
    """Keypoint-type outer, frame inner -- see module docstring for why.
    Uses one flat step counter over (keypoint_idx, frame_pos) pairs so undo
    is a single, uniform operation regardless of whether it crosses a
    frame or keypoint-type boundary: decrement the counter, remove
    whatever was recorded at the new (earlier) position. Mutates ``data``
    (and saves it to ``output_path``) in place; returns nothing."""
    n_frames = len(frame_indices)
    n_points = len(CANONICAL_KEYPOINTS)
    total_steps = n_points * n_frames

    click_state = _ClickState()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, click_state.callback)

    def points_for(frame_idx: int) -> dict[str, list[float] | None]:
        return data["frames"].setdefault(str(frame_idx), {})

    def save() -> None:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    last_announced_keypoint = -1
    step = 0
    while step < total_steps:
        keypoint_idx, frame_pos = divmod(step, n_frames)
        name, endpoint_idx = CANONICAL_KEYPOINTS[keypoint_idx]
        key = f"{name}|{endpoint_idx}"
        frame_idx = frame_indices[frame_pos]
        frame = frames[frame_idx]
        points = points_for(frame_idx)
        click_state.last_click = None
        color = _category_color(name)

        display = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if keypoint_idx != last_announced_keypoint:
            print(f"\n[point {keypoint_idx + 1}/{n_points}] {key} -- across {n_frames} frames")
            _show_new_point_interstitial(
                display.shape, key, keypoint_idx, n_points, n_frames, color
            )
            click_state.last_click = None  # ignore any stray click received during the interstitial
            last_announced_keypoint = keypoint_idx

        suggestion: tuple[float, float] | None = None
        if all_suggestions is not None:
            sx, sy, sconf = all_suggestions[frame_idx][keypoint_idx]
            if sconf >= suggestion_confidence:
                suggestion = (float(sx), float(sy))
        reference_points = (all_reference_points or {}).get(frame_idx)

        cv2.rectangle(
            display, (4, 4), (display.shape[1] - 5, display.shape[0] - 5), color, 6
        )
        if reference_points:
            for label, rx, ry in reference_points:
                rx_disp, ry_disp = int(rx * scale), int(ry * scale)
                cv2.circle(display, (rx_disp, ry_disp), 3, (255, 255, 0), -1)
                cv2.putText(
                    display,
                    f"R{label}",
                    (rx_disp + 6, ry_disp - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )
        for k, v in points.items():
            if v is not None:
                cv2.circle(display, (int(v[0] * scale), int(v[1] * scale)), 5, (0, 255, 0), -1)
        if suggestion is not None:
            sx_disp, sy_disp = int(suggestion[0] * scale), int(suggestion[1] * scale)
            cv2.drawMarker(
                display, (sx_disp, sy_disp), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2
            )

        _text_with_bg(
            display, f"[point {keypoint_idx + 1}/{n_points}] {key}", (20, 40), 1.0, color, 2
        )
        _text_with_bg(
            display, f"frame {frame_pos + 1}/{n_frames}  (index {frame_idx})", (20, 75), 0.7,
            color, 2,
        )
        controls = "click=record  s=skip  u/ctrl-z=undo  n=finish point  q=quit+save"
        if suggestion is not None:
            controls = "a=accept suggestion (yellow x)  " + controls
        _text_with_bg(display, controls, (20, 108), 0.55, (0, 0, 255), 2)
        if reference_points:
            _text_with_bg(
                display,
                "cyan dots = Roboflow reference (visual aid only, not clickable/acceptable)",
                (20, 136),
                0.5,
                (255, 255, 0),
                2,
            )
        cv2.imshow(WINDOW_NAME, display)

        advanced = False
        while not advanced:
            k = cv2.waitKey(30) & 0xFF
            if click_state.last_click is not None:
                x, y = click_state.last_click
                points[key] = [x / scale, y / scale]
                step += 1
                advanced = True
            elif k == ord("a") and suggestion is not None:
                points[key] = [suggestion[0], suggestion[1]]
                step += 1
                advanced = True
            elif k == ord("s"):
                points[key] = None
                step += 1
                advanced = True
            elif k == ord("u") or k == 26:  # 26 == Ctrl+Z
                if step > 0:
                    step -= 1
                    prev_kp_idx, prev_frame_pos = divmod(step, n_frames)
                    prev_name, prev_ep = CANONICAL_KEYPOINTS[prev_kp_idx]
                    prev_key = f"{prev_name}|{prev_ep}"
                    prev_frame_idx = frame_indices[prev_frame_pos]
                    points_for(prev_frame_idx).pop(prev_key, None)
                    last_announced_keypoint = -1  # re-announce on the way back in
                advanced = True
            elif k == ord("n"):
                step = (keypoint_idx + 1) * n_frames
                advanced = True
            elif k == ord("q"):
                save()
                cv2.destroyWindow(WINDOW_NAME)
                return

        save()

    cv2.destroyWindow(WINDOW_NAME)


def _detect_roboflow_reference(
    model: object, frame: np.ndarray, confidence: float
) -> list[tuple[str, float, float]]:
    results = model.predict(frame, verbose=False)[0]  # type: ignore[attr-defined]
    keypoints = results.keypoints
    if keypoints is None or keypoints.conf is None or len(keypoints.conf) == 0:
        return []
    xy = keypoints.xy[0].cpu().numpy()
    conf = keypoints.conf[0].cpu().numpy()
    return [
        (ROBOFLOW_KEYPOINT_LABELS[i], float(xy[i][0]), float(xy[i][1]))
        for i in range(len(conf))
        if conf[i] >= confidence
    ]


def main() -> None:
    args = parse_args()
    frame_indices = [int(f.strip()) for f in args.frames.split(",") if f.strip()]

    calibrator = None
    if args.model is not None:
        # Imported lazily so running without --model doesn't need onnxruntime.
        from agon.geometry.trained_pitch_calibrator import TrainedPitchCalibrator

        calibrator = TrainedPitchCalibrator(str(args.model))
        print(f"Model-assisted mode: suggestions from {args.model}")

    roboflow_model = None
    if args.roboflow_model is not None:
        # Imported lazily so running without --roboflow-model doesn't need ultralytics/torch.
        from ultralytics import YOLO

        roboflow_model = YOLO(str(args.roboflow_model))
        print(f"Reference overlay: Roboflow keypoints from {args.roboflow_model}")

    data = _load_existing(args.output)
    data["video"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_order == "point":
        print(f"Reading {len(frame_indices)} frames...")
        frames = {i: _read_frame(args.input, i) for i in frame_indices}

        all_suggestions = None
        if calibrator is not None:
            all_suggestions = {
                i: calibrator._detect_keypoints(f)  # noqa: SLF001 -- as in measurement scripts
                for i, f in frames.items()
            }
        all_reference_points = None
        if roboflow_model is not None:
            all_reference_points = {
                i: _detect_roboflow_reference(roboflow_model, f, args.roboflow_confidence)
                for i, f in frames.items()
            }

        print(
            f"Batch-by-point mode: {len(CANONICAL_KEYPOINTS)} keypoint types x "
            f"{len(frame_indices)} frames. Complete one keypoint type across every frame "
            f"before moving to the next (--batch-order frame for the old order)."
        )
        _annotate_by_point(
            frame_indices,
            frames,
            args.scale,
            all_suggestions,
            args.suggestion_confidence,
            all_reference_points,
            data,
            args.output,
        )
        recorded_total = sum(
            1 for pts in data["frames"].values() for v in pts.values() if v is not None
        )
        print(f"\nDone. {recorded_total} points recorded across {len(frame_indices)} frames.")
        print(f"Saved to {args.output}")
    else:
        for frame_idx in frame_indices:
            print(f"\nAnnotating frame {frame_idx} ({len(CANONICAL_KEYPOINTS)} keypoints)...")
            frame = _read_frame(args.input, frame_idx)
            suggestions = (
                calibrator._detect_keypoints(frame)  # noqa: SLF001 -- as in measurement scripts
                if calibrator is not None
                else None
            )
            reference_points = None
            if roboflow_model is not None:
                reference_points = _detect_roboflow_reference(
                    roboflow_model, frame, args.roboflow_confidence
                )
            points = _annotate_frame(
                frame, args.scale, suggestions, args.suggestion_confidence, reference_points
            )
            recorded = sum(1 for v in points.values() if v is not None)
            data["frames"][str(frame_idx)] = points
            print(f"  Recorded {recorded}/{len(points)} visible keypoints.")

            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved to {args.output}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
