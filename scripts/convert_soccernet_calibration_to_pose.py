"""Converts SoccerNet pitch-line annotations (SN-GSR-2025 sequences, and/or
the legacy Camera Calibration dataset) into Ultralytics pose-estimation
training data, for training a real pitch-calibration keypoint model
(project plan Phase 7 item 2 / Phase 12 item 5 -- replacing
``agon.geometry.pitch_keypoint_calibrator.PitchKeypointCalibrator``'s
classical-CV, center-circle-only first cut).

Each frame becomes one Ultralytics pose-format label: a single "pitch"
object (class 0) whose bounding box covers every visible canonical
keypoint, plus a fixed-length keypoint vector (see
``agon.geometry.pitch_keypoints.CANONICAL_KEYPOINTS`` for the order --
38 keypoints, one per (line, endpoint) pair, straight lines only, see
that module's docstring for what's deliberately excluded and why).
Keypoints not visible in a frame get visibility=0 (Ultralytics' "not
labeled") and (0, 0) coordinates, matching the standard convention.

**Frame-boundary-clipped points are excluded, not included with degraded
confidence** -- see ``agon.geometry.pitch_keypoints.is_frame_boundary_clipped``
and ``scripts/validate_pitch_keypoints_mapping.py`` for why: a line cut off
by the image edge before reaching its true real-world endpoint has an
annotated "endpoint" with no fixed real-world meaning at all, confirmed
to otherwise inject large, real position errors (16.85m mean before this
exclusion, on one measured frame, dropping to 2.03m after).

Frames with fewer than ``--min-keypoints`` visible canonical keypoints are
skipped entirely (not written with an all-zero label) -- not enough
signal to be a useful training example, and Ultralytics pose training
doesn't need every frame to have full coverage regardless.

**Point-order correction (project plan Phase 18)**: SoccerNet's raw
per-line point order is NOT globally consistent -- empirically checked
across 57 real SN-GSR-2025 train sequences and 400 legacy Calibration
images, roughly 43-44% of frames have a given 2-endpoint line's raw
points in the *reverse* of the order this project's canonical mapping
assumes, and it varies **per sequence**, not randomly per frame (e.g.
SNGS-060: 30/30 frames "expected" order; SNGS-061: 0/61, entirely
reversed). A real ground-truth accuracy check found this exact signature
in the trained model: several keypoint slots' predictions consistently
landed within a few px of their own *sibling* endpoint's true position,
not a real localization failure -- the model had learned a confused
mapping because a large fraction of its training labels had endpoints 0
and 1 silently swapped. Two independent, differently-shaped fixes:

1. **Per-frame reordering** for lines with a known coincident real-world
   corner shared with an adjacent named line (``_NEIGHBOR_CHECKS``,
   12 relationships covering both boxes' `main`/`top`/`bottom` and the
   four pitch corners) -- for each such line, checks whether its raw
   points already match the coincident corner within tolerance, and
   reverses them if the *other* orientation matches instead. Only acts
   when there's clear signal; a line with no checkable neighbor present
   in that frame, or an ambiguous match, is left as-is.
2. **Global name correction** for goal posts: unlike the box lines, the
   `Goal * post left`/`Goal * post right` confusion turned out to be
   *globally* consistent (191/191 checked frames, not sequence-varying)
   -- a genuine semantic mismatch between SoccerNet's own "left"/"right"
   convention and this project's (camera-facing) one, not a per-frame
   data problem. Fixed by swapping which raw line name feeds which
   canonical name, applied uniformly. Deliberately fixed only in this
   conversion script, not in ``agon.geometry.pitch_keypoints``' shared
   definitions -- those stay in this project's own (camera-facing,
   intuitive) convention, which is what the human-annotated ground truth
   and every inference-time consumer already use; only the raw-data
   ingestion needed correcting.

**Known, deliberately unfixed gap**: ``Middle line`` shows the same
per-sequence swap pattern as the box lines (checked: 85 expected vs 68
swapped across 9 sequences) but has no coincident-corner neighbor to
check against, and does not reliably track the same sequence's box-line
orientation (e.g. SNGS-066 showed box lines swapped while `Middle line`
was not) -- so no automatic correction is applied for it. Flagged, not
guessed at.

Usage:
    # SN-GSR-2025 (one call per extracted split):
    python scripts/convert_soccernet_calibration_to_pose.py gsr \\
        <sngs_root> <output_dir> <split_name>

    # Legacy Calibration dataset (one call per extracted split):
    python scripts/convert_soccernet_calibration_to_pose.py legacy \\
        <extracted_split_dir> <output_dir> <split_name>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.pitch_keypoints import (  # noqa: E402
    CANONICAL_KEYPOINTS,
    EXCLUDED_LINES,
    GOAL_POST_GROUND_POINT_INDEX,
    LINE_ENDPOINTS_M,
    is_frame_boundary_clipped,
)

PITCH_CATEGORY_ID = 5
CLASS_NAME = "pitch"
BBOX_PADDING_FRACTION = 0.05
"""Padding added around the tight bounding box of visible keypoints, as a
fraction of that box's own width/height -- an exactly-tight box (zero
margin) is a degenerate/unusual input for a detector to learn from, and
real pitch extent typically continues a bit beyond whatever keypoints
happen to be visible."""

_COINCIDENT_RAW_CHECKS: list[tuple[str, int, str, int]] = [
    ("Big rect. left main", 0, "Big rect. left top", -1),
    ("Big rect. left main", -1, "Big rect. left bottom", 0),
    ("Big rect. right main", 0, "Big rect. right top", -1),
    ("Big rect. right main", -1, "Big rect. right bottom", 0),
    ("Small rect. left main", 0, "Small rect. left top", -1),
    ("Small rect. left main", -1, "Small rect. left bottom", 0),
    ("Small rect. right main", 0, "Small rect. right top", -1),
    ("Small rect. right main", -1, "Small rect. right bottom", 0),
    ("Side line top", 0, "Side line left", -1),
    ("Side line bottom", 0, "Side line left", 0),
    ("Side line top", -1, "Side line right", -1),
    ("Side line bottom", -1, "Side line right", 0),
]
"""(line_a, raw_idx_a, line_b, raw_idx_b) pairs whose raw annotation points
should coincide at the same real-world corner (per ``LINE_ENDPOINTS_M``) --
used to detect and correct SoccerNet's per-sequence raw point-order
inconsistency. See this module's docstring, "Point-order correction"."""

_NEIGHBOR_CHECKS: dict[str, list[tuple[int, str, int]]] = {}
for _line_a, _idx_a, _line_b, _idx_b in _COINCIDENT_RAW_CHECKS:
    _NEIGHBOR_CHECKS.setdefault(_line_a, []).append((_idx_a, _line_b, _idx_b))
    _NEIGHBOR_CHECKS.setdefault(_line_b, []).append((_idx_b, _line_a, _idx_a))

GOAL_POST_NAME_CORRECTION: dict[str, str] = {
    "Goal left post left": "Goal left post right",
    "Goal left post right": "Goal left post left",
    "Goal right post left": "Goal right post right",
    "Goal right post right": "Goal right post left",
}
"""SoccerNet's raw ``Goal * post left``/``Goal * post right`` line names are
globally swapped relative to this project's (camera-facing) convention --
confirmed 191/191 checked frames agree, not sequence-varying like the
other point-order issue, i.e. a genuine semantic mismatch rather than a
per-frame data problem. Maps the raw line name to the canonical name it
actually corresponds to. Deliberately not fixed in
``agon.geometry.pitch_keypoints`` itself -- see module docstring."""


def _reorder_if_swapped(
    name: str, raw_pts: list[dict], lines: dict, width: int, height: int, tol_px: float = 8.0
) -> list[dict]:
    """Returns ``raw_pts`` in canonical (endpoint 0 first) order, reversing
    it if a coincident-corner neighbor line present in this same frame
    clearly matches the reversed orientation instead. Conservative by
    design: only acts when at least one neighbor check clearly supports
    reversing and none support keeping it as-is; otherwise returns
    ``raw_pts`` unchanged (including when no checkable neighbor is
    present in this frame at all)."""
    checks = _NEIGHBOR_CHECKS.get(name)
    if not checks or len(raw_pts) < 2:
        return raw_pts

    def px(pt: dict) -> tuple[float, float]:
        return pt["x"] * width, pt["y"] * height

    def dist(pt_a: dict, pt_b: dict) -> float:
        (ax, ay), (bx, by) = px(pt_a), px(pt_b)
        return math.hypot(ax - bx, ay - by)

    as_is_hits = 0
    reversed_hits = 0
    for own_idx, neighbor_name, neighbor_idx in checks:
        neighbor_pts = lines.get(neighbor_name)
        if neighbor_pts is None or len(neighbor_pts) < 2:
            continue
        neighbor_pt = neighbor_pts[neighbor_idx]
        other_idx = -1 if own_idx == 0 else 0
        if dist(raw_pts[own_idx], neighbor_pt) <= tol_px:
            as_is_hits += 1
        elif dist(raw_pts[other_idx], neighbor_pt) <= tol_px:
            reversed_hits += 1

    if reversed_hits > 0 and as_is_hits == 0:
        return list(reversed(raw_pts))
    return raw_pts


def _visible_keypoints_px(
    lines: dict, width: int, height: int
) -> dict[tuple[str, int], tuple[float, float]]:
    """Maps each visible, non-boundary-clipped canonical keypoint to its
    pixel position for one frame. Applies the point-order and goal-post
    name corrections described in this module's docstring before
    extracting endpoint positions."""
    visible = {}
    for raw_name, raw_pts in lines.items():
        if raw_name in EXCLUDED_LINES:
            continue
        name = GOAL_POST_NAME_CORRECTION.get(raw_name, raw_name)
        if name not in LINE_ENDPOINTS_M:
            continue
        real_endpoints = LINE_ENDPOINTS_M[name]
        pts = raw_pts
        if len(real_endpoints) == 1:
            raw_indices = (GOAL_POST_GROUND_POINT_INDEX,)
        else:
            pts = _reorder_if_swapped(name, raw_pts, lines, width, height)
            raw_indices = (0, -1)
        for real_idx, raw_idx in enumerate(raw_indices):
            px_val, py_val = pts[raw_idx]["x"] * width, pts[raw_idx]["y"] * height
            if is_frame_boundary_clipped(px_val, py_val, width, height):
                continue
            visible[(name, real_idx)] = (px_val, py_val)
    return visible


def _pose_label_line(
    visible: dict[tuple[str, int], tuple[float, float]], width: int, height: int
) -> str | None:
    xs = [p[0] for p in visible.values()]
    ys = [p[1] for p in visible.values()]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = max((x2 - x1) * BBOX_PADDING_FRACTION, 2.0)
    pad_y = max((y2 - y1) * BBOX_PADDING_FRACTION, 2.0)
    x1, x2 = max(0.0, x1 - pad_x), min(float(width), x2 + pad_x)
    y1, y2 = max(0.0, y1 - pad_y), min(float(height), y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None

    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height

    fields = [f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"]
    for key in CANONICAL_KEYPOINTS:
        point = visible.get(key)
        if point is None:
            fields.append("0.000000 0.000000 0")
        else:
            fields.append(f"{point[0] / width:.6f} {point[1] / height:.6f} 2")
    return " ".join(fields)


def _write_example(
    src_image: Path,
    stem: str,
    visible: dict,
    width: int,
    height: int,
    images_out: Path,
    labels_out: Path,
) -> bool:
    label_line = _pose_label_line(visible, width, height)
    if label_line is None:
        return False

    dst_image = images_out / f"{stem}{src_image.suffix}"
    dst_label = labels_out / f"{stem}.txt"
    if not dst_image.exists():
        relative_target = os.path.relpath(src_image.resolve(), dst_image.parent.resolve())
        dst_image.symlink_to(relative_target)
    dst_label.write_text(label_line + "\n")
    return True


def convert_gsr_sequence(
    seq_dir: Path, images_out: Path, labels_out: Path, min_keypoints: int
) -> tuple[int, int]:
    """Returns (num_frames_written, num_frames_skipped_too_few_keypoints)."""
    data = json.loads((seq_dir / "Labels-GameState.json").read_text())
    im_dir = data["info"]["im_dir"]
    images_by_id = {img["image_id"]: img for img in data["images"]}
    seq_name = seq_dir.name

    written = 0
    skipped = 0
    for ann in data["annotations"]:
        if ann.get("category_id") != PITCH_CATEGORY_ID:
            continue
        img = images_by_id.get(ann["image_id"])
        if img is None:
            continue
        src_image = seq_dir / im_dir / img["file_name"]
        if not src_image.exists():
            continue

        visible = _visible_keypoints_px(ann["lines"], img["width"], img["height"])
        if len(visible) < min_keypoints:
            skipped += 1
            continue

        stem = f"{seq_name}_{Path(img['file_name']).stem}"
        if _write_example(
            src_image, stem, visible, img["width"], img["height"], images_out, labels_out
        ):
            written += 1
        else:
            skipped += 1

    return written, skipped


def convert_legacy_split(
    split_dir: Path, images_out: Path, labels_out: Path, min_keypoints: int
) -> tuple[int, int]:
    """Returns (num_frames_written, num_frames_skipped_too_few_keypoints)."""
    import cv2

    written = 0
    skipped = 0
    for json_path in sorted(split_dir.glob("*.json")):
        src_image = json_path.with_suffix(".jpg")
        if not src_image.exists():
            continue
        lines = json.loads(json_path.read_text())
        img = cv2.imread(str(src_image))
        if img is None:
            continue
        height, width = img.shape[:2]

        visible = _visible_keypoints_px(lines, width, height)
        if len(visible) < min_keypoints:
            skipped += 1
            continue

        stem = f"legacy_{json_path.stem}"
        if _write_example(src_image, stem, visible, width, height, images_out, labels_out):
            written += 1
        else:
            skipped += 1

    return written, skipped


def _write_dataset_yaml(output_dir: Path) -> None:
    dataset_yaml = output_dir / "dataset.yaml"
    if dataset_yaml.exists():
        return
    num_kpts = len(CANONICAL_KEYPOINTS)
    names_comment = "\n".join(
        f"#   {i}: {name} pt{idx}" for i, (name, idx) in enumerate(CANONICAL_KEYPOINTS)
    )
    dataset_yaml.write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n  0: {CLASS_NAME}\n"
        f"kpt_shape: [{num_kpts}, 3]\n"
        f"# Keypoint order (index: line_name endpoint_index):\n{names_comment}\n"
    )
    print(f"\nWrote {dataset_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", choices=["gsr", "legacy"])
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("split_name", type=str)
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=4,
        help="Skip frames with fewer than this many visible canonical keypoints (default 4 -- "
        "cv2.findHomography needs at least 4 non-collinear correspondences at inference time, "
        "so a training frame with fewer isn't representative of a usable case).",
    )
    args = parser.parse_args()

    images_out = args.output_dir / "images" / args.split_name
    labels_out = args.output_dir / "labels" / args.split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0

    if args.source == "gsr":
        seq_dirs = sorted(p for p in args.input_dir.glob("SNGS-*") if p.is_dir())
        if not seq_dirs:
            print(f"No SNGS-* sequence directories found under {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        for seq_dir in seq_dirs:
            written, skipped = convert_gsr_sequence(
                seq_dir, images_out, labels_out, args.min_keypoints
            )
            print(f"{seq_dir.name}: {written} frames written, {skipped} skipped")
            total_written += written
            total_skipped += skipped
    else:
        written, skipped = convert_legacy_split(
            args.input_dir, images_out, labels_out, args.min_keypoints
        )
        print(f"{args.input_dir}: {written} frames written, {skipped} skipped")
        total_written += written
        total_skipped += skipped

    print(f"\nTotal ({args.split_name}): {total_written} frames written, {total_skipped} skipped")
    _write_dataset_yaml(args.output_dir)


if __name__ == "__main__":
    main()
