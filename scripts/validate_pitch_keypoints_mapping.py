"""Validates ``agon.geometry.pitch_keypoints``' canonical (line_name,
endpoint_index) -> real-world-position mapping against real SN-GSR-2025
annotations, independently of trusting the module's own docstring notes.

Method: for each richly-annotated real frame, take every canonical
keypoint visible in it (after excluding frame-boundary-clipped points --
see ``is_frame_boundary_clipped``), then run leave-one-out: solve a
homography from every keypoint but one, and check whether it correctly
predicts the held-out one's real pixel position. A systematically wrong
entry in the canonical mapping shows up as a keypoint with a
consistently large error across many frames; a merely noisy/unstable fit
(small per-frame sample sizes make homography solving numerically
sensitive) shows up as high variance without a consistent large median.

This is real, run-it-yourself validation, not just a claim -- see the
project plan's Phase 7 item 2 / Phase 12 item 5 notes for the actual
numbers this produced when it was first run (mean error dropped from
16.85m to 2.03m on one frame once boundary-clipped points were excluded;
aggregated leave-one-out across 25 frames showed no consistently-large
per-keypoint median error once boundary exclusion was applied, alongside
two direct point-order spot checks on real examples).

Usage:
    python scripts/validate_pitch_keypoints_mapping.py <gsr_root> [--num-sequences N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.geometry.pitch_keypoints import (  # noqa: E402
    EXCLUDED_LINES,
    GOAL_POST_GROUND_POINT_INDEX,
    LINE_ENDPOINTS_M,
    is_frame_boundary_clipped,
)

PITCH_CATEGORY_ID = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("gsr_root", type=Path, help="Directory containing SNGS-* sequence dirs.")
    parser.add_argument("--num-sequences", type=int, default=25)
    parser.add_argument("--min-keypoints-per-frame", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _correspondences_for_frame(
    lines: dict, width: int, height: int
) -> list[tuple[str, int, tuple[float, float], tuple[float, float]]]:
    correspondences = []
    for name, pts in lines.items():
        if name in EXCLUDED_LINES or name not in LINE_ENDPOINTS_M:
            continue
        real_endpoints = LINE_ENDPOINTS_M[name]
        raw_indices = (GOAL_POST_GROUND_POINT_INDEX,) if len(real_endpoints) == 1 else (0, -1)
        for real_idx, raw_idx in enumerate(raw_indices):
            px, py = pts[raw_idx]["x"] * width, pts[raw_idx]["y"] * height
            if is_frame_boundary_clipped(px, py, width, height):
                continue
            correspondences.append((name, real_idx, (px, py), real_endpoints[real_idx]))
    return correspondences


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    seq_dirs = sorted(p for p in args.gsr_root.glob("SNGS-*") if p.is_dir())
    random.shuffle(seq_dirs)

    per_key_errors: dict[str, list[float]] = defaultdict(list)
    frames_used = 0

    for seq_dir in seq_dirs[: args.num_sequences]:
        data = json.loads((seq_dir / "Labels-GameState.json").read_text())
        images_by_id = {img["image_id"]: img for img in data["images"]}
        pitch_anns = [a for a in data["annotations"] if a.get("category_id") == PITCH_CATEGORY_ID]
        if not pitch_anns:
            continue
        best = max(pitch_anns, key=lambda a: len(a["lines"]))
        img = images_by_id[best["image_id"]]

        correspondences = _correspondences_for_frame(best["lines"], img["width"], img["height"])
        if len(correspondences) < args.min_keypoints_per_frame:
            continue
        frames_used += 1

        for held_i in range(len(correspondences)):
            fit = [c for j, c in enumerate(correspondences) if j != held_i]
            held = correspondences[held_i]
            src_pts = np.array([c[2] for c in fit], dtype=np.float32)
            dst_pts = np.array([c[3] for c in fit], dtype=np.float32)
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is None:
                continue
            pred = cv2.perspectiveTransform(np.array([[held[2]]], dtype=np.float32), homography)[0][
                0
            ]
            error = float(np.hypot(pred[0] - held[3][0], pred[1] - held[3][1]))
            per_key_errors[f"{held[0]} pt{held[1]}"].append(error)

    print(f"Frames used: {frames_used}\n")
    rows = sorted(
        (
            (float(np.median(errs)), float(np.mean(errs)), len(errs), key)
            for key, errs in per_key_errors.items()
        ),
        reverse=True,
    )
    print(f"{'key':<30} {'median_m':>9} {'mean_m':>9} {'n':>4}")
    for median, mean, n, key in rows:
        print(f"{key:<30} {median:>9.2f} {mean:>9.2f} {n:>4}")


if __name__ == "__main__":
    main()
