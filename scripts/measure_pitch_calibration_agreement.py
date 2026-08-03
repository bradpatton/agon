"""Cross-checks two pitch-calibration methods against each other on the
same real footage, one point at a time.

This is deliberately NOT an absolute ground-truth accuracy measurement --
neither `ViewTransformer` (static, one fixed homography) nor
`PitchKeypointCalibrator` (dynamic, per-frame center-circle detection) has
its own correctness independently verified against a real surveyed pitch,
and both use fixed/derived scale references (the calibration file's
corner points; the center circle's known 9.15m radius) that make a
same-calibrator self-check tautological. Comparing the two INDEPENDENT
methods against each other on the same (frame, track) points is a
different, weaker, but still real and honest signal: substantial
disagreement definitely means at least one of them is wrong (and where);
close agreement is reassuring, though it doesn't prove either is
correct -- they could share a common bias. See agon.geometry's
HybridPitchCalibrator and the project plan's Phase 12 for the fuller
context (a real, ground-truth-based harness -- comparing transformed
distances between known pitch features against their real, standard
dimensions -- is scoped as a follow-on, once enough pitch features beyond
the center circle are detected to make that possible).

Usage:
    python scripts/measure_pitch_calibration_agreement.py \\
        --input input_videos/benchmark_clip.mp4 \\
        --model models/runs/train/soccernet-3/weights/best.onnx \\
        --calibration configs/calibration/benchmark_clip.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agon.config import CalibrationConfig, PipelineConfig  # noqa: E402
from agon.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--detection-imgsz", type=int, default=640)
    return parser.parse_args()


def _positions_by_key(tracks) -> dict[tuple[int, int], tuple[float, float]]:
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            pos = track.get("position_transformed")
            if pos is not None:
                positions[(frame_num, player_id)] = pos
    return positions


def main() -> None:
    args = parse_args()
    calibration = CalibrationConfig.from_json_file(args.calibration)

    results = {}
    for mode in ("static", "dynamic"):
        config = PipelineConfig(calibration_mode=mode, detection_imgsz=args.detection_imgsz)
        print(f"Running pipeline with calibration_mode={mode!r}...")
        result = run_pipeline(args.input, args.model, calibration, config=config)
        results[mode] = _positions_by_key(result.tracks)

    common_keys = set(results["static"]) & set(results["dynamic"])
    print(f"\nstatic resolved {len(results['static'])} points")
    print(f"dynamic resolved {len(results['dynamic'])} points")
    print(f"both resolved the same {len(common_keys)} points -- comparing those\n")

    if not common_keys:
        print("No overlapping points -- nothing to compare.")
        return

    distances = []
    for key in common_keys:
        sx, sy = results["static"][key]
        dx, dy = results["dynamic"][key]
        distances.append(math.hypot(sx - dx, sy - dy))

    distances.sort()
    n = len(distances)
    print(f"Disagreement between static and dynamic (meters), n={n}:")
    print(f"  mean:   {sum(distances) / n:.2f}")
    print(f"  median: {distances[n // 2]:.2f}")
    print(f"  p90:    {distances[int(n * 0.9)]:.2f}")
    print(f"  max:    {distances[-1]:.2f}")


if __name__ == "__main__":
    main()
