"""Orchestrates the full detect -> track -> analyze -> render pipeline.

Replaces the original tutorial's ``main.py`` script. Produces the
in-memory ``tracks`` structure (used by ``soccer_analysis.export`` to write
the ML-ready JSONL/Parquet data) and, optionally, an annotated video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soccer_analysis.analytics.player_ball_assigner import PlayerBallAssigner
from soccer_analysis.analytics.speed_distance import SpeedDistanceEstimator
from soccer_analysis.camera.camera_movement_estimator import CameraMovementEstimator
from soccer_analysis.config import CalibrationConfig, PipelineConfig, resolve_device
from soccer_analysis.detection.base import Tracks, add_position_to_tracks, interpolate_ball_positions
from soccer_analysis.geometry.bbox import Point
from soccer_analysis.geometry.pitch_keypoint_calibrator import PitchKeypointCalibrator
from soccer_analysis.geometry.view_transformer import ViewTransformer, add_transformed_position_to_tracks
from soccer_analysis.interfaces import Detector, PitchCalibrator, TeamClassifier
from soccer_analysis.io.video import read_video, save_video
from soccer_analysis.team.embedding_team_assigner import EmbeddingTeamClassifier
from soccer_analysis.team.team_assigner import TeamAssigner
from soccer_analysis.viz.annotate import draw_annotations

logger = logging.getLogger(__name__)


def _build_detector(model_path: Path, preferred_device: str | None, confidence: float) -> Detector:
    """Picks a Detector backend by model file extension.

    ``.onnx`` -> OnnxDetector (default, torch-free, runs on CPU via
    onnxruntime). ``.pt``/others -> UltralyticsDetector (needs the
    ``[train]`` extra; resolves a torch cuda/mps/cpu device). See the
    README's modernization notes for why onnxruntime is the default runtime
    path.
    """
    if model_path.suffix == ".onnx":
        from soccer_analysis.detection.onnx_tracker import OnnxDetector

        logger.info("Using OnnxDetector backend (CPUExecutionProvider)")
        return OnnxDetector(str(model_path), confidence=confidence)

    from soccer_analysis.detection.tracker import UltralyticsDetector

    device = resolve_device(preferred_device)
    logger.info("Using UltralyticsDetector backend (device=%s)", device)
    return UltralyticsDetector(str(model_path), device=device, confidence=confidence)


def _build_pitch_calibrator(
    calibration: CalibrationConfig, mode: str, video_frames: list
) -> PitchCalibrator:
    """Picks a PitchCalibrator per ``PipelineConfig.calibration_mode``.

    'dynamic' is a classical-CV first cut (center-circle detection), not a
    trained keypoint model -- see PitchKeypointCalibrator's docstring for
    what it actually solves and its real limitations before trusting its
    output over the static calibration.
    """
    if mode == "dynamic":
        calibrator = PitchKeypointCalibrator(court_width_m=calibration.court_width_m)
        calibrator.calibrate(video_frames)
        return calibrator

    calibrator = ViewTransformer(calibration)
    calibrator.calibrate(video_frames)
    return calibrator


def _build_team_classifier(mode: str, embedding_model_path: str, random_state: int) -> TeamClassifier:
    """Picks a TeamClassifier per ``PipelineConfig.team_classifier``.

    'embedding' needs a model exported by scripts/export_team_embedding_model.py
    -- see EmbeddingTeamClassifier's docstring for what it actually improves
    on and what it still can't do (referee separation).
    """
    if mode == "embedding":
        return EmbeddingTeamClassifier(embedding_model_path, random_state=random_state)

    return TeamAssigner(random_state=random_state)


@dataclass
class PipelineResult:
    tracks: Tracks
    team_ball_control: np.ndarray
    camera_movement_per_frame: list[Point]
    frame_rate: float


def run_pipeline(
    video_path: str | Path,
    model_path: str | Path,
    calibration: CalibrationConfig,
    config: PipelineConfig | None = None,
    stub_dir: str | Path | None = None,
    read_from_stub: bool = False,
    output_video_path: str | Path | None = None,
) -> PipelineResult:
    """Run detection/tracking/analytics, and optionally render an annotated video.

    Args:
        video_path: input match footage.
        model_path: YOLO checkpoint for player/referee/ball detection.
        calibration: per-video pixel-to-pitch calibration (see configs/calibration/).
        config: pipeline thresholds; defaults to ``PipelineConfig()``.
        stub_dir: optional directory to cache intermediate tracking/camera-movement
            results, keyed by video filename, so re-running is fast during
            development. See the README for why this is a *cache*, not the
            canonical data export (that's ``soccer_analysis.export``).
        read_from_stub: reuse a previously written stub cache if present.
        output_video_path: if given, renders an annotated video there.
    """
    config = config or PipelineConfig()

    video_path = Path(video_path)
    stub_dir = Path(stub_dir) if stub_dir else None
    tracks_stub = stub_dir / f"{video_path.stem}_tracks.pkl" if stub_dir else None
    camera_stub = stub_dir / f"{video_path.stem}_camera_movement.json" if stub_dir else None

    video_frames = read_video(video_path)

    detector = _build_detector(Path(model_path), config.device, config.detection_confidence)
    tracks = detector.get_object_tracks(
        video_frames, read_from_stub=read_from_stub, stub_path=tracks_stub
    )
    add_position_to_tracks(tracks)

    pitch_pixel_vertices = np.array(calibration.pixel_vertices, dtype=np.float32)
    camera_movement_estimator = CameraMovementEstimator(
        video_frames[0], pitch_pixel_vertices=pitch_pixel_vertices
    )
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=read_from_stub, stub_path=camera_stub
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    pitch_calibrator = _build_pitch_calibrator(calibration, config.calibration_mode, video_frames)
    add_transformed_position_to_tracks(tracks, pitch_calibrator)

    tracks["ball"] = interpolate_ball_positions(tracks["ball"])

    speed_distance_estimator = SpeedDistanceEstimator(
        frame_window=config.speed_frame_window, frame_rate=config.frame_rate
    )
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = _build_team_classifier(
        config.team_classifier, config.team_embedding_model_path, config.team_kmeans_random_state
    )
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            track["team"] = team
            track["team_color"] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner(
        max_player_ball_distance_px=config.ball_max_assignment_distance_px
    )
    team_ball_control: list[int] = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player is not None:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            # No one has been assigned the ball yet (or this frame): 0 means
            # "no team," not a guess — avoids the IndexError the original
            # tutorial code hit when this happened on frame 0.
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control_array = np.array(team_ball_control)

    if output_video_path is not None:
        output_frames = draw_annotations(video_frames, tracks, team_ball_control_array)
        output_frames = camera_movement_estimator.draw_camera_movement(
            output_frames, camera_movement_per_frame
        )
        speed_distance_estimator.draw_speed_and_distance(output_frames, tracks)
        save_video(output_frames, output_video_path, fps=config.frame_rate)

    return PipelineResult(
        tracks=tracks,
        team_ball_control=team_ball_control_array,
        camera_movement_per_frame=camera_movement_per_frame,
        frame_rate=config.frame_rate,
    )
