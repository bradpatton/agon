"""Orchestrates the full detect -> track -> analyze -> export/render pipeline.

Replaces the original tutorial's ``main.py`` script. The primary output is
the ML-ready JSONL/Parquet data (``agon.export``); an annotated
video is optional, not the point (see the README).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agon.analytics.player_ball_assigner import PlayerBallAssigner
from agon.analytics.speed_distance import SpeedDistanceEstimator
from agon.broadcast.clock_reader import ClockReader
from agon.broadcast.frame_filter import FrameClassification, classify_frame
from agon.camera.camera_movement_estimator import CameraMovementEstimator
from agon.config import (
    CalibrationConfig,
    ClockCalibrationConfig,
    PipelineConfig,
    resolve_device,
)
from agon.detection.base import (
    Tracks,
    add_position_to_tracks,
    interpolate_ball_positions,
)
from agon.export.schema import (
    FrameRecord,
    MatchStats,
    MatchSummary,
    accumulate_match_stats,
    build_frame_records,
    build_match_summary,
    finalize_match_summary,
)
from agon.export.writer import (
    JsonlWriter,
    ParquetChunkWriter,
    write_jsonl,
    write_match_summary,
    write_parquet,
    write_schema_json,
)
from agon.geometry.bbox import Point
from agon.geometry.pitch_keypoint_calibrator import PitchKeypointCalibrator
from agon.geometry.view_transformer import (
    ViewTransformer,
    add_transformed_position_to_tracks,
)
from agon.interfaces import Detector, JerseyClassifier, PitchCalibrator, TeamClassifier
from agon.io.video import (
    Frame,
    IncrementalVideoWriter,
    get_video_info,
    iter_video_chunks,
    read_video,
    save_video,
)
from agon.team.embedding_team_assigner import EmbeddingTeamClassifier
from agon.team.team_assigner import TeamAssigner
from agon.viz.annotate import draw_annotations, draw_annotations_on_frame

logger = logging.getLogger(__name__)


def _build_tracker(tracker_backend: str, frame_rate: float):
    """Picks a FrameTracker per ``PipelineConfig.tracker_backend``.

    'botsort' needs the [train] extra + boxmot -- see BoTSORTTracker's
    docstring for why (torch is a hard, unconditional import there, even in
    motion-only mode) and what it actually improves on over ByteTrack.
    """
    if tracker_backend == "botsort":
        from agon.detection.botsort_tracker import BoTSORTTracker

        logger.info("Using BoTSORTTracker (with_reid=False, use_cmc=True)")
        return BoTSORTTracker(frame_rate=max(1, round(frame_rate)))

    return None  # each Detector defaults to ByteTrackAdapter itself


def _build_detector(
    model_path: Path,
    preferred_device: str | None,
    confidence: float,
    imgsz: int = 640,
    tracker=None,
) -> Detector:
    """Picks a Detector backend by model file extension.

    ``.onnx`` -> OnnxDetector (default, torch-free, runs on CPU via
    onnxruntime). ``.pt``/others -> UltralyticsDetector (needs the
    ``[train]`` extra; resolves a torch cuda/mps/cpu device). See the
    README's modernization notes for why onnxruntime is the default runtime
    path. ``tracker`` overrides the default ByteTrack tracking (see
    ``_build_tracker``). ``imgsz`` must match the checkpoint's actual
    trained/exported resolution (``PipelineConfig.detection_imgsz``) --
    silently defaulting to 640 regardless of the checkpoint was a real bug
    this parameter exists to close off.
    """
    if model_path.suffix == ".onnx":
        from agon.detection.onnx_tracker import OnnxDetector

        logger.info("Using OnnxDetector backend (CPUExecutionProvider), imgsz=%d", imgsz)
        return OnnxDetector(
            str(model_path),
            confidence=confidence,
            input_size=(imgsz, imgsz),
            tracker=tracker,
        )

    from agon.detection.tracker import UltralyticsDetector

    device = resolve_device(preferred_device)
    logger.info("Using UltralyticsDetector backend (device=%s), imgsz=%d", device, imgsz)
    return UltralyticsDetector(
        str(model_path), device=device, confidence=confidence, imgsz=imgsz, tracker=tracker
    )


def _first_frame_with_enough_players(player_tracks: list[dict], min_players: int = 2) -> int:
    """Finds the first frame with enough confirmed player tracks to seed
    team-color KMeans clustering (needs >=2 samples for 2 clusters).

    Frame 0 isn't reliably usable for this: trackers like ByteTrack often
    require a short hit-streak before confirming a track, so frame 0 can
    have zero confirmed player tracks even when raw detections exist --
    confirmed by hitting this exact crash validating UltralyticsDetector
    end-to-end in Docker, where frame 0 detected 17 people but confirmed
    zero tracks yet.
    """
    for frame_num, players in enumerate(player_tracks):
        if len(players) >= min_players:
            return frame_num
    raise ValueError(
        f"No frame has >= {min_players} tracked players -- can't seed team-color clustering."
    )


def _make_pitch_calibrator(calibration: CalibrationConfig, mode: str) -> PitchCalibrator:
    """Picks a PitchCalibrator per ``PipelineConfig.calibration_mode`` --
    constructs it, but doesn't calibrate() it yet (the caller decides
    whether that's one whole-clip call or several chunked calls).

    'dynamic' is a classical-CV first cut (center-circle detection), not a
    trained keypoint model -- see PitchKeypointCalibrator's docstring for
    what it actually solves and its real limitations before trusting its
    output over the static calibration.
    """
    if mode == "dynamic":
        return PitchKeypointCalibrator(court_width_m=calibration.court_width_m)
    return ViewTransformer(calibration)


def _build_pitch_calibrator(
    calibration: CalibrationConfig, mode: str, video_frames: list
) -> PitchCalibrator:
    """Whole-clip convenience wrapper around _make_pitch_calibrator, for the
    non-streaming pipeline."""
    calibrator = _make_pitch_calibrator(calibration, mode)
    calibrator.calibrate(video_frames)
    return calibrator


def _build_team_classifier(
    mode: str, embedding_model_path: str, random_state: int
) -> TeamClassifier:
    """Picks a TeamClassifier per ``PipelineConfig.team_classifier``.

    'embedding' needs a model exported by scripts/export_team_embedding_model.py
    -- see EmbeddingTeamClassifier's docstring for what it actually improves
    on and what it still can't do (referee separation).
    """
    if mode == "embedding":
        return EmbeddingTeamClassifier(embedding_model_path, random_state=random_state)

    return TeamAssigner(random_state=random_state)


def _build_jersey_classifier(config: PipelineConfig) -> JerseyClassifier | None:
    """None when config.jersey_backend == 'off' (the default) -- jersey
    classification is entirely opt-in, matching clock_reader's pattern."""
    if config.jersey_backend == "off":
        return None

    if config.jersey_backend == "ocr":
        from agon.jersey.ocr_reader import EasyOcrJerseyReader

        return EasyOcrJerseyReader()

    if config.jersey_model_path is None:
        raise ValueError("jersey_backend='onnx' requires jersey_model_path to be set.")

    from agon.jersey.onnx_classifier import OnnxJerseyClassifier

    return OnnxJerseyClassifier(config.jersey_model_path)


def _assign_jersey_numbers(
    tracks: Tracks,
    jersey_classifier: JerseyClassifier,
    video_frames: list[Frame],
    min_confidence: float,
    min_votes: int = 1,
) -> None:
    """Runs jersey_classifier.classify() once per player/goalkeeper track
    per frame, aggregates each track's predictions across every frame it
    appears in (see agon.jersey.aggregator -- single-frame predictions
    alone are unreliable by the underlying task's own nature, not a bug
    here), then writes the one aggregated number back into every one of
    that track's frame entries."""
    from agon.jersey.aggregator import aggregate_track_jersey_numbers

    predictions_by_track: dict[int, list[tuple[int | None, float]]] = {}
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            prediction = jersey_classifier.classify(video_frames[frame_num], track["bbox"])
            predictions_by_track.setdefault(player_id, []).append(prediction)

    aggregated = aggregate_track_jersey_numbers(
        predictions_by_track, min_confidence=min_confidence, min_votes=min_votes
    )

    for player_track in tracks["players"]:
        for player_id, track in player_track.items():
            track["jersey_number"] = aggregated.get(player_id)


def _build_clock_reader(config: PipelineConfig) -> ClockReader | None:
    if config.clock_calibration_path is None:
        return None
    clock_calibration = ClockCalibrationConfig.from_json_file(Path(config.clock_calibration_path))
    return ClockReader(clock_calibration)


def _classify_frames(
    video_frames: list[Frame],
    config: PipelineConfig,
    clock_reader: ClockReader | None,
    frame_id_base: int,
) -> tuple[list[Frame], list[int] | None, list[str] | None, list[float | None] | None]:
    """No-op passthrough when ``config.frame_filter_mode == 'off'``
    (returns ``video_frames`` unchanged and None for the other three).
    Otherwise classifies every frame (see agon.broadcast) and,
    in 'strip' mode, drops everything that isn't LIVE_PLAY.

    Returns ``(kept_frames, kept_frame_ids, frame_classifications,
    game_clock_s_per_frame)``, all aligned to ``kept_frames``.
    ``kept_frame_ids`` (the *original*, source-video-relative index of
    each kept frame -- ``frame_id_base + local_idx`` in the caller's
    frame numbering) is None unless frames were actually dropped, since
    every other caller can keep assuming contiguous frame_offset+local_idx
    numbering (see build_frame_records).
    """
    if config.frame_filter_mode == "off":
        return video_frames, None, None, None

    classifications: list[FrameClassification] = []
    clocks: list[float | None] = []
    for frame in video_frames:
        classification, game_clock = classify_frame(frame, config.min_grass_fraction, clock_reader)
        classifications.append(classification)
        clocks.append(game_clock)

    if config.frame_filter_mode == "tag":
        return video_frames, None, [c.value for c in classifications], clocks

    keep = [i for i, c in enumerate(classifications) if c == FrameClassification.LIVE_PLAY]
    return (
        [video_frames[i] for i in keep],
        [frame_id_base + i for i in keep],
        [classifications[i].value for i in keep],
        [clocks[i] for i in keep],
    )


EXPORT_FORMATS = ("jsonl", "parquet", "summary", "schema", "video")


@dataclass
class PipelineResult:
    tracks: Tracks
    team_ball_control: np.ndarray
    camera_movement_per_frame: list[Point]
    frame_rate: float
    frame_records: list[FrameRecord] | None = None
    match_summary: MatchSummary | None = None
    kept_frame_ids: list[int] | None = None
    """Original (source-video) frame indices of ``tracks``' frames, when
    ``config.frame_filter_mode == 'strip'`` dropped some. None otherwise
    (every source frame was kept, in order)."""


def run_pipeline(
    video_path: str | Path,
    model_path: str | Path,
    calibration: CalibrationConfig,
    config: PipelineConfig | None = None,
    stub_dir: str | Path | None = None,
    read_from_stub: bool = False,
    output_video_path: str | Path | None = None,
    export_dir: str | Path | None = None,
    export_formats: list[str] | None = None,
    video_id: str | None = None,
) -> PipelineResult:
    """Run detection/tracking/analytics, and optionally render an annotated
    video and/or write the ML-ready data export.

    Args:
        video_path: input match footage.
        model_path: YOLO checkpoint for player/referee/ball detection.
        calibration: per-video pixel-to-pitch calibration (see configs/calibration/).
        config: pipeline thresholds; defaults to ``PipelineConfig()``.
        stub_dir: optional directory to cache intermediate tracking/camera-movement
            results, keyed by video filename, so re-running is fast during
            development. See the README for why this is a *cache*, not the
            canonical data export (that's ``agon.export``, driven
            by ``export_dir``/``export_formats`` below).
        read_from_stub: reuse a previously written stub cache if present.
        output_video_path: if given, renders an annotated video there.
        export_dir: if given, writes the ML-ready data export here.
        export_formats: subset of ``EXPORT_FORMATS`` to write into
            ``export_dir``; "video" is handled via ``output_video_path``, not
            here (kept in this tuple since the CLI's ``--format`` flag covers
            both with one option). Defaults to jsonl+parquet+summary.
        video_id: identifier stamped into export records; defaults to the
            input video's filename stem.

    ``config.frame_filter_mode`` ('off' by default): 'tag' attaches
    frame_classification/game_clock_s to every exported record without
    dropping anything; 'strip' additionally drops non-live-play frames
    (ads/replays/graphics -- see agon.broadcast) before
    detection/tracking/video rendering even run on them, so the annotated
    video and every downstream stat only reflect live play. Dropped frames
    are gaps, not renumbered away: exported frame_id always refers to the
    original source-video frame index (see PipelineResult.kept_frame_ids),
    but camera-movement/pitch-calibration continuity across a strip gap
    inherits the same "some discontinuity at the seam" trade-off already
    documented for chunk boundaries in run_pipeline_streaming.
    """
    config = config or PipelineConfig()

    video_path = Path(video_path)
    stub_dir = Path(stub_dir) if stub_dir else None
    tracks_stub = stub_dir / f"{video_path.stem}_tracks.json" if stub_dir else None
    camera_stub = stub_dir / f"{video_path.stem}_camera_movement.json" if stub_dir else None

    video_frames = read_video(video_path)
    frame_rate = config.frame_rate or get_video_info(video_path).fps

    clock_reader = _build_clock_reader(config)
    video_frames, kept_frame_ids, frame_classifications, game_clock_s_per_frame = _classify_frames(
        video_frames, config, clock_reader, frame_id_base=0
    )
    if not video_frames:
        raise ValueError(
            "frame_filter_mode='strip' classified every frame in this video as "
            "non-live-play -- check min_grass_fraction/clock_calibration_path."
        )

    tracker = _build_tracker(config.tracker_backend, frame_rate)
    detector = _build_detector(
        Path(model_path),
        config.device,
        config.detection_confidence,
        imgsz=config.detection_imgsz,
        tracker=tracker,
    )
    tracks = detector.get_object_tracks(
        video_frames, read_from_stub=read_from_stub, stub_path=tracks_stub
    )
    # Ball interpolation must happen before position/camera/pitch steps below,
    # not after (the original tutorial did it after -- a latent bug: an
    # interpolated-only ball frame would never get "position"/
    # "position_transformed" computed, since those steps had already run over
    # the pre-interpolation, mostly-empty ball tracks by the time interpolation
    # replaced them).
    tracks["ball"] = interpolate_ball_positions(tracks["ball"])
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

    speed_distance_estimator = SpeedDistanceEstimator(
        frame_window=config.speed_frame_window, frame_rate=frame_rate
    )
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = _build_team_classifier(
        config.team_classifier, config.team_embedding_model_path, config.team_kmeans_random_state
    )
    seed_frame = _first_frame_with_enough_players(tracks["players"])
    team_assigner.assign_team_color(video_frames[seed_frame], tracks["players"][seed_frame])
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

    jersey_classifier = _build_jersey_classifier(config)
    if jersey_classifier is not None:
        _assign_jersey_numbers(
            tracks,
            jersey_classifier,
            video_frames,
            min_confidence=config.jersey_min_confidence,
            min_votes=config.jersey_min_votes,
        )

    if output_video_path is not None:
        output_frames = draw_annotations(video_frames, tracks, team_ball_control_array)
        output_frames = camera_movement_estimator.draw_camera_movement(
            output_frames, camera_movement_per_frame
        )
        speed_distance_estimator.draw_speed_and_distance(output_frames, tracks)
        save_video(output_frames, output_video_path, fps=frame_rate)

    frame_records = None
    match_summary = None
    if export_dir is not None:
        formats = export_formats or ["jsonl", "parquet", "summary"]
        video_id = video_id or Path(video_path).stem
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        needs_records = {"jsonl", "parquet"} & set(formats)
        if needs_records:
            frame_records = build_frame_records(
                tracks,
                team_ball_control_array.tolist(),
                camera_movement_per_frame,
                video_id,
                frame_rate,
                frame_ids=kept_frame_ids,
                frame_classifications=frame_classifications,
                game_clock_s_per_frame=game_clock_s_per_frame,
            )
            if "jsonl" in formats:
                write_jsonl(frame_records, export_dir / f"{video_id}_frames.jsonl")
            if "parquet" in formats:
                write_parquet(frame_records, export_dir / f"{video_id}_frames.parquet")

        if "summary" in formats:
            match_summary = build_match_summary(
                tracks, team_ball_control_array.tolist(), video_id, frame_rate
            )
            write_match_summary(match_summary, export_dir / f"{video_id}_summary.json")

        if "schema" in formats:
            write_schema_json(export_dir / "schema.json")

    return PipelineResult(
        tracks=tracks,
        team_ball_control=team_ball_control_array,
        camera_movement_per_frame=camera_movement_per_frame,
        frame_rate=frame_rate,
        frame_records=frame_records,
        match_summary=match_summary,
        kept_frame_ids=kept_frame_ids,
    )


def run_pipeline_streaming(
    video_path: str | Path,
    model_path: str | Path,
    calibration: CalibrationConfig,
    config: PipelineConfig | None = None,
    chunk_size: int = 1000,
    stub_dir: str | Path | None = None,
    read_from_stub: bool = False,
    output_video_path: str | Path | None = None,
    export_dir: str | Path | None = None,
    export_formats: list[str] | None = None,
    video_id: str | None = None,
) -> MatchSummary:
    """Chunked/bounded-memory variant of ``run_pipeline`` for footage that
    doesn't fit in memory as a single frame list -- see ``io.video``'s
    module docstring for why: a full-length match at native broadcast
    resolution/framerate is on the order of terabytes as one in-memory
    list. Same arguments as ``run_pipeline`` plus ``chunk_size`` (frames per
    chunk -- pick this so ``chunk_size * height * width * 3`` bytes
    comfortably fits in memory; a few thousand frames is reasonable at
    1080p). Returns the match summary (per-frame records are written
    directly to disk, not held in memory or returned).

    Trades some cross-chunk continuity for boundedness. What carries
    across chunk boundaries correctly: track IDs (the same detector/tracker
    instance is reused for every chunk), camera-movement optical flow (see
    ``CameraMovementEstimator.get_camera_movement_chunk``), pitch
    calibration's temporal continuity (see ``PitchKeypointCalibrator``),
    team-color KMeans (fit once, on the first chunk with enough players),
    ball-possession's carry-forward fallback, and cumulative distance (see
    ``SpeedDistanceEstimator.total_distance``). What does NOT, as a
    deliberate, documented, minor trade-off rather than a hidden bug:
    - Ball-position interpolation only fills gaps *within* one chunk. A gap
      spanning a chunk boundary leaves those frames' ball position
      unresolved (null), same treatment as "ball never detected in this
      chunk at all" (already handled gracefully -- see
      ``interpolate_ball_positions``).
    - The last ``frame_window`` frames of every chunk but the last don't get
      a speed/distance *reading* from that chunk's own window (no "future"
      frame within the same chunk to measure against) -- a handful of
      frames every ``chunk_size / frame_rate`` seconds. Cumulative distance
      itself is unaffected; only an instantaneous speed/distance display
      value at each seam is missing.

    Detection/tracking results are cached per chunk (in ``stub_dir``, if
    given), so an interrupted long run can be restarted without redoing the
    expensive part. The continuity state above (camera flow, calibration,
    cumulative distance, ...) is not itself persisted across a restart --
    cheap to resume, not perfectly seamless at the exact interruption
    point. Acceptable for what this is: recovering from "the process died
    partway through a multi-hour run," not routine operation.

    ``config.frame_filter_mode`` ('off' by default): see ``run_pipeline``'s
    docstring -- same 'tag'/'strip' semantics here, applied chunk by chunk.
    A chunk that strips down to zero live-play frames is skipped entirely
    (no detection call, nothing written for it) but still advances the
    match's frame-position bookkeeping by its original (pre-strip) length,
    so later chunks' frame numbering stays correct regardless of how much
    was stripped earlier.
    """
    config = config or PipelineConfig()
    video_path = Path(video_path)
    stub_dir_path = Path(stub_dir) if stub_dir else None
    video_id = video_id or video_path.stem
    formats = export_formats or ["jsonl", "parquet", "summary"]

    info = get_video_info(video_path)
    frame_rate = config.frame_rate or info.fps
    total_chunks = -(-info.frame_count // chunk_size) if info.frame_count > 0 else None
    logger.info(
        "Streaming pipeline starting: %s frames, %.1f fps, chunk_size=%d (~%s chunks)",
        info.frame_count or "unknown",
        info.fps,
        chunk_size,
        total_chunks if total_chunks is not None else "unknown",
    )

    tracker = _build_tracker(config.tracker_backend, frame_rate)
    detector = _build_detector(
        Path(model_path),
        config.device,
        config.detection_confidence,
        imgsz=config.detection_imgsz,
        tracker=tracker,
    )
    pitch_calibrator = _make_pitch_calibrator(calibration, config.calibration_mode)
    team_classifier = _build_team_classifier(
        config.team_classifier, config.team_embedding_model_path, config.team_kmeans_random_state
    )
    speed_distance_estimator = SpeedDistanceEstimator(
        frame_window=config.speed_frame_window, frame_rate=frame_rate
    )
    player_assigner = PlayerBallAssigner(
        max_player_ball_distance_px=config.ball_max_assignment_distance_px
    )
    pitch_pixel_vertices = np.array(calibration.pixel_vertices, dtype=np.float32)

    camera_movement_estimator: CameraMovementEstimator | None = None
    camera_flow_state = None
    team_color_seeded = False
    team_ball_control: list[int] = []  # running, whole-match-so-far
    match_stats = MatchStats()
    video_writer: IncrementalVideoWriter | None = None

    export_dir_path = Path(export_dir) if export_dir else None
    if export_dir_path is not None:
        export_dir_path.mkdir(parents=True, exist_ok=True)
    jsonl_writer = (
        JsonlWriter(export_dir_path / f"{video_id}_frames.jsonl")
        if export_dir_path is not None and "jsonl" in formats
        else None
    )
    parquet_writer = (
        ParquetChunkWriter(export_dir_path / f"{video_id}_frames.parquet")
        if export_dir_path is not None and "parquet" in formats
        else None
    )

    def carry_forward_team(chunk_control: list[int]) -> int:
        if chunk_control:
            return chunk_control[-1]
        return team_ball_control[-1] if team_ball_control else 0

    clock_reader = _build_clock_reader(config)
    frame_offset = 0
    start_time = time.monotonic()

    try:
        for chunk_idx, raw_video_frames in enumerate(iter_video_chunks(video_path, chunk_size)):
            original_chunk_len = len(raw_video_frames)
            video_frames, kept_frame_ids, frame_classifications_chunk, game_clock_s_chunk = (
                _classify_frames(raw_video_frames, config, clock_reader, frame_id_base=frame_offset)
            )
            if not video_frames:
                logger.info(
                    "Chunk %d: all %d frames classified as non-live-play, skipping.",
                    chunk_idx,
                    original_chunk_len,
                )
                frame_offset += original_chunk_len
                continue

            chunk_stub = (
                stub_dir_path / f"{video_path.stem}_chunk{chunk_idx:05d}_tracks.json"
                if stub_dir_path is not None
                else None
            )

            tracks = detector.get_object_tracks(
                video_frames, read_from_stub=read_from_stub, stub_path=chunk_stub
            )
            tracks["ball"] = interpolate_ball_positions(tracks["ball"])
            add_position_to_tracks(tracks)

            if camera_movement_estimator is None:
                camera_movement_estimator = CameraMovementEstimator(
                    video_frames[0], pitch_pixel_vertices=pitch_pixel_vertices
                )
            camera_movement_chunk, camera_flow_state = (
                camera_movement_estimator.get_camera_movement_chunk(video_frames, camera_flow_state)
            )
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_chunk)

            pitch_calibrator.calibrate(video_frames, frame_offset=frame_offset)
            add_transformed_position_to_tracks(tracks, pitch_calibrator, frame_offset=frame_offset)

            speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

            if not team_color_seeded:
                try:
                    seed_frame = _first_frame_with_enough_players(tracks["players"])
                    team_classifier.assign_team_color(
                        video_frames[seed_frame], tracks["players"][seed_frame]
                    )
                    team_color_seeded = True
                except ValueError:
                    logger.info(
                        "Chunk %d: still no frame with enough players to seed team colors",
                        chunk_idx,
                    )

            if team_color_seeded:
                for local_idx, player_track in enumerate(tracks["players"]):
                    for player_id, track in player_track.items():
                        player_team = team_classifier.get_player_team(
                            video_frames[local_idx], track["bbox"], player_id
                        )
                        track["team"] = player_team
                        track["team_color"] = team_classifier.team_colors[player_team]

            chunk_team_ball_control: list[int] = []
            for local_idx, player_track in enumerate(tracks["players"]):
                ball_bbox = tracks["ball"][local_idx][1]["bbox"]
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                team = None
                if assigned_player is not None:
                    tracks["players"][local_idx][assigned_player]["has_ball"] = True
                    team = tracks["players"][local_idx][assigned_player].get("team")

                chunk_team_ball_control.append(
                    team if team is not None else carry_forward_team(chunk_team_ball_control)
                )

            team_ball_control.extend(chunk_team_ball_control)
            accumulate_match_stats(match_stats, tracks, chunk_team_ball_control)

            if jsonl_writer is not None or parquet_writer is not None:
                frame_records = build_frame_records(
                    tracks,
                    chunk_team_ball_control,
                    camera_movement_chunk,
                    video_id,
                    frame_rate,
                    frame_offset=frame_offset,
                    frame_ids=kept_frame_ids,
                    frame_classifications=frame_classifications_chunk,
                    game_clock_s_per_frame=game_clock_s_chunk,
                )
                if jsonl_writer is not None:
                    jsonl_writer.write_chunk(frame_records)
                if parquet_writer is not None:
                    parquet_writer.write_chunk(frame_records)

            if output_video_path is not None:
                # Indexes team_ball_control (which only ever grows by *kept*
                # frames -- see the strip-mode note above) by how many kept
                # frames preceded this chunk, not by frame_offset (the true
                # source-video position, which diverges from the kept count
                # once frame_filter_mode='strip' has dropped anything).
                kept_frames_before_chunk = len(team_ball_control) - len(chunk_team_ball_control)
                output_frames: list[Frame] = []
                for local_idx, frame in enumerate(video_frames):
                    annotated = draw_annotations_on_frame(
                        frame,
                        tracks["players"][local_idx],
                        tracks["referees"][local_idx],
                        tracks["ball"][local_idx],
                        team_ball_control[: kept_frames_before_chunk + local_idx + 1],
                    )
                    annotated = camera_movement_estimator.draw_camera_movement_on_frame(
                        annotated, camera_movement_chunk[local_idx]
                    )
                    speed_distance_estimator.draw_speed_and_distance_on_frame(
                        annotated,
                        {
                            "players": tracks["players"][local_idx],
                            "referees": tracks["referees"][local_idx],
                            "ball": tracks["ball"][local_idx],
                        },
                    )
                    output_frames.append(annotated)

                if video_writer is None:
                    height, width = output_frames[0].shape[:2]
                    video_writer = IncrementalVideoWriter(
                        output_video_path, fps=frame_rate, frame_size=(width, height)
                    )
                video_writer.write(output_frames)

            frame_offset += original_chunk_len
            elapsed = time.monotonic() - start_time
            fps_so_far = frame_offset / elapsed if elapsed > 0 else 0.0
            eta = (
                (info.frame_count - frame_offset) / fps_so_far
                if info.frame_count > 0 and fps_so_far > 0
                else None
            )
            logger.info(
                "Chunk %d done: %d frames processed, %.1fs elapsed, %.1f fps%s",
                chunk_idx,
                frame_offset,
                elapsed,
                fps_so_far,
                f", ETA {eta / 60:.1f} min" if eta is not None else "",
            )
    finally:
        if jsonl_writer is not None:
            jsonl_writer.close()
        if parquet_writer is not None:
            parquet_writer.close()
        if video_writer is not None:
            video_writer.close()

    # frame_count reflects frames actually analyzed (post frame-filter, if
    # any), matching run_pipeline/build_match_summary's semantics -- not
    # frame_offset, which is the true source-video position and can be
    # larger once frame_filter_mode='strip' has dropped anything.
    match_summary = finalize_match_summary(
        match_stats, video_id, len(team_ball_control), frame_rate
    )
    if export_dir_path is not None and "summary" in formats:
        write_match_summary(match_summary, export_dir_path / f"{video_id}_summary.json")
    if export_dir_path is not None and "schema" in formats:
        write_schema_json(export_dir_path / "schema.json")

    logger.info(
        "Streaming pipeline done: %d source frames scanned, %d frames analyzed",
        frame_offset,
        len(team_ball_control),
    )
    return match_summary
