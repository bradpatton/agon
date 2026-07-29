# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
versions the export schema independently via `SCHEMA_VERSION` in
`src/agon/export/schema.py`, not just the package version.

## [0.1.0] - Unreleased

### Added
- ML-ready data export: versioned pydantic schema (`FrameRecord`/
  `ObjectRecord`/`MatchSummary`), JSONL + Parquet + match-summary + JSON
  Schema writers — the project's core new capability, replacing "annotated
  video only" as the primary output.
- `agon` CLI (Typer), `--format` flag selecting any combination
  of `jsonl`/`parquet`/`summary`/`schema`/`video`.
- Installable `src/agon` package (`pyproject.toml`, `uv.lock`)
  replacing the original tutorial's flat, `sys.path`-hacked scripts.
- `OnnxDetector`: torch-free default detector backend (onnxruntime), plus
  `UltralyticsDetector` (torch-backed) and `BoTSORTTracker` behind the
  `[train]` extra.
- `PitchKeypointCalibrator`: classical-CV per-frame pitch calibration
  (center-circle + halfway-line detection), an alternative to the static
  per-video `ViewTransformer`. See its docstring for real limitations.
- `EmbeddingTeamClassifier`: small-CNN-embedding team clustering, an
  alternative to raw-pixel-color `TeamAssigner`.
- `Detector`/`PitchCalibrator`/`TeamClassifier`/`FrameTracker` protocols
  (`interfaces.py`, `detection/base.py`) making every stage above swappable
  via `PipelineConfig` without touching `pipeline.py`.
- Test suite (pytest, pure-Python/synthetic-input, no video/model file
  needed), GitHub Actions CI (ruff + mypy + pytest on Python 3.11-3.13),
  `.pre-commit-config.yaml`.
- `Dockerfile` for validating the `[train]`-extra-dependent backends on
  platforms (e.g. Intel macOS) where current torch/onnxruntime/boxmot have
  no wheels.
- `scripts/download_assets.py`, `scripts/export_team_embedding_model.py`.
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR
  templates.

### Fixed
Bugs inherited from the original tutorial, found while restructuring and
validating end-to-end against real footage:
- Team-color KMeans clustering was hardcoded to a specific player ID
  (`player_id == 91`) overfit to one video.
- Ball-possession fallback indexed `[-1]` into an empty list on frame 0,
  raising `IndexError` if the ball wasn't assigned to anyone yet.
- `ViewTransformer`'s pixel calibration points and `CameraMovementEstimator`'s
  feature-mask region were hardcoded to one video's resolution/framing.
- `KMeans` calls had no `random_state`, making team clustering
  non-deterministic between runs.
- Track-result caching used `pickle` (unsafe to load from an untrusted
  source); replaced with JSON.
- `interpolate_ball_positions` crashed (`ValueError`) when the ball was
  never detected in any frame of a clip.
- Ball interpolation ran *after* position/camera-compensation/pitch-calibration
  steps instead of before, so an interpolated ball frame never actually got
  those fields computed.
- Team-color seeding assumed frame 0 always has confirmed player tracks;
  false when a tracker's confirmation warm-up delays past frame 0.
- `UltralyticsDetector` and `OnnxDetector` had diverged, inconsistent
  default class-name-to-object-type mappings — running a generic COCO
  checkpoint through `UltralyticsDetector` silently tracked zero players.
- `SpeedDistanceEstimator` divided by zero on the tail window whenever
  `(number_of_frames - 1) % frame_window == 0`.
- `save_video` silently reported success while writing nothing when the
  local OpenCV build lacked a compatible video codec/backend.
- `resolve_device()` unconditionally imported torch, breaking the
  torch-free `OnnxDetector` path even when torch was never needed.

### Changed
- Naming typos throughout (`draw_traingle` → `draw_triangle`,
  `persepctive_trasnformer` → `perspective_transform`,
  `SpeedAndDistance_Estimator` → `SpeedDistanceEstimator`, etc.).
- Notebooks moved to `notebooks/training/` and `notebooks/exploration/`.
- Default video codec changed from XVID/`.avi` to H.264/`.mp4` (more
  broadly supported across OpenCV builds).

## [Unreleased] - post-0.1.0

### Added
- Streaming/chunked pipeline (`run_pipeline_streaming`, `--chunk-size`):
  bounded-memory processing of full-length matches, validated against a
  real ~108-minute match at native 1920x1080/50fps.
- `agon.broadcast`: classical-CV frame classification
  (live-play/replay/graphic) plus Tesseract-OCR match-clock reading,
  `frame_filter_mode` (`off`/`tag`/`strip`) on `PipelineConfig` — filters
  ads/replays/graphics out of the export and annotated video, tags frames
  with real elapsed game-time (`game_clock_s`).
- SoccerNet-driven model training pipeline (Phase 7): `scripts/
  download_soccernet_gsr.py`, `scripts/convert_soccernet_gsr_to_yolo.py`,
  `scripts/convert_soccernet_gsr_to_jersey_crops.py`, `scripts/
  convert_soccernet_gsr_to_calibration.py`, `scripts/train_detector.py`,
  `scripts/train_jersey_classifier.py`, `scripts/prepare_training_data.py`
  (one-command download+convert pipeline). `ObjectRecord.jersey_number`
  added to the export schema (not yet wired into inference).
- `PipelineConfig.detection_imgsz`: configurable inference resolution,
  threaded through both detector backends.
- `docs/TRAINING.md`: quick guide for deploying the training image to a
  GPU machine and training end to end.

### Fixed
- `OnnxDetector` always supported a configurable input resolution, but
  nothing above it in the pipeline ever passed one through — inference
  silently ran at 640x640 regardless of what resolution a checkpoint was
  actually trained/exported at.
- Pipeline defaulted `frame_rate` to a hardcoded 24fps instead of the
  input video's actual fps, corrupting speed/distance/timestamp math and
  the annotated video's playback speed on any non-24fps footage.
- `IncrementalVideoWriter` didn't remove a stale file at the output path
  first; macOS's AVFoundation backend refuses to open a `VideoWriter` at a
  path that already exists instead of overwriting it.
- `ViewTransformer`/`PitchKeypointCalibrator` crashed (`ValueError`) on a
  NaN ball position (the ball never detected anywhere in a chunk).
- `Dockerfile`'s `ENTRYPOINT` silently prepended itself to any command
  passed to `docker run`, breaking every training-script invocation.
- SoccerNet data-conversion scripts baked in absolute host paths (dataset
  config and image symlinks), breaking the moment the same output
  directory was mounted into a container at a different path.
- `classify_frame` called OCR (~150ms) unconditionally on every frame
  regardless of whether a frame had pitch visible at all — an unforced
  multi-hour runtime for a 10-minute clip with clock-reading enabled.
