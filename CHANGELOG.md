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
- `--workers` in both training scripts now defaults to an auto-detected
  value based on available *system* RAM (`_auto_workers`, using
  `psutil`, already a transitive dependency via `ultralytics` -- no new
  dependency needed) instead of Ultralytics' fixed default of 8 --
  motivated by wanting the scripts to be safe by default on any future
  machine (more/less RAM, different GPU count) without manual tuning,
  not just the one machine this project happened to hit an OOM on.
  Deliberately conservative (2GB/worker budget, 3GB reserved for the
  main process before any of it goes to workers) since a slower run is
  a minor cost and a run that gets killed hours in is not -- on the
  actual machine/dataset that hit the original OOM (~13GB available
  RAM), this computes 5 workers instead of the fixed 8 that crashed.
  Explicit `--workers <N>` always overrides, including in `--resume`
  mode (extends the existing override-on-resume mechanism). Real,
  stated limitation: checks memory available once, at start, not growth
  over a multi-hour run -- reduces OOM risk, doesn't guarantee against
  it.
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
- Project renamed from `soccer-analysis` to **Agon** (ἀγών, Greek for
  "contest") throughout: package (`src/agon`), CLI command, Docker image,
  env var prefix.
- Two more SoccerNet training data sources, combined with SN-GSR-2025 into
  the same training directories: `scripts/download_soccernet_legacy.py`,
  `scripts/convert_soccernet_calibration_to_pixels.py` (2,719 standalone
  pitch-calibration images, more scene diversity than GSR's video-sequence
  frames), `scripts/convert_soccernet_jersey2023_to_crops.py` (the richer,
  tracklet-based jersey number dataset, evenly sampled per tracklet).
  `prepare_training_data.py` rewritten to orchestrate all three sources.

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
- SoccerNet's legacy Tracking/Calibration/Re-ID/Jersey-2023 datasets were
  wrongly documented as broken — they were tested with the wrong password
  (a personal NDA video-download password, not the generic public one
  these specific datasets actually use); re-tested and confirmed working.
- `docs/TRAINING.md`'s Docker volume mounts didn't match where the scripts
  actually write: Step 3 mounted `/data` but `prepare_training_data.py`
  writes under `/app/data` (relative to its own location in the image), so
  downloaded/converted data never left the container and was lost on
  `--rm`; Step 4 didn't override `train_detector.py`/
  `train_jersey_classifier.py`'s default `--project runs/train` (also not
  under any mounted volume), so a real training run would complete with
  correct metrics and then lose every weight file the same way. Both fixed
  by mounting `/app/data` and adding an explicit `--project
  /app/models/runs/train`. Also added the previously-undocumented
  prerequisite that `--gpus all` needs the NVIDIA Container Toolkit
  installed and Docker configured for it beforehand — without it the
  command fails outright, not just falls back to CPU.
- `README.md`'s Docker development example was missing the `agon` command
  name — since the image has no `ENTRYPOINT`, `docker run <image> --input
  ...` tried to exec `--input` itself as the container's process and failed
  immediately.
- `huggingface_hub` and the `SoccerNet` pip package (used by
  `download_soccernet_gsr.py`/`download_soccernet_legacy.py`) were never
  declared as dependencies anywhere — interactive testing on dev machines
  worked because they'd been manually `pip install`ed into a throwaway
  venv, but a fresh `uv sync --extra train` (i.e. building the Docker
  image from scratch) never installed them, so `prepare_training_data.py`
  failed for every data source inside a real container. Found and fixed by
  actually building the image fresh and running it, not just reasoning
  through the dependency graph — added both to the `[train]` extra and
  re-validated with a real download+convert run (2,719 images, matching
  the earlier manually-tested count) confirming data now actually lands on
  the host via the volume mount.
- `docs/TRAINING.md`'s commands assumed a bash/zsh shell (`$(pwd)`, `\`
  line continuation) with no mention of Windows — hit for real on an
  actual ML machine running Command Prompt, where `$(pwd)` isn't expanded
  at all and Docker fails with `includes invalid characters for a local
  volume name`. Added a shell-substitution table (`%cd%`/`^` for
  `cmd.exe`, `${PWD}`/backtick for PowerShell) and a Windows-specific GPU
  passthrough note (needs Docker Desktop's WSL2 backend with Linux
  containers, not Windows-containers mode).
- `train_detector.py` hard-exited if `--base-model` (default
  `models/yolo11n.pt`) didn't already exist locally -- hit for real on an
  ML machine with a freshly-mounted, empty `models/` directory. This
  directly contradicted `docs/TRAINING.md`'s own claim that the base
  checkpoint "auto-downloads on first training run": Ultralytics does
  auto-download a recognized checkpoint name to the exact given path, but
  only if the script ever reaches `YOLO(...)` -- the pre-check was exiting
  before that could happen. Changed to a warning instead of a hard exit,
  restoring the documented self-heal behavior.
- `docs/TRAINING.md`'s example training commands hardcoded `--device
  0,1,2` -- hit for real on an ML machine with only 1 GPU configured
  ("invalid CUDA device"). `train_detector.py`/`train_jersey_classifier.py`
  now auto-detect every visible CUDA GPU (`torch.cuda.device_count()`) and
  use all of them by default, so the command doesn't need editing as GPUs
  are added or removed; `--device` remains available to explicitly
  restrict to a subset. Verified the detection logic's branches (0/1/N
  GPUs) with a mocked `torch.cuda` since no real multi-GPU hardware is
  available in this environment.
- Training `docker run` commands were missing `--ipc=host` -- hit for real
  on an ML machine as `RuntimeError: unable to allocate shared memory
  (shm)... No space left on device`, which reads like a disk problem but
  is actually Docker's default 64MB `/dev/shm` being far too small for
  PyTorch DataLoader workers passing real image batches between
  processes. `--ipc=host` (PyTorch's own recommended fix) added to both
  training commands in `docs/TRAINING.md`.
- Same class of bug again, this time with batch size: the documented
  `--batch 32` example OOM'd for real (`torch.AcceleratorError: CUDA
  error: out of memory`) on a GPU with less VRAM than whatever card 32 was
  sized for. `train_detector.py`/`train_jersey_classifier.py`'s `--batch`
  default changed from a fixed number to `-1`, which triggers Ultralytics'
  AutoBatch (a few trial passes pick the largest batch that actually fits
  in available VRAM) instead of guessing a number that happens to work on
  one card and not another.
- AutoBatch alone wasn't enough on a small-VRAM GPU -- training ran
  cleanly for ~15 real iterations (memory climbing) then OOM'd in the
  DataLoader's pin-memory step, a fragmentation/worst-case-batch pattern
  rather than "never fits." Added `--workers` to both training scripts
  (fewer DataLoader workers = less pinned memory held for prefetching)
  and documented `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (PyTorch's own fix for this exact symptom) in `docs/TRAINING.md`'s
  training command and troubleshooting section.
- AutoBatch (`--batch -1`) doesn't work at all for multi-GPU training --
  confirmed via Ultralytics' own error on real 2-GPU hardware (`AutoBatch
  with batch<1 not supported for Multi-GPU training, please specify a
  valid batch size multiple of GPU count`). Both training scripts now
  detect this case and fall back to a conservative explicit batch (8 per
  GPU) automatically instead of crashing, printing a note when the
  fallback is used. Verified the fallback logic's branches (1 GPU, 2/3
  GPUs, explicit override) directly since no real multi-GPU hardware was
  available in this environment at the time.

### Added
- `train_detector.py --resume <path/to/last.pt>`: resumes an interrupted
  training run (a real power outage killed a run mid-epoch-8 on the ML
  machine) from Ultralytics' own checkpoint, picking up at the next epoch
  using the checkpoint's saved training args rather than restarting from
  scratch. The ONNX-export step now reads the actual trained `imgsz` back
  off the model's trainer args instead of trusting `--imgsz` on the CLI,
  since that flag is meaningless (still at its argparse default) in
  `--resume` mode.
- `train_jersey_classifier.py --resume <path>`: same mechanism, added
  after a real host-RAM OOM kill (kernel OOM killer, not GPU VRAM --
  `dmesg` showed the training process at ~8.3GB resident on a 15GB-total
  system) killed a run mid-epoch-11 on the ML machine's 1.29M-image
  jersey-crop dataset.
- Both `--resume` implementations now accept `--workers` as an override
  even in resume mode -- caught before it caused a repeat failure: plain
  `model.train(resume=True)` reuses the checkpoint's *saved* `workers`
  value, which would have silently repeated the exact OOM the resume was
  meant to recover from. `--workers` is a pure DataLoader setting (unlike
  `--data`/`--imgsz`, which must match the checkpoint's architecture), so
  overriding it on resume is safe.
- **Jersey number recognition, inference side** (`agon.jersey`): a real
  first training run surfaced that per-frame classification alone is
  fundamentally unreliable for this task -- confirmed against the official
  SoccerNet Jersey Number Recognition task description
  (github.com/SoccerNet/sn-jersey), which states the numbers "might be
  visible on a very small subset of the whole tracklet" and its own
  reference solutions score ~93% by aggregating across a whole tracklet,
  not classifying isolated frames (this project measured ~1-1.5% top1 on
  isolated frames, consistent with that). Added `agon.interfaces.
  JerseyClassifier`, `agon.jersey.onnx_classifier.OnnxJerseyClassifier`
  (mirrors `EmbeddingTeamClassifier`'s ONNX pattern), and `agon.jersey.
  aggregator.aggregate_track_jersey_numbers` (confidence-weighted voting
  across every frame a track appears in, using the track IDs the tracker
  already provides -- the natural, already-available piece of
  infrastructure for this). Wired into `run_pipeline` via new
  `PipelineConfig.jersey_model_path`/`jersey_min_confidence` fields,
  opt-in and off by default (`ObjectRecord.jersey_number` stays null
  unless configured, unchanged from before). `train_jersey_classifier.py`
  gained `--export-onnx` (it had no ONNX export at all previously) plus a
  `classes.json` sidecar, since Ultralytics' output-index -> label-string
  order isn't reliably recoverable from the ONNX file alone.
- Real, targeted training-data fix on the ML machine (not yet in a
  script, done by hand and documented in the project plan): removed
  legacy Jersey-2023 crops from the combined jersey training set for
  every class except the 3 (`1`, `18`, `2`) where SN-GSR-2025 provided
  zero coverage -- Jersey-2023 labels the *entire tracklet* with one
  ground-truth number regardless of whether it's visible in any given
  sampled frame, while SN-GSR-2025's `attributes.jersey` is null (mapped
  to "unknown") per-frame when illegible, a much cleaner training signal
  wherever it has sufficient volume (97%+ of most classes already).
  Dropped ~20K noisy crops; a fresh `jersey-gsr` training run is in
  progress on the cleaned data.

- **`OnnxJerseyClassifier` preprocessing bug, found by the empirical
  cross-check the previous entry flagged as outstanding**: ran 32 real
  jersey crops (8 classes) through both `OnnxJerseyClassifier` and
  `YOLO(best.pt).predict()` on the ML machine. Output format (assumed
  post-softmax) was correct -- confirmed by reading Ultralytics'
  `Classify.forward()`, which applies `softmax(1)` in export mode. But
  preprocessing was wrong on two counts, both taken from assumption
  rather than Ultralytics' actual `classify_transforms`: (1) it applied
  ImageNet mean/std normalization, but Ultralytics' classification
  pipeline uses `DEFAULT_MEAN=(0,0,0)`/`DEFAULT_STD=(1,1,1)` -- i.e. plain
  `[0, 1]` scaling, no normalization at all; (2) it did a direct
  stretch-resize to the target size, but Ultralytics does a shortest-edge
  resize (preserving aspect ratio) followed by a center-crop for a square
  target. Together these produced near-random predictions (3/32 label
  matches, mean confidence gap 0.536 against the reference model). Fixed
  both in `_preprocess()`; re-validated at 32/32 label matches, mean
  confidence gap 0.0087 (float32 rounding-order noise between torch and
  onnxruntime, not a remaining mismatch).

### Known limitations, honestly documented
- `agon.jersey.onnx_classifier.OnnxJerseyClassifier` has now been
  empirically validated (see the fix entry above) -- no longer an open
  caveat.

### Verified
- GPU passthrough (`docker run --gpus all`), confirmed end-to-end for the
  first time against real hardware (2x RTX 3090, Ubuntu) after carrying an
  "unverified, no GPU available" caveat through every prior phase of this
  project. `torch.cuda.is_available()` returns `True` with both GPUs
  correctly enumerated inside the `agon:train` container. Updated
  `Dockerfile`'s header comment and `docs/TRAINING.md`'s opening note, and
  removed the corresponding "Not yet done" bullet from `README.md`.
