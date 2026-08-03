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
- Annotated video now labels each tracked player with their predicted
  jersey number when `PipelineConfig.jersey_model_path` is configured,
  e.g. `"12 (3)"` (predicted jersey number, tracker's own track_id in
  parentheses) instead of just `"3"` -- lets a viewer visually cross-check
  the model's prediction against the real shirt on screen frame-by-frame,
  rather than only being able to check it by reading the JSON export.
  `draw_ellipse`'s label box is now sized to the actual label text
  (`cv2.getTextSize`) instead of a fixed 40px width, since jersey labels
  are meaningfully wider than a bare track_id. No visual change when
  jersey classification isn't configured (falls back to the plain
  track_id label, unchanged).
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
- **First real-footage review of both `models/runs/train/soccernet-3`
  (the full 50-epoch/960-imgsz detector trained on the ML machine's real
  GPU hardware) and `jersey-gsr`, run together against
  `benchmark_clip.mp4`** (real Premier League broadcast footage, outside
  both models' training data). Detector: strong, validated result -- ball
  detected in 100% of frames (180/180), every player correctly boxed and
  tracked, referee correctly identified; confirmed both numerically and
  by visually inspecting an annotated frame. Jersey classifier: a real,
  separate accuracy problem found, distinct from the preprocessing bug
  fixed earlier the same day -- pulled a player's actual crop directly
  from the source video, visually confirmed the real shirt number was
  12, then ran the exact same crop through the model, which predicted 36
  with up to 98% confidence sustained across dozens of frames of that
  track. The earlier fix confirmed the model's *output* is being read
  correctly; this shows the *model itself* doesn't generalize well from
  SoccerNet's own broadcast footage to a visually different broadcast
  (different league, kit designs, camera/lighting) -- a domain-gap
  problem, not a code bug. Flagged as a real, open limitation rather than
  worked around.

### Fixed
- **Root cause of the jersey classifier's near-random accuracy, found by
  checking the raw SoccerNet label files directly rather than assuming
  the earlier "domain gap" diagnosis was the whole story.** Confirmed
  across 167 real player tracks (8 sequences): `attributes.jersey` in
  SN-GSR-2025 is assigned per *track*, not per frame -- every annotation
  instance of a track carries the same value regardless of whether the
  number is visible in that specific frame (zero tracks showed more than
  one distinct value). Pulled real examples confirming this in practice:
  a crop labeled "8" with the player facing forward and no number visible
  at all, sitting next to a crop labeled "8" with the number clearly
  printed on the back. A large fraction of the ~630K training crops teach
  the classifier to associate irrelevant visual noise (pose, kit color,
  background) with arbitrary digits -- this, not domain gap, is why
  training-time validation accuracy (2.41% top1) was *worse* than a
  trivial "always guess the most common class" baseline (16.4%).
- **Replaced the from-scratch classifier with EasyOCR**
  (`agon.jersey.ocr_reader.EasyOcrJerseyReader`), a pretrained
  general-purpose scene-text-recognition model with no training on this
  project's noisy labels at all. `PipelineConfig.jersey_backend` selects
  `"off"` (default) / `"ocr"` (recommended) / `"onnx"` (the old
  classifier, kept for existing checkpoints, documented as not
  recommended). Validated against real crops with known ground truth:
  correct reads at 93-100% confidence on clearly-visible numbers, and
  correctly returned nothing at all on a crop where the player faced away
  -- exactly the "only label if confident" behavior needed. Real,
  observed failure modes: a 93%-confidence misread (true 36 read as 35,
  a classic 6/5 digit confusion) and lower-confidence misreads on
  two-digit numbers -- confidence alone doesn't guarantee correctness.
- `agon.jersey.aggregator.aggregate_track_jersey_numbers` gained
  `min_votes` (`PipelineConfig.jersey_min_votes`, default 2): the winning
  candidate must be predicted by at least this many separate frames, not
  just clear a confidence threshold on one. Directly motivated by the
  93%-confidence single-frame misread above -- `min_confidence` alone
  wouldn't have caught it, since 0.93 clears any reasonable threshold.
  Requiring a second frame to agree is cheap for any track spanning more
  than a couple of frames (the normal case) and directly protects against
  this exact failure mode.
- **Full pipeline, end-to-end, re-run against the exact real player the
  old classifier got wrong.** Same track (`benchmark_clip.mp4`, track_id
  3, the player independently confirmed by eye to be wearing a real #12
  shirt) that the old `OnnxJerseyClassifier` confidently mislabeled #36:
  re-processed the whole clip through the full pipeline (detection →
  tracking → OCR jersey reading → track-level aggregation) with
  `jersey_backend="ocr"`, and this exact track now correctly comes out as
  **12**. Confirmed it's the same physical player, not a coincidence or a
  different track_id, by re-pulling the crop from the new run's own
  bounding box. 17/43 player tracks in this short (12s) clip cleared both
  the confidence and `min_votes` thresholds -- the rest legitimately
  never showed a clearly-readable number, which is the intended "don't
  guess" behavior, not a shortfall. Run on GPU (`--gpus all`) for speed
  -- EasyOCR runs one full forward pass per player per frame, ~66 seconds
  for 180 frames vs. an estimated ~an hour on CPU for the same clip.

### Investigated
- **Pitch-position (`position_pitch_m`) coverage, measured directly
  rather than assumed.** Ran the same real clip (`benchmark_clip.mp4`)
  through both calibrators: static (`ViewTransformer`, one fixed
  homography for the whole clip) covered only 34.4% of player detections
  with a non-null pitch position; dynamic (`PitchKeypointCalibrator`,
  per-frame center-circle detection) covered 56.9% -- meaningfully
  better, but still leaves nearly half of all detections uncalibrated.
  Checked whether the two calibrators fail on the *same* frames or
  *different* ones: mostly different -- a simple hybrid (try dynamic,
  fall back to static) would cover 73.5% of the same detections, more
  than double the static-only baseline, using two calibrators that
  already exist with zero new CV/ML work. Five improvements identified
  and scoped in the project plan's Phase 12, priority order: hybrid
  fallback, multi-feature full homography instead of a similarity
  transform, camera-motion-compensated carry-forward between circle
  detections, a real accuracy-measurement harness, and the already-planned
  trained keypoint model.

### Added
- **`agon.geometry.hybrid_pitch_calibrator.HybridPitchCalibrator`**
  (Phase 12 item 1): tries a primary `PitchCalibrator` per point, falls
  back to a secondary when the primary returns None. Selectable via
  `PipelineConfig.calibration_mode = "hybrid"` (dynamic primary, static
  fallback). Re-validated on a fresh run against the same real clip:
  77.0% pitch-position coverage (2228/2892 player detections), confirming
  and slightly exceeding the 73.5% estimate above.
- **`scripts/measure_pitch_calibration_agreement.py`** (Phase 12 item 4,
  a simpler version of the originally-scoped harness -- the full
  known-pitch-dimension version needs more detected features than exist
  yet): runs the pipeline twice (static, dynamic) and cross-checks the
  two independent calibrators' transformed positions against each other
  on the same real (frame, track) points. Real result on
  `benchmark_clip.mp4`: mean disagreement of 65.79m (n=587 overlapping
  points) -- but that clip's calibration file is explicitly
  self-documented as "UNCALIBRATED / APPROXIMATE," so this large number
  is dominated by that already-known-bad static calibration, not
  necessarily the dynamic calibrator's own accuracy. Real, useful finding
  regardless: confirms concretely why that file was never trustworthy
  (consistent with the previously-observed 100-200 km/h nonsense speeds),
  and surfaced a genuine gap -- this project has no properly-annotated
  pitch calibration file anywhere to run a clean accuracy check against.
- **`PitchKeypointCalibrator.inverse_transform_point()`**: the inverse of
  `transform_point()` (pitch-space meters -> pixel space), added to
  support a genuine ground-truth self-consistency check -- round-trip
  tested against a real `calibrate()` call on a synthetic frame, not just
  algebraically. Used by
  **`scripts/validate_pitch_calibration_self_consistency.py`**: predicts
  where a touchline should be in pixel space (a known real distance,
  court_width_m/2, from the resolved circle center along the halfway
  line -- something the transform is never actually built from), then
  checks whether the frame really has a pitch-line pixel there. A
  sharper validation than cross-calibrator agreement above, since it
  doesn't depend on a second calibrator's own (possibly also wrong)
  output.

  Real results against two real clips: `benchmark_clip.mp4` had zero
  checkable predictions across all 180 frames -- diagnosed, not just
  observed: at that clip's zoom level the touchlines are ~1979px apart,
  wider than the 1916px frame itself, so the prediction correctly never
  lands in-frame given that camera's shot. `match_10min_sample.mp4`
  (sampled across ten 500-frame windows spanning the whole match): 55
  checkable predictions, only 5 (~9%) landed within 25px of a real line
  pixel -- confirms the similarity-transform approximation really is
  inaccurate on real match footage, not just theoretically.

  **A bigger finding than the accuracy number itself, found by pulling
  and looking at the actual source frames rather than trusting the
  statistics alone**: some of the "resolved" detections in that sample
  weren't calibrating against the pitch at all -- a sustained NBC
  broadcast bumper/transition graphic (the animated peacock logo) was
  confidently matched as the center circle (its glowing near-white
  outline over a green-ish background passes the same filters a real
  circle would). A separate sample showed a different form of the same
  underlying issue: real pitch footage, but near a goal, where the
  detected "circle" is almost certainly the penalty box. Both are
  consistent with an already-documented limitation, but the
  broadcast-graphic case is new and more serious -- this project already
  has a broadcast-frame classifier
  (`agon.broadcast.frame_filter`, live-play vs. replay vs. graphic) that
  isn't wired into pitch calibration at all today, so it never gets a
  chance to reject a frame like this before calibration runs on it.
  Flagged as a new Phase 12 item 6 in the project plan.
- **Phase 12 item 6, built and honestly tested against the original
  failing frame**: `PitchKeypointCalibrator` gained a `min_grass_fraction`
  gate (default 0.35, reusing `PipelineConfig.min_grass_fraction`) that
  skips circle detection on any frame without enough pitch-green
  coverage, via the existing `agon.broadcast.frame_filter.grass_fraction`
  (cheap, no OCR, no new dependency). Deliberately *not* gated on
  `classify_frame`'s LIVE_PLAY/REPLAY distinction instead -- testing that
  option against the same real footage first showed it would inherit the
  already-documented Phase 8 bug (a confirmed-real live goal-mouth
  scramble got misclassified as REPLAY because clock OCR failed on 4 of
  5 sampled frames), trading one false positive for a worse one.

  **Reported honestly rather than oversold**: re-tested directly against
  the original bumper frame -- this gate does *not* catch it. Its
  `grass_fraction` measures 0.645, comfortably above the 0.35 threshold,
  because the bumper animation overlays a still-visible, blurred-green
  pitch background rather than replacing it with a solid non-green
  graphic. Two other frames originally suspected of being more instances
  of the same bug turned out, on closer inspection, to be a different,
  already-documented issue instead (real live footage near a goal, where
  the penalty box was likely matched as the center circle -- item 2's
  territory, not a frame-classification problem). A "vivid non-grass-hue
  pixel fraction" heuristic was tried as a more targeted fix and did not
  cleanly separate the bumper frame from real pitch frames on this
  sample -- not pursued further, not shipped. This gate is a real,
  useful improvement for genuine no-pitch content (ads, lineup cards,
  studio shots), just not a complete fix for the specific case that
  motivated it.
- **`scripts/render_pitch_markings_overlay.py`** (Phase 12 item 7):
  draws every standard pitch marking (touchlines, goal lines, penalty
  boxes, six-yard boxes, penalty spots, penalty arcs, corner arcs,
  halfway line, center circle) onto real footage, computed purely from
  `PitchKeypointCalibrator`'s resolved transform plus fixed, real-world
  FIFA pitch dimensions -- extends item 4's touchline self-consistency
  check to every marking, and makes it visual instead of a single
  hit-rate percentage. Circular features are drawn as true circles
  deliberately, not as an oversight -- the transform is similarity-only
  (no perspective correction), so this is an honest visualization of
  that documented limitation, not a rendering bug.

  Real findings from looking at the actual rendered output: on the known
  bumper-graphic frame, the overlay draws the "center circle" directly
  on the peacock logo animation -- an immediate, unambiguous visual
  confirmation, clearer than the earlier abstract statistic. On a
  confirmed real free-kick frame, the overlay circle lands on the goal,
  not center pitch. **New, more specific finding from `benchmark_clip.mp4`
  (checked across 3 frames)**: the circle's position and scale are
  genuinely close to correct, but the drawn halfway line is consistently
  rotated ~90 degrees from the real one -- decomposes what the earlier
  aggregate "~9% accurate" touchline number couldn't distinguish: the
  circle-based position/scale resolution is reasonably trustworthy on
  this footage, the halfway-line-angle resolution specifically isn't.
  Root cause not yet investigated -- a clear, now visually-demonstrated
  target for item 2 or a follow-on fix.

### Fixed
- **Root cause of the ~90-degree halfway-line rotation error above,
  found and fixed.** Dumped every Hough-line candidate on the actual
  failing frame instead of guessing: the real halfway line *was* being
  detected (multiple candidates at ~91 degrees, 12-18px from the circle
  center) but `_detect_halfway_line_angle`'s `minLineLength` (`radius *
  1.3`) rejected all of them -- the longest real unbroken segment found
  was only ~0.89x the radius (players standing on the line, motion blur,
  and Hough's own segmentation fragment a long line into shorter
  pieces). With every candidate rejected, the code silently fell back to
  a **hardcoded `angle = 0.0`** whenever there was no previous frame to
  reuse -- wrong, not just imprecise. Fixed: `minLineLength` loosened to
  `radius * 0.7` (confirmed to recover ~91-92 degrees on all 5 tested
  frames); the silent `0.0` fallback removed (a frame with nothing to go
  on now correctly resolves nothing, matching this project's "don't
  guess when unsure" rule elsewhere). Loosening (1) introduced a real
  second-order bug caught before shipping -- a wide, flattened ellipse's
  own boundary can itself register as a spurious Hough candidate under
  the loosened threshold (confirmed via a synthetic test with no
  halfway line drawn at all) -- fixed by also tightening the
  proximity-to-center filter (`radius * 0.6` -> `radius * 0.15`),
  justified by a wide empirical margin: real detections measured
  6.7-39.3px against a ~530px radius across 5 real frames; the synthetic
  ellipse-boundary artifact measured ~60px against a 182px radius.

  **Re-validated after the fix**: on `benchmark_clip.mp4`, the rendered
  overlay's halfway line now visibly overlaps the real one (previously
  off by ~90 degrees); coverage unchanged (113/180 frames). On the
  harder `match_10min_sample.mp4` sample, the touchline hit rate improved
  modestly (5/37, 13.5%, vs. the earlier 5/55, 9.1%) -- most of that
  dataset's remaining failures are the *other* already-documented causes
  (wrong pitch arc matched as the circle, broadcast-graphic false
  positives), which this fix doesn't address on its own.

### Added
- **Data pipeline for a trained pitch-calibration keypoint model**
  (Phase 13): a full audit of already-downloaded SoccerNet labeled data
  confirmed 115 SN-GSR-2025 sequences with per-frame annotations for up
  to 26 named pitch-line features (far more than the classical
  calibrator's center-circle-only input), tagged by real match event
  type -- confirmed goals (14) and corners (11) are both well
  represented, directly answering whether those situations' distinctive
  camera angles are covered. Plus 17,309 more images from the legacy
  Calibration dataset.

  `agon.geometry.pitch_keypoints`: 38 canonical real-world keypoints (21
  straight-line features' endpoints -- circles and crossbars deliberately
  excluded, see module docstring), with point order verified directly
  against real pixel data, not assumed. Verification found a real,
  consequential bug along the way: SoccerNet clips line annotations to
  the visible frame, so a clipped "endpoint" can be an arbitrary
  frame-boundary position with no fixed real-world meaning --
  `is_frame_boundary_clipped` excludes these; fixed a homography-fit
  validation's mean error from 16.85m to 2.03m on one real frame. A
  direct visual check (real annotation pixels drawn on a real frame, no
  fitting involved) confirmed the full mapping independently of
  homography-fit noise.

  `scripts/convert_soccernet_calibration_to_pose.py` converts this into
  Ultralytics pose-format training data (validated by drawing a
  converted label back onto its image before scaling up) --
  **45,781 train + 37,164 val = 82,945 frames, 655MB** (symlinked, no
  image duplication). `scripts/train_pitch_calibration.py` mirrors
  `train_detector.py`'s established pattern (auto-device, auto-workers,
  resume, ONNX export) using Ultralytics pose-estimation mode.

  Status: 1-epoch smoke test completed successfully on real GPU hardware
  (training + validation against all 37,164 val images + ONNX export all
  succeeded). A real 30-epoch run (imgsz=960) is now in progress in the
  background, ~5 hours estimated. The inference side
  (`TrainedPitchCalibrator`) isn't built yet -- deliberately deferred
  until the real trained/exported model exists, since its ONNX
  output-decoding needs to be verified against real output, not assumed.
