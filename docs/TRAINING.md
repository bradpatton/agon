# Quick guide: deploy to Docker on the ML machine, train, deploy the result

Fine-tunes the detector (player/goalkeeper/referee/ball) and the jersey
number classifier on SoccerNet data, using the `agon:train`
Docker image, on a separate GPU machine (training is CPU-impractical — see
the project plan).

**GPU passthrough is confirmed working end-to-end** (2026-08-01, against a
real 2x RTX 3090 Ubuntu machine — `torch.cuda.is_available()` returned
`True` with both GPUs correctly enumerated inside the container). Earlier
drafts of this guide carried this as an unverified assumption; it no
longer is. **Step 2 is still worth running on any new machine** — it's
the fastest way to catch a host-level GPU/driver/toolkit misconfiguration
before spending time on data prep.

**Prerequisites on the ML machine, before Step 1** (not optional — `--gpus
all` fails outright, before any Python runs, if these aren't in place):
- Docker Engine 19.03+ (for the `--gpus` flag itself).
- A current NVIDIA driver (CUDA 13-compatible — this image's `torch` build
  bundles its own CUDA 13 runtime, but still needs a host driver new enough
  to support it; an older driver already on the machine may need updating).
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed **and** the Docker daemon configured for it (`sudo nvidia-ctk
  runtime configure --runtime=docker && sudo systemctl restart docker` on
  most Linux distros) — without this, `docker run --gpus all` errors
  immediately (`could not select device driver` / `unknown or invalid
  runtime name: nvidia`), it does not just fall back to CPU.
- A few GB of free disk for the image build alone (`torch`'s Linux wheel
  pulls in the full CUDA toolkit as dependencies — cuBLAS, cuDNN, NCCL,
  etc.) — separate from the ~20GB+ Step 3 budgets for training data.

**On Windows, read this first.** Every command below is written for
bash/zsh (macOS/Linux, or Git Bash on Windows). If your ML machine's
terminal is **Command Prompt (`cmd.exe`)** or **PowerShell**, `$(pwd)` is
not valid syntax there — `cmd.exe` doesn't expand it at all and passes the
literal text `$(pwd)/data` to Docker, which then fails with `includes
invalid characters for a local volume name` (a real error hit and fixed
while writing this guide, not a hypothetical). Two substitutions to make
throughout every command below that uses `-v "$(pwd)/...`:

| Shell | Current directory | Line continuation |
|---|---|---|
| bash/zsh (this guide's default) | `$(pwd)` | `\` |
| Command Prompt | `%cd%` | `^` |
| PowerShell | `${PWD}` | `` ` `` (backtick) |

Example — Step 1's build command needs no change (no volume mount), but
any later command like `docker run --rm -v "$(pwd)/data:/app/data" ...`
becomes, on Command Prompt:
```cmd
docker run --rm -v "%cd%/data:/app/data" -w /app agon:train python scripts/prepare_training_data.py --split train valid
```
When in doubt, sidestep the substitution entirely and hardcode the
absolute path instead (works identically in every shell):
```
docker run --rm -v "C:/full/path/to/agon/data:/app/data" -w /app agon:train python scripts/prepare_training_data.py --split train valid
```
Also worth checking on Windows specifically: GPU passthrough (`--gpus
all`, Step 2) needs Docker Desktop's **WSL2 backend** with Linux containers
— Windows-containers mode doesn't support it. If Step 2 fails outright
(not just prints `False`), confirm Docker Desktop is set to Linux
containers with the WSL2 engine before chasing the NVIDIA Container
Toolkit steps above, which target Linux hosts specifically.

## 1. Get the code and build the image

```bash
git clone https://github.com/bradpatton/agon.git && cd agon
docker build -t agon:train .
```

`models/` and `data/` are gitignored (nothing large comes over via git) —
the base checkpoint (`models/yolo11n.pt`) auto-downloads on first training
run, so the ML machine needs internet access at least once.

## 2. Verify the GPU is actually visible

```bash
docker run --rm --gpus all agon:train \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expect `True <N>` (`N` = your GPU count, e.g. 3). If `False`:
- Confirm the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on the host.
- Sanity-check GPU passthrough independent of this project: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`.
- Don't proceed past this point until it's `True` — training will silently fall back to CPU otherwise (impractically slow, not just slower).

## 3. Pull and prepare the training data

One command — downloads from every SoccerNet source this project can
currently train from, extracts, and converts to training format:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -w /app agon:train \
  python scripts/prepare_training_data.py --split train valid
```

**Mount at `/app/data`, not `/data`.** `prepare_training_data.py` resolves
its output relative to its own location inside the image (`/app`), so it
always writes under `/app/data/...` regardless of what you mount — mounting
anywhere else (including plain `/data`) means the container silently
downloads and converts everything into its own throwaway filesystem, and
all of it is gone the moment `--rm` removes the container. This is a real
mistake this guide made in an earlier draft; if a training run mysteriously
can't find its dataset in Step 4, this mismatch is the first thing to check.

Three sources, combined into the same output directories regardless of
which one a given example came from:
- **SN-GSR-2025** (HuggingFace, public, no NDA) — detection boxes +
  jersey numbers + pitch lines, from a curated subset of games.
- **Legacy Calibration** — standalone action/replay images with pitch-line
  annotations, curated from many different games/broadcasts (more scene
  diversity per image than GSR's video-sequence frames, genuinely
  complementary, not just more of the same).
- **Legacy Jersey-2023** — the original, richer jersey-number dataset:
  full per-player tracklets (dozens of crops per player over time, not
  one static crop), sampled evenly per tracklet rather than using every
  frame (they're highly redundant at video frame rate).

All three were originally assumed NDA-gated and were nearly skipped —
turned out they download fine with SoccerNet's **generic public password**
(distinct from the personal NDA password issued for raw broadcast video),
confirmed by actually downloading and inspecting real data, not just
checking HTTP status. See `download_soccernet_legacy.py`'s docstring.

- Budget **~20GB+** and bandwidth-proportional time (more if you add
  `legacy-jersey`, which is many small files and slower to extract); set
  `HF_TOKEN` in your environment for faster/more reliable HuggingFace
  downloads.
- Idempotent — safe to re-run, skips what's already done.
- `-v "$(pwd)/data:/app/data"` is required — without it, downloaded data
  dies with the container instead of persisting.
- `--sources gsr` (or `legacy-calibration`/`legacy-jersey`) to pull just
  one. `--skip-jersey` skips GSR's jersey conversion specifically. `--split
  test` for a small validation pass across all three before committing to
  `train`+`valid`.
- The raw broadcast videos (a separate SoccerNet asset, genuinely NDA-gated
  with your *personal* password) are **not** what this pulls, deliberately
  — they carry no bounding-box labels of their own, so they're not usable
  training data for this pipeline on their own.
- **Not pulled by this script, available but not yet integrated**: the
  full Tracking dataset (would extend detection training with more volume,
  beyond what GSR already covers) and Re-Identification (a genuinely new
  capability, not something this project does today). Both confirmed
  downloadable the same way if you want to extend this further — see the
  project plan.

## 4. Train

**Detection:**
```bash
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  -w /app agon:train \
  python scripts/train_detector.py \
    --data /app/data/soccernet_yolo/dataset.yaml \
    --project /app/models/runs/train \
    --imgsz 960 --epochs 50
```

**Jersey number classifier:**
```bash
docker run --rm --gpus all --ipc=host \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  -w /app agon:train \
  python scripts/train_jersey_classifier.py \
    --data /app/data/soccernet_jersey \
    --project /app/models/runs/train \
    --epochs 30
```

**Pitch-calibration keypoint model:** first convert the same SoccerNet
line-annotation sources Step 3 already pulled into Ultralytics pose-format
labels (one call per source; both write into the same output directory,
so both a GSR extraction and the legacy Calibration dataset can feed one
combined training set):
```bash
docker run --rm -v "$(pwd)/data:/app/data" -w /app agon:train \
  python scripts/convert_soccernet_calibration_to_pose.py \
    gsr /app/data/soccernet/gamestate-2025/train_extracted \
    /app/data/soccernet_pose train
docker run --rm -v "$(pwd)/data:/app/data" -w /app agon:train \
  python scripts/convert_soccernet_calibration_to_pose.py \
    legacy /app/data/soccernet/calibration/train \
    /app/data/soccernet_pose train
# repeat both for the valid/val split
```
Then train (same auto-device/auto-workers/AutoBatch behavior as the two
scripts above; Ultralytics pose-estimation mode, not a custom loop):
```bash
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  -w /app agon:train \
  python scripts/train_pitch_calibration.py \
    --data /app/data/soccernet_pose/dataset.yaml \
    --project /app/models/runs/train \
    --imgsz 960 --epochs 30
```
Real result from the run that validated this path (2026-08-03, 82,945
combined frames, RTX 3090, ~7.3 hours): pose mAP50 **0.842**, mAP50-95
**0.379** on the held-out validation set, still improving at epoch 30 (no
plateau) — a longer run would likely help further. `best.pt`/`best.onnx`
land under `models/runs/train/pitch_calibration/weights/`, same layout as
the other two training runs. **The inference side
(`TrainedPitchCalibrator`) doesn't exist yet** — decoding this checkpoint's
keypoint output into a homography and wiring it into
`PipelineConfig.calibration_mode` is still open; see the project plan.

**`--ipc=host` is required, not optional.** PyTorch's DataLoader passes
batches between worker processes through `/dev/shm` (shared memory) —
Docker defaults that to a tiny 64MB, nowhere near enough for real image
batches, and training crashes partway in with `RuntimeError: unable to
allocate shared memory(shm)... No space left on device`, which reads like
a disk-space problem but isn't. `--ipc=host` shares the host's IPC
namespace (PyTorch's own recommended fix) rather than trying to guess a
`--shm-size` value that's enough for every batch size/resolution
combination.

**After training, validate the ONNX export against the reference model
before trusting it.** `OnnxJerseyClassifier` reimplements Ultralytics'
classification preprocessing by hand rather than calling Ultralytics
itself, so a preprocessing mismatch would degrade accuracy silently — a
real bug of exactly this kind was caught and fixed this way (see
CHANGELOG). Run:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/src:/app/src" -v "$(pwd)/scripts:/app/scripts" \
  -w /app agon:train \
  python scripts/validate_jersey_onnx.py \
    --pt /app/models/runs/train/jersey-gsr/weights/best.pt \
    --onnx /app/models/runs/train/jersey-gsr/weights/best.onnx \
    --data /app/data/soccernet_jersey/train
```
A healthy result looks like every (or nearly every) sampled crop's ONNX
prediction matching the reference model's prediction, with a small
(<0.05) confidence gap. Wide disagreement or a large confidence gap means
`OnnxJerseyClassifier._preprocess()` doesn't match this checkpoint's real
preprocessing — fix that before using the export in the pipeline.

**`--project /app/models/runs/train` is required, not optional.** Both
scripts default `--project` to `runs/train`, a path relative to the
container's working directory (`/app`) — that's `/app/runs/train`, which
isn't under either mounted volume. Without overriding it explicitly, a
training run completes successfully, prints real metrics, and then loses
every weight file the moment `--rm` removes the container. `--base-model
models/yolo11n.pt` (detection's default) is fine as-is — it's already
relative to the mounted `/app/models`.

- **GPU selection is automatic** — both scripts detect every visible CUDA
  GPU (`torch.cuda.device_count()`) and use all of them by default, so the
  command above doesn't need editing as GPUs are added or removed from the
  machine. A hardcoded `--device 0,1,2` was here in an earlier draft of
  this guide and broke on a real machine with only 1 GPU configured
  ("invalid CUDA device" — index 1/2 didn't exist). Pass `--device`
  explicitly only if you want to *restrict* training to a subset, e.g.
  `--device 0` to use just the first GPU.
- **Batch size is also automatic on a single GPU** (`--batch -1`, the
  default) — Ultralytics' AutoBatch runs a few trial passes and picks the
  largest batch that actually fits in whatever VRAM this GPU has. A
  hardcoded `--batch 32` was here in an earlier draft and OOM'd
  (`torch.AcceleratorError: CUDA error: out of memory`) on a real GPU with
  less VRAM than whatever card 32 was sized for. Pass `--batch <N>`
  explicitly if you want a fixed, reproducible batch size across runs
  instead.
- **AutoBatch does not work at all across multiple GPUs** — confirmed via
  Ultralytics' own error on real 2-GPU hardware: `AutoBatch with batch<1
  not supported for Multi-GPU training`. Both scripts detect this
  automatically and fall back to a conservative explicit batch (8 per
  GPU — a valid multiple of the GPU count, which Ultralytics requires) so
  training still starts rather than crashing; you'll see a printed note
  when this fallback kicks in. Pass `--batch <N>` yourself (a multiple of
  your GPU count) if 8-per-GPU isn't the right tradeoff for your cards.
- **On a small-VRAM GPU, AutoBatch can still OOM a few iterations in**
  (not on the first batch) — hit for real: training ran cleanly for ~15
  iterations, memory climbing, then died in the DataLoader's pin-memory
  step. That pattern points at CUDA memory fragmentation or worst-case
  batches (soccer frames vary a lot in player-instance count, so a
  higher-instance batch costs more than whatever AutoBatch benchmarked)
  rather than "never enough VRAM at all." Three levers, roughly in order
  of how non-invasive they are:
  1. `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in the
     command above) — PyTorch's own fix for exactly this
     works-for-a-while-then-OOMs fragmentation pattern.
  2. `--workers <N>` — both training scripts now auto-detect a worker
     count from available *system* RAM by default (not a fixed number —
     see `_auto_workers` in either script), so this usually doesn't need
     manual tuning at all; pass it explicitly only to go even lower than
     the auto-detected value.
  3. A smaller fixed `--batch <N>` instead of AutoBatch, or a lower
     `--imgsz` (960 costs ~2.25x the memory of 640) if the GPU genuinely
     doesn't have the VRAM for 960px training even with 1-2 applied —
     a real tradeoff against small-object (ball) detection quality, not
     a free fix.
- `--imgsz 960`+ matters specifically for the ball — it's a tiny object in
  broadcast frames, and this project's own runs (a 5-epoch/CPU/small-subset
  proof of concept) already show ball detection as the clear weak point
  (mAP50 0.25 vs. 0.83-0.97 for player/goalkeeper/referee) even after real
  fine-tuning. Higher resolution costs ~quadratic compute (1280 ≈ 4x 640).
  Budget ~1-2 hours at 640 on a single 3090; more at higher resolution,
  less split across 3 GPUs.
- ONNX export happens automatically at the end, at the same `--imgsz` you
  trained with.
- Watch the per-epoch loss log for divergence or stalling — standard
  training hygiene.

## 5. Deploy the trained model

Weights land in `models/` on the host (via the volume mount, given the
`--project /app/models/runs/train` override from Step 4): `best.pt` and
`best.onnx`, under `models/runs/train/<name>/weights/`.

**Set `detection_imgsz` to match whatever `--imgsz` you trained with.** A
mismatch fails loudly (a clear onnxruntime shape error), not silently — but
you still need to set it correctly for the new resolution to do anything:

```yaml
# your-config.yaml
pipeline:
  detection_imgsz: 960   # match --imgsz from training
```

```bash
agon --input <video> --model models/best.onnx \
  --calibration <calibration.json> --config your-config.yaml
```

**The pitch-calibration checkpoint deploys the same way, via
`calibration_mode: trained`** (no `detection_imgsz`-style resolution
setting needed -- `TrainedPitchCalibrator` always runs at the 960x960 it
was exported at):

```yaml
# your-config.yaml
pipeline:
  calibration_mode: "trained"
  pitch_calibration_model_path: models/runs/train/pitch_calibration/weights/best.onnx
```

Real, measured tradeoff (see `scripts/validate_trained_pitch_calibrator.py`
and the README's "Pitch calibration" section): much higher coverage than
`calibration_mode: dynamic` on typical wide match footage (92.5% vs. 13.7%
on one real sample), but *lower* on a tight center-circle-only shot (13.3%
vs. 62.8%) -- pick based on the footage you actually have, not blindly.

## Known gaps (honest, not hidden)

- **Pitch calibration model is trained and wired in
  (`calibration_mode: "trained"`), but isn't the default yet.** Real
  result: pose mAP50 0.842 on held-out data, and 92.5% pitch-position
  coverage on typical wide match footage (vs. 13.7% for `dynamic` on the
  same window) -- but *lower* coverage than `dynamic` on tight
  center-circle-only shots (13.3% vs. 62.8%), so it's a real, measured
  tradeoff, not a strict upgrade. See Step 5's deploy subsection and the
  README's "Pitch calibration" section for the full numbers. Not yet
  wired into `hybrid`'s fallback chain, and the export schema has no
  camera-pose field yet even though `agon.geometry.camera_pose` can now
  produce one for every `trained`-resolved frame (100% success rate on
  the same sample). `PipelineConfig.calibration_mode = "hybrid"` (dynamic
  + static, no model needed) remains the recommended default when you
  don't have or don't want to manage the pitch-calibration checkpoint --
  see the project plan's Phase 12.
- **The trained jersey classifier (`OnnxJerseyClassifier`) is not
  recommended — use `PipelineConfig.jersey_backend = "ocr"` instead.**
  Root cause found: SN-GSR-2025's `attributes.jersey` label is assigned
  per *track*, not per frame, so a large fraction of training crops show
  no visible number at all while being confidently labeled with a real
  digit anyway. Training-time validation accuracy (2.4% top1) was
  actually worse than trivially guessing the most common class (16.4%).
  `agon.jersey.ocr_reader.EasyOcrJerseyReader` (needs the `[train]`
  extra) replaces it with a pretrained scene-text reader that needs no
  training on this project's data at all — validated at 93-100%
  confidence on real, clearly-visible crops, correctly abstaining when
  the number isn't visible. The `onnx` backend is kept only for anyone
  with an existing checkpoint; the training scripts and
  `scripts/validate_jersey_onnx.py` below are still accurate for that
  path, just not the recommended one anymore.
- **Frame-filter clock-reliability issue** (unrelated to training, but
  relevant if you also run the main pipeline here): known limitation,
  queued for a future fix — see the project plan.

## Troubleshooting

- **`docker: Error response from daemon: could not select device driver`**
  or **`unknown or invalid runtime name: nvidia`** on `docker run --gpus
  all`: the NVIDIA Container Toolkit isn't installed/configured on this
  host — see Prerequisites above. This is a host setup issue, not something
  wrong with the image.
- **Step 4 fails immediately with the dataset config not found**
  (`--data /app/data/soccernet_yolo/dataset.yaml`): almost always means
  Step 3 was run with the wrong mount (`/data` instead of `/app/data`) and
  its output never left the container. Re-run Step 3 with the corrected
  mount and confirm `data/soccernet_yolo/dataset.yaml` actually exists on
  the *host* before moving on.
- **Training finishes with real metrics printed, but `models/runs/train/`
  is empty afterward**: `--project` wasn't set (or was set to something not
  under `/app/models`) — see the note under Step 4. The run itself worked;
  its output was just written outside any mounted volume and discarded with
  the container.
- **`torch.AcceleratorError: CUDA error: out of memory`** immediately /
  within the first batch: the batch size (and/or `--imgsz`) is too large
  for this GPU's VRAM. Default `--batch -1` (AutoBatch) should avoid this
  on its own — if you still hit it, you've likely overridden `--batch`
  with a fixed value; drop the override.
- **Same error, but a number of iterations into training, not the first
  batch**: see the "small-VRAM GPU" bullet under Step 4 — try
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then `--workers`,
  then a smaller fixed `--batch`/`--imgsz`, in that order.
- **`ValueError: AutoBatch with batch<1 not supported for Multi-GPU
  training`**: shouldn't happen anymore — both scripts now detect
  multi-GPU and pick an explicit batch automatically (see the AutoBatch
  bullet under Step 4). If you still hit this, you're likely running an
  image built before that fix; rebuild (`docker build -t agon:train .`)
  and confirm `git log -1` shows a commit at or after the "multi-GPU
  AutoBatch fallback" change.
- **`RuntimeError: unable to allocate shared memory(shm)... No space left
  on device`** partway through a training run: not actual disk space —
  Docker's default 64MB `/dev/shm` is too small for PyTorch's DataLoader
  workers. Add `--ipc=host` to the `docker run` command (Step 4 already
  has it; add it too if you've customized the command).
- **Container exits with no error, just `Exited (137)`** (`docker ps -a`):
  that's SIGKILL, almost always the Linux kernel's own OOM killer for
  *system* RAM, not a CUDA/GPU error at all — confirm with `dmesg | grep
  -i "out of memory\|oom"` on the host. Both training scripts now
  auto-detect a safe `--workers` count from available RAM by default
  (see the AutoBatch/`--workers` bullet under Step 4), which should
  prevent this on its own; if you still hit it, pass `--workers` even
  lower explicitly, or `--resume` from the last checkpoint once it's
  resolved (both scripts support this).
- **`onnxruntime...InvalidArgument: Got invalid dimensions`** at inference:
  `detection_imgsz` doesn't match the ONNX export resolution — fix per Step 5.
- **Stale-cache crash** (`IndexError: list index out of range`) after
  changing frame-filter config on a previously-cached streaming run: the
  detection stub cache doesn't know frame-classification logic changed.
  Clear the cache directory and rerun.
- **No internet on the training machine** for the auto-downloaded base
  checkpoint: pass `--base-model` with a path to one you've copied over
  manually.
