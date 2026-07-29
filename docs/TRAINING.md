# Quick guide: deploy to Docker on the ML machine, train, deploy the result

Fine-tunes the detector (player/goalkeeper/referee/ball) and the jersey
number classifier on SoccerNet data, using the `soccer-analysis:train`
Docker image, on a separate GPU machine (training is CPU-impractical — see
the project plan).

**Before you start, know this**: the image resolves to a real CUDA-enabled
`torch` build (`torch.version.cuda == "13.0"`, confirmed), but GPU
passthrough via `--gpus all` has not been run against real GPU hardware
while building this — no GPU was available to test it. **Step 2 is exactly
that check** — run it first, don't skip it.

## 1. Get the code and build the image

```bash
git clone <this-repo-url> && cd soccer-analysis
docker build -t soccer-analysis:train .
```

`models/` and `data/` are gitignored (nothing large comes over via git) —
the base checkpoint (`models/yolo11n.pt`) auto-downloads on first training
run, so the ML machine needs internet access at least once.

## 2. Verify the GPU is actually visible

```bash
docker run --rm --gpus all soccer-analysis:train \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expect `True <N>` (`N` = your GPU count, e.g. 3). If `False`:
- Confirm the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on the host.
- Sanity-check GPU passthrough independent of this project: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`.
- Don't proceed past this point until it's `True` — training will silently fall back to CPU otherwise (impractically slow, not just slower).

## 3. Pull and prepare the training data

One command — downloads SoccerNet SN-GSR-2025 (HuggingFace, public, no NDA),
extracts it, and converts it to both training formats:

```bash
docker run --rm -v "$(pwd)/data:/data" -w /app soccer-analysis:train \
  python scripts/prepare_training_data.py --split train valid
```

- Budget **~20GB** and bandwidth-proportional time; set `HF_TOKEN` in your
  environment for faster/more reliable downloads.
- Idempotent — safe to re-run, skips what's already done.
- `-v "$(pwd)/data:/data"` is required — without it, downloaded data dies
  with the container instead of persisting.
- Add `--skip-jersey` for detection-only, or `--split test` for a small
  (~8.85GB) validation pass first.
- The raw broadcast videos (a separate SoccerNet asset) are **not** what
  this pulls, deliberately — they carry no bounding-box labels, so they're
  not usable training data for this pipeline. This is the real thing.

## 4. Train

**Detection:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/data" -v "$(pwd)/models:/app/models" \
  -w /app soccer-analysis:train \
  python scripts/train_detector.py \
    --data /data/soccernet_yolo/dataset.yaml \
    --imgsz 960 --epochs 50 --batch 32 --device 0,1,2
```

**Jersey number classifier:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/data" -v "$(pwd)/models:/app/models" \
  -w /app soccer-analysis:train \
  python scripts/train_jersey_classifier.py \
    --data /data/soccernet_jersey --epochs 30 --device 0,1,2
```

- `--device 0,1,2` uses all 3 GPUs (drop it to auto-select one).
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

Weights land in `models/` on the host (via the volume mount): `best.pt` and
`best.onnx`, under `runs/train/<name>/weights/`.

**Set `detection_imgsz` to match whatever `--imgsz` you trained with.** A
mismatch fails loudly (a clear onnxruntime shape error), not silently — but
you still need to set it correctly for the new resolution to do anything:

```yaml
# your-config.yaml
pipeline:
  detection_imgsz: 960   # match --imgsz from training
```

```bash
soccer-analysis --input <video> --model models/best.onnx \
  --calibration <calibration.json> --config your-config.yaml
```

## Known gaps (honest, not hidden)

- **Pitch calibration model**: not built yet (only the data-prep script
  exists). Detection + jersey number training are the complete, validated
  paths today.
- **Jersey classifier isn't wired into inference yet.** Training is done
  and validated; `ObjectRecord.jersey_number` exists in the export schema
  ready to receive it, but no pipeline code loads a trained jersey model
  and calls it yet — a small follow-up once you have a checkpoint worth
  wiring in.
- **Frame-filter clock-reliability issue** (unrelated to training, but
  relevant if you also run the main pipeline here): known limitation,
  queued for a future fix — see the project plan.

## Troubleshooting

- **`onnxruntime...InvalidArgument: Got invalid dimensions`** at inference:
  `detection_imgsz` doesn't match the ONNX export resolution — fix per Step 5.
- **Stale-cache crash** (`IndexError: list index out of range`) after
  changing frame-filter config on a previously-cached streaming run: the
  detection stub cache doesn't know frame-classification logic changed.
  Clear the cache directory and rerun.
- **No internet on the training machine** for the auto-downloaded base
  checkpoint: pass `--base-model` with a path to one you've copied over
  manually.
