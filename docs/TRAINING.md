# Quick guide: deploy to Docker on the ML machine, train, deploy the result

Fine-tunes the detector (player/goalkeeper/referee/ball) and the jersey
number classifier on SoccerNet data, using the `agon:train`
Docker image, on a separate GPU machine (training is CPU-impractical — see
the project plan).

**Before you start, know this**: the image resolves to a real CUDA-enabled
`torch` build (`torch.version.cuda == "13.0"`, confirmed), but GPU
passthrough via `--gpus all` has not been run against real GPU hardware
while building this — no GPU was available to test it. **Step 2 is exactly
that check** — run it first, don't skip it.

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
docker run --rm -v "$(pwd)/data:/data" -w /app agon:train \
  python scripts/prepare_training_data.py --split train valid
```

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
- `-v "$(pwd)/data:/data"` is required — without it, downloaded data dies
  with the container instead of persisting.
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
docker run --rm --gpus all \
  -v "$(pwd)/data:/data" -v "$(pwd)/models:/app/models" \
  -w /app agon:train \
  python scripts/train_detector.py \
    --data /data/soccernet_yolo/dataset.yaml \
    --imgsz 960 --epochs 50 --batch 32 --device 0,1,2
```

**Jersey number classifier:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/data" -v "$(pwd)/models:/app/models" \
  -w /app agon:train \
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
agon --input <video> --model models/best.onnx \
  --calibration <calibration.json> --config your-config.yaml
```

## Known gaps (honest, not hidden)

- **Pitch calibration model**: not built yet (only the data-prep scripts
  exist, now fed by two sources — see Step 3). Detection + jersey number
  training are the complete, validated, trainable paths today.
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
