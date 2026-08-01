# Image for the torch-dependent paths this project's default install
# deliberately doesn't need: UltralyticsDetector, BoT-SORT tracking (via
# boxmot), and model training/fine-tuning (scripts/train_detector.py,
# scripts/train_jersey_classifier.py). See the project plan's Phase 2.6 for
# why this exists -- on some platforms (e.g. Intel macOS, as of this
# writing) current torch, current onnxruntime, and boxmot simply have no
# wheels at all, so this is the reliable way to actually run and validate
# those paths rather than only sanity-checking them in isolation. The
# default onnxruntime-only install (see pyproject.toml) doesn't need this
# image.
#
# Despite the plain python:3.11-slim base (no CUDA toolkit baked in): the
# Linux torch wheel this resolves to (see uv.lock) pulls in its own bundled
# CUDA runtime (nvidia-cudnn-cu13, nvidia-nccl-cu13, etc. as real
# dependencies, confirmed in uv.lock) -- the modern PyPI-wheel pattern of
# not needing a system-wide CUDA toolkit install, just a compatible host
# NVIDIA driver + the NVIDIA Container Toolkit for GPU passthrough
# (`docker run --gpus all`). Confirmed working end-to-end on real GPU
# hardware (2026-08-01, 2x RTX 3090, Ubuntu) -- see docs/TRAINING.md for
# the verification command and troubleshooting if a *different* machine's
# GPU doesn't show up the same way.
FROM python:3.11-slim

# libgl1/libglib2.0-0: runtime deps for opencv-python's video/image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# --extra train pulls in torch/ultralytics/boxmot -- the whole point of this
# image (see header comment). --frozen: install exactly what's locked,
# don't re-resolve.
RUN uv sync --extra train --frozen

ENV PATH="/app/.venv/bin:${PATH}"

# No ENTRYPOINT (deliberately -- see docs/TRAINING.md): this image is used
# both as the inference CLI (`docker run <image>` with no args runs the
# default CMD below) and as a training environment (`docker run <image>
# python scripts/train_detector.py ...`). An ENTRYPOINT pinned to
# `agon` would silently prepend itself to that second form,
# turning a training command into a broken CLI invocation -- confirmed by
# hitting exactly that failure mode, which is why this is CMD, not
# ENTRYPOINT (CMD is replaced outright by any command given to `docker
# run`; ENTRYPOINT's argv is only ever appended to).
CMD ["agon", "--help"]
