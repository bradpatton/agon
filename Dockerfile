# CPU image for the torch-dependent paths this project's default install
# deliberately doesn't need: UltralyticsDetector and BoT-SORT tracking (via
# boxmot). See the project plan's Phase 2.6 for why this exists -- on some
# platforms (e.g. Intel macOS, as of this writing) current torch, current
# onnxruntime, and boxmot simply have no wheels at all, so this is the
# reliable way to actually run and validate those paths rather than only
# sanity-checking them in isolation. The default onnxruntime-only install
# (see pyproject.toml) doesn't need this image.
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

ENTRYPOINT ["soccer-analysis"]
CMD ["--help"]
