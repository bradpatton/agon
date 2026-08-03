"""EasyOCR-backed jersey number reader.

Satisfies ``agon.interfaces.JerseyClassifier`` -- a drop-in replacement
for ``agon.jersey.onnx_classifier.OnnxJerseyClassifier`` (see
``agon.jersey.aggregator`` for why a single-frame reader's output always
gets aggregated across a track, never trusted standalone).

Why this exists instead of fixing the trained classifier: the trained
classifier scored *worse* than a trivial "always guess the most common
class" baseline (2.4% vs. 16.4% -- see CHANGELOG). Root cause, confirmed
against the raw SoccerNet label files: SN-GSR-2025's ``attributes.jersey``
is assigned per *track*, not per frame -- every annotation instance of a
given track carries the same jersey value regardless of whether the
number is actually visible in that specific frame (checked across 167
real tracks across 8 sequences: zero showed more than one distinct value).
So a large fraction of the training crops show no visible number at all
while being confidently labeled with a real digit, teaching the
classifier to associate irrelevant visual noise (pose, kit color,
background) with arbitrary numbers rather than actually reading digits.

EasyOCR sidesteps this entirely: it's a general-purpose, pretrained
scene-text-recognition model, not trained on any of this project's noisy
per-frame labels at all. Empirically validated against real crops with
known ground truth (see CHANGELOG): correctly read clearly-visible
numbers at 93-100% confidence, and correctly returned nothing at all on a
crop where the player faced away and no number was visible -- exactly the
"only label if confident" behavior this project needs. Real, observed
failure modes, not hypothetical: a 93%-confidence misread (true 36 read
as 35, a classic 6/5 digit confusion) and lower-confidence misreads on
multi-digit numbers. Confidence alone doesn't guarantee correctness --
that's why ``agon.jersey.aggregator``'s ``min_votes`` (requiring several
frames to agree, not just one high-confidence read) exists and matters
here specifically.

Needs the ``[train]`` extra (``easyocr`` pulls in ``torch``) -- this is a
deliberately optional, torch-backed capability like ``UltralyticsDetector``
and ``BoTSORTTracker``, not part of the core onnxruntime-only install.
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from agon.config import resolve_device
from agon.geometry.bbox import BBox
from agon.io.video import Frame

logger = logging.getLogger(__name__)

_JERSEY_PATTERN = re.compile(r"^\d{1,2}$")
_UPSCALE_FACTOR = 6
"""EasyOCR needs meaningfully more pixels than a native player crop has --
validated empirically: a crop with a clearly-visible "7" wasn't detected
at all below 6x upscaling, and was read correctly (93% confidence) at 6x.
Lower factors (2x-4x) missed real, visible numbers in testing."""


class EasyOcrJerseyReader:
    def __init__(self, min_confidence: float = 0.0, device: str | None = None):
        """``min_confidence``: below this, ``classify()`` returns
        ``(None, confidence)`` even for a numeric read. 0.0 (default)
        never abstains here -- ``agon.jersey.aggregator`` applies its own
        track-level ``min_confidence``/``min_votes`` regardless (see that
        module's docstring for why confidence alone isn't a safe per-frame
        gate), so a low per-call value is fine; set this higher only if
        you also want single-frame calls used outside that aggregator to
        self-filter.

        ``device``: 'cuda', 'cpu', or None to auto-detect via
        ``agon.config.resolve_device`` (EasyOCR has no MPS support, so an
        auto-detected 'mps' is treated as 'cpu' here).
        """
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "EasyOcrJerseyReader needs the 'train' extra (pulls in easyocr + torch): "
                "pip install 'agon[train]'"
            ) from e

        resolved = resolve_device(device)
        use_gpu = resolved == "cuda"
        self._reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.min_confidence = min_confidence

    def classify(self, frame: Frame, bbox: BBox) -> tuple[int | None, float]:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        upscaled = self._upscale(crop)
        results = self._reader.readtext(upscaled, allowlist="0123456789")
        if not results:
            return None, 0.0

        # Highest-confidence detected region, not the first -- readtext can
        # return multiple spurious regions per crop (sponsor logos, shorts
        # numbers, background digits on hoardings).
        _, text, confidence = max(results, key=lambda r: r[2])

        if not _JERSEY_PATTERN.match(text) or confidence < self.min_confidence:
            return None, confidence
        return int(text), confidence

    def _upscale(self, crop: Frame) -> Frame:
        h, w = crop.shape[:2]
        resized = cv2.resize(
            crop, (w * _UPSCALE_FACTOR, h * _UPSCALE_FACTOR), interpolation=cv2.INTER_CUBIC
        )
        return resized.astype(np.uint8)
