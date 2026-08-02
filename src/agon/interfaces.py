"""Structural interfaces (typing.Protocol) marking the pipeline's swap points.

Concrete implementations satisfy these protocols structurally (Python
Protocols don't require explicit inheritance — matching method signatures is
enough). They document where a different backend can be substituted via
config without changing pipeline.py. See the README's modernization notes
for the specific alternatives called out below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agon.geometry.bbox import BBox, Point
from agon.io.video import Frame


@runtime_checkable
class Detector(Protocol):
    """Detects and tracks players/referees/ball across a clip.

    Two implementations: ``agon.detection.onnx_tracker.OnnxDetector``
    (onnxruntime-backed, the default -- no torch needed) and
    ``agon.detection.tracker.UltralyticsDetector`` (torch-backed,
    needs the ``[train]`` extra). Both default to ``supervision.ByteTrack``
    for tracking but accept any ``agon.detection.base.FrameTracker``
    via their ``tracker=`` argument --
    ``agon.detection.botsort_tracker.BoTSORTTracker`` is a
    BoT-SORT alternative (Kalman motion model + optional camera-motion
    compensation, needs the ``[train]`` extra too; see that module's
    docstring) for more robust tracking through occlusion (players
    clustering during set pieces).
    """

    def get_object_tracks(
        self,
        frames: list[Frame],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> dict[str, list[dict[int, dict[str, Any]]]]: ...


@runtime_checkable
class TeamClassifier(Protocol):
    """Assigns each player track to one of two teams.

    Two implementations: ``agon.team.team_assigner.TeamAssigner``
    (KMeans over raw jersey-crop pixel colors) and
    ``agon.team.embedding_team_assigner.EmbeddingTeamClassifier``
    (KMeans over small-CNN embeddings of the same crops -- holds up better
    under similar kit colors, lighting, and motion blur; see that class's
    docstring for what it still can't do).
    """

    team_colors: dict[int, Any]

    def assign_team_color(
        self, frame: Frame, player_detections: dict[int, dict[str, Any]]
    ) -> None: ...

    def get_player_team(self, frame: Frame, player_bbox: BBox, player_id: int) -> int: ...


@runtime_checkable
class PitchCalibrator(Protocol):
    """Maps pixel-space positions to pitch-space meters, per frame.

    ``calibrate()`` does any one-time, whole-clip work up front (a no-op for
    a static calibrator); ``transform_point()`` is then called per point per
    frame. ``frame_idx``/``frame_offset`` exist specifically so a *dynamic*
    calibrator can use a different transform per frame, including across
    multiple ``calibrate()`` calls in chunked/streaming processing (each
    chunk's frames start over at local index 0, so ``frame_offset`` is that
    chunk's first frame's global/match-relative index).

    Two implementations: ``agon.geometry.view_transformer.ViewTransformer``
    (one static per-video homography from four calibrated corner points --
    ignores ``frame_idx``/``frame_offset``) and
    ``agon.geometry.pitch_keypoint_calibrator.PitchKeypointCalibrator``
    (classical-CV per-frame center-circle detection -- see that class's
    docstring for what it actually solves and its real limitations).
    """

    def calibrate(self, frames: list[Frame], frame_offset: int = 0) -> None: ...

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None: ...


@runtime_checkable
class JerseyClassifier(Protocol):
    """Reads a jersey number from one player/goalkeeper crop.

    One implementation: ``agon.jersey.onnx_classifier.OnnxJerseyClassifier``
    (onnxruntime-backed, trained via ``scripts/train_jersey_classifier.py``).

    Deliberately single-frame, not track-aware -- a single crop often
    doesn't show the number at all (player facing away from the camera,
    motion blur, low resolution; see that training script's docstring for
    why this is common, not an edge case). ``agon.jersey.aggregator``
    combines many single-frame calls (one per frame a track appears in)
    into one per-track answer via confidence-weighted voting, which is
    what actually determines real-world accuracy -- see the project plan
    for why single-frame accuracy alone is the wrong metric to optimize.
    """

    def classify(self, frame: Frame, bbox: BBox) -> tuple[int | None, float]:
        """Returns (jersey_number, confidence). jersey_number is None for
        the "unknown"/illegible class; confidence is that class's own
        softmax probability either way, so a caller can threshold on it."""
        ...
