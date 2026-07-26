"""Structural interfaces (typing.Protocol) marking the pipeline's swap points.

Concrete implementations satisfy these protocols structurally (Python
Protocols don't require explicit inheritance — matching method signatures is
enough). They document where a different backend can be substituted via
config without changing pipeline.py. See the README's modernization notes
for the specific alternatives called out below.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from soccer_analysis.geometry.bbox import BBox, Point
from soccer_analysis.io.video import Frame


@runtime_checkable
class Detector(Protocol):
    """Detects and tracks players/referees/ball across a clip.

    Two implementations: ``soccer_analysis.detection.onnx_tracker.OnnxDetector``
    (onnxruntime-backed, the default -- no torch needed) and
    ``soccer_analysis.detection.tracker.UltralyticsDetector`` (torch-backed,
    needs the ``[train]`` extra). Both use ``supervision.ByteTrack`` for
    tracking; BoT-SORT is a further alternative worth benchmarking for more
    robust tracking through occlusion (players clustering during set pieces).
    """

    def get_object_tracks(
        self,
        frames: list[Frame],
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> dict[str, list[dict[int, dict[str, Any]]]]: ...


@runtime_checkable
class TeamClassifier(Protocol):
    """Assigns each player track to one of two teams.

    Current implementation: ``soccer_analysis.team.team_assigner.TeamAssigner``
    (KMeans over raw jersey-crop pixel colors). A more robust alternative is
    embedding-based clustering (e.g. SigLIP crop embeddings + KMeans), which
    holds up better under similar kit colors, lighting, and motion blur.
    """

    def assign_team_color(self, frame: Frame, player_detections: dict[int, dict[str, Any]]) -> None: ...

    def get_player_team(self, frame: Frame, player_bbox: BBox, player_id: int) -> int: ...


@runtime_checkable
class PitchCalibrator(Protocol):
    """Maps pixel-space positions to pitch-space meters, per frame.

    ``calibrate()`` does any one-time, whole-clip work up front (a no-op for
    a static calibrator); ``transform_point()`` is then called per point per
    frame. ``frame_idx`` exists specifically so a *dynamic* calibrator can
    use a different transform per frame.

    Two implementations: ``soccer_analysis.geometry.view_transformer.ViewTransformer``
    (one static per-video homography from four calibrated corner points --
    ignores ``frame_idx``) and
    ``soccer_analysis.geometry.pitch_keypoint_calibrator.PitchKeypointCalibrator``
    (classical-CV per-frame center-circle detection -- see that class's
    docstring for what it actually solves and its real limitations).
    """

    def calibrate(self, frames: list[Frame]) -> None: ...

    def transform_point(self, point: Point, frame_idx: int = 0) -> Point | None: ...
