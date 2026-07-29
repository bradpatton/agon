"""Video reading/writing helpers.

``read_video`` loads the whole clip into memory as a list of frames --
fine for short clips, but at native broadcast resolution/framerate a full
~90-minute match is on the order of terabytes of frames, which will not
fit in memory. ``iter_video_chunks`` is the streaming alternative: it never
holds more than one bounded chunk of frames at a time. See
``agon.pipeline.run_pipeline_streaming`` for the pipeline built
on top of it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Frame = npt.NDArray[np.uint8]


@dataclass
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


def get_video_info(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    try:
        return VideoInfo(
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 24.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_video(video_path: str | Path) -> list[Frame]:
    """Read every frame of a video into memory, in order."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frames: list[Frame] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)  # type: ignore[arg-type]  # cv2 stubs: Mat|ndarray vs our uint8 alias
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video file: {video_path}")

    logger.info("Read %d frames from %s", len(frames), video_path)
    return frames


def iter_video_chunks(video_path: str | Path, chunk_size: int) -> Iterator[list[Frame]]:
    """Yields the video's frames in bounded-size chunks, never holding more
    than one chunk in memory. Same frame ordering/content as ``read_video``,
    just delivered incrementally. The final chunk may be shorter than
    ``chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    try:
        chunk: list[Frame] = []
        any_frames = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            any_frames = True
            chunk.append(frame)  # type: ignore[arg-type]  # cv2 stubs: Mat|ndarray vs our uint8 alias
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
        if not any_frames:
            raise ValueError(f"No frames could be read from video file: {video_path}")
    finally:
        cap.release()


class IncrementalVideoWriter:
    """Context manager wrapping cv2.VideoWriter so callers can write frames
    chunk-by-chunk instead of accumulating a full output list first. See
    ``save_video``'s docstring for why H.264/.mp4 (``avc1``) is the default
    and why ``isOpened()`` is checked explicitly rather than trusting a
    silent no-op write.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        fourcc: str = "avc1",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # macOS's AVFoundation backend (used when the opencv-python build has
        # no FFMPEG support -- see save_video's docstring) refuses to open a
        # VideoWriter at a path that already exists ("AVAssetWriter status:
        # Cannot Save", isOpened() False, no other indication why) instead of
        # truncating/overwriting it -- confirmed directly against this exact
        # failure re-running the pipeline at the same --output-dir. Remove
        # any stale file from a previous run first so re-running with the
        # same output path doesn't fail for a reason that has nothing to do
        # with this run's frames.
        self.output_path.unlink(missing_ok=True)
        self._frames_written = 0

        fourcc_code = cv2.VideoWriter.fourcc(*fourcc)
        width, height = frame_size
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc_code, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open for {self.output_path} with fourcc={fourcc!r}. "
                "This usually means the local OpenCV build lacks a compatible video backend "
                "(check cv2.getBuildInformation() for 'Video I/O') -- try a different "
                "fourcc/extension, or use PyAV instead of cv2 for video writing."
            )

    def write(self, frames: list[Frame]) -> None:
        for frame in frames:
            self._writer.write(frame)
        self._frames_written += len(frames)

    def close(self) -> None:
        self._writer.release()
        logger.info("Wrote %d frames to %s", self._frames_written, self.output_path)

    def __enter__(self) -> IncrementalVideoWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def save_video(
    frames: list[Frame], output_path: str | Path, fps: float = 24.0, fourcc: str = "avc1"
) -> None:
    """Write a list of frames out as an H.264 .mp4 by default.

    OpenCV's video I/O backend is highly platform/build-dependent — some
    ``opencv-python`` wheels ship without FFMPEG support at all (only
    AVFoundation on macOS, for instance), in which case codec/container
    combinations that work elsewhere (e.g. XVID/.avi) silently fail to open
    and ``cv2.VideoWriter.write()`` becomes a silent no-op rather than an
    error. H.264/.mp4 (``avc1``) is the most broadly supported combination
    across builds; ``IncrementalVideoWriter`` (which this uses internally)
    still checks ``isOpened()`` explicitly and raises rather than reporting
    success with nothing written, since that failure mode is exactly what
    caused this to need fixing in the first place.
    """
    if not frames:
        raise ValueError("Cannot save an empty list of frames")

    height, width = frames[0].shape[:2]
    with IncrementalVideoWriter(output_path, fps, (width, height), fourcc) as writer:
        writer.write(frames)
