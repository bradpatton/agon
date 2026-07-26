"""Video reading/writing helpers.

``read_video`` currently loads the whole clip into memory as a list of
frames, same as the original tutorial. That's fine for short clips but will
not scale to a full 90-minute match — see the streaming/windowed pipeline
stretch goal in the project README before pointing this at long footage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Frame = npt.NDArray[np.uint8]


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
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video file: {video_path}")

    logger.info("Read %d frames from %s", len(frames), video_path)
    return frames


def save_video(frames: list[Frame], output_path: str | Path, fps: float = 24.0) -> None:
    """Write a list of frames out as a video file."""
    if not frames:
        raise ValueError("Cannot save an empty list of frames")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    logger.info("Wrote %d frames to %s", len(frames), output_path)
