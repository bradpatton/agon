"""Draws tracked players/referees/ball and match stats onto video frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from soccer_analysis.geometry.bbox import BBox, get_bbox_width, get_center_of_bbox
from soccer_analysis.io.video import Frame


def draw_ellipse(
    frame: Frame, bbox: BBox, color: tuple[int, int, int], track_id: int | None = None
) -> Frame:
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(int(x_center), y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    if track_id is not None:
        rectangle_width, rectangle_height = 40, 20
        x1_rect = int(x_center - rectangle_width // 2)
        x2_rect = int(x_center + rectangle_width // 2)
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (x1_text, y1_rect + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame


def draw_triangle(frame: Frame, bbox: BBox, color: tuple[int, int, int]) -> Frame:
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)
    x = int(x)

    triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame


def draw_team_ball_control(
    frame: Frame, frame_num: int, team_ball_control: npt.NDArray[np.int_]
) -> Frame:
    """Overlays running ball-possession percentage per team.

    ``team_ball_control`` entries are 1, 2, or 0 (no team had the ball yet).
    Frames with no assignment are excluded from the percentage rather than
    causing a divide-by-zero when nobody has had the ball yet.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    control_so_far = team_ball_control[: frame_num + 1]
    team_1_frames = int((control_so_far == 1).sum())
    team_2_frames = int((control_so_far == 2).sum())
    total = team_1_frames + team_2_frames

    team_1_pct = (team_1_frames / total) if total else 0.0
    team_2_pct = (team_2_frames / total) if total else 0.0

    cv2.putText(
        frame, f"Team 1 Ball Control: {team_1_pct * 100:.2f}%", (1400, 900),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3,
    )
    cv2.putText(
        frame, f"Team 2 Ball Control: {team_2_pct * 100:.2f}%", (1400, 950),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3,
    )

    return frame


def draw_annotations(
    video_frames: list[Frame],
    tracks: dict[str, list[dict[int, dict[str, Any]]]],
    team_ball_control: npt.NDArray[np.int_],
) -> list[Frame]:
    output_frames = []

    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get("has_ball", False):
                frame = draw_triangle(frame, player["bbox"], (0, 0, 255))

        for referee in referee_dict.values():
            frame = draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        for ball in ball_dict.values():
            frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))

        frame = draw_team_ball_control(frame, frame_num, team_ball_control)

        output_frames.append(frame)

    return output_frames
