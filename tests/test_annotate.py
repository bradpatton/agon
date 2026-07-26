import math

import numpy as np

from soccer_analysis.viz.annotate import _is_valid_bbox, draw_annotations


def test_is_valid_bbox_true_for_normal_box():
    assert _is_valid_bbox((0, 0, 10, 10)) is True


def test_is_valid_bbox_false_when_any_value_is_nan():
    assert _is_valid_bbox((0, 0, math.nan, 10)) is False


def test_draw_annotations_skips_nan_ball_without_crashing():
    # Regression test: a never-detected ball ends up with a NaN bbox after
    # interpolation (see detection/base.py), which used to crash on
    # int(nan) inside draw_triangle -- found validating UltralyticsDetector
    # end-to-end in Docker.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracks = {
        "players": [{}],
        "referees": [{}],
        "ball": [{1: {"bbox": [math.nan, math.nan, math.nan, math.nan]}}],
    }

    output_frames = draw_annotations([frame], tracks, team_ball_control=np.array([0]))

    assert len(output_frames) == 1
    assert output_frames[0].shape == frame.shape


def test_draw_annotations_with_players_and_valid_ball():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    tracks = {
        "players": [{1: {"bbox": [10, 10, 30, 30], "team_color": (255, 0, 0), "has_ball": True}}],
        "referees": [{2: {"bbox": [50, 50, 70, 70]}}],
        "ball": [{1: {"bbox": [15, 15, 20, 20]}}],
    }

    output_frames = draw_annotations([frame], tracks, team_ball_control=np.array([1]))

    assert len(output_frames) == 1
    # Something was actually drawn (frame is no longer all zeros).
    assert output_frames[0].any()
