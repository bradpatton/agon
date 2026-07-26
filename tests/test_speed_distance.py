import pytest

from soccer_analysis.analytics.speed_distance import SpeedDistanceEstimator


def _frame(position) -> dict:
    return {1: {"position_transformed": position}}


def test_speed_and_distance_over_a_window():
    # frame_window=2, frame_rate=1.0 frame/s: 5 frames, track 1 moves
    # (0,0) -> (3,4) -> (3,4). Also exercises the tail-alignment edge case
    # below (number_of_frames - 1 == 4 is an exact multiple of window=2).
    tracks = {
        "players": [
            _frame((0, 0)),
            _frame((1, 1)),  # position doesn't matter, never read directly
            _frame((3, 4)),
            _frame((2, 2)),
            _frame((3, 4)),
        ],
        "referees": [{}] * 5,
        "ball": [{}] * 5,
    }

    SpeedDistanceEstimator(frame_window=2, frame_rate=1.0).add_speed_and_distance_to_tracks(tracks)

    # frame 0-1: distance 5 over 2s -> 9 km/h, cumulative distance 5.
    assert tracks["players"][0][1]["speed"] == pytest.approx(9.0)
    assert tracks["players"][0][1]["distance"] == pytest.approx(5.0)
    assert tracks["players"][1][1]["speed"] == pytest.approx(9.0)

    # frame 2-3: same position (3,4) -> (3,4) at the second window's start
    # and end, so distance covered in *this* window is 0 -> 0 km/h, but
    # cumulative distance carries forward.
    assert tracks["players"][2][1]["speed"] == pytest.approx(0.0)
    assert tracks["players"][2][1]["distance"] == pytest.approx(5.0)

    # frame 4 is the tail edge case (frame_num == last_frame): must not
    # crash, and gets no speed/distance written since no window completes.
    assert "speed" not in tracks["players"][4][1]
    assert "distance" not in tracks["players"][4][1]


def test_ball_and_referees_are_never_touched():
    tracks = {
        "players": [_frame((0, 0)), _frame((3, 4))],
        "referees": [_frame((0, 0)), _frame((3, 4))],
        "ball": [_frame((0, 0)), _frame((3, 4))],
    }

    SpeedDistanceEstimator(frame_window=1, frame_rate=1.0).add_speed_and_distance_to_tracks(tracks)

    assert "speed" not in tracks["referees"][0][1]
    assert "speed" not in tracks["ball"][0][1]


def test_none_position_transformed_is_skipped_not_crashed():
    tracks = {
        "players": [_frame(None), _frame((3, 4))],
        "referees": [{}] * 2,
        "ball": [{}] * 2,
    }

    SpeedDistanceEstimator(frame_window=1, frame_rate=1.0).add_speed_and_distance_to_tracks(tracks)

    assert "speed" not in tracks["players"][0][1]
