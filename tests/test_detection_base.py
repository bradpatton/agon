import math

from agon.detection.base import (
    _tracks_from_jsonable,
    _tracks_to_jsonable,
    interpolate_ball_positions,
)


def test_interpolate_fills_gaps_between_detections():
    ball_positions = [
        {1: {"bbox": [0, 0, 10, 10]}},
        {},  # missed detection
        {1: {"bbox": [20, 20, 30, 30]}},
    ]

    result = interpolate_ball_positions(ball_positions)

    assert result[0][1]["bbox"] == [0, 0, 10, 10]
    assert result[1][1]["bbox"] == [10, 10, 20, 20]  # linearly interpolated midpoint
    assert result[2][1]["bbox"] == [20, 20, 30, 30]


def test_interpolate_backfills_leading_gap():
    ball_positions = [{}, {1: {"bbox": [5, 5, 15, 15]}}]

    result = interpolate_ball_positions(ball_positions)

    assert result[0][1]["bbox"] == [5, 5, 15, 15]


def test_interpolate_never_detected_does_not_crash():
    # Regression test: this used to raise
    # `ValueError: 4 columns passed, passed data had 0 columns` -- found
    # validating UltralyticsDetector end-to-end against a real clip where
    # the ball was never detected in any frame at all.
    ball_positions = [{}, {}, {}]

    result = interpolate_ball_positions(ball_positions)

    assert len(result) == 3
    for entry in result:
        bbox = entry[1]["bbox"]
        assert len(bbox) == 4
        assert all(math.isnan(v) for v in bbox)


def test_interpolate_stamps_class_name_ball():
    result = interpolate_ball_positions([{1: {"bbox": [0, 0, 1, 1]}}])
    assert result[0][1]["class_name"] == "ball"


def test_tracks_jsonable_round_trip():
    tracks = {
        "players": [{1: {"bbox": [0, 0, 10, 10], "class_name": "player"}}],
        "referees": [{}],
        "ball": [{1: {"bbox": [5, 5, 6, 6], "class_name": "ball"}}],
    }

    restored = _tracks_from_jsonable(_tracks_to_jsonable(tracks))

    assert restored == tracks
    # Track-id keys must survive the round trip as ints, not strings.
    assert list(restored["players"][0].keys()) == [1]
    assert isinstance(list(restored["players"][0].keys())[0], int)
