from agon.jersey.aggregator import aggregate_track_jersey_numbers


def test_picks_the_only_confident_prediction():
    predictions = {1: [(10, 0.9)]}
    assert aggregate_track_jersey_numbers(predictions) == {1: 10}


def test_drops_predictions_below_min_confidence():
    predictions = {1: [(10, 0.2), (7, 0.1)]}
    assert aggregate_track_jersey_numbers(predictions, min_confidence=0.5) == {1: None}


def test_ignores_unknown_class_predictions():
    predictions = {1: [(None, 0.99), (None, 0.95)]}
    assert aggregate_track_jersey_numbers(predictions) == {1: None}


def test_confidence_weighted_vote_beats_plain_majority():
    # 7 has more votes (3 vs 2) -- a plain majority-count vote would pick
    # it -- but 10's total confidence is higher (1.80 vs 1.53), which is
    # what should actually win: the whole point of weighting by confidence
    # instead of just counting frames.
    predictions = {
        1: [
            (7, 0.51),
            (7, 0.51),
            (7, 0.51),
            (10, 0.90),
            (10, 0.90),
        ]
    }
    assert aggregate_track_jersey_numbers(predictions) == {1: 10}


def test_returns_none_when_no_prediction_clears_threshold():
    predictions = {1: [(None, 0.9), (7, 0.1)]}
    assert aggregate_track_jersey_numbers(predictions, min_confidence=0.5) == {1: None}


def test_handles_multiple_tracks_independently():
    predictions = {
        1: [(10, 0.9)],
        2: [(7, 0.8)],
        3: [],
    }
    assert aggregate_track_jersey_numbers(predictions) == {1: 10, 2: 7, 3: None}


def test_empty_input_returns_empty_output():
    assert aggregate_track_jersey_numbers({}) == {}
