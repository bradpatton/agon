from agon.analytics.player_ball_assigner import PlayerBallAssigner


def test_assigns_closest_player_within_threshold():
    assigner = PlayerBallAssigner(max_player_ball_distance_px=70.0)
    players = {
        1: {"bbox": (0, 0, 20, 20)},  # bottom-right corner at (20, 20)
        2: {"bbox": (100, 100, 120, 120)},
    }
    ball_bbox = (15, 15, 25, 25)  # center (20, 20)

    assert assigner.assign_ball_to_player(players, ball_bbox) == 1


def test_returns_none_when_no_player_within_threshold():
    assigner = PlayerBallAssigner(max_player_ball_distance_px=10.0)
    players = {1: {"bbox": (0, 0, 20, 20)}}
    ball_bbox = (500, 500, 520, 520)

    assert assigner.assign_ball_to_player(players, ball_bbox) is None


def test_returns_none_for_no_players():
    assigner = PlayerBallAssigner()
    assert assigner.assign_ball_to_player({}, (0, 0, 10, 10)) is None


def test_picks_nearer_of_two_players_in_range():
    assigner = PlayerBallAssigner(max_player_ball_distance_px=1000.0)
    players = {
        1: {"bbox": (0, 0, 10, 10)},  # bottom corners far from the ball
        2: {"bbox": (19, 19, 21, 21)},  # bottom corners right next to the ball
    }
    ball_bbox = (18, 18, 22, 22)  # center (20, 20)

    assert assigner.assign_ball_to_player(players, ball_bbox) == 2
