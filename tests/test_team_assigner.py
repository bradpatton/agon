import numpy as np

from soccer_analysis.team.team_assigner import TeamAssigner


def _frame_with_two_colored_players():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:40, 0:20] = (0, 0, 200)  # red-ish (BGR), player 1
    frame[0:40, 50:70] = (200, 0, 0)  # blue-ish (BGR), player 2
    return frame


def test_assigns_visually_distinct_players_to_different_teams():
    frame = _frame_with_two_colored_players()
    players = {1: {"bbox": (0, 0, 20, 40)}, 2: {"bbox": (50, 0, 70, 40)}}

    assigner = TeamAssigner(random_state=0)
    assigner.assign_team_color(frame, players)

    team1 = assigner.get_player_team(frame, players[1]["bbox"], player_id=1)
    team2 = assigner.get_player_team(frame, players[2]["bbox"], player_id=2)

    assert team1 != team2
    assert {team1, team2} == {1, 2}
    assert 1 in assigner.team_colors
    assert 2 in assigner.team_colors


def test_get_player_team_is_cached_per_player_id():
    frame = _frame_with_two_colored_players()
    players = {1: {"bbox": (0, 0, 20, 40)}, 2: {"bbox": (50, 0, 70, 40)}}

    assigner = TeamAssigner(random_state=0)
    assigner.assign_team_color(frame, players)

    first = assigner.get_player_team(frame, players[1]["bbox"], player_id=1)
    # Even with a bbox that would plausibly classify differently, the
    # cached result for this player_id must win.
    second = assigner.get_player_team(frame, players[2]["bbox"], player_id=1)

    assert first == second == assigner.player_team_dict[1]


def test_get_player_team_before_assign_raises():
    assigner = TeamAssigner()
    frame = _frame_with_two_colored_players()
    try:
        assigner.get_player_team(frame, (0, 0, 20, 40), player_id=99)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
