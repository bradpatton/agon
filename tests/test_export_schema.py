import json

import pytest

from soccer_analysis.export.schema import (
    ObjectClass,
    build_frame_records,
    build_match_summary,
    object_class_for,
)
from soccer_analysis.export.writer import write_jsonl, write_match_summary, write_parquet


def _sample_tracks():
    return {
        "players": [
            {
                1: {
                    "bbox": [0, 0, 10, 10],
                    "position": (5, 10),
                    "position_transformed": (1.0, 2.0),
                    "class_name": "player",
                    "team": 1,
                    "speed": 10.0,
                    "distance": 5.0,
                    "has_ball": True,
                }
            },
            {
                1: {
                    "bbox": [1, 1, 11, 11],
                    "position": (6, 11),
                    "position_transformed": None,
                    "class_name": "goalkeeper",
                    "team": 2,
                }
            },
        ],
        "referees": [
            {2: {"bbox": [50, 50, 60, 60], "position": (55, 60), "class_name": "referee"}},
            {},
        ],
        "ball": [
            {
                1: {
                    "bbox": [20, 20, 22, 22],
                    "position": (21, 21),
                    "position_transformed": (3.0, 4.0),
                    "class_name": "ball",
                }
            },
            {
                1: {
                    "bbox": [21, 21, 23, 23],
                    "position": (22, 22),
                    "position_transformed": None,
                    "class_name": "ball",
                }
            },
        ],
    }


class TestObjectClassFor:
    def test_goalkeeper_distinguished_from_player(self):
        assert object_class_for("players", "goalkeeper") == ObjectClass.GOALKEEPER
        assert object_class_for("players", "player") == ObjectClass.PLAYER

    def test_generic_coco_person_maps_to_player(self):
        assert object_class_for("players", "person") == ObjectClass.PLAYER

    def test_referee_and_ball(self):
        assert object_class_for("referees", "referee") == ObjectClass.REFEREE
        assert object_class_for("ball", "sports ball") == ObjectClass.BALL


class TestBuildFrameRecords:
    def test_frame_and_object_fields(self):
        records = build_frame_records(
            _sample_tracks(),
            team_ball_control=[1, 0],
            camera_movement_per_frame=[(0.0, 0.0), (1.0, -1.0)],
            video_id="test_vid",
            frame_rate=10.0,
        )

        assert len(records) == 2
        frame0, frame1 = records

        assert frame0.video_id == "test_vid"
        assert frame0.frame_id == 0
        assert frame0.timestamp_s == 0.0
        assert frame0.camera_movement_px == (0.0, 0.0)
        assert frame0.team_ball_control == 1
        assert len(frame0.objects) == 3

        player = next(o for o in frame0.objects if o.object_class == ObjectClass.PLAYER)
        assert player.track_id == 1
        assert player.team == 1
        assert player.bbox_px == (0, 0, 10, 10)
        assert player.position_pitch_m == (1.0, 2.0)
        assert player.speed_kmh == 10.0
        assert player.has_ball is True

        referee = next(o for o in frame0.objects if o.object_class == ObjectClass.REFEREE)
        assert referee.team is None
        assert referee.position_pitch_m is None
        assert referee.has_ball is False

        # frame 1: timestamp scales with frame_rate, goalkeeper class
        # distinguished, empty referees frame contributes no object.
        assert frame1.timestamp_s == pytest.approx(0.1)
        goalkeeper = next(o for o in frame1.objects if o.object_class == ObjectClass.GOALKEEPER)
        assert goalkeeper.team == 2
        assert goalkeeper.position_pitch_m is None
        assert sum(1 for o in frame1.objects if o.object_class == ObjectClass.REFEREE) == 0


class TestBuildMatchSummary:
    def test_possession_and_player_aggregates(self):
        tracks = {
            "players": [
                {1: {"team": 1, "distance": 5.0, "speed": 10.0}},
                {1: {"team": 1, "distance": 8.0, "speed": 20.0}},
            ],
            "referees": [{}, {}],
            "ball": [{}, {}],
        }

        summary = build_match_summary(
            tracks, team_ball_control=[1, 1, 2, 0], video_id="v", frame_rate=24.0
        )

        assert summary.team_1_possession_pct == pytest.approx(200 / 3)
        assert summary.team_2_possession_pct == pytest.approx(100 / 3)
        assert summary.frame_count == 4

        [player] = summary.players
        assert player.track_id == 1
        assert player.team == 1
        assert player.total_distance_m == pytest.approx(8.0)
        assert player.avg_speed_kmh == pytest.approx(15.0)
        assert player.max_speed_kmh == pytest.approx(20.0)

    def test_no_possession_yet_does_not_divide_by_zero(self):
        tracks = {"players": [{}], "referees": [{}], "ball": [{}]}
        summary = build_match_summary(tracks, team_ball_control=[0], video_id="v", frame_rate=24.0)
        assert summary.team_1_possession_pct == 0.0
        assert summary.team_2_possession_pct == 0.0


class TestWriters:
    def test_jsonl_round_trip(self, tmp_path):
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        path = tmp_path / "frames.jsonl"
        write_jsonl(records, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["video_id"] == "v"
        assert first["objects"][0]["class"] in ("player", "goalkeeper", "referee", "ball")

    def test_parquet_round_trip(self, tmp_path):
        pd = pytest.importorskip("pandas")
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        path = tmp_path / "frames.parquet"
        write_parquet(records, path)

        df = pd.read_parquet(path)
        assert len(df) == 5  # 3 objects in frame 0 + 2 in frame 1
        assert set(df["frame_id"]) == {0, 1}

    def test_parquet_empty_objects_frame_still_gets_a_row(self, tmp_path):
        pd = pytest.importorskip("pandas")
        tracks = {"players": [{}], "referees": [{}], "ball": [{}]}
        records = build_frame_records(tracks, [0], [(0.0, 0.0)], "v", 24.0)
        path = tmp_path / "empty.parquet"
        write_parquet(records, path)

        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["track_id"] is None or pd.isna(df.iloc[0]["track_id"])

    def test_write_match_summary(self, tmp_path):
        tracks = {"players": [{}], "referees": [{}], "ball": [{}]}
        summary = build_match_summary(tracks, [0], "v", 24.0)
        path = tmp_path / "summary.json"
        write_match_summary(summary, path)

        data = json.loads(path.read_text())
        assert data["video_id"] == "v"
