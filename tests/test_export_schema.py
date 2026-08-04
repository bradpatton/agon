import json

import pytest

from agon.export.schema import (
    CameraPoseRecord,
    MatchStats,
    ObjectClass,
    accumulate_match_stats,
    build_frame_records,
    build_match_summary,
    finalize_match_summary,
    object_class_for,
)
from agon.export.writer import (
    JsonlWriter,
    ParquetChunkWriter,
    write_jsonl,
    write_match_summary,
    write_parquet,
)


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

    def test_jersey_number_defaults_to_none(self):
        # No jersey-number recognizer is wired into the pipeline yet (see
        # ObjectRecord.jersey_number's docstring) -- _sample_tracks doesn't
        # set it, so every object should come through null.
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        assert all(o.jersey_number is None for r in records for o in r.objects)

    def test_jersey_number_passes_through_when_present(self):
        tracks = {
            "players": [{7: {"bbox": [0, 0, 10, 10], "position": (5, 10), "jersey_number": 10}}],
            "referees": [{}],
            "ball": [{}],
        }
        [record] = build_frame_records(tracks, [0], [(0.0, 0.0)], "v", 24.0)
        [player] = record.objects
        assert player.jersey_number == 10

    def test_camera_pose_defaults_to_none(self):
        # No calibrator with a real per-frame homography was involved --
        # camera_poses isn't passed at all (distinct from an explicit list
        # of Nones, same convention as frame_classifications/game_clock_s).
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        assert all(r.camera_pose is None for r in records)

    def test_camera_pose_passes_through_when_present(self):
        pose = CameraPoseRecord(
            pan_degrees=1.0,
            tilt_degrees=100.0,
            roll_degrees=0.5,
            position_m=(0.0, -60.0, -20.0),
            x_focal_length_px=1400.0,
            y_focal_length_px=1400.0,
        )
        records = build_frame_records(
            _sample_tracks(),
            [1, 0],
            [(0.0, 0.0), (0.0, 0.0)],
            "v",
            24.0,
            camera_poses=[pose, None],
        )
        assert records[0].camera_pose == pose
        assert records[1].camera_pose is None


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


class TestFrameOffset:
    def test_frame_ids_and_timestamps_shift_by_offset(self):
        records = build_frame_records(
            _sample_tracks(),
            team_ball_control=[1, 0],
            camera_movement_per_frame=[(0.0, 0.0), (1.0, -1.0)],
            video_id="v",
            frame_rate=10.0,
            frame_offset=100,
        )

        assert [r.frame_id for r in records] == [100, 101]
        assert records[0].timestamp_s == pytest.approx(10.0)
        assert records[1].timestamp_s == pytest.approx(10.1)

    def test_explicit_frame_ids_overrides_offset_for_non_contiguous_frames(self):
        # frame_ids is what agon.broadcast's strip mode uses --
        # the surviving frames after stripping aren't contiguously numbered,
        # so a single frame_offset int can't reconstruct their true indices.
        records = build_frame_records(
            _sample_tracks(),
            team_ball_control=[1, 0],
            camera_movement_per_frame=[(0.0, 0.0), (1.0, -1.0)],
            video_id="v",
            frame_rate=10.0,
            frame_offset=999,  # ignored when frame_ids is given
            frame_ids=[5, 42],
        )

        assert [r.frame_id for r in records] == [5, 42]
        assert records[0].timestamp_s == pytest.approx(0.5)
        assert records[1].timestamp_s == pytest.approx(4.2)

    def test_frame_classification_and_game_clock_default_to_none(self):
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        assert all(r.frame_classification is None for r in records)
        assert all(r.game_clock_s is None for r in records)

    def test_frame_classification_and_game_clock_pass_through_when_given(self):
        records = build_frame_records(
            _sample_tracks(),
            team_ball_control=[1, 0],
            camera_movement_per_frame=[(0.0, 0.0), (1.0, -1.0)],
            video_id="v",
            frame_rate=10.0,
            frame_classifications=["live_play", "replay"],
            game_clock_s_per_frame=[123.0, None],
        )

        assert [r.frame_classification for r in records] == ["live_play", "replay"]
        assert [r.game_clock_s for r in records] == [123.0, None]


class TestStreamingMatchStats:
    def test_accumulate_across_chunks_matches_whole_clip_summary(self):
        chunk1 = {
            "players": [{1: {"team": 1, "distance": 5.0, "speed": 10.0}}],
            "referees": [{}],
            "ball": [{}],
        }
        chunk2 = {
            "players": [{1: {"team": 1, "distance": 8.0, "speed": 20.0}}],
            "referees": [{}],
            "ball": [{}],
        }

        stats = MatchStats()
        accumulate_match_stats(stats, chunk1, [1, 1])
        accumulate_match_stats(stats, chunk2, [2, 0])
        summary = finalize_match_summary(stats, "v", frame_count=4, frame_rate=24.0)

        whole_clip_tracks = {
            "players": [
                {1: {"team": 1, "distance": 5.0, "speed": 10.0}},
                {1: {"team": 1, "distance": 8.0, "speed": 20.0}},
            ],
            "referees": [{}, {}],
            "ball": [{}, {}],
        }
        expected = build_match_summary(
            whole_clip_tracks, team_ball_control=[1, 1, 2, 0], video_id="v", frame_rate=24.0
        )

        assert summary.team_1_possession_pct == pytest.approx(expected.team_1_possession_pct)
        assert summary.team_2_possession_pct == pytest.approx(expected.team_2_possession_pct)
        assert summary.frame_count == 4
        [player] = summary.players
        [expected_player] = expected.players
        assert player.total_distance_m == pytest.approx(expected_player.total_distance_m)
        assert player.avg_speed_kmh == pytest.approx(expected_player.avg_speed_kmh)
        assert player.max_speed_kmh == pytest.approx(expected_player.max_speed_kmh)

    def test_empty_stats_produces_zeroed_summary(self):
        summary = finalize_match_summary(MatchStats(), "v", frame_count=0, frame_rate=24.0)
        assert summary.team_1_possession_pct == 0.0
        assert summary.team_2_possession_pct == 0.0
        assert summary.players == []


class TestIncrementalWriters:
    def test_jsonl_writer_multiple_chunks_matches_single_write(self, tmp_path):
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        chunked_path = tmp_path / "chunked.jsonl"
        with JsonlWriter(chunked_path) as writer:
            writer.write_chunk(records[:1])
            writer.write_chunk(records[1:])

        whole_path = tmp_path / "whole.jsonl"
        write_jsonl(records, whole_path)

        assert chunked_path.read_text() == whole_path.read_text()

    def test_parquet_writer_multiple_chunks_matches_single_write(self, tmp_path):
        pd = pytest.importorskip("pandas")
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        chunked_path = tmp_path / "chunked.parquet"
        with ParquetChunkWriter(chunked_path) as writer:
            writer.write_chunk(records[:1])
            writer.write_chunk(records[1:])

        whole_path = tmp_path / "whole.parquet"
        write_parquet(records, whole_path)

        chunked_df = pd.read_parquet(chunked_path).reset_index(drop=True)
        whole_df = pd.read_parquet(whole_path).reset_index(drop=True)
        pd.testing.assert_frame_equal(chunked_df, whole_df)

    def test_parquet_writer_skips_empty_chunk_without_breaking_schema(self, tmp_path):
        pd = pytest.importorskip("pandas")
        records = build_frame_records(_sample_tracks(), [1, 0], [(0.0, 0.0), (0.0, 0.0)], "v", 24.0)
        path = tmp_path / "with_empty_chunk.parquet"
        with ParquetChunkWriter(path) as writer:
            writer.write_chunk([])  # e.g. a chunk with zero frames processed
            writer.write_chunk(records)

        df = pd.read_parquet(path)
        assert len(df) == 5
