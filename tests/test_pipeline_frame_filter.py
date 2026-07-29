import numpy as np

from agon.config import PipelineConfig
from agon.pipeline import _classify_frames


def _pitch_green_frame(size=(20, 20)) -> np.ndarray:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)
    return frame


def _graphic_frame(size=(20, 20)) -> np.ndarray:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = (200, 40, 20)
    return frame


class TestClassifyFramesOff:
    def test_passthrough_returns_original_frames_and_no_metadata(self):
        frames = [_pitch_green_frame(), _graphic_frame()]
        config = PipelineConfig(frame_filter_mode="off")

        kept, frame_ids, classifications, clocks = _classify_frames(
            frames, config, clock_reader=None, frame_id_base=10
        )

        assert kept is frames
        assert frame_ids is None
        assert classifications is None
        assert clocks is None


class TestClassifyFramesTag:
    def test_tags_every_frame_without_dropping_any(self):
        frames = [_pitch_green_frame(), _graphic_frame(), _pitch_green_frame()]
        config = PipelineConfig(frame_filter_mode="tag")

        kept, frame_ids, classifications, clocks = _classify_frames(
            frames, config, clock_reader=None, frame_id_base=0
        )

        assert kept == frames
        assert frame_ids is None
        assert classifications == ["live_play", "graphic", "live_play"]
        assert clocks == [None, None, None]


class TestClassifyFramesStrip:
    def test_drops_non_live_play_frames_and_tracks_true_ids(self):
        frames = [_graphic_frame(), _pitch_green_frame(), _graphic_frame(), _pitch_green_frame()]
        config = PipelineConfig(frame_filter_mode="strip")

        kept, frame_ids, classifications, clocks = _classify_frames(
            frames, config, clock_reader=None, frame_id_base=200
        )

        assert len(kept) == 2
        assert frame_ids == [201, 203]
        assert classifications == ["live_play", "live_play"]
        assert clocks == [None, None]

    def test_dropping_everything_returns_empty_lists_not_an_error(self):
        frames = [_graphic_frame(), _graphic_frame()]
        config = PipelineConfig(frame_filter_mode="strip")

        kept, frame_ids, classifications, clocks = _classify_frames(
            frames, config, clock_reader=None, frame_id_base=0
        )

        assert kept == []
        assert frame_ids == []
        assert classifications == []
        assert clocks == []
