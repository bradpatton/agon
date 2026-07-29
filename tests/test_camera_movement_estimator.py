import numpy as np
import pytest

from soccer_analysis.camera.camera_movement_estimator import (
    CameraFlowState,
    CameraMovementEstimator,
)


def _panning_frames(num_frames: int, step: tuple[int, int], size=(240, 320), seed=0):
    """Crops a sliding window across one large noise canvas, one frame per
    step -- simulates a camera panning across a textured background so
    goodFeaturesToTrack/optical flow have real correspondences to find,
    without needing actual video footage."""
    rng = np.random.default_rng(seed)
    height, width = size
    dx, dy = step
    margin = max(abs(dx), abs(dy)) * (num_frames - 1) + 1
    canvas = rng.integers(0, 255, size=(height + margin, width + margin, 3), dtype=np.uint8)
    frames = []
    for i in range(num_frames):
        y0, x0 = i * dy, i * dx
        frames.append(canvas[y0 : y0 + height, x0 : x0 + width].copy())
    return frames


class TestGetCameraMovementChunk:
    def test_first_chunk_seeds_zero_movement_for_frame_zero(self):
        frames = _panning_frames(4, step=(3, 2))
        estimator = CameraMovementEstimator(frames[0])
        movement, _ = estimator.get_camera_movement_chunk(frames, state=None)
        assert movement[0] == (0.0, 0.0)
        assert len(movement) == len(frames)

    def test_chunking_reproduces_whole_clip_result(self):
        """The whole point of get_camera_movement_chunk: splitting one clip
        into chunks and carrying CameraFlowState across them must give the
        same per-frame movement as processing it in one call -- otherwise
        run_pipeline_streaming's output would depend on chunk_size."""
        frames = _panning_frames(6, step=(4, 3))

        baseline_estimator = CameraMovementEstimator(frames[0])
        baseline_movement, _ = baseline_estimator.get_camera_movement_chunk(frames, state=None)

        chunked_estimator = CameraMovementEstimator(frames[0])
        chunk1_movement, state = chunked_estimator.get_camera_movement_chunk(frames[:3], state=None)
        chunk2_movement, _ = chunked_estimator.get_camera_movement_chunk(frames[3:], state=state)
        chunked_movement = chunk1_movement + chunk2_movement

        assert len(chunked_movement) == len(baseline_movement)
        for (bx, by), (cx, cy) in zip(baseline_movement, chunked_movement, strict=True):
            assert cx == pytest.approx(bx, abs=1e-6)
            assert cy == pytest.approx(by, abs=1e-6)

    def test_state_carries_last_frame_and_features(self):
        frames = _panning_frames(3, step=(2, 2))
        estimator = CameraMovementEstimator(frames[0])
        _, state = estimator.get_camera_movement_chunk(frames, state=None)

        assert isinstance(state, CameraFlowState)
        assert state.gray.shape == frames[0].shape[:2]
        assert state.features is not None

    def test_empty_continuation_chunk_returns_no_movement(self):
        frames = _panning_frames(2, step=(2, 2))
        estimator = CameraMovementEstimator(frames[0])
        _, state = estimator.get_camera_movement_chunk(frames, state=None)

        movement, new_state = estimator.get_camera_movement_chunk([], state=state)
        assert movement == []
        np.testing.assert_array_equal(new_state.gray, state.gray)
        np.testing.assert_array_equal(new_state.features, state.features)


class TestGetCameraMovementStubCache:
    def test_delegates_to_chunk_variant_with_no_state(self, tmp_path):
        frames = _panning_frames(3, step=(2, 1))
        estimator = CameraMovementEstimator(frames[0])
        via_stub_path = estimator.get_camera_movement(frames, stub_path=tmp_path / "stub.json")

        estimator2 = CameraMovementEstimator(frames[0])
        via_chunk, _ = estimator2.get_camera_movement_chunk(frames, state=None)

        assert via_stub_path == via_chunk
        assert (tmp_path / "stub.json").exists()
