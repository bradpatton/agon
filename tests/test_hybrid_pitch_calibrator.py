from agon.geometry.hybrid_pitch_calibrator import HybridPitchCalibrator


class _FakeCalibrator:
    """A calibrator whose calibrate()/transform_point() behavior is
    entirely controlled by the test -- avoids depending on real pixel
    geometry or CV detection just to test the fallback wiring itself."""

    def __init__(self, transform_result=None):
        self.transform_result = transform_result
        self.calibrate_calls: list[tuple[list, int]] = []

    def calibrate(self, frames, frame_offset=0):
        self.calibrate_calls.append((frames, frame_offset))

    def transform_point(self, point, frame_idx=0):
        return self.transform_result


class _FakeCalibratorWithHomography(_FakeCalibrator):
    """Like TrainedPitchCalibrator, exposes a real .homography() method
    beyond the PitchCalibrator protocol -- controlled by the test."""

    def __init__(self, transform_result=None, homography_result=None):
        super().__init__(transform_result)
        self.homography_result = homography_result

    def homography(self, frame_idx=0):
        return self.homography_result


def test_uses_primary_result_when_available():
    primary = _FakeCalibrator(transform_result=(1.0, 2.0))
    fallback = _FakeCalibrator(transform_result=(9.0, 9.0))
    hybrid = HybridPitchCalibrator(primary, fallback)

    assert hybrid.transform_point((10, 20)) == (1.0, 2.0)


def test_falls_back_when_primary_returns_none():
    primary = _FakeCalibrator(transform_result=None)
    fallback = _FakeCalibrator(transform_result=(5.0, 6.0))
    hybrid = HybridPitchCalibrator(primary, fallback)

    assert hybrid.transform_point((10, 20)) == (5.0, 6.0)


def test_returns_none_when_both_fail():
    primary = _FakeCalibrator(transform_result=None)
    fallback = _FakeCalibrator(transform_result=None)
    hybrid = HybridPitchCalibrator(primary, fallback)

    assert hybrid.transform_point((10, 20)) is None


def test_calibrate_calls_both_calibrators():
    primary = _FakeCalibrator()
    fallback = _FakeCalibrator()
    hybrid = HybridPitchCalibrator(primary, fallback)

    frames = [object(), object()]
    hybrid.calibrate(frames, frame_offset=42)

    assert primary.calibrate_calls == [(frames, 42)]
    assert fallback.calibrate_calls == [(frames, 42)]


def test_frame_idx_is_passed_through_to_both_calibrators():
    class _RecordingCalibrator(_FakeCalibrator):
        def __init__(self):
            super().__init__(transform_result=None)
            self.seen_frame_idx: list[int] = []

        def transform_point(self, point, frame_idx=0):
            self.seen_frame_idx.append(frame_idx)
            return super().transform_point(point, frame_idx)

    primary = _RecordingCalibrator()
    fallback = _RecordingCalibrator()
    hybrid = HybridPitchCalibrator(primary, fallback)

    hybrid.transform_point((1, 1), frame_idx=7)

    assert primary.seen_frame_idx == [7]
    assert fallback.seen_frame_idx == [7]


class TestHomographyPassthrough:
    def test_uses_primary_homography_when_available(self):
        primary = _FakeCalibratorWithHomography(homography_result="primary-H")
        fallback = _FakeCalibratorWithHomography(homography_result="fallback-H")
        hybrid = HybridPitchCalibrator(primary, fallback)

        assert hybrid.homography(frame_idx=0) == "primary-H"

    def test_falls_back_when_primary_has_no_homography_method(self):
        primary = _FakeCalibrator()  # no .homography() at all, e.g. ViewTransformer
        fallback = _FakeCalibratorWithHomography(homography_result="fallback-H")
        hybrid = HybridPitchCalibrator(primary, fallback)

        assert hybrid.homography(frame_idx=0) == "fallback-H"

    def test_falls_back_when_primary_homography_returns_none_for_this_frame(self):
        primary = _FakeCalibratorWithHomography(homography_result=None)
        fallback = _FakeCalibratorWithHomography(homography_result="fallback-H")
        hybrid = HybridPitchCalibrator(primary, fallback)

        assert hybrid.homography(frame_idx=0) == "fallback-H"

    def test_none_when_neither_calibrator_exposes_homography(self):
        primary = _FakeCalibrator()
        fallback = _FakeCalibrator()
        hybrid = HybridPitchCalibrator(primary, fallback)

        assert hybrid.homography(frame_idx=0) is None

    def test_recurses_through_a_nested_hybrid_calibrator(self):
        inner_primary = _FakeCalibratorWithHomography(homography_result="inner-H")
        inner_fallback = _FakeCalibrator()
        inner_hybrid = HybridPitchCalibrator(inner_primary, inner_fallback)

        outer_fallback = _FakeCalibratorWithHomography(homography_result="outer-H")
        outer_hybrid = HybridPitchCalibrator(primary=inner_hybrid, fallback=outer_fallback)

        assert outer_hybrid.homography(frame_idx=0) == "inner-H"
