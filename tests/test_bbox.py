from soccer_analysis.geometry.bbox import (
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)


def test_get_center_of_bbox():
    assert get_center_of_bbox((0, 0, 10, 20)) == (5.0, 10.0)


def test_get_bbox_width():
    assert get_bbox_width((10, 0, 30, 50)) == 20


def test_get_foot_position_is_bottom_center():
    assert get_foot_position((0, 0, 10, 40)) == (5.0, 40)


def test_measure_distance_pythagorean():
    assert measure_distance((0, 0), (3, 4)) == 5.0


def test_measure_distance_zero():
    assert measure_distance((1, 1), (1, 1)) == 0.0


def test_measure_xy_distance():
    assert measure_xy_distance((5, 10), (2, 3)) == (3, 7)
