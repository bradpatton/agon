from agon.geometry.pitch_keypoints import (
    CANONICAL_KEYPOINTS,
    EXCLUDED_LINES,
    GOAL_POST_GROUND_POINT_INDEX,
    LINE_ENDPOINTS_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    canonical_keypoint_real_xy,
    is_frame_boundary_clipped,
)


class TestLineEndpointsM:
    def test_every_straight_line_has_two_endpoints(self):
        # Goal posts are the deliberate exception -- only the ground-level
        # point is kept (see GOAL_POST_GROUND_POINT_INDEX).
        for name, endpoints in LINE_ENDPOINTS_M.items():
            if "post" in name:
                assert len(endpoints) == 1, name
            else:
                assert len(endpoints) == 2, name

    def test_excludes_circles_and_crossbars(self):
        for name in EXCLUDED_LINES:
            assert name not in LINE_ENDPOINTS_M

    def test_all_points_within_pitch_bounds(self):
        half_l, half_w = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
        for endpoints in LINE_ENDPOINTS_M.values():
            for x, y in endpoints:
                assert -half_l <= x <= half_l
                assert -half_w <= y <= half_w

    def test_box_edges_share_corners_with_adjacent_edges(self):
        # "Big rect. left main" and "Big rect. left top" should agree on
        # the box's far-top corner -- confirmed against real pixel data
        # (see module docstring); this is the same claim expressed as a
        # regression test on the *geometry*, not the pixels.
        main = LINE_ENDPOINTS_M["Big rect. left main"]
        top = LINE_ENDPOINTS_M["Big rect. left top"]
        assert main[0] == top[1]

    def test_goal_posts_sit_on_the_goal_line(self):
        half_l = PITCH_LENGTH_M / 2
        for name in (
            "Goal left post left",
            "Goal left post right",
            "Goal right post left",
            "Goal right post right",
        ):
            x, _y = LINE_ENDPOINTS_M[name][0]
            assert abs(x) == half_l


class TestCanonicalKeypoints:
    def test_derived_from_line_endpoints_m_in_order(self):
        expected = [
            (name, idx)
            for name, endpoints in LINE_ENDPOINTS_M.items()
            for idx in range(len(endpoints))
        ]
        assert expected == CANONICAL_KEYPOINTS

    def test_no_duplicate_keys(self):
        assert len(CANONICAL_KEYPOINTS) == len(set(CANONICAL_KEYPOINTS))

    def test_canonical_keypoint_real_xy_matches_line_endpoints_m(self):
        for name, idx in CANONICAL_KEYPOINTS:
            assert canonical_keypoint_real_xy(name, idx) == LINE_ENDPOINTS_M[name][idx]


class TestGoalPostGroundPointIndex:
    def test_is_index_zero(self):
        # Empirically verified against real annotations (see module
        # docstring) -- point 0 is ground level, point 1 matches the
        # crossbar and is therefore off the ground plane.
        assert GOAL_POST_GROUND_POINT_INDEX == 0


class TestIsFrameBoundaryClipped:
    def test_interior_point_is_not_clipped(self):
        assert is_frame_boundary_clipped(500, 500, width=1920, height=1080) is False

    def test_left_edge_is_clipped(self):
        assert is_frame_boundary_clipped(0, 500, width=1920, height=1080) is True

    def test_right_edge_is_clipped(self):
        assert is_frame_boundary_clipped(1920, 500, width=1920, height=1080) is True

    def test_top_edge_is_clipped(self):
        assert is_frame_boundary_clipped(500, 0, width=1920, height=1080) is True

    def test_bottom_edge_is_clipped(self):
        assert is_frame_boundary_clipped(500, 1080, width=1920, height=1080) is True

    def test_just_inside_margin_is_not_clipped(self):
        assert is_frame_boundary_clipped(10, 10, width=1920, height=1080, margin=3) is False

    def test_custom_margin_is_respected(self):
        assert is_frame_boundary_clipped(4, 500, width=1920, height=1080, margin=5) is True
