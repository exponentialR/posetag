from __future__ import annotations

import unittest

import numpy as np

from posetag.workflows.calibration_guidance import (
    CELL_PRIORITY,
    analyze_calibration_guidance,
    auto_capture_decision,
    cell_label,
    guidance_overlay_lines,
    observation_from_charuco_corners,
    parse_grid_shape,
)


class CalibrationGuidanceTests(unittest.TestCase):
    def test_observation_maps_corners_to_cell_and_scale_bucket(self) -> None:
        observation = _observation(0.5, 0.5, size=0.28)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.cell, "center")
        self.assertEqual(observation.scale_bucket, "medium")
        self.assertTrue(observation.has_enough_corners)
        self.assertFalse(observation.clipped)
        self.assertAlmostEqual(observation.center_x, 0.5)
        self.assertAlmostEqual(observation.center_y, 0.5)

    def test_no_detection_prompts_user_to_bring_board_into_view(self) -> None:
        state = analyze_calibration_guidance(
            None,
            (),
            min_samples=30,
        )

        self.assertEqual(state.recommendation.code, "searching")
        self.assertIn("fully into view", state.recommendation.message)

    def test_too_few_corners_prompts_closer_or_better_lighting(self) -> None:
        observation = observation_from_charuco_corners(
            _corners(0.5, 0.5, size=0.28, count=3),
            (900, 900),
            min_corners=4,
        )

        state = analyze_calibration_guidance(
            observation,
            (),
            min_samples=30,
        )

        self.assertEqual(state.recommendation.code, "too_few_corners")
        self.assertIn("Move closer", state.recommendation.message)

    def test_first_clear_view_prompts_first_sample(self) -> None:
        current = _observation(0.5, 0.5)

        state = analyze_calibration_guidance(
            current,
            (),
            min_samples=30,
        )

        self.assertEqual(state.recommendation.code, "first_sample")
        self.assertIn("Press SPACE", state.recommendation.message)

    def test_guided_auto_changes_first_sample_prompt(self) -> None:
        current = _observation(0.5, 0.5)

        state = analyze_calibration_guidance(
            current,
            (),
            min_samples=30,
            auto_capture=True,
        )

        self.assertEqual(state.recommendation.code, "first_sample")
        self.assertIn("automatic", state.recommendation.message)

    def test_missing_frame_region_drives_specific_next_prompt(self) -> None:
        accepted = (_observation(0.5, 0.5),)
        current = _observation(0.5, 0.5)

        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=30,
        )

        self.assertEqual(state.recommendation.code, "move_to_missing_cell")
        self.assertEqual(state.recommendation.target_cell, "top-left")
        self.assertIn("top-left corner", state.recommendation.message)

    def test_current_missing_region_prompts_user_to_accept_sample(self) -> None:
        accepted = (_observation(0.5, 0.5),)
        current = _observation(0.16, 0.16)

        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=30,
        )

        self.assertEqual(state.recommendation.code, "accept_target_cell")
        self.assertEqual(state.recommendation.target_cell, "top-left")
        self.assertIn("press SPACE", state.recommendation.message)

    def test_balanced_coverage_but_low_scale_diversity_prompts_distance_change(self) -> None:
        accepted = tuple(_observation_for_cell(cell, size=0.28) for cell in CELL_PRIORITY)
        current = _observation(0.5, 0.5, size=0.28)

        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=9,
        )

        self.assertEqual(state.coverage_count, 9)
        self.assertEqual(state.recommendation.code, "change_distance")
        self.assertIn("distance", state.recommendation.message)

    def test_balanced_samples_and_scale_diversity_mark_ready_to_solve(self) -> None:
        accepted = [
            *(_observation_for_cell(cell, size=0.28) for cell in CELL_PRIORITY),
            _observation(0.5, 0.5, size=0.45),
        ]
        current = _observation(0.5, 0.5, size=0.45)

        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=10,
        )

        self.assertEqual(state.coverage_count, 9)
        self.assertEqual(state.sample_count, 10)
        self.assertTrue(state.recommendation.ready_to_solve)
        self.assertIn("Press ENTER", state.recommendation.message)

    def test_overlay_lines_include_coverage_samples_and_recommendation(self) -> None:
        current = _observation(0.5, 0.5)
        state = analyze_calibration_guidance(
            current,
            (),
            min_samples=30,
        )

        lines = guidance_overlay_lines(state)

        self.assertEqual(len(lines), 3)
        self.assertIn("Guide:", lines[0])
        self.assertIn("Coverage: 0/9", lines[1])
        self.assertIn("Samples: 0/30", lines[1])

    def test_parse_grid_shape_accepts_square_and_rectangular_values(self) -> None:
        self.assertEqual(parse_grid_shape("4"), (4, 4))
        self.assertEqual(parse_grid_shape("4x5"), (4, 5))

    def test_parse_grid_shape_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "ROWSxCOLS"):
            parse_grid_shape("wide")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_grid_shape("0x3")

    def test_non_default_grid_maps_observation_to_generic_cell(self) -> None:
        observation = observation_from_charuco_corners(
            _corners(0.75, 0.25, size=0.12),
            (900, 900),
            min_corners=4,
            grid_shape=(4, 4),
        )

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.cell, "r2c4")
        self.assertEqual(observation.row, 1)
        self.assertEqual(observation.col, 3)
        self.assertEqual(cell_label(observation.cell, (4, 4)), "row 2, column 4")

    def test_samples_per_cell_controls_coverage_completion(self) -> None:
        current = _observation(0.5, 0.5)
        accepted = (_observation(0.5, 0.5),)

        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=10,
            samples_per_cell=2,
        )

        self.assertNotIn("center", state.covered_cells)
        self.assertIn("center", state.missing_cells)

    def test_auto_capture_accepts_current_undercovered_cell(self) -> None:
        current = _observation(0.5, 0.5)
        state = analyze_calibration_guidance(
            current,
            (),
            min_samples=30,
            auto_capture=True,
        )

        decision = auto_capture_decision(state)

        self.assertTrue(decision.should_capture)
        self.assertEqual(decision.reason, "covers_missing_cell")

    def test_auto_capture_respects_cooldown(self) -> None:
        current = _observation(0.5, 0.5)
        state = analyze_calibration_guidance(
            current,
            (),
            min_samples=30,
            auto_capture=True,
        )

        decision = auto_capture_decision(state, cooldown_frames_remaining=3)

        self.assertFalse(decision.should_capture)
        self.assertEqual(decision.reason, "cooldown")

    def test_auto_capture_waits_for_missing_cells_before_extra_samples(self) -> None:
        current = _observation(0.5, 0.5)
        state = analyze_calibration_guidance(
            current,
            (_observation(0.5, 0.5),),
            min_samples=30,
            auto_capture=True,
        )

        decision = auto_capture_decision(state)

        self.assertFalse(decision.should_capture)
        self.assertEqual(decision.reason, "waiting_for_missing_cell")

    def test_auto_capture_adds_scale_diversity_after_coverage(self) -> None:
        accepted = tuple(_observation_for_cell(cell, size=0.28) for cell in CELL_PRIORITY)
        current = _observation(0.5, 0.5, size=0.45)
        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=9,
            auto_capture=True,
        )

        decision = auto_capture_decision(state)

        self.assertTrue(decision.should_capture)
        self.assertEqual(decision.reason, "adds_scale_diversity")

    def test_auto_capture_balances_samples_until_required_count(self) -> None:
        accepted = [
            *(_observation_for_cell(cell, size=0.28) for cell in CELL_PRIORITY),
            _observation(0.5, 0.5, size=0.45),
        ]
        current = _observation(0.5, 0.5, size=0.45)
        state = analyze_calibration_guidance(
            current,
            accepted,
            min_samples=30,
            auto_capture=True,
        )

        decision = auto_capture_decision(state)

        self.assertTrue(decision.should_capture)
        self.assertEqual(decision.reason, "balances_cell_samples")


def _observation(
    center_x: float,
    center_y: float,
    *,
    size: float = 0.28,
):
    return observation_from_charuco_corners(
        _corners(center_x, center_y, size=size),
        (900, 900),
        min_corners=4,
    )


def _observation_for_cell(cell: str, *, size: float = 0.28):
    centers = {
        "top-left": (0.16, 0.16),
        "top-center": (0.5, 0.16),
        "top-right": (0.84, 0.16),
        "middle-left": (0.16, 0.5),
        "center": (0.5, 0.5),
        "middle-right": (0.84, 0.5),
        "bottom-left": (0.16, 0.84),
        "bottom-center": (0.5, 0.84),
        "bottom-right": (0.84, 0.84),
    }
    center = centers[cell]
    return _observation(center[0], center[1], size=size)


def _corners(
    center_x: float,
    center_y: float,
    *,
    size: float,
    count: int = 4,
):
    half = size / 2.0
    points = [
        (center_x - half, center_y - half),
        (center_x + half, center_y - half),
        (center_x - half, center_y + half),
        (center_x + half, center_y + half),
        (center_x, center_y - half),
        (center_x + half, center_y),
    ][:count]
    return np.asarray(
        [[[x * 900.0, y * 900.0]] for x, y in points],
        dtype=np.float32,
    )


if __name__ == "__main__":
    unittest.main()
