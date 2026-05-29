from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from posetag.workflows.calibration_flow import (
    CALIBRATION_PROCESS_FAILED_CANCELLED,
    CALIBRATION_PROCESS_FINISHED,
    SOURCE_OPENCV,
    SOURCE_REALSENSE,
    SOURCE_VIDEO,
    CameraCalibrationConfig,
    build_camera_calibration_arguments,
    build_camera_calibration_command,
    build_camera_calibration_launch,
    camera_calibration_config_from_project,
    generated_charuco_metadata_paths,
    inspect_camera_calibration_output,
    inspect_camera_calibration_readiness,
    inspect_charuco_metadata,
    load_charuco_metadata,
    read_camera_calibration_yaml_text,
    summarize_camera_calibration_process_result,
)
from posetag.pipelines.charuco_calibration import (
    build_calibration_yaml,
    write_calibration_yaml,
)


class WorkflowCalibrationFlowTests(unittest.TestCase):
    def test_metadata_autofills_calibration_config(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            metadata_path = _write_metadata(
                project_root,
                squares_x=4,
                squares_y=6,
                square_length_mm=25.5,
                marker_length_mm=18.5,
                dictionary="6X6_250",
            )

            metadata = load_charuco_metadata(metadata_path)
            config = camera_calibration_config_from_project(project_root)

        self.assertEqual(metadata.squares_x, 4)
        self.assertEqual(metadata.squares_y, 6)
        self.assertEqual(metadata.square_length_mm, 25.5)
        self.assertEqual(metadata.marker_length_mm, 18.5)
        self.assertEqual(metadata.dictionary_name, "6X6_250")
        self.assertEqual(config.squares_x, 4)
        self.assertEqual(config.squares_y, 6)
        self.assertEqual(config.square_length_mm, 25.5)
        self.assertEqual(config.marker_length_mm, 18.5)
        self.assertEqual(config.dictionary_name, "6X6_250")

    def test_command_construction_quotes_paths_and_uses_webcam_fields(self) -> None:
        command = build_camera_calibration_command(
            CameraCalibrationConfig(
                project_root=Path("/tmp/PoseTag project"),
                source=SOURCE_OPENCV,
                camera_index=2,
                squares_x=4,
                squares_y=6,
                square_length_mm=25.5,
                marker_length_mm=18.5,
                dictionary_name="6X6_250",
            )
        )

        self.assertIn("posetag-calib-charuco", command)
        self.assertIn("'/tmp/PoseTag project'", command)
        self.assertIn("--source opencv --cam 2", command)
        self.assertIn("--squares-x 4 --squares-y 6", command)
        self.assertIn("--square-length-mm 25.5", command)
        self.assertIn("--marker-length-mm 18.5", command)
        self.assertIn("--dict 6X6_250", command)

    def test_launch_spec_uses_current_python_module_and_argument_list(self) -> None:
        launch = build_camera_calibration_launch(
            CameraCalibrationConfig(
                project_root=Path("/tmp/PoseTag project"),
                source=SOURCE_VIDEO,
                video_path="sample clips/calib.mp4",
                squares_x=4,
                squares_y=6,
                square_length_mm=25.5,
                marker_length_mm=18.5,
                dictionary_name="6X6_250",
            ),
            python_executable="/opt/posetag/python",
        )

        self.assertEqual(launch.program, "/opt/posetag/python")
        self.assertEqual(launch.arguments[:2], ("-m", "posetag.cli.charuco"))
        self.assertIn("--video", launch.arguments)
        self.assertIn("sample clips/calib.mp4", launch.arguments)
        self.assertNotIn("posetag-calib-charuco", launch.arguments)
        self.assertIn("posetag-calib-charuco", launch.display_command)
        self.assertEqual(
            launch.expected_output,
            Path("/tmp/PoseTag project") / "calib" / "calib_color.yaml",
        )

    def test_argument_construction_returns_shell_free_argv_items(self) -> None:
        arguments = build_camera_calibration_arguments(
            CameraCalibrationConfig(
                project_root="project with spaces",
                source=SOURCE_OPENCV,
                camera_index=3,
            )
        )

        self.assertEqual(arguments[0:2], ("--project_root", "project with spaces"))
        self.assertIn("--cam", arguments)
        self.assertIn("3", arguments)
        self.assertIn("--coverage-grid", arguments)
        self.assertIn("3x3", arguments)

    def test_argument_construction_includes_guided_capture_options(self) -> None:
        arguments = build_camera_calibration_arguments(
            CameraCalibrationConfig(
                project_root="project",
                coverage_grid="4x5",
                samples_per_cell=2,
                guided_auto=False,
                guided_auto_cooldown=5,
            )
        )

        self.assertIn("--coverage-grid", arguments)
        self.assertIn("4x5", arguments)
        self.assertIn("--samples-per-cell", arguments)
        self.assertIn("2", arguments)
        self.assertIn("--guided-auto-cooldown", arguments)
        self.assertIn("5", arguments)
        self.assertIn("--no-guided-auto", arguments)

    def test_video_command_requires_video_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "--video path is required"):
            build_camera_calibration_command(
                CameraCalibrationConfig(
                    project_root="project",
                    source=SOURCE_VIDEO,
                    video_path="",
                )
            )

        command = build_camera_calibration_command(
            CameraCalibrationConfig(
                project_root="project",
                source=SOURCE_VIDEO,
                video_path="sample clips/calib.mp4",
            )
        )

        self.assertIn("--source video --video 'sample clips/calib.mp4'", command)
        self.assertNotIn("--cam", command)

    def test_realsense_command_omits_webcam_and_video_fields(self) -> None:
        command = build_camera_calibration_command(
            CameraCalibrationConfig(
                project_root="project",
                source=SOURCE_REALSENSE,
            )
        )

        self.assertIn("--source realsense", command)
        self.assertNotIn("--cam", command)
        self.assertNotIn("--video", command)

    def test_command_construction_rejects_invalid_board_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "marker-length-mm"):
            build_camera_calibration_command(
                CameraCalibrationConfig(
                    project_root="project",
                    square_length_mm=10.0,
                    marker_length_mm=20.0,
                )
            )

    def test_command_construction_rejects_unsupported_dictionary(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported ArUco dictionary"):
            build_camera_calibration_command(
                CameraCalibrationConfig(
                    project_root="project",
                    dictionary_name="NOT_A_DICT",
                )
            )

    def test_command_construction_rejects_invalid_guided_capture_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "--coverage-grid"):
            build_camera_calibration_command(
                CameraCalibrationConfig(
                    project_root="project",
                    coverage_grid="0x3",
                )
            )

        with self.assertRaisesRegex(ValueError, "--samples-per-cell"):
            build_camera_calibration_command(
                CameraCalibrationConfig(
                    project_root="project",
                    samples_per_cell=0,
                )
            )

    def test_readiness_reports_missing_metadata_before_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            result = inspect_camera_calibration_readiness(
                CameraCalibrationConfig(project_root=project_root),
                realsense_available=True,
            )

        self.assertFalse(result.ready)
        self.assertEqual(result.command_preview, "")
        self.assertTrue(
            any("Missing ChArUco board metadata" in error for error in result.errors)
        )
        self.assertIn("Calibration output is missing", "\n".join(result.warnings))

    def test_readiness_builds_command_when_metadata_exists(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            metadata_path = _write_metadata(project_root)

            result = inspect_camera_calibration_readiness(
                camera_calibration_config_from_project(project_root),
                realsense_available=True,
            )

        self.assertTrue(result.ready)
        self.assertEqual(result.metadata_path, metadata_path)
        self.assertIn("posetag-calib-charuco", result.command_preview)
        self.assertIn("--squares-x 3 --squares-y 5", result.command_preview)
        self.assertFalse(result.output_exists)
        self.assertIn(result.expected_output, result.checked_paths)

    def test_readiness_rejects_invalid_guided_capture_options(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_metadata(project_root)

            result = inspect_camera_calibration_readiness(
                CameraCalibrationConfig(
                    project_root=project_root,
                    coverage_grid="bad",
                ),
                realsense_available=True,
            )

        self.assertFalse(result.ready)
        self.assertEqual(result.command_preview, "")
        self.assertTrue(any("--coverage-grid" in error for error in result.errors))

    def test_readiness_rejects_edited_invalid_board_lengths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_metadata(project_root)

            result = inspect_camera_calibration_readiness(
                CameraCalibrationConfig(
                    project_root=project_root,
                    square_length_mm=10.0,
                    marker_length_mm=20.0,
                ),
                realsense_available=True,
            )

        self.assertFalse(result.ready)
        self.assertEqual(result.command_preview, "")
        self.assertTrue(
            any("marker-length-mm" in error for error in result.errors)
        )

    def test_readiness_rejects_edited_unsupported_dictionary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_metadata(project_root)

            result = inspect_camera_calibration_readiness(
                CameraCalibrationConfig(
                    project_root=project_root,
                    dictionary_name="NOT_A_DICT",
                ),
                realsense_available=True,
            )

        self.assertFalse(result.ready)
        self.assertEqual(result.command_preview, "")
        self.assertTrue(
            any("Unsupported ArUco dictionary" in error for error in result.errors)
        )

    def test_readiness_rejects_video_without_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_metadata(project_root)

            result = inspect_camera_calibration_readiness(
                camera_calibration_config_from_project(
                    project_root,
                    source=SOURCE_VIDEO,
                ),
                realsense_available=True,
            )

        self.assertFalse(result.ready)
        self.assertEqual(result.command_preview, "")
        self.assertTrue(any("--video path is required" in e for e in result.errors))

    def test_readiness_reports_unavailable_realsense_dependency(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_metadata(project_root)

            result = inspect_camera_calibration_readiness(
                camera_calibration_config_from_project(
                    project_root,
                    source=SOURCE_REALSENSE,
                ),
                realsense_available=False,
            )

        self.assertFalse(result.ready)
        self.assertEqual(result.command_preview, "")
        self.assertTrue(any("pyrealsense2 is not available" in e for e in result.errors))

    def test_latest_metadata_is_selected_when_multiple_exist(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            first = _write_metadata(project_root, dictionary="5X5_50")
            time.sleep(0.001)
            second = _write_metadata(project_root, dictionary="7X7_100")

            inspection = inspect_charuco_metadata(project_root)
            paths = generated_charuco_metadata_paths(project_root)

        self.assertEqual(paths, tuple(sorted([first, second])))
        self.assertEqual(inspection.selected_path, second)
        self.assertIsNotNone(inspection.metadata)
        self.assertEqual(inspection.metadata.dictionary_name, "7X7_100")
        self.assertTrue(inspection.warnings)

    def test_malformed_metadata_blocks_readiness(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            metadata_path = (
                project_root
                / "calib"
                / "boards"
                / "charuco_3x5_square50mm_marker37mm_7X7_50_A4_300dpi.yaml"
            )
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text("squares_x: 3\n", encoding="utf-8")

            result = inspect_camera_calibration_readiness(
                CameraCalibrationConfig(project_root=project_root),
                realsense_available=True,
            )

        self.assertFalse(result.ready)
        self.assertTrue(any("squares_y" in error for error in result.errors))
        self.assertEqual(result.command_preview, "")

    def test_process_result_requires_valid_calibration_yaml_for_success(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_calibration(project_root / "calib" / "calib_color.yaml")

            state = summarize_camera_calibration_process_result(
                project_root,
                exit_code=0,
            )

        self.assertEqual(state.state, CALIBRATION_PROCESS_FINISHED)
        self.assertTrue(state.success)
        self.assertIn("passed", state.message)

    def test_process_result_does_not_treat_old_yaml_as_failed_rerun_success(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            _write_calibration(calib_path)
            previous_mtime = calib_path.stat().st_mtime_ns

            state = summarize_camera_calibration_process_result(
                project_root,
                exit_code=0,
                previous_output_mtime_ns=previous_mtime,
                require_output_update=True,
            )

        self.assertEqual(state.state, CALIBRATION_PROCESS_FAILED_CANCELLED)
        self.assertFalse(state.success)
        self.assertIn("not updated", state.message)

    def test_process_result_failed_when_exit_nonzero_even_with_valid_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_calibration(project_root / "calib" / "calib_color.yaml")

            state = summarize_camera_calibration_process_result(
                project_root,
                exit_code=2,
            )

        self.assertEqual(state.state, CALIBRATION_PROCESS_FAILED_CANCELLED)
        self.assertFalse(state.success)
        self.assertIn("code 2", state.message)

    def test_process_result_failed_when_zero_exit_without_valid_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"

            state = summarize_camera_calibration_process_result(
                project_root,
                exit_code=0,
            )

        self.assertEqual(state.state, CALIBRATION_PROCESS_FAILED_CANCELLED)
        self.assertFalse(state.success)
        self.assertIn("missing", state.message)

    def test_process_result_failed_when_yaml_schema_is_invalid(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            calib_path.parent.mkdir(parents=True)
            calib_path.write_text("not_camera_matrix: true\n", encoding="utf-8")

            state = summarize_camera_calibration_process_result(
                project_root,
                exit_code=0,
            )

        self.assertEqual(state.state, CALIBRATION_PROCESS_FAILED_CANCELLED)
        self.assertFalse(state.success)
        self.assertIn("did not pass", state.message)

    def test_output_summary_parses_valid_calibration_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            _write_calibration(calib_path)
            run_dir = project_root / "calib" / "runs" / "2026-05-29T08-19-06Z"
            run_dir.mkdir(parents=True)

            summary = inspect_camera_calibration_output(project_root)

        self.assertTrue(summary.exists)
        self.assertTrue(summary.valid)
        self.assertEqual(summary.path, calib_path)
        self.assertEqual(summary.image_width, 640)
        self.assertEqual(summary.image_height, 480)
        self.assertEqual(summary.model, "plumb_bob")
        self.assertEqual(summary.reproj_rms, 0.12)
        self.assertEqual(summary.camera_params, (600.0, 610.0, 320.0, 240.0))
        self.assertIn(("k1", 0.0), summary.distortion_coefficients)
        self.assertEqual(summary.latest_run_dir, run_dir)

    def test_output_summary_reports_missing_calibration_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"

            summary = inspect_camera_calibration_output(project_root)

        self.assertFalse(summary.exists)
        self.assertFalse(summary.valid)
        self.assertIn("missing", summary.message)

    def test_read_calibration_yaml_text_returns_raw_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            calib_path = Path(tmpdir) / "calib_color.yaml"
            _write_calibration(calib_path)

            text = read_camera_calibration_yaml_text(calib_path)

        self.assertIn("camera_matrix", text)
        self.assertIn("reproj_rms", text)


def _write_metadata(
    project_root: Path,
    *,
    squares_x: int = 3,
    squares_y: int = 5,
    square_length_mm: float = 50.0,
    marker_length_mm: float = 37.0,
    dictionary: str = "7X7_50",
) -> Path:
    path = (
        project_root
        / "calib"
        / "boards"
        / (
            f"charuco_{squares_x}x{squares_y}_square{square_length_mm:g}mm_"
            f"marker{marker_length_mm:g}mm_{dictionary}_A4_300dpi.yaml"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "squares_x": squares_x,
                "squares_y": squares_y,
                "square_length_mm": square_length_mm,
                "marker_length_mm": marker_length_mm,
                "dictionary": dictionary,
                "paper": "A4",
                "dpi": 300,
                "png": path.with_suffix(".png").name,
                "notes": "Print at 100% / Actual Size.",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_calibration(path: Path) -> None:
    data = build_calibration_yaml(
        image_width=640,
        image_height=480,
        camera_matrix=[
            [600.0, 0.0, 320.0],
            [0.0, 610.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        distortion_coefficients=[[0.0, 0.0, 0.0, 0.0, 0.0]],
        reproj_rms=0.12,
        notes="Synthetic calibration for process-state tests.",
    )
    write_calibration_yaml(path, data)


if __name__ == "__main__":
    unittest.main()
