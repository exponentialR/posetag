from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from posetag.workflows.calibration_flow import (
    SOURCE_OPENCV,
    SOURCE_REALSENSE,
    SOURCE_VIDEO,
    CameraCalibrationConfig,
    build_camera_calibration_command,
    camera_calibration_config_from_project,
    generated_charuco_metadata_paths,
    inspect_camera_calibration_readiness,
    inspect_charuco_metadata,
    load_charuco_metadata,
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


if __name__ == "__main__":
    unittest.main()
