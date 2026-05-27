from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from posetag.pipelines.charuco_calibration import (
    build_calibration_yaml,
    write_calibration_yaml,
)
from posetag.pipelines.make_board import build_board_yaml, write_board_yaml
from posetag.workflows.status import WorkflowStatus, inspect_project


def _stage_by_id(project_root: Path, stage_id: int):
    summaries = inspect_project(project_root)
    return {summary.stage_id: summary for summary in summaries}[stage_id]


def _write_valid_calibration(path: Path) -> None:
    camera_matrix = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 610.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    data = build_calibration_yaml(
        image_width=640,
        image_height=480,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        reproj_rms=0.12,
        notes="Synthetic calibration for workflow status tests.",
    )
    write_calibration_yaml(path, data)


def _write_valid_board_and_registry(project_root: Path) -> tuple[Path, Path]:
    board_path = project_root / "boards" / "connection_plate_white_sideA.yaml"
    registry_path = project_root / "boards" / "tag_registry.yaml"
    board = build_board_yaml(
        object_name="connection_plate_white_sideA",
        family="tag36h11",
        tag_size_mm=80.0,
        origin_id=52,
        entries=[
            {"id": 52, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
            {"id": 53, "cx": 0.1, "cy": -0.2, "yaw_deg": 180.0},
        ],
    )
    write_board_yaml(board_path, board)
    registry = {
        "version": 1,
        "updated": "2026-05-27T12:00:00Z",
        "tags": {
            "52": {"object": "connection_plate_white_sideA", "yaml": str(board_path)},
            "53": {"object": "connection_plate_white_sideA", "yaml": str(board_path)},
        },
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return board_path, registry_path


class WorkflowStatusTests(unittest.TestCase):
    def test_empty_project_reports_steps_0_to_2_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            summaries = inspect_project(project_root)
            by_id = {summary.stage_id: summary for summary in summaries}

            self.assertEqual(by_id[0].status, WorkflowStatus.MISSING)
            self.assertEqual(by_id[1].status, WorkflowStatus.MISSING)
            self.assertEqual(by_id[2].status, WorkflowStatus.MISSING)
            self.assertEqual(len(summaries), 7)
            for summary in summaries:
                rendered = summary.to_dict()
                for key in (
                    "stage_id",
                    "key",
                    "name",
                    "status",
                    "message",
                    "checked_paths",
                    "warnings",
                    "errors",
                    "next_action",
                ):
                    self.assertIn(key, rendered)

    def test_patterns_png_marks_step0_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            patterns_dir = project_root / "boards" / "patterns"
            patterns_dir.mkdir(parents=True)
            (patterns_dir / "apriltag_sheet.png").write_bytes(
                b"not-an-image-but-a-png-path"
            )

            stage = _stage_by_id(project_root, 0)

            self.assertEqual(stage.status, WorkflowStatus.COMPLETE)
            self.assertIn(str(patterns_dir / "apriltag_sheet.png"), stage.checked_paths)

    def test_valid_calib_color_yaml_marks_step1_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")

            stage = _stage_by_id(project_root, 1)

            self.assertEqual(stage.status, WorkflowStatus.COMPLETE)
            self.assertEqual(stage.errors, ())

    def test_malformed_calib_color_yaml_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            calib_path.parent.mkdir(parents=True)
            calib_path.write_text("not_camera_matrix: true\n", encoding="utf-8")

            stage = _stage_by_id(project_root, 1)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(stage.errors)
            self.assertIn("Malformed calibration YAML", stage.errors[0])

    def test_valid_board_yaml_and_registry_mark_step2_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path, registry_path = _write_valid_board_and_registry(project_root)

            stage = _stage_by_id(project_root, 2)

            self.assertEqual(stage.status, WorkflowStatus.COMPLETE)
            self.assertEqual(stage.errors, ())
            self.assertIn(str(registry_path), stage.checked_paths)
            self.assertIn(str(board_path), stage.checked_paths)

    def test_registry_referencing_missing_board_yaml_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            registry_path = project_root / "boards" / "tag_registry.yaml"
            missing_board = project_root / "boards" / "missing_board.yaml"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "updated": "2026-05-27T12:00:00Z",
                        "tags": {
                            "52": {
                                "object": "connection_plate_white_sideA",
                                "yaml": str(missing_board),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stage = _stage_by_id(project_root, 2)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(stage.errors)
            self.assertIn("was not found", stage.errors[0])

    def test_workflow_helpers_import_without_gui_dependency(self) -> None:
        code = (
            "from posetag.workflows.status import inspect_project\n"
            "import sys\n"
            "raise SystemExit(1 if 'PySide6' in sys.modules else 0)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
