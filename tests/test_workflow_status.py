from __future__ import annotations

import json
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
from posetag.pipelines.capture_face import (
    append_capture_manifest,
    build_capture_metadata,
    build_shot_paths,
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


def _write_two_face_board_and_registry(project_root: Path) -> tuple[dict[str, Path], Path]:
    board_paths: dict[str, Path] = {}
    tags: dict[str, dict[str, str]] = {}
    next_id = 52
    for side in ("sideA", "sideB"):
        object_name = f"connection_plate_white_{side}"
        board_path = project_root / "boards" / f"{object_name}.yaml"
        board = build_board_yaml(
            object_name=object_name,
            family="tag36h11",
            tag_size_mm=80.0,
            origin_id=next_id,
            entries=[
                {"id": next_id, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
                {"id": next_id + 1, "cx": 0.1, "cy": -0.2, "yaw_deg": 180.0},
            ],
        )
        write_board_yaml(board_path, board)
        tags[str(next_id)] = {"object": object_name, "yaml": str(board_path)}
        tags[str(next_id + 1)] = {"object": object_name, "yaml": str(board_path)}
        board_paths[side] = board_path
        next_id += 2

    registry_path = project_root / "boards" / "tag_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "updated": "2026-05-30T12:00:00Z",
                "tags": tags,
            }
        ),
        encoding="utf-8",
    )
    return board_paths, registry_path


def _write_face_capture(
    project_root: Path,
    board_path: Path,
    *,
    timestamp: str,
    write_raw: bool = True,
) -> None:
    data = yaml.safe_load(board_path.read_text(encoding="utf-8"))
    object_full = str(data["object"])
    tag_ids = [int(tag["id"]) for tag in data["tags"]]
    shot_paths = build_shot_paths(
        layout="by_object_side",
        out_dir=project_root / "shots",
        object_full=object_full,
        timestamp=timestamp,
    )
    metadata = build_capture_metadata(
        face={"yaml": str(board_path), "object": object_full, "tag_ids": tag_ids},
        shot_paths=shot_paths,
        detected_tag_ids=tag_ids,
        validation_ok=True,
        auto_face=True,
        frame_shape=(480, 640, 3),
        camera_params=(600.0, 610.0, 320.0, 240.0),
        timestamp=timestamp,
    )
    shot_paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    if write_raw:
        shot_paths.raw_path.write_bytes(b"raw")
    shot_paths.ann_path.write_bytes(b"ann")
    shot_paths.meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    append_capture_manifest(project_root / "shots" / "manifest.csv", metadata)


def _write_valid_keypoints(project_root: Path, object_name: str) -> Path:
    path = project_root / "objects" / object_name / "keypoints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "units_to_m": 1.0,
                "points": {
                    "xmin_ymin_zmin": [0.0, 0.0, 0.0],
                    "xmin_ymin_zmax": [0.0, 0.0, 1.0],
                    "xmin_ymax_zmin": [0.0, 1.0, 0.0],
                    "xmin_ymax_zmax": [0.0, 1.0, 1.0],
                    "xmax_ymin_zmin": [1.0, 0.0, 0.0],
                    "xmax_ymin_zmax": [1.0, 0.0, 1.0],
                    "xmax_ymax_zmin": [1.0, 1.0, 0.0],
                    "xmax_ymax_zmax": [1.0, 1.0, 1.0],
                },
                "faces": {
                    f"{object_name}_sideA": [
                        "xmax_ymax_zmin",
                        "xmax_ymax_zmax",
                        "xmax_ymin_zmax",
                        "xmax_ymin_zmin",
                    ],
                    f"{object_name}_sideB": [
                        "xmin_ymax_zmin",
                        "xmin_ymax_zmax",
                        "xmin_ymin_zmax",
                        "xmin_ymin_zmin",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class WorkflowStatusTests(unittest.TestCase):
    def test_empty_project_reports_calibration_first_gui_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            summaries = inspect_project(project_root)
            by_id = {summary.stage_id: summary for summary in summaries}

            self.assertEqual(by_id[0].status, WorkflowStatus.MISSING)
            self.assertEqual(by_id[1].status, WorkflowStatus.MISSING)
            self.assertEqual(by_id[2].status, WorkflowStatus.NOT_APPLICABLE)
            self.assertEqual(by_id[3].status, WorkflowStatus.NOT_APPLICABLE)
            self.assertEqual(by_id[4].status, WorkflowStatus.NOT_APPLICABLE)
            self.assertEqual(len(summaries), 10)
            self.assertEqual(
                [summary.name for summary in summaries],
                [
                    "Project Setup",
                    "Generate ChArUco Calibration Board",
                    "Calibrate Camera",
                    "Generate Object AprilTags",
                    "Build Object Board Definitions",
                    "Capture Face Shots",
                    "Generate Mesh Keypoints / Object Geometry",
                    "Annotate Faces / Board-To-Object Transforms",
                    "Collect Dataset / Estimate Poses",
                    "Review And Export",
                ],
            )
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

    def test_patterns_png_marks_object_tag_stage_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            patterns_dir = project_root / "boards" / "patterns"
            patterns_dir.mkdir(parents=True)
            (patterns_dir / "apriltag_sheet.png").write_bytes(
                b"not-an-image-but-a-png-path"
            )

            stage = _stage_by_id(project_root, 3)

            self.assertEqual(stage.status, WorkflowStatus.COMPLETE)
            self.assertIn(str(patterns_dir / "apriltag_sheet.png"), stage.checked_paths)

    def test_valid_calib_color_yaml_marks_calibration_stage_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")

            stage = _stage_by_id(project_root, 2)

            self.assertEqual(stage.status, WorkflowStatus.COMPLETE)
            self.assertEqual(stage.errors, ())

    def test_generated_charuco_metadata_is_stage_before_calibration(self) -> None:
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

            charuco_stage = _stage_by_id(project_root, 1)
            calibration_stage = _stage_by_id(project_root, 2)

            self.assertEqual(charuco_stage.status, WorkflowStatus.COMPLETE)
            self.assertIn(str(metadata_path), charuco_stage.checked_paths)
            self.assertIn(
                "generated ChArUco calibration board metadata",
                charuco_stage.message,
            )
            self.assertIn("Actual Size", charuco_stage.next_action)
            self.assertEqual(calibration_stage.status, WorkflowStatus.MISSING)
            self.assertIn(str(metadata_path), calibration_stage.checked_paths)

    def test_malformed_calib_color_yaml_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            calib_path.parent.mkdir(parents=True)
            calib_path.write_text("not_camera_matrix: true\n", encoding="utf-8")

            stage = _stage_by_id(project_root, 2)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(stage.errors)
            self.assertIn("Malformed calibration YAML", stage.errors[0])

    def test_calib_color_yaml_missing_workflow_fields_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            calib_path.parent.mkdir(parents=True)
            calib_path.write_text(
                yaml.safe_dump(
                    {
                        "camera_matrix": {
                            "fx": 600.0,
                            "fy": 610.0,
                            "cx": 320.0,
                            "cy": 240.0,
                        },
                        "distortion_coefficients": {
                            "k1": 0.0,
                            "k2": 0.0,
                            "p1": 0.0,
                            "p2": 0.0,
                            "k3": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            stage = _stage_by_id(project_root, 2)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(any("image_width" in error for error in stage.errors))
            self.assertTrue(any("reproj_rms" in error for error in stage.errors))

    def test_valid_board_yaml_and_registry_mark_board_definition_stage_complete(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path, registry_path = _write_valid_board_and_registry(project_root)

            stage = _stage_by_id(project_root, 4)

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

            stage = _stage_by_id(project_root, 4)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(stage.errors)
            self.assertIn("was not found", stage.errors[0])

    def test_registry_tag_missing_from_board_yaml_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path = project_root / "boards" / "connection_plate_white_sideA.yaml"
            registry_path = project_root / "boards" / "tag_registry.yaml"
            board = build_board_yaml(
                object_name="connection_plate_white_sideA",
                family="tag36h11",
                tag_size_mm=80.0,
                origin_id=52,
                entries=[{"id": 52, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0}],
            )
            write_board_yaml(board_path, board)
            registry_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "updated": "2026-05-27T12:00:00Z",
                        "tags": {
                            "99": {
                                "object": "connection_plate_white_sideA",
                                "yaml": str(board_path),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stage = _stage_by_id(project_root, 4)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(any("not present" in error for error in stage.errors))

    def test_registry_object_mismatch_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path, registry_path = _write_valid_board_and_registry(project_root)
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            registry["tags"]["52"]["object"] = "different_object"
            registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

            stage = _stage_by_id(project_root, 4)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(any("different_object" in error for error in stage.errors))
            self.assertTrue(any(str(board_path) in error for error in stage.errors))

    def test_board_definitions_without_face_shots_marks_stage5_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_board_and_registry(project_root)

            stage5 = _stage_by_id(project_root, 5)
            stage6 = _stage_by_id(project_root, 6)
            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage5.status, WorkflowStatus.MISSING)
            self.assertIn("No face-shot manifest", stage5.message)
            self.assertEqual(stage6.status, WorkflowStatus.NOT_APPLICABLE)
            self.assertEqual(stage7.status, WorkflowStatus.NOT_APPLICABLE)

    def test_partial_face_shot_coverage_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_paths, _registry_path = _write_two_face_board_and_registry(
                project_root
            )
            _write_face_capture(
                project_root,
                board_paths["sideA"],
                timestamp="20260530_120000",
            )

            stage = _stage_by_id(project_root, 5)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertIn("1/2 registered faces", stage.message)
            self.assertIn("connection_plate_white_sideB", stage.message)

    def test_valid_face_shots_for_all_registered_faces_mark_stage5_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_paths, _registry_path = _write_two_face_board_and_registry(project_root)
            _write_face_capture(
                project_root,
                board_paths["sideA"],
                timestamp="20260530_120000",
            )
            _write_face_capture(
                project_root,
                board_paths["sideB"],
                timestamp="20260530_120100",
            )

            stage5 = _stage_by_id(project_root, 5)
            stage6 = _stage_by_id(project_root, 6)
            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage5.status, WorkflowStatus.COMPLETE)
            self.assertIn("valid face-shot coverage for 2 registered faces", stage5.message)
            self.assertEqual(stage6.status, WorkflowStatus.MISSING)
            self.assertIn("Missing annotation-ready keypoints", stage6.message)
            self.assertIn(
                str(project_root / "objects" / "connection_plate_white" / "keypoints.json"),
                stage6.checked_paths,
            )
            self.assertEqual(stage7.status, WorkflowStatus.NOT_APPLICABLE)

    def test_valid_keypoints_after_face_shots_mark_stage6_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_paths, _registry_path = _write_two_face_board_and_registry(project_root)
            _write_face_capture(
                project_root,
                board_paths["sideA"],
                timestamp="20260530_120000",
            )
            _write_face_capture(
                project_root,
                board_paths["sideB"],
                timestamp="20260530_120100",
            )
            keypoints_path = _write_valid_keypoints(
                project_root,
                "connection_plate_white",
            )

            stage6 = _stage_by_id(project_root, 6)
            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage6.status, WorkflowStatus.COMPLETE)
            self.assertIn("annotation-ready keypoints for 1 object", stage6.message)
            self.assertIn(str(keypoints_path), stage6.checked_paths)
            self.assertEqual(stage7.status, WorkflowStatus.MISSING)
            self.assertIn("Face annotation status", stage7.message)

    def test_keypoints_missing_captured_side_keep_stage6_invalid(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_paths, _registry_path = _write_two_face_board_and_registry(project_root)
            _write_face_capture(
                project_root,
                board_paths["sideA"],
                timestamp="20260530_120000",
            )
            _write_face_capture(
                project_root,
                board_paths["sideB"],
                timestamp="20260530_120100",
            )
            keypoints_path = _write_valid_keypoints(project_root, "connection_plate_white")
            payload = json.loads(keypoints_path.read_text(encoding="utf-8"))
            del payload["faces"]["connection_plate_white_sideB"]
            keypoints_path.write_text(json.dumps(payload), encoding="utf-8")

            stage6 = _stage_by_id(project_root, 6)
            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage6.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(
                any("connection_plate_white_sideB" in error for error in stage6.errors)
            )
            self.assertEqual(stage7.status, WorkflowStatus.NOT_APPLICABLE)

    def test_invalid_keypoints_after_face_shots_need_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path, _registry_path = _write_valid_board_and_registry(project_root)
            _write_face_capture(
                project_root,
                board_path,
                timestamp="20260530_120000",
            )
            keypoints_path = project_root / "objects" / "connection_plate_white" / "keypoints.json"
            keypoints_path.parent.mkdir(parents=True, exist_ok=True)
            keypoints_path.write_text(
                json.dumps({"units_to_m": 1.0, "points": {}, "faces": {}}),
                encoding="utf-8",
            )

            stage6 = _stage_by_id(project_root, 6)
            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage6.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(any("points" in error for error in stage6.errors))
            self.assertEqual(stage7.status, WorkflowStatus.NOT_APPLICABLE)

    def test_broken_face_shot_output_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path, _registry_path = _write_valid_board_and_registry(project_root)
            _write_face_capture(
                project_root,
                board_path,
                timestamp="20260530_120000",
                write_raw=False,
            )

            stage = _stage_by_id(project_root, 5)

            self.assertEqual(stage.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertIn("0/1 registered faces", stage.message)
            self.assertTrue(any("raw image was not found" in warning for warning in stage.warnings))

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
