from __future__ import annotations

import json
import csv
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
from posetag.pipelines.collect_dataset import (
    Intrinsics,
    build_frame_annotation_record,
    compose_T_cam_object,
)
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


def _write_face_annotation(
    project_root: Path,
    *,
    object_name: str,
    side: str,
    board_path: Path,
    image_path: Path,
    rms_px: float = 0.25,
) -> Path:
    face_key = f"{object_name}_{side}"
    path = project_root / "faces" / object_name / side / f"{face_key}_T_board_object.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "object": object_name,
                "face_key": face_key,
                "board_yaml": str(board_path),
                "image": str(image_path),
                "rms_px": rms_px,
                "T_board_object": {"matrix": np.eye(4).tolist()},
                "corner_uv": {
                    "p0": [10.0, 20.0],
                    "p1": [30.0, 20.0],
                    "p2": [30.0, 40.0],
                    "p3": [10.0, 40.0],
                },
                "pnp": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 1.0]},
                "clicked_uv_raw": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
                "face_corner_names": ["p0", "p1", "p2", "p3"],
                "assignment": {
                    "p0": [10.0, 20.0],
                    "p1": [30.0, 20.0],
                    "p2": [30.0, 40.0],
                    "p3": [10.0, 40.0],
                },
                "diagnostics": {
                    "board_tag_size_mm": 80.0,
                    "tag_scale_ratio": 1.0,
                    "tag_scale_pairs": 1,
                    "tag_scale_auto_corrected": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_face_manifest(project_root: Path, annotation_paths: list[Path]) -> Path:
    path = project_root / "faces" / "face_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "object",
                "side",
                "face_key",
                "yaml_path",
                "board_yaml",
                "image",
                "rms_px",
                "tag_size_m",
            ],
        )
        writer.writeheader()
        for annotation_path in annotation_paths:
            data = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
            writer.writerow(
                {
                    "timestamp": "2026-06-06T12:00:00",
                    "object": data["object"],
                    "side": data["face_key"].rsplit("_", 1)[-1],
                    "face_key": data["face_key"],
                    "yaml_path": str(annotation_path.relative_to(project_root)),
                    "board_yaml": data["board_yaml"],
                    "image": data["image"],
                    "rms_px": str(data["rms_px"]),
                    "tag_size_m": "0.080000",
                }
            )
    return path


def _write_object_tag_pattern(project_root: Path) -> Path:
    pattern_path = project_root / "boards" / "patterns" / "apriltag_sheet.png"
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    pattern_path.write_bytes(b"synthetic pattern")
    return pattern_path


def _write_collection_ready_project(project_root: Path) -> dict[str, Path]:
    _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
    _write_object_tag_pattern(project_root)
    board_paths, registry_path = _write_two_face_board_and_registry(project_root)
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
    _write_valid_keypoints(project_root, "connection_plate_white")
    side_a_image = (
        project_root
        / "shots"
        / "connection_plate_white"
        / "sideA"
        / "connection_plate_white_sideA_20260530_120000_raw.png"
    )
    side_b_image = (
        project_root
        / "shots"
        / "connection_plate_white"
        / "sideB"
        / "connection_plate_white_sideB_20260530_120100_raw.png"
    )
    annotation_paths = [
        _write_face_annotation(
            project_root,
            object_name="connection_plate_white",
            side="sideA",
            board_path=board_paths["sideA"],
            image_path=side_a_image,
        ),
        _write_face_annotation(
            project_root,
            object_name="connection_plate_white",
            side="sideB",
            board_path=board_paths["sideB"],
            image_path=side_b_image,
        ),
    ]
    manifest_path = _write_face_manifest(project_root, annotation_paths)
    return {
        "manifest": manifest_path,
        "registry": registry_path,
        "sideA_board": board_paths["sideA"],
        "sideB_board": board_paths["sideB"],
    }


def _write_dataset_session(
    project_root: Path,
    board_path: Path,
    *,
    break_composition: bool = False,
) -> Path:
    session_dir = project_root / "datasets" / "run01"
    for folder in ("images", "annotated", "reproj", "annotations"):
        (session_dir / folder).mkdir(parents=True, exist_ok=True)

    image_name = "Run_20260607-120000_rgb_frame_000000.png"
    annotated_name = "Run_20260607-120000_annotated_frame_000000.png"
    reproj_name = "Run_20260607-120000_reproj_frame_000000.png"
    (session_dir / "images" / image_name).write_bytes(b"raw")
    (session_dir / "annotated" / annotated_name).write_bytes(b"annotated")
    (session_dir / "reproj" / reproj_name).write_bytes(b"reproj")

    intrinsics = Intrinsics(
        path=project_root / "calib" / "calib_color.yaml",
        fx=600.0,
        fy=610.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        dist=np.zeros((1, 5), dtype=float),
    )
    T_cam_board = np.eye(4)
    T_cam_board[:3, 3] = [0.2, -0.1, 1.5]
    T_board_object = np.eye(4)
    T_board_object[:3, 3] = [0.03, 0.02, 0.01]
    T_cam_object = compose_T_cam_object(T_cam_board, T_board_object)
    result = {
        "object": "connection_plate_white",
        "face_key": "connection_plate_white_sideA",
        "board_yaml": str(board_path),
        "tag_used": 52,
        "num_tags_visible": 2,
        "score": 2500.0,
        "score_tags": 2,
        "score_area_px": 500,
        "bbox_xywh": [64.0, 48.0, 128.0, 96.0],
        "bbox_source": "face_kps",
        "T_cam_board": {"matrix": T_cam_board.tolist()},
        "T_board_object": {"matrix": T_board_object.tolist()},
        "T_cam_object": {"matrix": T_cam_object.tolist()},
        "diagnostics": {
            "tag_scale_ratio": 1.0,
            "tag_scale_pairs": 1,
            "tag_scale_mad": 0.0,
            "tag_scale_auto_corrected": False,
        },
    }
    record = build_frame_annotation_record(
        dataset_name="run01",
        frame_index=0,
        timestamp=123.456,
        tag_family="tag36h11",
        image_filename=image_name,
        annotated_image=annotated_name,
        reproj_image=reproj_name,
        depth=None,
        intrinsics=intrinsics,
        results={"connection_plate_white_sideA": result},
    )
    if break_composition:
        record["objects"][0]["transforms"]["T_cam_object"]["matrix"][0][3] += 1.0
    (session_dir / "annotations" / "Run_20260607-120000_frame_000000.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    (session_dir / "run01.jsonl").write_text(
        json.dumps({"idx": 0, "image": image_name, "objects": ["connection_plate_white"]}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "session.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "name": "run01",
                "camera": {
                    "kind": "opencv_webcam",
                    "fx": 600.0,
                    "fy": 610.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "width": 640,
                    "height": 480,
                },
                "pose_convention": {
                    "composition": "T_cam_object = T_cam_board @ T_board_object",
                    "translation_units": "meters",
                    "quaternion_order": "x, y, z, w",
                    "rpy_order": "roll, pitch, yaw",
                    "angle_units": "degrees",
                },
                "frames_captured": 1,
            }
        ),
        encoding="utf-8",
    )
    return session_dir


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
            self.assertIn("Face annotations are incomplete: 0/2", stage7.message)
            self.assertIn(
                str(
                    project_root
                    / "faces"
                    / "connection_plate_white"
                    / "sideA"
                    / "connection_plate_white_sideA_T_board_object.yaml"
                ),
                stage7.checked_paths,
            )

    def test_valid_face_annotations_after_keypoints_mark_stage7_complete(self) -> None:
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
            _write_valid_keypoints(project_root, "connection_plate_white")
            side_a_image = (
                project_root
                / "shots"
                / "connection_plate_white"
                / "sideA"
                / "connection_plate_white_sideA_20260530_120000_raw.png"
            )
            side_b_image = (
                project_root
                / "shots"
                / "connection_plate_white"
                / "sideB"
                / "connection_plate_white_sideB_20260530_120100_raw.png"
            )
            annotation_paths = [
                _write_face_annotation(
                    project_root,
                    object_name="connection_plate_white",
                    side="sideA",
                    board_path=board_paths["sideA"],
                    image_path=side_a_image,
                ),
                _write_face_annotation(
                    project_root,
                    object_name="connection_plate_white",
                    side="sideB",
                    board_path=board_paths["sideB"],
                    image_path=side_b_image,
                ),
            ]
            manifest_path = _write_face_manifest(project_root, annotation_paths)

            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage7.status, WorkflowStatus.COMPLETE)
            self.assertIn("valid board-to-object annotations for 2 faces", stage7.message)
            self.assertIn(str(manifest_path), stage7.checked_paths)

    def test_collection_ready_project_marks_stage8_missing_not_not_applicable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            paths = _write_collection_ready_project(project_root)

            stage8 = _stage_by_id(project_root, 8)

            self.assertEqual(stage8.status, WorkflowStatus.MISSING)
            self.assertIn("inputs are ready", stage8.message)
            self.assertIn(str(paths["manifest"].resolve()), stage8.checked_paths)
            self.assertIn(str(paths["registry"].resolve()), stage8.checked_paths)
            self.assertIn("posetag-collect --dry-run", stage8.next_action)

    def test_valid_dataset_session_marks_stage8_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            paths = _write_collection_ready_project(project_root)
            session_dir = _write_dataset_session(project_root, paths["sideA_board"])

            stage8 = _stage_by_id(project_root, 8)
            stage9 = _stage_by_id(project_root, 9)

            self.assertEqual(stage8.status, WorkflowStatus.COMPLETE)
            self.assertIn("valid pose-labelled frame", stage8.message)
            self.assertIn(str((session_dir / "session.yaml").resolve()), stage8.checked_paths)
            self.assertIn(
                str((session_dir / "annotations" / "Run_20260607-120000_frame_000000.json").resolve()),
                stage8.checked_paths,
            )
            self.assertEqual(stage9.status, WorkflowStatus.MISSING)
            self.assertIn("manual review/export", stage9.message)

    def test_dataset_session_with_bad_pose_composition_needs_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            paths = _write_collection_ready_project(project_root)
            _write_dataset_session(
                project_root,
                paths["sideA_board"],
                break_composition=True,
            )

            stage8 = _stage_by_id(project_root, 8)

            self.assertEqual(stage8.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(
                any("T_cam_object = T_cam_board @ T_board_object" in error for error in stage8.errors),
                stage8.errors,
            )

    def test_stale_face_manifest_row_needs_attention(self) -> None:
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
            _write_valid_keypoints(project_root, "connection_plate_white")
            side_a_image = (
                project_root
                / "shots"
                / "connection_plate_white"
                / "sideA"
                / "connection_plate_white_sideA_20260530_120000_raw.png"
            )
            side_b_image = (
                project_root
                / "shots"
                / "connection_plate_white"
                / "sideB"
                / "connection_plate_white_sideB_20260530_120100_raw.png"
            )
            annotation_paths = [
                _write_face_annotation(
                    project_root,
                    object_name="connection_plate_white",
                    side="sideA",
                    board_path=board_paths["sideA"],
                    image_path=side_a_image,
                ),
                _write_face_annotation(
                    project_root,
                    object_name="connection_plate_white",
                    side="sideB",
                    board_path=board_paths["sideB"],
                    image_path=side_b_image,
                ),
            ]
            manifest_path = _write_face_manifest(project_root, annotation_paths)
            with manifest_path.open("r", newline="", encoding="utf-8") as handle:
                fieldnames = csv.DictReader(handle).fieldnames
            self.assertIsNotNone(fieldnames)
            with manifest_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "timestamp": "2026-06-06T12:10:00",
                        "object": "connection_plate_white",
                        "side": "old",
                        "face_key": "connection_plate_white_old",
                        "yaml_path": (
                            "faces/connection_plate_white/old/"
                            "connection_plate_white_old_T_board_object.yaml"
                        ),
                        "board_yaml": str(board_paths["sideA"]),
                        "image": str(side_a_image),
                        "rms_px": "0.25",
                        "tag_size_m": "0.080000",
                    }
                )

            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage7.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertTrue(
                any("stale annotation row" in error for error in stage7.errors),
                stage7.errors,
            )

    def test_face_annotation_manifest_missing_needs_attention(self) -> None:
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
            _write_valid_keypoints(project_root, "connection_plate_white")
            side_a_image = (
                project_root
                / "shots"
                / "connection_plate_white"
                / "sideA"
                / "connection_plate_white_sideA_20260530_120000_raw.png"
            )
            side_b_image = (
                project_root
                / "shots"
                / "connection_plate_white"
                / "sideB"
                / "connection_plate_white_sideB_20260530_120100_raw.png"
            )
            _write_face_annotation(
                project_root,
                object_name="connection_plate_white",
                side="sideA",
                board_path=board_paths["sideA"],
                image_path=side_a_image,
            )
            _write_face_annotation(
                project_root,
                object_name="connection_plate_white",
                side="sideB",
                board_path=board_paths["sideB"],
                image_path=side_b_image,
            )

            stage7 = _stage_by_id(project_root, 7)

            self.assertEqual(stage7.status, WorkflowStatus.NEEDS_ATTENTION)
            self.assertIn("outputs exist but need attention", stage7.message)
            self.assertTrue(any("Face manifest was not found" in error for error in stage7.errors))

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
