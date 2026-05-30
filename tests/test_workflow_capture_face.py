from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from posetag.pipelines.capture_face import (
    append_capture_manifest,
    build_capture_metadata,
    build_shot_paths,
)
from posetag.pipelines.charuco_calibration import (
    build_calibration_yaml,
    write_calibration_yaml,
)
from posetag.pipelines.make_board import build_board_yaml, write_board_yaml
from posetag.workflows.capture_face import (
    SOURCE_REALSENSE,
    SOURCE_VIDEO,
    CaptureFaceConfig,
    build_capture_face_command,
    build_capture_face_launch,
    capture_face_process_not_started,
    capture_face_process_running,
    inspect_capture_face_outputs,
    inspect_capture_face_readiness,
    summarize_capture_face_process_result,
)


class WorkflowCaptureFaceTests(unittest.TestCase):
    def test_readiness_builds_editable_install_safe_launch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            _write_board_and_registry(project_root, ("sideA",))
            video = Path(tmpdir) / "capture.mp4"
            video.touch()
            config = CaptureFaceConfig(
                project_root=project_root,
                object_name="connection_plate_white",
                source=SOURCE_VIDEO,
                video_path=video,
            )

            readiness = inspect_capture_face_readiness(config)
            command = build_capture_face_command(config)
            launch = build_capture_face_launch(config, python_executable="python-test")

        self.assertTrue(readiness.ready, readiness.errors)
        self.assertIn("connection_plate_white", readiness.registered_bases)
        self.assertEqual(readiness.selected_faces, ("connection_plate_white_sideA",))
        self.assertIn("posetag-capture-face", command)
        self.assertIn("--source video", command)
        self.assertEqual(launch.program, "python-test")
        self.assertEqual(launch.arguments[:2], ("-m", "posetag.cli.capture_face"))
        self.assertEqual(launch.expected_manifest.name, "manifest.csv")

    def test_readiness_reports_missing_selection_video_and_realsense(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            _write_board_and_registry(project_root, ("sideA",))

            missing_selection = inspect_capture_face_readiness(
                CaptureFaceConfig(project_root=project_root)
            )
            missing_video = inspect_capture_face_readiness(
                CaptureFaceConfig(
                    project_root=project_root,
                    object_name="connection_plate_white",
                    source=SOURCE_VIDEO,
                )
            )
            missing_realsense = inspect_capture_face_readiness(
                CaptureFaceConfig(
                    project_root=project_root,
                    object_name="connection_plate_white",
                    source=SOURCE_REALSENSE,
                ),
                realsense_available=False,
            )

        self.assertFalse(missing_selection.ready)
        self.assertTrue(
            any("Choose a registered object" in error for error in missing_selection.errors)
        )
        self.assertFalse(missing_video.ready)
        self.assertTrue(any("--video path is required" in error for error in missing_video.errors))
        self.assertFalse(missing_realsense.ready)
        self.assertTrue(any("pyrealsense2 is not available" in error for error in missing_realsense.errors))

    def test_output_status_requires_valid_coverage_for_each_registered_face(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            board_paths, _registry_path = _write_board_and_registry(
                project_root,
                ("sideA", "sideB"),
            )

            empty = inspect_capture_face_outputs(project_root)
            _write_capture(project_root, board_paths["sideA"], timestamp="20260530_120000")
            partial = inspect_capture_face_outputs(project_root)
            _write_capture(project_root, board_paths["sideB"], timestamp="20260530_120100")
            complete = inspect_capture_face_outputs(project_root)

        self.assertFalse(empty.manifest_exists)
        self.assertEqual(
            empty.missing_faces,
            ("connection_plate_white_sideA", "connection_plate_white_sideB"),
        )
        self.assertEqual(partial.covered_faces, ("connection_plate_white_sideA",))
        self.assertEqual(partial.missing_faces, ("connection_plate_white_sideB",))
        self.assertFalse(partial.complete)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.valid_shot_count, 2)
        self.assertEqual(complete.missing_faces, ())

    def test_broken_metadata_does_not_count_toward_coverage(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            board_paths, _registry_path = _write_board_and_registry(
                project_root,
                ("sideA",),
            )
            _write_capture(
                project_root,
                board_paths["sideA"],
                timestamp="20260530_120000",
                write_raw=False,
            )

            status = inspect_capture_face_outputs(project_root)

        self.assertFalse(status.complete)
        self.assertEqual(status.valid_shot_count, 0)
        self.assertEqual(status.invalid_shot_count, 1)
        self.assertEqual(status.missing_faces, ("connection_plate_white_sideA",))
        self.assertTrue(any("raw image was not found" in warning for warning in status.warnings))

    def test_process_state_summarizes_manifest_update(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "project" / "shots" / "manifest.csv"
            idle = capture_face_process_not_started(manifest)
            running = capture_face_process_running(manifest)
            missing = summarize_capture_face_process_result(
                exit_code=0,
                expected_manifest=manifest,
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text("timestamp,object_full\n", encoding="utf-8")
            success = summarize_capture_face_process_result(
                exit_code=0,
                expected_manifest=manifest,
                previous_manifest_mtime_ns=None,
                require_output_update=True,
            )

        self.assertEqual(idle.state, "not_started")
        self.assertTrue(running.running)
        self.assertFalse(missing.success)
        self.assertTrue(success.success)


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
        notes="Synthetic calibration for workflow capture-face tests.",
    )
    write_calibration_yaml(path, data)


def _write_board_and_registry(
    project_root: Path,
    sides: tuple[str, ...],
) -> tuple[dict[str, Path], Path]:
    board_paths: dict[str, Path] = {}
    tags: dict[str, dict[str, str]] = {}
    next_id = 52
    for side in sides:
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


def _write_capture(
    project_root: Path,
    board_path: Path,
    *,
    timestamp: str,
    write_raw: bool = True,
) -> None:
    object_full = board_path.stem
    base, _side = object_full.rsplit("_", 1)
    shot_paths = build_shot_paths(
        layout="by_object_side",
        out_dir=project_root / "shots",
        object_full=object_full,
        timestamp=timestamp,
    )
    raw_ids = [int(tag["id"]) for tag in _read_board_tags(board_path)]
    metadata = build_capture_metadata(
        face={"yaml": str(board_path), "object": object_full, "tag_ids": raw_ids},
        shot_paths=shot_paths,
        detected_tag_ids=raw_ids,
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
    assert base == "connection_plate_white"


def _read_board_tags(board_path: Path) -> list[dict[str, object]]:
    data = yaml.safe_load(board_path.read_text(encoding="utf-8"))
    return list(data["tags"])


if __name__ == "__main__":
    unittest.main()
