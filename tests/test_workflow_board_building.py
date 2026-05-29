from __future__ import annotations

import os
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
from posetag.workflows.board_building import (
    SOURCE_REALSENSE,
    SOURCE_VIDEO,
    BoardBatchRow,
    BoardBuildingConfig,
    BoardBuildingFlowError,
    NativeBoardCaptureSession,
    build_board_batch_items,
    build_board_building_arguments,
    build_board_building_command,
    build_board_building_launch,
    board_building_process_not_started,
    board_building_process_running,
    compose_board_object_name,
    default_board_batch_draft_path,
    default_boards_dir,
    default_calibration_path,
    delete_board_batch_draft,
    expected_board_yaml_path,
    expected_registry_path,
    infer_latest_object_tag_size_mm,
    inspect_board_building,
    inspect_board_building_outputs,
    load_board_batch_draft,
    next_default_side_label,
    parse_board_batch_rows,
    save_board_batch_draft,
    summarize_board_building_process_result,
)


class FakeFrameSource:
    def __init__(self, frames: tuple[np.ndarray, ...]) -> None:
        self._frames = list(frames)
        self.closed = False

    def read(self):
        if not self._frames:
            return None
        return self._frames.pop(0)

    def stop(self) -> None:
        self.closed = True


class FakeNativeDetector:
    def __init__(self, detections: tuple[object, ...]) -> None:
        self._detections = detections

    def detect(self, *args, **kwargs):
        return list(self._detections)


class FakeNativeDetection:
    def __init__(self, tag_id: int, xyz: tuple[float, float, float]) -> None:
        self.tag_id = tag_id
        self.pose_R = np.eye(3, dtype=float)
        self.pose_t = np.array(xyz, dtype=float).reshape(3, 1)
        base = float(tag_id)
        self.corners = np.array(
            [
                [base, base],
                [base + 10.0, base],
                [base + 10.0, base + 10.0],
                [base, base + 10.0],
            ],
            dtype=float,
        )


class WorkflowBoardBuildingTests(unittest.TestCase):
    def test_object_and_side_labels_compose_existing_cli_object_name(self) -> None:
        self.assertEqual(
            compose_board_object_name("connection plate white", "sideA"),
            "connection_plate_white_sideA",
        )
        self.assertEqual(
            compose_board_object_name("connection_plate_white", ""),
            "connection_plate_white",
        )

        with self.assertRaises(BoardBuildingFlowError):
            compose_board_object_name("../connection_plate_white", "sideA")

    def test_next_default_side_label_skips_existing_board_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            boards_dir = project_root / "boards"
            boards_dir.mkdir(parents=True)
            (boards_dir / "connection_plate_white_sideA.yaml").write_text(
                "object: connection_plate_white_sideA\n",
                encoding="utf-8",
            )

            side_label = next_default_side_label(
                project_root,
                "connection_plate_white",
            )

        self.assertEqual(side_label, "sideB")

    def test_batch_rows_expand_object_instances_and_sides(self) -> None:
        items = build_board_batch_items(
            (
                BoardBatchRow(
                    object_label="connection_plate",
                    instances="01-02",
                    sides="sideA-sideB",
                ),
                BoardBatchRow(
                    object_label="column_to_column",
                    instances="",
                    sides="front, back",
                    tag_size_mm=80.0,
                ),
            )
        )

        self.assertEqual(
            [item.object_name for item in items],
            [
                "connection_plate_01_sideA",
                "connection_plate_01_sideB",
                "connection_plate_02_sideA",
                "connection_plate_02_sideB",
                "column_to_column_front",
                "column_to_column_back",
            ],
        )
        self.assertIsNone(items[0].tag_size_mm)
        self.assertEqual(items[-1].tag_size_mm, 80.0)

    def test_batch_rows_reject_duplicate_board_names(self) -> None:
        with self.assertRaises(BoardBuildingFlowError):
            build_board_batch_items(
                (
                    BoardBatchRow("connection_plate", "01", "sideA"),
                    BoardBatchRow("connection_plate_01", "", "sideA"),
                )
            )

    def test_parse_multi_object_batch_rows(self) -> None:
        rows = parse_board_batch_rows(
            """
            connection_plate | 01-02 | sideA, sideB
            column | 01 | sideA-sideC
            # comments are ignored
            column_to_column |  | front, back
            plate | 01 | small, large | 40
            """
        )
        items = build_board_batch_items(rows)

        self.assertEqual(len(items), 11)
        self.assertEqual(items[0].object_name, "connection_plate_01_sideA")
        self.assertEqual(items[4].object_name, "column_01_sideA")
        self.assertEqual(items[-3].object_name, "column_to_column_back")
        self.assertEqual(items[-2].object_name, "plate_01_small")
        self.assertEqual(items[-2].tag_size_mm, 40.0)

    def test_board_batch_draft_persists_queue_without_board_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            rows = (
                BoardBatchRow("connection_plate_white", "", "sideA, sideB", 80.0),
                BoardBatchRow("column_white", "", "front, back", 40.0),
            )
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="connection_plate_white_sideA",
                family="tag36h11",
                tag_size_mm=80.0,
                source=SOURCE_VIDEO,
                video_path=Path("capture.mp4"),
                width=800,
                height=600,
                fps=24,
                calibration_path=Path("calib/calib_color.yaml"),
                allow_nonplanar=True,
                require_object_tags=False,
            )

            path = save_board_batch_draft(project_root, rows, config)
            loaded = load_board_batch_draft(project_root)

            self.assertEqual(path, default_board_batch_draft_path(project_root))
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.rows, rows)
            self.assertEqual(loaded.tag_size_mm, 80.0)
            self.assertEqual(loaded.source, SOURCE_VIDEO)
            self.assertEqual(loaded.video_path, "capture.mp4")
            self.assertEqual(loaded.width, 800)
            self.assertEqual(loaded.height, 600)
            self.assertEqual(loaded.fps, 24)
            self.assertEqual(loaded.calibration_path, "calib/calib_color.yaml")
            self.assertTrue(loaded.allow_nonplanar)
            self.assertFalse(
                (project_root / "boards" / "connection_plate_white_sideA.yaml")
                .exists()
            )
            self.assertFalse((project_root / "boards" / "tag_registry.yaml").exists())

    def test_board_batch_draft_delete_removes_queue_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            save_board_batch_draft(
                project_root,
                (BoardBatchRow("column", "", "sideA", 40.0),),
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="column_sideA",
                    require_object_tags=False,
                ),
            )

            removed = delete_board_batch_draft(project_root)
            removed_again = delete_board_batch_draft(project_root)

            self.assertTrue(removed)
            self.assertFalse(removed_again)
            self.assertIsNone(load_board_batch_draft(project_root))

    def test_latest_object_tag_size_is_inferred_from_stage3_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            patterns = project_root / "boards" / "patterns"
            patterns.mkdir(parents=True)
            small = patterns / "apriltag_36h11_IDs1-2_40mm_A4_600dpi.png"
            large = patterns / "apriltag_36h11_IDs3-4_80mm_A4_600dpi.png"
            small.write_bytes(b"small")
            large.write_bytes(b"large")

            inferred = infer_latest_object_tag_size_mm(project_root)

        self.assertEqual(inferred, 80.0)

    def test_native_capture_session_auto_captures_and_saves_existing_schema(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            source = FakeFrameSource((frame.copy(), frame.copy()))
            detector = FakeNativeDetector(
                (
                    FakeNativeDetection(52, (0.0, 0.0, 1.0)),
                    FakeNativeDetection(53, (0.1, 0.0, 1.0)),
                )
            )
            session = NativeBoardCaptureSession(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="connection_plate_white_sideA",
                    require_object_tags=False,
                ),
                detector=detector,
                frame_source=source,
                min_tags=2,
                stable_frames_required=2,
            )

            first = session.read_observation()
            second = session.read_observation()
            captured = session.capture_current(second)
            result = session.save_capture(selected_ids=(52, 53), origin_id=52)
            session.close()

            board = yaml.safe_load(result.board_yaml_path.read_text(encoding="utf-8"))
            registry = yaml.safe_load(result.registry_path.read_text(encoding="utf-8"))

        self.assertFalse(first.ready_to_capture)
        self.assertTrue(second.ready_to_capture)
        self.assertEqual(captured.pose_ready_ids, (52, 53))
        self.assertEqual(board["object"], "connection_plate_white_sideA")
        self.assertEqual([entry["id"] for entry in board["tags"]], [52, 53])
        self.assertEqual(registry["tags"]["52"]["object"], "connection_plate_white_sideA")
        self.assertEqual(result.registry_updated, 2)
        self.assertTrue(source.closed)

    def test_native_capture_session_saves_multiple_batch_items_without_reopening_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            source = FakeFrameSource(
                (frame.copy(), frame.copy(), frame.copy(), frame.copy())
            )
            detector = FakeNativeDetector(
                (
                    FakeNativeDetection(52, (0.0, 0.0, 1.0)),
                    FakeNativeDetection(53, (0.1, 0.0, 1.0)),
                )
            )
            base_config = BoardBuildingConfig(
                project_root=project_root,
                object_name="connection_plate_01_sideA",
                require_object_tags=False,
            )
            session = NativeBoardCaptureSession(
                base_config,
                detector=detector,
                frame_source=source,
                min_tags=2,
                stable_frames_required=2,
            )

            session.read_observation()
            first_capture = session.read_observation()
            session.capture_current(first_capture)
            first = session.save_capture(selected_ids=(52, 53), origin_id=52)
            session.set_capture_config(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="connection_plate_01_sideB",
                    tag_size_mm=80.0,
                    require_object_tags=False,
                )
            )
            self.assertEqual(session.tag_size_m, 0.08)
            session.read_observation()
            second_capture = session.read_observation()
            session.capture_current(second_capture)
            second = session.save_capture(
                selected_ids=(52, 53),
                origin_id=52,
                config=BoardBuildingConfig(
                    project_root=project_root,
                    object_name="connection_plate_01_sideB",
                    tag_size_mm=80.0,
                    require_object_tags=False,
                ),
            )
            second_board = yaml.safe_load(
                second.board_yaml_path.read_text(encoding="utf-8")
            )
            first_exists = first.board_yaml_path.exists()
            second_exists = second.board_yaml_path.exists()
            session.close()

        self.assertTrue(first.board_yaml_path.name.endswith("sideA.yaml"))
        self.assertTrue(second.board_yaml_path.name.endswith("sideB.yaml"))
        self.assertTrue(first_exists)
        self.assertTrue(second_exists)
        self.assertEqual(second_board["tag_size_m"], 0.08)
        self.assertTrue(source.closed)

    def test_native_capture_recapture_replaces_board_registry_membership(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            source = FakeFrameSource((frame.copy(), frame.copy()))
            session = NativeBoardCaptureSession(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="connection_plate_white_sideA",
                    require_object_tags=False,
                ),
                detector=FakeNativeDetector(
                    (
                        FakeNativeDetection(52, (0.0, 0.0, 1.0)),
                        FakeNativeDetection(53, (0.1, 0.0, 1.0)),
                    )
                ),
                frame_source=source,
                min_tags=2,
                stable_frames_required=1,
            )

            first_capture = session.read_observation()
            session.capture_current(first_capture)
            first = session.save_capture(selected_ids=(52, 53), origin_id=52)
            session.set_capture_config(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="connection_plate_white_sideA",
                    require_object_tags=False,
                )
            )
            session.detector = FakeNativeDetector(
                (
                    FakeNativeDetection(54, (0.0, 0.0, 1.0)),
                    FakeNativeDetection(55, (0.1, 0.0, 1.0)),
                )
            )
            second_capture = session.read_observation()
            session.capture_current(second_capture)
            second = session.save_capture(selected_ids=(54, 55), origin_id=54)
            board = yaml.safe_load(
                second.board_yaml_path.read_text(encoding="utf-8")
            )
            registry = yaml.safe_load(
                second.registry_path.read_text(encoding="utf-8")
            )
            session.close()

        self.assertEqual(first.board_yaml_path, second.board_yaml_path)
        self.assertEqual([entry["id"] for entry in board["tags"]], [54, 55])
        self.assertNotIn("52", registry["tags"])
        self.assertNotIn("53", registry["tags"])
        self.assertEqual(
            registry["tags"]["54"]["yaml"],
            str(second.board_yaml_path),
        )
        self.assertEqual(
            registry["tags"]["55"]["yaml"],
            str(second.board_yaml_path),
        )
        self.assertTrue(source.closed)

    def test_module_launch_target_invokes_make_board_cli(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = _pythonpath_with_src(env)
        result = subprocess.run(
            [sys.executable, "-m", "posetag.cli.make_board", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Interactive AprilTag board builder", result.stdout)
        self.assertIn("--object_name", result.stdout)

    def test_readiness_reports_missing_calibration_and_default_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="connection_plate_white_sideA",
            )

            readiness = inspect_board_building(
                config,
                realsense_available=False,
            )

        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.command_preview, "")
        self.assertEqual(readiness.calibration_path, default_calibration_path(project_root))
        self.assertEqual(
            readiness.expected_board_yaml,
            project_root / "boards" / "connection_plate_white_sideA.yaml",
        )
        self.assertEqual(
            readiness.expected_registry,
            project_root / "boards" / "tag_registry.yaml",
        )
        self.assertTrue(any("Calibration YAML not found" in err for err in readiness.errors))
        self.assertTrue(any("Stage 3 object AprilTag" in err for err in readiness.errors))

    def test_opencv_command_uses_existing_make_board_arguments(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            _write_object_tag_sheet(project_root)
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="connection_plate_white_sideA",
                camera_index=1,
                tag_size_mm=40.0,
            )

            readiness = inspect_board_building(
                config,
                realsense_available=False,
            )
            args = build_board_building_arguments(config)
            command = build_board_building_command(config)

        self.assertTrue(readiness.ready, readiness.errors)
        self.assertIn("posetag-make-board", readiness.command_preview)
        self.assertIn("--source", args)
        self.assertIn("opencv", args)
        self.assertIn("--cam", args)
        self.assertIn("1", args)
        self.assertIn("--object_name", args)
        self.assertIn("connection_plate_white_sideA", args)
        self.assertIn("--calib", args)
        self.assertIn(str(project_root / "calib" / "calib_color.yaml"), command)
        self.assertIn("--tag_size_mm 40", command)
        self.assertIn("--z_thresh 0.01", command)

    def test_launch_spec_uses_current_python_module_and_argument_list(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="sideA",
            )

            launch = build_board_building_launch(
                config,
                python_executable="/tmp/python",
            )

        self.assertEqual(launch.program, "/tmp/python")
        self.assertEqual(launch.arguments[:2], ("-m", "posetag.cli.make_board"))
        self.assertIn("--object_name", launch.arguments)
        self.assertIn("sideA", launch.arguments)
        self.assertIn("posetag-make-board", launch.display_command)
        self.assertEqual(launch.expected_board_yaml, project_root / "boards" / "sideA.yaml")
        self.assertEqual(launch.expected_registry, project_root / "boards" / "tag_registry.yaml")

    def test_video_source_requires_existing_video_path_for_readiness(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            _write_object_tag_sheet(project_root)

            missing_video = inspect_board_building(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="sideA",
                    source=SOURCE_VIDEO,
                ),
                realsense_available=False,
            )

            video_path = Path(tmpdir) / "board_capture.mp4"
            video_path.write_bytes(b"not a real video, just a path preflight")
            ready_video = inspect_board_building(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="sideA",
                    source=SOURCE_VIDEO,
                    video_path=video_path,
                ),
                realsense_available=False,
            )
            args = build_board_building_arguments(
                BoardBuildingConfig(
                    project_root=project_root,
                    object_name="sideA",
                    source=SOURCE_VIDEO,
                    video_path=video_path,
                )
            )

        self.assertFalse(missing_video.ready)
        self.assertTrue(any("--video path" in err for err in missing_video.errors))
        self.assertTrue(ready_video.ready, ready_video.errors)
        self.assertIn("--video", args)
        self.assertIn(str(video_path), args)
        self.assertNotIn("--cam", args)

    def test_realsense_readiness_checks_optional_dependency(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            _write_object_tag_sheet(project_root)
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="sideA",
                source=SOURCE_REALSENSE,
                width=848,
                height=480,
                fps=30,
            )

            unavailable = inspect_board_building(
                config,
                realsense_available=False,
            )
            available = inspect_board_building(
                config,
                realsense_available=True,
            )
            args = build_board_building_arguments(config)

        self.assertFalse(unavailable.ready)
        self.assertTrue(any("pyrealsense2" in err for err in unavailable.errors))
        self.assertTrue(available.ready, available.errors)
        self.assertIn("--width", args)
        self.assertIn("848", args)
        self.assertIn("--height", args)
        self.assertIn("480", args)

    def test_custom_output_paths_are_reflected_in_command_and_status(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            _write_object_tag_sheet(project_root)
            out_dir = base / "boards_out"
            registry = base / "custom_registry.yaml"
            shots_dir = base / "audit_shots"
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="sideA",
                out_dir=out_dir,
                registry_path=registry,
                save_shot=True,
                shots_dir=shots_dir,
            )

            readiness = inspect_board_building(
                config,
                realsense_available=False,
            )
            args = build_board_building_arguments(config)

            out_dir.mkdir(parents=True)
            registry.write_text("version: 1\ntags: {}\n", encoding="utf-8")
            expected_board_yaml_path(config).write_text("object: sideA\n", encoding="utf-8")
            shots_dir.mkdir()
            outputs = inspect_board_building_outputs(config)

        self.assertTrue(readiness.ready, readiness.errors)
        self.assertTrue(any("Custom registry paths" in item for item in readiness.warnings))
        self.assertEqual(expected_registry_path(config), registry)
        self.assertIn("--out_dir", args)
        self.assertIn(str(out_dir), args)
        self.assertIn("--registry", args)
        self.assertIn(str(registry), args)
        self.assertIn("--shots_dir", args)
        self.assertIn(str(shots_dir), args)
        self.assertIn("--save_shot", args)
        self.assertTrue(outputs.board_yaml_exists)
        self.assertTrue(outputs.registry_exists)
        self.assertTrue(outputs.shots_dir_exists)

    def test_shots_dir_argument_is_only_included_when_save_shot_is_enabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            shots_dir = Path(tmpdir) / "audit_shots"
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="sideA",
                shots_dir=shots_dir,
                save_shot=False,
                require_object_tags=False,
            )

            args = build_board_building_arguments(config)

        self.assertNotIn("--shots_dir", args)
        self.assertNotIn(str(shots_dir), args)

    def test_process_states_describe_prompt_boundary(self) -> None:
        board_yaml = Path("/tmp/project/boards/sideA.yaml")
        registry = Path("/tmp/project/boards/tag_registry.yaml")

        idle = board_building_process_not_started(board_yaml, registry)
        running = board_building_process_running(board_yaml, registry)

        self.assertFalse(idle.running)
        self.assertTrue(running.running)
        self.assertIn("selected tag IDs", running.message)
        self.assertEqual(running.expected_board_yaml, board_yaml)
        self.assertEqual(running.expected_registry, registry)

    def test_process_result_requires_board_yaml_and_registry_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_yaml = project_root / "boards" / "sideA.yaml"
            registry = project_root / "boards" / "tag_registry.yaml"

            missing = summarize_board_building_process_result(
                exit_code=0,
                expected_board_yaml=board_yaml,
                expected_registry=registry,
            )

            board_yaml.parent.mkdir(parents=True)
            board_yaml.write_text("object: sideA\n", encoding="utf-8")
            registry.write_text("version: 1\ntags: {}\n", encoding="utf-8")
            success = summarize_board_building_process_result(
                exit_code=0,
                expected_board_yaml=board_yaml,
                expected_registry=registry,
            )
            failed = summarize_board_building_process_result(
                exit_code=2,
                expected_board_yaml=board_yaml,
                expected_registry=registry,
            )

        self.assertFalse(missing.success)
        self.assertIn("missing", missing.message)
        self.assertTrue(success.success)
        self.assertIn("present", success.message)
        self.assertFalse(failed.success)
        self.assertIn("code 2", failed.message)

    def test_invalid_object_name_fails_before_command_construction(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="../sideA",
                require_object_tags=False,
            )

            readiness = inspect_board_building(config, realsense_available=False)

            with self.assertRaises(BoardBuildingFlowError):
                build_board_building_arguments(config)

        self.assertFalse(readiness.ready)
        self.assertTrue(any("filename-safe" in err for err in readiness.errors))

    def test_object_tag_requirement_can_be_warning_for_external_prints(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            config = BoardBuildingConfig(
                project_root=project_root,
                object_name="sideA",
                require_object_tags=False,
            )

            readiness = inspect_board_building(
                config,
                realsense_available=False,
            )

        self.assertTrue(readiness.ready, readiness.errors)
        self.assertTrue(any("outside the default project folder" in item for item in readiness.warnings))
        self.assertIn(str(default_boards_dir(project_root)), str(readiness.expected_board_yaml))


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
        notes="Synthetic calibration for board-building workflow tests.",
    )
    write_calibration_yaml(path, data)


def _write_object_tag_sheet(project_root: Path) -> Path:
    path = project_root / "boards" / "patterns" / "apriltag_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png placeholder")
    return path


def _pythonpath_with_src(env: dict[str, str]) -> str:
    src_root = Path(__file__).resolve().parents[1] / "src"
    existing = env.get("PYTHONPATH")
    if existing:
        return f"{src_root}{os.pathsep}{existing}"
    return str(src_root)


if __name__ == "__main__":
    unittest.main()
