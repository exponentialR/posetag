from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from posetag.gui import app as gui_app
from posetag.gui.main_window import (
    _board_building_command_card_note,
    _board_building_command_ready_message,
    _board_building_run_button_label,
    _capture_face_command_card_note,
    _capture_face_command_ready_message,
    _capture_face_run_button_label,
    _capture_face_saved_shot_collection_label,
    _capture_face_saved_shot_group_label,
    _capture_face_saved_shot_label,
    _command_button_label,
    _command_card_note,
    _command_card_title,
    _command_ready_message,
    _copy_confirmation_message,
    _calibration_run_button_label,
    _checked_tag_ids_for_update,
    _format_calibration_result_summary,
    _format_calibration_result_summary_html,
    _format_calibration_process_state,
    _format_board_building_outputs,
    _format_board_building_process_state,
    _format_board_building_readiness,
    _format_board_batch_preview,
    _format_capture_face_outputs,
    _format_capture_face_process_state,
    _format_capture_face_readiness,
    _format_capture_face_saved_shot_group_details,
    _format_capture_face_saved_shot_details,
    _format_charuco_outputs,
    _format_health_counts,
    _format_mesh_keypoint_details,
    _format_object_tag_expected_output,
    _format_object_tag_outputs,
    _format_object_tag_readiness,
    _group_capture_face_saved_shots,
    _health_status_style,
    _mesh_keypoint_command_card_note,
    _mesh_keypoint_command_ready_message,
    _mesh_keypoint_list_item,
    _project_root_hint,
    _stage_list_label,
    _stage_rail_dot_style,
    _stage_rail_entry_style,
    _stage_rail_name,
    _stage_rail_status_label,
    _stage_rail_number_style,
    _stage_status_chip_style,
    _style_sheet,
)
from posetag.workflows.calibration_flow import (
    CameraCalibrationOutputSummary,
    calibration_process_not_started,
    calibration_process_running,
)
from posetag.workflows.board_building import (
    BoardBatchItem,
    BoardBuildingOutputStatus,
    BoardBuildingProcessState,
    BoardBuildingReadiness,
    board_building_process_not_started,
    board_building_process_running,
)
from posetag.pipelines.capture_face import CaptureFacePaths
from posetag.workflows.capture_face import (
    CaptureFaceOutputStatus,
    CaptureFaceProcessState,
    CaptureFaceReadiness,
    CaptureFaceSavedShot,
    capture_face_process_not_started,
    capture_face_process_running,
)
from posetag.workflows.object_tags import (
    ObjectTagGenerationReadiness,
    ObjectTagGenerationResult,
)
from posetag.workflows.mesh_keypoints import MeshKeypointObjectStatus
from posetag.gui.models import inspect_project_view, project_health_view
from posetag.workflows.commands import command_preview


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


class WorkflowGuiTests(unittest.TestCase):
    def test_pyproject_declares_optional_gui_extra_and_entry_point(self) -> None:
        with PYPROJECT_PATH.open("rb") as handle:
            pyproject = tomllib.load(handle)

        self.assertEqual(
            pyproject["project"]["optional-dependencies"]["gui"],
            ["PySide6"],
        )
        self.assertEqual(
            pyproject["project"]["scripts"]["posetag-gui"],
            "posetag.gui.app:main",
        )
        self.assertIn("posetag.gui", pyproject["tool"]["setuptools"]["packages"])

    def test_posetag_gui_help_does_not_require_pyside6(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = _pythonpath_with_src(env)
        result = subprocess.run(
            [sys.executable, "-m", "posetag.gui.app", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("posetag-gui", result.stdout)
        self.assertIn("--project_root", result.stdout)
        self.assertNotIn("PySide6 is required", result.stderr)

    def test_app_accepts_project_root_aliases(self) -> None:
        parser = gui_app.build_parser()

        underscore = parser.parse_args(["--project_root", "project_a"])
        hyphen = parser.parse_args(["--project-root", "project_b"])

        self.assertEqual(underscore.project_root, "project_a")
        self.assertEqual(hyphen.project_root, "project_b")

    def test_missing_pyside6_error_is_clear(self) -> None:
        with TemporaryDirectory() as tmpdir:
            stderr = StringIO()
            with patch.object(
                gui_app,
                "_load_qt_modules",
                side_effect=gui_app.MissingGuiDependency("PySide6 is required."),
            ):
                with redirect_stderr(stderr):
                    rc = gui_app.main(["--project_root", tmpdir])

        self.assertEqual(rc, 2)
        self.assertIn("PySide6 is required", stderr.getvalue())
        self.assertIn('pip install -e ".[gui]"', stderr.getvalue())

    def test_terminal_shutdown_handlers_quit_qt_application(self) -> None:
        fake_app = _FakeQtApp()
        fake_qt_core = _FakeQtCore()

        with patch.object(gui_app.signal, "signal") as signal_fn:
            gui_app._install_terminal_shutdown_handlers(fake_app, fake_qt_core)

        self.assertGreaterEqual(signal_fn.call_count, 2)
        handler = signal_fn.call_args_list[0].args[1]
        handler(2, None)

        self.assertEqual(fake_app.quit_calls, 1)
        self.assertTrue(fake_app._posetag_signal_timer.started)
        self.assertEqual(fake_app._posetag_signal_timer.interval, 200)

    def test_view_models_are_built_from_workflow_status_helpers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project with spaces"
            models = inspect_project_view(project_root)

        self.assertEqual(len(models), 10)
        self.assertEqual(models[0].stage_id, 0)
        self.assertEqual(models[0].status, "missing")
        self.assertIn("posetag project new", models[0].command_preview)
        self.assertIn("posetag-gen-charuco", models[1].command_preview)
        self.assertIn("posetag-calib-charuco", models[2].command_preview)
        self.assertIn("posetag-gen-tags", models[3].command_preview)
        self.assertIn("--project_root", models[3].command_preview)
        self.assertIn("posetag-make-board", models[4].command_preview)
        self.assertIn("posetag-gen-keypoints", models[6].command_preview)
        self.assertIn("posetag-annotate", models[7].command_preview)
        self.assertEqual(models[9].command_preview, "")

    def test_project_health_view_handles_empty_project_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "empty-project"
            health = project_health_view(inspect_project_view(project_root))

        self.assertEqual(health.status, "missing")
        self.assertEqual(health.headline, "No workflow outputs found")
        self.assertEqual(health.error_count, 0)
        self.assertEqual(health.next_stage_label, "Stage 0: Project Setup")
        self.assertIn("posetag project new", health.next_action)

    def test_project_health_view_prioritizes_attention_states(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            calib_path.parent.mkdir(parents=True)
            calib_path.write_text("not_camera_matrix: true\n", encoding="utf-8")

            health = project_health_view(inspect_project_view(project_root))

        self.assertEqual(health.status, "needs_attention")
        self.assertEqual(health.headline, "Needs attention")
        self.assertGreater(health.error_count, 0)
        self.assertEqual(health.next_stage_label, "Stage 2: Calibrate Camera")
        self.assertIn("calib_color.yaml", health.next_action)

    def test_project_health_view_keeps_mixed_missing_outputs_incomplete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = project_root / "calib" / "calib_color.yaml"
            calib_path.parent.mkdir(parents=True)
            calib_path.write_text(
                yaml.safe_dump(
                    {
                        "image_width": 640,
                        "image_height": 480,
                        "camera_matrix": {
                            "fx": 600.0,
                            "fy": 610.0,
                            "cx": 320.0,
                            "cy": 240.0,
                            "data": [
                                [600.0, 0.0, 320.0],
                                [0.0, 610.0, 240.0],
                                [0.0, 0.0, 1.0],
                            ],
                        },
                        "distortion_coefficients": {
                            "k1": 0.0,
                            "k2": 0.0,
                            "p1": 0.0,
                            "p2": 0.0,
                            "k3": 0.0,
                            "data": [[0.0, 0.0, 0.0, 0.0, 0.0]],
                        },
                        "reproj_rms": 0.12,
                        "model": "pinhole",
                        "notes": "Synthetic calibration for GUI health tests.",
                    }
                ),
                encoding="utf-8",
            )

            health = project_health_view(inspect_project_view(project_root))

        self.assertEqual(health.status, "missing")
        self.assertEqual(health.headline, "Workflow outputs missing")
        self.assertEqual(
            health.next_stage_label,
            "Stage 1: Generate ChArUco Calibration Board",
        )
        self.assertIn("required outputs are still missing", health.message)

    def test_command_preview_quotes_project_paths(self) -> None:
        preview = command_preview(
            "build_boards",
            Path("/tmp/PoseTag project"),
        )

        self.assertIn("posetag-make-board", preview)
        self.assertIn("'/tmp/PoseTag project'", preview)
        self.assertIn("OBJECT_FACE", preview)

    def test_legacy_command_previews_use_selected_project_env(self) -> None:
        project_root = Path("/tmp/PoseTag project")

        keypoints_preview = command_preview("generate_mesh_keypoints", project_root)
        annotate_preview = command_preview("annotate_faces", project_root)
        collect_preview = command_preview("collect_dataset", project_root)

        self.assertIn("posetag-gen-keypoints", keypoints_preview)
        self.assertIn("OBJECT.obj", keypoints_preview)
        self.assertIn("OBJECT_NAME", keypoints_preview)
        self.assertIn("env 'POSETAG_PROJECT=/tmp/PoseTag project'", annotate_preview)
        self.assertIn("posetag-annotate --browse", annotate_preview)
        self.assertIn("env 'POSETAG_PROJECT=/tmp/PoseTag project'", collect_preview)
        self.assertIn("posetag-collect --mode live", collect_preview)

    def test_mesh_keypoint_command_preview_uses_inferred_object(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            boards_dir = project_root / "boards"
            boards_dir.mkdir(parents=True)
            for object_name in (
                "column_white_front",
                "column_white_back",
                "connection_plate_white_sideA",
            ):
                (boards_dir / f"{object_name}.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "object": object_name,
                            "family": "tag36h11",
                            "tag_size_m": 0.04,
                            "origin_id": 1,
                            "tags": [{"id": 1, "cx": 0.0, "cy": 0.0}],
                        }
                    ),
                    encoding="utf-8",
                )

            preview = command_preview("generate_mesh_keypoints", project_root)

            self.assertIn("posetag-gen-keypoints", preview)
            self.assertIn("meshes/column_white.obj", preview)
            self.assertIn("--object_name column_white", preview)
            self.assertNotIn("column_white_front.obj", preview)

    def test_command_copy_confirmation_text_is_explicit(self) -> None:
        self.assertEqual(
            _copy_confirmation_message("posetag-gen-tags --project_root project"),
            "Copied command preview to the clipboard.",
        )
        self.assertEqual(
            _copy_confirmation_message(
                "posetag-gen-tags --project_root project",
                stage_complete=True,
            ),
            "Copied rerun command to the clipboard.",
        )
        self.assertEqual(
            _copy_confirmation_message(""),
            "No command preview is available for this stage.",
        )

    def test_complete_stage_command_copy_reads_as_secondary_rerun(self) -> None:
        complete = SimpleNamespace(
            status="complete",
            key="generate_charuco_board",
            command_preview="posetag-gen-charuco --project_root project",
        )
        missing = SimpleNamespace(
            status="missing",
            key="generate_object_tags",
            command_preview="posetag-gen-tags --project_root project",
        )
        no_command = SimpleNamespace(
            status="not_applicable",
            key="review_export",
            command_preview="",
        )

        self.assertEqual(_command_card_title(complete), "Rerun command")
        self.assertEqual(_command_button_label(complete), "Copy Rerun")
        self.assertIn("already complete", _command_card_note(complete))
        self.assertIn("checked paths", _command_card_note(complete))
        self.assertEqual(
            _command_ready_message(
                complete.command_preview,
                stage_complete=True,
            ),
            "Stage complete. Copy only if you need to regenerate outputs.",
        )

        self.assertEqual(_command_card_title(missing), "Command preview")
        self.assertEqual(_command_button_label(missing), "Copy Command")
        self.assertIn("Copy the preview", _command_card_note(missing))
        self.assertEqual(
            _command_ready_message(missing.command_preview),
            "Ready to copy. Replace placeholder values before running it.",
        )

        self.assertEqual(_command_card_title(no_command), "Command preview")
        self.assertEqual(_command_button_label(no_command), "Copy Command")
        self.assertEqual(
            _command_card_note(no_command),
            "No command preview is defined for this stage.",
        )

    def test_mesh_keypoint_panel_text_makes_obj_import_visible(self) -> None:
        status = MeshKeypointObjectStatus(
            object_name="column_white",
            faces=("back", "front"),
            sources=("boards/*.yaml",),
            mesh_path=Path("/tmp/project/meshes/column_white.obj"),
            keypoints_path=Path("/tmp/project/objects/column_white/keypoints.json"),
            mesh_exists=False,
            keypoints_exists=False,
            keypoints_valid=False,
            keypoints_errors=(),
            command_preview=(
                "posetag-gen-keypoints --project_root /tmp/project "
                "--mesh /tmp/project/meshes/column_white.obj "
                "--object_name column_white"
            ),
        )
        model = SimpleNamespace(status="missing", key="generate_mesh_keypoints")

        row = _mesh_keypoint_list_item(status)
        details = _format_mesh_keypoint_details(status)
        note = _mesh_keypoint_command_card_note((status,), model)
        ready = _mesh_keypoint_command_ready_message((status,))

        self.assertIn("column_white", row)
        self.assertIn("mesh missing", row)
        self.assertIn("keypoints missing", row)
        self.assertIn("Mesh OBJ:", details)
        self.assertIn("meshes/column_white.obj", details)
        self.assertIn(
            "Stage status: mesh input and keypoints JSON still need generation.",
            details,
        )
        self.assertIn("Import OBJ", note)
        self.assertIn("objects/<object>/keypoints.json", note)
        self.assertIn("Stage 6 remains missing", note)
        self.assertEqual(
            ready,
            "1 object needs OBJ mesh input; 1 object needs keypoints JSON.",
        )
        self.assertIn("object geometry inputs panel", _command_card_note(model))

        staged_status = replace(status, mesh_exists=True)
        staged_details = _format_mesh_keypoint_details(staged_status)
        staged_ready = _mesh_keypoint_command_ready_message((staged_status,))
        self.assertIn(
            "Stage status: mesh input is staged; keypoints JSON still needs "
            "generation.",
            staged_details,
        )
        self.assertEqual(
            staged_ready,
            "0 objects need OBJ mesh input; 1 object needs keypoints JSON.",
        )

    def test_project_root_hint_does_not_create_empty_project(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing_project = Path(tmpdir) / "project"
            hint = _project_root_hint(missing_project)

            self.assertIn("not found", hint)
            self.assertFalse(missing_project.exists())

    def test_selected_stage_row_keeps_readable_text_style(self) -> None:
        style = _style_sheet()

        self.assertIn("QListWidget::item:selected", style)
        self.assertIn("background: transparent", style)
        self.assertIn("color: #0b1724", style)
        self.assertIn("QListWidget::item:selected:!active", style)

    def test_batch_capture_queue_has_visible_selection_style(self) -> None:
        style = _style_sheet()

        self.assertIn("QListWidget#BatchCaptureQueue", style)
        self.assertIn("QListWidget#BatchCaptureQueue::item:selected", style)
        self.assertIn("QListWidget#BoardObjectRows", style)
        self.assertIn("QListWidget#BoardObjectRows::item:selected", style)
        self.assertIn("background: #d9efff", style)
        self.assertIn("border: 2px solid #0b6f8f", style)

    def test_dashboard_style_uses_professional_instrumentation_layout(self) -> None:
        style = _style_sheet()

        self.assertIn('"Aptos"', style)
        self.assertIn('"Aptos Display"', style)
        self.assertIn('"Segoe UI"', style)
        self.assertIn('"Segoe UI Variable Display"', style)
        self.assertNotIn('"Virgil"', style)
        self.assertNotIn('"Comic Sans MS"', style)
        self.assertIn("QFrame#HeaderPanel", style)
        self.assertIn("QFrame#RailPanel", style)
        self.assertIn("QFrame#HealthPanel", style)
        self.assertIn("QScrollArea#HealthScroll", style)
        self.assertIn("QFrame#ObjectTagPreviewColumn", style)
        self.assertIn("QGroupBox#CollapsibleGroup", style)
        self.assertIn("background: #f2f7fb", style)
        self.assertIn("background: #f6fafc", style)
        self.assertIn("QLabel#StageTitle", style)
        self.assertIn("QLineEdit#CommandPreview", style)
        self.assertIn("QPlainTextEdit#CalibrationLog", style)
        self.assertIn("QLabel#HealthMessage", style)
        self.assertIn("QLabel#HealthSectionTitle", style)
        self.assertIn("QLabel#HealthSectionBody", style)
        self.assertIn("QLabel#RailStageName", style)
        self.assertIn("QLabel#RailStageNumber", style)
        self.assertIn("QLabel#RailStageDot", style)
        self.assertIn("QLabel#RailStageStatus", style)
        self.assertIn("QLabel#CharucoPreviewCanvas", style)
        self.assertIn("QLabel#ObjectTagPreviewCanvas", style)
        self.assertIn("QLabel#MeshKeypointPreview", style)
        self.assertIn("QListWidget#MeshKeypointObjectList", style)
        self.assertIn("QPlainTextEdit#CalibrationYamlView", style)
        self.assertIn("monospace", style)
        self.assertIn("font-weight: 800", style)
        self.assertIn("border-left: 3px solid #54708a", style)
        self.assertIn("border-radius: 8px", style)

    def test_charuco_output_summary_reflects_pdf_format_choice(self) -> None:
        result = SimpleNamespace(
            png=Path("board.png"),
            yaml=Path("board.yaml"),
            pdf=None,
            requested_pdf=False,
        )

        summary = _format_charuco_outputs(result)

        self.assertIn("Folder:", summary)
        self.assertIn("PNG:    board.png", summary)
        self.assertIn("YAML:   board.yaml", summary)
        self.assertIn("PDF:    not requested", summary)

    def test_object_tag_output_summary_lists_generated_sheets(self) -> None:
        result = ObjectTagGenerationResult(
            png_paths=(Path("apriltag_36h11_IDs1-4_40mm_A4_80dpi.png"),),
            pdf_paths=(Path("apriltag_36h11_IDs1-4_40mm_A4_80dpi.pdf"),),
            out_dir=Path("/tmp/project/boards/patterns"),
            parsed_ids=(1, 2, 3, 4),
        )

        summary = _format_object_tag_outputs(result)

        self.assertIn("Folder:", summary)
        self.assertIn("IDs:    1, 2, 3, 4", summary)
        self.assertIn("PNG:    1 sheet", summary)
        self.assertIn("PDF:    1 sheet", summary)

    def test_object_tag_readiness_summary_reports_errors_and_output(self) -> None:
        readiness = ObjectTagGenerationReadiness(
            ready=False,
            command_preview="",
            expected_output_dir=Path("/tmp/project/boards/patterns"),
            parsed_ids=(1, 2),
            checked_paths=(Path("/tmp/project/calib/calib_color.yaml"),),
            warnings=("Printer scaling must be checked.",),
            errors=("Complete Stage 2 camera calibration first.",),
        )

        summary = _format_object_tag_readiness(readiness)
        output = _format_object_tag_expected_output(readiness)

        self.assertIn("Not ready yet", summary)
        self.assertIn("IDs: 1, 2", summary)
        self.assertIn("Complete Stage 2", summary)
        self.assertIn("Printer scaling", summary)
        self.assertIn("boards/patterns", output)
        self.assertIn("Status: missing", output)

    def test_board_building_readiness_summary_reports_outputs(self) -> None:
        outputs = BoardBuildingOutputStatus(
            board_yaml_path=Path("/tmp/project/boards/sideA.yaml"),
            registry_path=Path("/tmp/project/boards/tag_registry.yaml"),
            shots_dir=Path("/tmp/project/boards/shots"),
            board_yaml_exists=False,
            registry_exists=False,
            shots_dir_exists=False,
            checked_paths=(
                Path("/tmp/project/boards/sideA.yaml"),
                Path("/tmp/project/boards/tag_registry.yaml"),
                Path("/tmp/project/boards/shots"),
            ),
        )
        readiness = BoardBuildingReadiness(
            ready=False,
            command_preview="",
            calibration_path=Path("/tmp/project/calib/calib_color.yaml"),
            output_status=outputs,
            checked_paths=(
                Path("/tmp/project/calib/calib_color.yaml"),
                Path("/tmp/project/boards/sideA.yaml"),
                Path("/tmp/project/boards/tag_registry.yaml"),
            ),
            warnings=("Custom registry paths may not mark Stage 4 complete.",),
            errors=("Complete Stage 3 object AprilTag generation first.",),
        )

        summary = _format_board_building_readiness(readiness)
        output = _format_board_building_outputs(readiness)

        self.assertIn("Not ready yet", summary)
        self.assertIn("Complete Stage 3", summary)
        self.assertIn("Custom registry", summary)
        self.assertIn("Calibration YAML", output)
        self.assertIn("Board YAML", output)
        self.assertIn("Tag registry", output)
        self.assertIn("Audit shots", output)
        self.assertIn("Status: missing", output)

    def test_board_batch_preview_summarizes_queue(self) -> None:
        items = tuple(
            BoardBatchItem(
                object_label="connection_plate",
                instance_label=f"{index:02d}",
                side_label="sideA",
                object_name=f"connection_plate_{index:02d}_sideA",
            )
            for index in range(1, 11)
        )

        preview = _format_board_batch_preview(items)

        self.assertIn("10 boards queued", preview)
        self.assertIn("connection_plate_01_sideA", preview)
        self.assertIn("2 more", preview)

    def test_board_batch_preview_shows_per_side_tag_sizes(self) -> None:
        items = (
            BoardBatchItem(
                object_label="column",
                instance_label="01",
                side_label="sideA",
                object_name="column_01_sideA",
                tag_size_mm=40.0,
            ),
            BoardBatchItem(
                object_label="column",
                instance_label="01",
                side_label="sideC",
                object_name="column_01_sideC",
                tag_size_mm=80.0,
            ),
        )

        preview = _format_board_batch_preview(items)

        self.assertIn("column_01_sideA (40 mm)", preview)
        self.assertIn("column_01_sideC (80 mm)", preview)

    def test_captured_board_ids_default_to_all_pose_ready_tags(self) -> None:
        live_preserved = _checked_tag_ids_for_update(
            (2, 3),
            {2},
            force_all=False,
        )
        captured = _checked_tag_ids_for_update(
            (2, 3),
            {2},
            force_all=True,
        )
        changed_ids = _checked_tag_ids_for_update(
            (0, 1),
            {2},
            force_all=False,
        )

        self.assertEqual(live_preserved, {2})
        self.assertEqual(captured, {2, 3})
        self.assertEqual(changed_ids, {0, 1})

    def test_board_building_command_note_keeps_existing_workflow_boundary(self) -> None:
        outputs = BoardBuildingOutputStatus(
            board_yaml_path=Path("/tmp/project/boards/sideA.yaml"),
            registry_path=Path("/tmp/project/boards/tag_registry.yaml"),
            shots_dir=Path("/tmp/project/boards/shots"),
            board_yaml_exists=False,
            registry_exists=False,
            shots_dir_exists=False,
            checked_paths=(),
        )
        readiness = BoardBuildingReadiness(
            ready=True,
            command_preview="posetag-make-board --project_root /tmp/project",
            calibration_path=Path("/tmp/project/calib/calib_color.yaml"),
            output_status=outputs,
            checked_paths=(),
            warnings=(),
            errors=(),
        )
        active = SimpleNamespace(status="missing")
        complete = SimpleNamespace(status="complete")

        note = _board_building_command_card_note(readiness, active)
        complete_note = _board_building_command_card_note(readiness, complete)

        self.assertIn("Guided Capture", note)
        self.assertIn("Start Batch", note)
        self.assertIn("Run Board Builder", note)
        self.assertIn("posetag-make-board", note)
        self.assertIn("ENTER", note)
        self.assertIn("selected-ID", note)
        self.assertIn("origin-ID", note)
        self.assertIn("prompt box", note)
        self.assertIn("already pass", complete_note)
        self.assertEqual(
            _board_building_command_ready_message(readiness),
            "Ready for batch capture, single-board capture, CLI launch, or copy.",
        )

    def test_board_building_process_state_text_and_run_button_labels(self) -> None:
        board_yaml = Path("/tmp/project/boards/sideA.yaml")
        registry = Path("/tmp/project/boards/tag_registry.yaml")
        idle = board_building_process_not_started(board_yaml, registry)
        running = board_building_process_running(board_yaml, registry)
        success = BoardBuildingProcessState(
            state="finished",
            label="finished",
            message="Board outputs are present.",
            success=True,
            expected_board_yaml=board_yaml,
            expected_registry=registry,
            exit_code=0,
        )

        self.assertEqual(_board_building_run_button_label(idle), "Run Board Builder")
        self.assertEqual(
            _board_building_run_button_label(running),
            "Board Builder Running...",
        )
        self.assertEqual(
            _board_building_run_button_label(success),
            "Run Board Builder Again",
        )
        self.assertIn("not started", _format_board_building_process_state(idle))
        self.assertIn("Expected board:", _format_board_building_process_state(idle))
        self.assertIn("selected tag IDs", _format_board_building_process_state(running))

    def test_capture_face_readiness_summary_reports_coverage_outputs(self) -> None:
        saved_shot = CaptureFaceSavedShot(
            row_index=2,
            object_full="connection_plate_white_sideA",
            object_base="connection_plate_white",
            side="sideA",
            timestamp="20260531_120000",
            raw_path=Path("/tmp/project/shots/connection_plate_white/sideA/raw.png"),
            annotated_path=Path("/tmp/project/shots/connection_plate_white/sideA/ann.png"),
            metadata_path=Path("/tmp/project/shots/connection_plate_white/sideA/meta.json"),
            face_yaml=Path("/tmp/project/boards/connection_plate_white_sideA.yaml"),
            validation_ok=True,
            coverage_ok=True,
            expected_tag_ids=(52, 53),
            detected_tag_ids=(52, 53),
        )
        outputs = CaptureFaceOutputStatus(
            registry_path=Path("/tmp/project/boards/tag_registry.yaml"),
            manifest_path=Path("/tmp/project/shots/manifest.csv"),
            out_dir=Path("/tmp/project/shots"),
            checked_paths=(
                Path("/tmp/project/boards/tag_registry.yaml"),
                Path("/tmp/project/shots/manifest.csv"),
            ),
            registered_faces=(
                "connection_plate_white_sideA",
                "connection_plate_white_sideB",
            ),
            covered_faces=("connection_plate_white_sideA",),
            missing_faces=("connection_plate_white_sideB",),
            valid_shot_count=1,
            invalid_shot_count=1,
            manifest_exists=True,
            errors=(),
            warnings=("One manifest row is missing a raw image.",),
            saved_shots=(saved_shot,),
        )
        paths = CaptureFacePaths(
            project_root=Path("/tmp/project"),
            calib_path=Path("/tmp/project/calib/calib_color.yaml"),
            registry_path=outputs.registry_path,
            out_dir=outputs.out_dir,
            manifest_path=outputs.manifest_path,
            log_path=Path("/tmp/project/logs/capture_face.log"),
        )
        readiness = CaptureFaceReadiness(
            ready=False,
            command_preview="",
            calibration_path=paths.calib_path,
            paths=paths,
            output_status=outputs,
            registered_bases=("connection_plate_white",),
            registered_faces=outputs.registered_faces,
            selected_faces=("connection_plate_white_sideA",),
            checked_paths=(paths.calib_path, outputs.registry_path),
            warnings=outputs.warnings,
            errors=("Choose a registered object or face before capture.",),
        )

        summary = _format_capture_face_readiness(readiness)
        output = _format_capture_face_outputs(readiness)

        self.assertIn("Not ready yet", summary)
        self.assertIn("Registered objects", summary)
        self.assertIn("Selected faces", summary)
        self.assertIn("Choose a registered object", summary)
        self.assertIn("Calibration YAML", output)
        self.assertIn("Tag registry", output)
        self.assertIn("Manifest", output)
        self.assertIn("1/2 registered faces", output)
        self.assertIn("connection_plate_white_sideB", output)
        self.assertIn("Invalid rows", output)
        self.assertIn("Saved shots", output)
        self.assertIn("connection_plate_white_sideA", _capture_face_saved_shot_label(saved_shot))
        details = _format_capture_face_saved_shot_details(saved_shot)
        self.assertIn("valid coverage shot", details)
        self.assertIn("Expected tags: 52, 53", details)
        self.assertIn("Metadata:", details)
        older_shot = replace(
            saved_shot,
            row_index=1,
            timestamp="20260531_115900",
            metadata_path=Path("/tmp/project/shots/older_meta.json"),
        )
        grouped = _group_capture_face_saved_shots((saved_shot, older_shot))
        group_label = _capture_face_saved_shot_group_label(grouped[0][1])
        collection_label = _capture_face_saved_shot_collection_label(
            (saved_shot, older_shot)
        )
        group_details = _format_capture_face_saved_shot_group_details(grouped[0][1])
        self.assertEqual(len(grouped), 1)
        self.assertIn("2 shots", group_label)
        self.assertIn("latest 20260531_120000", group_label)
        self.assertIn("Saved face shots", collection_label)
        self.assertIn("2 shots", collection_label)
        self.assertIn("1 face", collection_label)
        self.assertIn("Saved shots: 2", group_details)
        self.assertIn("Previous shots", group_details)

    def test_capture_face_command_note_keeps_existing_workflow_boundary(self) -> None:
        outputs = CaptureFaceOutputStatus(
            registry_path=Path("/tmp/project/boards/tag_registry.yaml"),
            manifest_path=Path("/tmp/project/shots/manifest.csv"),
            out_dir=Path("/tmp/project/shots"),
            checked_paths=(),
            registered_faces=("connection_plate_white_sideA",),
            covered_faces=(),
            missing_faces=("connection_plate_white_sideA",),
            valid_shot_count=0,
            invalid_shot_count=0,
            manifest_exists=False,
            errors=(),
            warnings=(),
        )
        paths = CaptureFacePaths(
            project_root=Path("/tmp/project"),
            calib_path=Path("/tmp/project/calib/calib_color.yaml"),
            registry_path=outputs.registry_path,
            out_dir=outputs.out_dir,
            manifest_path=outputs.manifest_path,
            log_path=Path("/tmp/project/logs/capture_face.log"),
        )
        readiness = CaptureFaceReadiness(
            ready=True,
            command_preview="posetag-capture-face --project_root /tmp/project",
            calibration_path=paths.calib_path,
            paths=paths,
            output_status=outputs,
            registered_bases=("connection_plate_white",),
            registered_faces=outputs.registered_faces,
            selected_faces=outputs.registered_faces,
            checked_paths=(),
            warnings=(),
            errors=(),
        )
        active = SimpleNamespace(status="missing")
        complete = SimpleNamespace(status="complete")

        note = _capture_face_command_card_note(readiness, active)
        complete_note = _capture_face_command_card_note(readiness, complete)

        self.assertIn("Start Batch", note)
        self.assertIn("Capture Selected", note)
        self.assertIn("Capture Current", note)
        self.assertIn("posetag-capture-face", note)
        self.assertIn("registered face queue", note)
        self.assertIn("native guided", note)
        self.assertIn("CLI fallback", note)
        self.assertIn("raw image", note)
        self.assertIn("annotated image", note)
        self.assertIn("manifest row", note)
        self.assertIn("every registered face", complete_note)
        self.assertEqual(
            _capture_face_command_ready_message(readiness),
            "Ready to open guided capture or copy a face-shot capture command.",
        )

    def test_capture_face_process_state_text_and_run_button_labels(self) -> None:
        manifest = Path("/tmp/project/shots/manifest.csv")
        idle = capture_face_process_not_started(manifest)
        running = capture_face_process_running(manifest)
        success = CaptureFaceProcessState(
            state="finished",
            label="finished",
            message="Face-shot manifest was updated.",
            success=True,
            expected_manifest=manifest,
            exit_code=0,
        )

        self.assertEqual(_capture_face_run_button_label(idle), "Start Batch")
        self.assertEqual(
            _capture_face_run_button_label(running),
            "Capture Running...",
        )
        self.assertEqual(
            _capture_face_run_button_label(success),
            "Start Batch Again",
        )
        self.assertIn("not started", _format_capture_face_process_state(idle))
        self.assertIn("Expected manifest:", _format_capture_face_process_state(idle))
        self.assertIn("ENTER", _format_capture_face_process_state(running))

    def test_calibration_process_state_text_and_run_button_labels(self) -> None:
        expected_output = Path("/tmp/project/calib/calib_color.yaml")
        idle = calibration_process_not_started(expected_output)
        running = calibration_process_running(expected_output)

        self.assertEqual(_calibration_run_button_label(idle), "Run Calibration")
        self.assertEqual(
            _calibration_run_button_label(running),
            "Calibration Running...",
        )
        self.assertIn("not started", _format_calibration_process_state(idle))
        self.assertIn("Expected output:", _format_calibration_process_state(idle))
        self.assertIn("Guided auto-capture", _format_calibration_process_state(running))

    def test_calibration_result_summary_is_human_readable(self) -> None:
        summary = CameraCalibrationOutputSummary(
            path=Path("/tmp/project/calib/calib_color.yaml"),
            exists=True,
            valid=True,
            message="Colour-camera calibration YAML exists and passed schema checks.",
            image_width=640,
            image_height=480,
            model="plumb_bob",
            reproj_rms=0.1234,
            camera_params=(600.0, 610.0, 320.0, 240.0),
            distortion_coefficients=(("k1", -0.1), ("k2", 0.02)),
            latest_run_dir=Path("/tmp/project/calib/runs/2026-05-29T08-19-06Z"),
        )

        text = _format_calibration_result_summary(summary)

        self.assertIn("Calibration valid", text)
        self.assertIn("640 x 480", text)
        self.assertIn("RMS", text)
        self.assertIn("0.123 px", text)
        self.assertIn("fx=600.000", text)
        self.assertIn("k1=-0.1", text)
        self.assertIn("Run snapshot", text)

        html = _format_calibration_result_summary_html(summary)

        self.assertIn("font-family", html)
        self.assertIn("Menlo", html)
        self.assertIn("fx=600.000", html)
        self.assertIn("Calibration valid", html)

    def test_health_panel_counts_and_status_are_compact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            health = project_health_view(inspect_project_view(Path(tmpdir) / "project"))

        counts = _format_health_counts(health)
        self.assertIn("AT A GLANCE", counts)
        self.assertIn("<table", counts)
        self.assertIn("href='stage-status:missing'", counts)
        self.assertIn(">Missing</a>", counts)
        self.assertIn("<b>2</b>", counts)
        self.assertIn("href='stage-status:not_applicable'", counts)
        self.assertIn(">Not applicable</a>", counts)
        self.assertIn("<b>8</b>", counts)
        self.assertIn("<td>Complete</td>", counts)

        status_style = _health_status_style(
            foreground=health.status_color,
            background=health.status_background,
            border=health.status_border,
        )
        self.assertIn("font-size: 15px", status_style)
        self.assertIn("border-left: 4px", status_style)
        self.assertNotIn("font-size: 18px", status_style)

    def test_stage_rail_has_compact_labels_without_progress_strip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            models = inspect_project_view(Path(tmpdir) / "project")

        stage0 = models[0]

        self.assertEqual(
            _stage_list_label(stage0),
            "Stage 0\nProject Setup\nMissing",
        )
        stage3 = models[3]
        stage5 = models[5]

        self.assertEqual(_stage_rail_name(stage0), "Project")
        self.assertEqual(_stage_rail_status_label(stage0), "Missing")
        self.assertEqual(_stage_rail_name(stage3), "Tags")
        self.assertEqual(_stage_rail_status_label(stage3), "N/A")
        self.assertEqual(_stage_rail_name(stage5), "Shots")
        self.assertIn("#eef8ff", _stage_rail_entry_style(stage0, selected=True))
        self.assertIn("#0b6f8f", _stage_rail_entry_style(stage0, selected=True))
        self.assertIn(
            "border-left: 3px",
            _stage_rail_entry_style(stage0, selected=True),
        )
        self.assertIn(
            "border: 1px solid transparent",
            _stage_rail_entry_style(stage0, selected=False),
        )
        self.assertIn("background: transparent", _stage_rail_number_style(stage0))
        self.assertIn("border: none", _stage_rail_number_style(stage0))
        self.assertIn("font-size: 12px", _stage_rail_number_style(stage0))
        self.assertIn(stage0.status_color, _stage_rail_dot_style(stage0))
        self.assertIn("border-radius: 3px", _stage_rail_dot_style(stage0))
        self.assertIn(stage0.status_color, _stage_status_chip_style(stage0))
        self.assertIn("background: transparent", _stage_status_chip_style(stage0))
        self.assertIn("font-size: 9px", _stage_status_chip_style(stage0))

        style = _style_sheet()
        self.assertIn("QListWidget#WorkflowRail", style)
        self.assertNotIn("ProgressStrip", style)
        self.assertNotIn("ProgressStep", style)

    def test_backend_workflow_modules_do_not_import_gui(self) -> None:
        backend_files = list((SRC_ROOT / "posetag" / "workflows").glob("*.py"))
        backend_files.extend((SRC_ROOT / "posetag" / "pipelines").glob("*.py"))

        offenders: list[str] = []
        for path in backend_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "posetag.gui" or module.startswith("posetag.gui."):
                        offenders.append(str(path.relative_to(REPO_ROOT)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "posetag.gui" or alias.name.startswith(
                            "posetag.gui."
                        ):
                            offenders.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(offenders, [])

    def test_workflow_status_import_does_not_import_gui_or_pyside6(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = _pythonpath_with_src(env)
        code = (
            "from posetag.workflows.status import inspect_project\n"
            "import sys\n"
            "bad = [name for name in sys.modules "
            "if name.startswith('posetag.gui') or name.startswith('PySide6')]\n"
            "raise SystemExit(1 if bad else 0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)


def _pythonpath_with_src(env: dict[str, str]) -> str:
    existing = env.get("PYTHONPATH")
    if existing:
        return f"{SRC_ROOT}{os.pathsep}{existing}"
    return str(SRC_ROOT)


class _FakeSignal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback


class _FakeTimer:
    def __init__(self) -> None:
        self.interval = None
        self.started = False
        self.timeout = _FakeSignal()

    def setInterval(self, interval: int) -> None:
        self.interval = interval

    def start(self) -> None:
        self.started = True


class _FakeQtCore:
    QTimer = _FakeTimer


class _FakeQtApp:
    def __init__(self) -> None:
        self.quit_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1


if __name__ == "__main__":
    unittest.main()
