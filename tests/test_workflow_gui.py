from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from posetag.gui import app as gui_app
from posetag.gui.main_window import (
    _copy_confirmation_message,
    _format_charuco_outputs,
    _format_health_counts,
    _health_status_style,
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

        self.assertEqual(len(models), 7)
        self.assertEqual(models[0].stage_id, 0)
        self.assertEqual(models[0].status, "missing")
        self.assertIn("posetag-gen-tags", models[0].command_preview)
        self.assertIn("--project_root", models[0].command_preview)
        self.assertEqual(models[6].command_preview, "")

    def test_project_health_view_handles_empty_project_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "empty-project"
            health = project_health_view(inspect_project_view(project_root))

        self.assertEqual(health.status, "missing")
        self.assertEqual(health.headline, "No workflow outputs found")
        self.assertEqual(health.error_count, 0)
        self.assertEqual(health.next_stage_label, "Stage 0: Generate AprilTag Sheets")
        self.assertIn("posetag-gen-tags", health.next_action)

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
        self.assertEqual(health.next_stage_label, "Stage 1: Calibrate Camera")
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
        self.assertEqual(health.next_stage_label, "Stage 0: Generate AprilTag Sheets")
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

        annotate_preview = command_preview("annotate_faces", project_root)
        collect_preview = command_preview("collect_dataset", project_root)

        self.assertIn("env 'POSETAG_PROJECT=/tmp/PoseTag project'", annotate_preview)
        self.assertIn("posetag-annotate --browse", annotate_preview)
        self.assertIn("env 'POSETAG_PROJECT=/tmp/PoseTag project'", collect_preview)
        self.assertIn("posetag-collect --mode live", collect_preview)

    def test_command_copy_confirmation_text_is_explicit(self) -> None:
        self.assertEqual(
            _copy_confirmation_message("posetag-gen-tags --project_root project"),
            "Copied command preview to the clipboard.",
        )
        self.assertEqual(
            _copy_confirmation_message(""),
            "No command preview is available for this stage.",
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
        self.assertIn("QLabel#StageTitle", style)
        self.assertIn("QLineEdit#CommandPreview", style)
        self.assertIn("QLabel#HealthMessage", style)
        self.assertIn("QLabel#HealthSectionTitle", style)
        self.assertIn("QLabel#HealthSectionBody", style)
        self.assertIn("QLabel#RailStageName", style)
        self.assertIn("QLabel#RailStageNumber", style)
        self.assertIn("QLabel#RailStageDot", style)
        self.assertIn("QLabel#RailStageStatus", style)
        self.assertIn("QLabel#CharucoPreviewCanvas", style)
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

    def test_health_panel_counts_and_status_are_compact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            health = project_health_view(inspect_project_view(Path(tmpdir) / "project"))

        counts = _format_health_counts(health)
        self.assertIn("AT A GLANCE", counts)
        self.assertIn("<table", counts)
        self.assertIn("<td>Missing</td>", counts)
        self.assertIn("<b>3</b>", counts)

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
            "Stage 0\nGenerate AprilTag Sheets\nMissing",
        )
        stage3 = models[3]

        self.assertEqual(_stage_rail_name(stage0), "Tags")
        self.assertEqual(_stage_rail_status_label(stage0), "Missing")
        self.assertEqual(_stage_rail_name(stage3), "Shots")
        self.assertEqual(_stage_rail_status_label(stage3), "N/A")
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
