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
from tempfile import TemporaryDirectory
from unittest.mock import patch

from posetag.gui import app as gui_app
from posetag.gui.main_window import _style_sheet
from posetag.gui.models import inspect_project_view
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

    def test_selected_stage_row_keeps_readable_text_style(self) -> None:
        style = _style_sheet()

        self.assertIn("QListWidget::item:selected", style)
        self.assertIn("background: #e8f1fb", style)
        self.assertIn("color: #0b1724", style)
        self.assertIn("QListWidget::item:selected:!active", style)

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


if __name__ == "__main__":
    unittest.main()
