from __future__ import annotations

import importlib
import os
import tomllib
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


class RenamePhase0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with PYPROJECT_PATH.open("rb") as handle:
            cls.pyproject = tomllib.load(handle)

    def test_project_metadata_is_canonical(self) -> None:
        project = self.pyproject["project"]
        scripts = project["scripts"]

        self.assertEqual(project["name"], "posetag")
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(scripts["posetag"], "posetag.cli.posetag:main")
        self.assertEqual(scripts["posetag-gen-tags"], "posetag.cli.gen_tags:main")
        self.assertEqual(scripts["posetag-gen-charuco"], "posetag.cli.gen_charuco:main")
        self.assertEqual(scripts["posetag-gui"], "posetag.gui.app:main")
        self.assertEqual(scripts["gtat"], "gt6dof_atag.cli.gtat:main")

    def test_script_targets_are_importable(self) -> None:
        scripts = self.pyproject["project"]["scripts"]
        expected = [
            "posetag",
            "posetag-gen-tags",
            "posetag-gen-charuco",
            "posetag-calib-charuco",
            "posetag-make-board",
            "posetag-capture-face",
            "posetag-annotate",
            "posetag-collect",
            "posetag-gui",
            "gtat",
            "gtat-gen-tags",
            "gtat-calib-charuco",
            "gtat-make-board",
            "gtat-capture-face",
            "gtat-annotate",
            "gtat-collect",
        ]

        for script_name in expected:
            module_name, func_name = scripts[script_name].split(":")
            module = importlib.import_module(module_name)
            self.assertTrue(
                callable(getattr(module, func_name)),
                msg=f"{script_name} did not resolve to a callable target",
            )

    def test_canonical_and_legacy_imports_work(self) -> None:
        import gt6dof_atag
        import posetag
        from gt6dof_atag.utils.project_config import resolve_project_root as legacy_resolve
        from posetag.utils.project_config import _norm, resolve_project_root

        self.assertEqual(posetag.__version__, gt6dof_atag.__version__)
        self.assertTrue(callable(resolve_project_root))
        self.assertTrue(callable(legacy_resolve))
        self.assertTrue(callable(_norm))

    def test_project_home_cli_uses_canonical_directory_for_both_commands(self) -> None:
        from gt6dof_atag.cli.gtat import main as legacy_main
        from posetag.cli.posetag import main as canonical_main

        with TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            env = {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
            }

            with patch.dict(os.environ, env, clear=False):
                canonical_stdout = StringIO()
                with redirect_stdout(canonical_stdout):
                    canonical_rc = canonical_main(["project", "home"])

                legacy_stdout = StringIO()
                with redirect_stdout(legacy_stdout):
                    legacy_rc = legacy_main(["project", "home"])

            expected = str((home / "posetag").resolve())
            self.assertEqual(canonical_rc, 0)
            self.assertEqual(legacy_rc, 0)
            self.assertEqual(canonical_stdout.getvalue().strip(), expected)
            self.assertEqual(legacy_stdout.getvalue().strip(), expected)

    def test_legacy_env_var_still_resolves_project_root(self) -> None:
        from posetag.utils.project_config import resolve_project_root

        with TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            legacy_project = home / "legacy-project"
            env = {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "GTAT_PROJECT": str(legacy_project),
            }

            with patch.dict(os.environ, env, clear=False):
                resolved = resolve_project_root()

            self.assertEqual(resolved, legacy_project.resolve())
            for subdir in ("boards", "shots", "objects", "datasets"):
                self.assertTrue((legacy_project / subdir).is_dir(), msg=f"missing {subdir}")

    def test_canonical_env_var_takes_precedence_over_legacy(self) -> None:
        from posetag.utils.project_config import resolve_project_root

        with TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            canonical_project = home / "canonical-project"
            legacy_project = home / "legacy-project"
            env = {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "POSETAG_PROJECT": str(canonical_project),
                "GTAT_PROJECT": str(legacy_project),
            }

            with patch.dict(os.environ, env, clear=False):
                resolved = resolve_project_root()

            self.assertEqual(resolved, canonical_project.resolve())
            self.assertTrue((canonical_project / "boards").is_dir())
            self.assertFalse((legacy_project / "boards").exists())


if __name__ == "__main__":
    unittest.main()
