from __future__ import annotations

import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import yaml

from posetag.cli import gen_charuco as gen_charuco_cli
from posetag.pipelines import generate_charuco


REPO_ROOT = Path(__file__).resolve().parents[1]


class GenerateCharucoStep1Tests(unittest.TestCase):
    def run_with_isolated_config(
        self,
        tmpdir: str,
        argv: list[str],
    ) -> generate_charuco.CharucoBoardOutputs:
        base = Path(tmpdir)
        env = {
            "HOME": str(base / "home"),
            "XDG_CONFIG_HOME": str(base / ".config"),
        }
        args = generate_charuco.build_parser().parse_args(argv)
        with patch.dict(os.environ, env, clear=False):
            return generate_charuco.run(args)

    def test_cli_help_resolves(self) -> None:
        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                gen_charuco_cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Generate printable ChArUco", stdout.getvalue())
        self.assertIn("--square-length-mm", stdout.getvalue())

    def test_module_entrypoint_help_smoke(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "posetag.cli.gen_charuco", "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("posetag-gen-charuco", result.stdout)

    def test_pyproject_exposes_console_script_entry_point(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn(
            'posetag-gen-charuco = "posetag.cli.gen_charuco:main"',
            pyproject,
        )

    def test_valid_generation_writes_png_yaml_and_readable_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            outputs = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--squares-x",
                    "3",
                    "--squares-y",
                    "4",
                    "--square-length-mm",
                    "20",
                    "--marker-length-mm",
                    "14",
                    "--dict",
                    "7X7_50",
                    "--paper-mm",
                    "80x120",
                    "--dpi",
                    "80",
                ],
            )

            expected_dir = project_root.resolve() / "calib" / "boards"
            self.assertEqual(outputs.png.parent.resolve(), expected_dir)
            self.assertEqual(outputs.yaml.parent.resolve(), expected_dir)
            self.assertTrue(outputs.png.exists())
            self.assertTrue(outputs.yaml.exists())
            self.assertEqual(
                outputs.png.name,
                "charuco_3x4_square20mm_marker14mm_7X7_50_CUSTOM_80dpi.png",
            )

            metadata = yaml.safe_load(outputs.yaml.read_text(encoding="utf-8"))
            for key in (
                "squares_x",
                "squares_y",
                "square_length_mm",
                "marker_length_mm",
                "dictionary",
                "paper",
                "paper_width_mm",
                "paper_height_mm",
                "dpi",
                "image_width_px",
                "image_height_px",
                "png",
                "notes",
            ):
                self.assertIn(key, metadata)
            self.assertEqual(metadata["squares_x"], 3)
            self.assertEqual(metadata["squares_y"], 4)
            self.assertEqual(metadata["square_length_mm"], 20.0)
            self.assertEqual(metadata["marker_length_mm"], 14.0)
            self.assertEqual(metadata["dictionary"], "7X7_50")
            self.assertEqual(metadata["paper"], "CUSTOM")
            self.assertEqual(metadata["png"], outputs.png.name)
            self.assertIn("Actual Size", metadata["notes"])

            if outputs.pdf is None:
                self.assertNotIn("pdf", metadata)
            else:
                self.assertTrue(outputs.pdf.exists())
                self.assertEqual(metadata["pdf"], outputs.pdf.name)

            image = cv2.imread(str(outputs.png), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[1], metadata["image_width_px"])
            self.assertEqual(image.shape[0], metadata["image_height_px"])

    def test_invalid_dictionary_name_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(generate_charuco.CharucoBoardGenerationError) as ctx:
                self.run_with_isolated_config(
                    tmpdir,
                    [
                        "--out_dir",
                        str(Path(tmpdir) / "out"),
                        "--dict",
                        "not-a-dictionary",
                        "--paper-mm",
                        "80x120",
                        "--dpi",
                        "80",
                    ],
                )

        self.assertIn("Unsupported ArUco dictionary", str(ctx.exception))
        self.assertIn("7X7_50", str(ctx.exception))

    def test_dictionary_too_small_for_board_fails_before_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            with self.assertRaises(generate_charuco.CharucoBoardGenerationError) as ctx:
                self.run_with_isolated_config(
                    tmpdir,
                    [
                        "--out_dir",
                        str(out_dir),
                        "--squares-x",
                        "20",
                        "--squares-y",
                        "20",
                        "--square-length-mm",
                        "5",
                        "--marker-length-mm",
                        "3",
                        "--dict",
                        "4X4_50",
                        "--paper-mm",
                        "200x200",
                        "--dpi",
                        "150",
                    ],
                )

            self.assertIn("provides 50 marker IDs", str(ctx.exception))
            self.assertIn("requires 200", str(ctx.exception))
            self.assertFalse(out_dir.exists())

    def test_cli_dictionary_capacity_error_exits_without_traceback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "posetag.cli.gen_charuco",
                    "--out_dir",
                    str(out_dir),
                    "--squares-x",
                    "20",
                    "--squares-y",
                    "20",
                    "--square-length-mm",
                    "5",
                    "--marker-length-mm",
                    "3",
                    "--dict",
                    "4X4_50",
                    "--paper-mm",
                    "200x200",
                    "--dpi",
                    "150",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "Error: Dictionary 4X4_50 provides 50 marker IDs",
                result.stderr,
            )
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(out_dir.exists())

    def test_invalid_dimensions_and_lengths_fail_clearly(self) -> None:
        cases = [
            (
                ["--squares-x", "1", "--paper-mm", "80x120", "--dpi", "80"],
                "at least 2",
            ),
            (
                [
                    "--square-length-mm",
                    "20",
                    "--marker-length-mm",
                    "20",
                    "--paper-mm",
                    "80x120",
                    "--dpi",
                    "80",
                ],
                "smaller than --square-length-mm",
            ),
            (
                [
                    "--square-length-mm",
                    "200",
                    "--marker-length-mm",
                    "120",
                    "--paper-mm",
                    "80x120",
                    "--dpi",
                    "80",
                ],
                "does not fit",
            ),
            (
                [
                    "--marker-length-mm",
                    "0",
                    "--paper-mm",
                    "80x120",
                    "--dpi",
                    "80",
                ],
                "--marker-length-mm",
            ),
        ]

        for argv, expected in cases:
            with self.subTest(argv=argv):
                with TemporaryDirectory() as tmpdir:
                    with self.assertRaises(
                        generate_charuco.CharucoBoardGenerationError
                    ) as ctx:
                        self.run_with_isolated_config(
                            tmpdir,
                            ["--out_dir", str(Path(tmpdir) / "out"), *argv],
                        )
                    self.assertIn(expected, str(ctx.exception))

    def test_project_root_default_output_path_is_deterministic(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            outputs = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--squares-x",
                    "3",
                    "--squares-y",
                    "4",
                    "--square-length-mm",
                    "20",
                    "--marker-length-mm",
                    "14",
                    "--paper-mm",
                    "80x120",
                    "--dpi",
                    "80",
                ],
            )

            self.assertEqual(
                outputs.png.resolve(),
                (
                    project_root
                    / "calib"
                    / "boards"
                    / "charuco_3x4_square20mm_marker14mm_7X7_50_CUSTOM_80dpi.png"
                ).resolve(),
            )

    def test_explicit_output_directory_overrides_project_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            out_dir = base / "custom-charuco"
            outputs = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--out_dir",
                    str(out_dir),
                    "--squares-x",
                    "3",
                    "--squares-y",
                    "4",
                    "--square-length-mm",
                    "20",
                    "--marker-length-mm",
                    "14",
                    "--paper-mm",
                    "80x120",
                    "--dpi",
                    "80",
                ],
            )

            self.assertTrue((project_root / "boards").is_dir())
            self.assertEqual(outputs.png.parent, out_dir)
            self.assertTrue(outputs.png.exists())
            self.assertFalse((project_root / "calib" / "boards").exists())


if __name__ == "__main__":
    unittest.main()
