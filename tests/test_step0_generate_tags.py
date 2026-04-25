from __future__ import annotations

import os
import site
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2

from posetag.pipelines import generate_tags


REPO_ROOT = Path(__file__).resolve().parents[1]


def _installed_posetag_gen_tags() -> str | None:
    candidates = [
        Path(site.getuserbase()) / "bin" / "posetag-gen-tags",
        Path(sys.executable).resolve().parent / "posetag-gen-tags",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


class GenerateTagsStep0Tests(unittest.TestCase):
    def run_with_isolated_config(
        self, tmpdir: str, argv: list[str]
    ) -> tuple[list[Path], list[Path]]:
        base = Path(tmpdir)
        env = {
            "HOME": str(base / "home"),
            "XDG_CONFIG_HOME": str(base / ".config"),
        }
        with patch.dict(os.environ, env, clear=False):
            return generate_tags.run(generate_tags.build_parser().parse_args(argv))

    def test_parse_ids_comma_list(self) -> None:
        self.assertEqual(generate_tags.parse_ids_string("1,2,3"), [1, 2, 3])

    def test_parse_ids_range(self) -> None:
        self.assertEqual(generate_tags.parse_ids_string("1-4"), [1, 2, 3, 4])

    def test_parse_ids_mixed(self) -> None:
        self.assertEqual(generate_tags.parse_ids_string("1-3,7,9-10"), [1, 2, 3, 7, 9, 10])

    def test_parse_ids_descending_range_fails(self) -> None:
        with self.assertRaises(generate_tags.TagGenerationError) as ctx:
            generate_tags.parse_ids_string("4-1")
        self.assertIn("end < start", str(ctx.exception))

    def test_project_root_run_creates_expected_directories_and_png(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            png_paths, pdf_paths = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "1-4",
                    "--dpi",
                    "150",
                ],
            )

            self.assertTrue((project_root / "boards").is_dir())
            self.assertTrue((project_root / "boards" / "patterns").is_dir())
            self.assertEqual(len(png_paths), 1)
            self.assertTrue(png_paths[0].exists())
            self.assertEqual(
                png_paths[0].parent.resolve(),
                (project_root / "boards" / "patterns").resolve(),
            )
            for pdf_path in pdf_paths:
                self.assertTrue(pdf_path.exists())

    def test_explicit_out_dir_overrides_project_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            out_dir = base / "custom-out"
            png_paths, _ = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--out_dir",
                    str(out_dir),
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "1,2,3",
                    "--dpi",
                    "150",
                ],
            )

            self.assertTrue((project_root / "boards").is_dir())
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(len(png_paths), 1)
            self.assertTrue(png_paths[0].exists())
            self.assertEqual(png_paths[0].parent, out_dir)

    def test_explicit_default_named_out_dir_overrides_project_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            cwd = base / "work"
            cwd.mkdir()
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }
            args = generate_tags.build_parser().parse_args(
                [
                    "--project_root",
                    str(project_root),
                    "--out_dir",
                    "apriltags_out",
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "1",
                    "--dpi",
                    "50",
                ]
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, env, clear=False):
                    png_paths, _ = generate_tags.run(args)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(len(png_paths), 1)
            self.assertEqual(png_paths[0].parent, Path("apriltags_out"))
            self.assertTrue((cwd / "apriltags_out").is_dir())
            self.assertFalse((project_root / "boards" / "patterns").exists())

    def test_impossible_page_configuration_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            args = generate_tags.build_parser().parse_args(
                [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "200",
                    "--paper-mm",
                    "50x50",
                    "--ids",
                    "1",
                    "--dpi",
                    "150",
                ]
            )
            with self.assertRaises(generate_tags.TagGenerationError) as ctx:
                with patch.dict(
                    os.environ,
                    {"HOME": str(Path(tmpdir) / "home"), "XDG_CONFIG_HOME": str(Path(tmpdir) / ".config")},
                    clear=False,
                ):
                    generate_tags.run(args)
            self.assertIn("cannot fit", str(ctx.exception))

    def test_invalid_tag_id_fails_before_opencv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            args = generate_tags.build_parser().parse_args(
                [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "9999",
                    "--dpi",
                    "150",
                ]
            )
            with self.assertRaises(generate_tags.TagGenerationError) as ctx:
                with patch.dict(
                    os.environ,
                    {"HOME": str(Path(tmpdir) / "home"), "XDG_CONFIG_HOME": str(Path(tmpdir) / ".config")},
                    clear=False,
                ):
                    generate_tags.run(args)
            self.assertIn("IDs must be between", str(ctx.exception))

    def test_non_positive_numeric_inputs_fail_before_opencv(self) -> None:
        parser = generate_tags.build_parser()
        bad_cases = [
            (["--tag-size-mm", "0", "--ids", "1"], "--tag-size-mm"),
            (["--tag-size-mm", "0.001", "--ids", "1", "--dpi", "1"], "at least 1 pixel"),
            (["--tag-size-mm", "40", "--ids", "1", "--dpi", "0"], "--dpi"),
            (["--tag-size-mm", "40", "--ids", "1", "--margin-frac", "0.9"], "--margin-frac"),
            (["--tag-size-mm", "40", "--ids", "1", "--label-gap-frac", "-0.1"], "--label-gap-frac"),
        ]
        for argv, expected in bad_cases:
            with self.subTest(argv=argv):
                with self.assertRaises(generate_tags.TagGenerationError) as ctx:
                    generate_tags.run(parser.parse_args(argv))
                self.assertIn(expected, str(ctx.exception))

    def test_generated_png_can_be_opened_with_opencv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            png_paths, _ = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "1-4",
                    "--dpi",
                    "150",
                ],
            )
            image = cv2.imread(str(png_paths[0]))
            self.assertIsNotNone(image)
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[1], 0)

    def test_generated_png_records_requested_dpi_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            png_paths, _ = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "1",
                    "--dpi",
                    "150",
                ],
            )
            self.assertIn(b"pHYs", png_paths[0].read_bytes())

    def test_opencv_label_text_fits_within_tag_cell(self) -> None:
        _, _, text_w, _, _ = generate_tags._fit_cv_text("ID 9999 | 300 mm", 250)
        self.assertLessEqual(text_w, 250)

    def test_more_ids_than_one_page_creates_additional_png_sheets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            png_paths, _ = self.run_with_isolated_config(
                tmpdir,
                [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "40",
                    "--paper-mm",
                    "80x80",
                    "--ids",
                    "1-3",
                    "--dpi",
                    "100",
                ],
            )

            self.assertEqual(len(png_paths), 3)
            self.assertTrue(all(path.exists() for path in png_paths))
            self.assertTrue(all("_page" in path.stem for path in png_paths))

    def test_cli_help_smoke(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        command = (
            [_installed_posetag_gen_tags(), "--help"]
            if _installed_posetag_gen_tags() is not None
            else [sys.executable, "-m", "posetag.cli.gen_tags", "--help"]
        )
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--tag-size-mm", result.stdout + result.stderr)

    def test_module_entrypoint_help_smoke(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "posetag.cli.gen_tags", "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--tag-size-mm", result.stdout)

    def test_cli_generation_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["XDG_CONFIG_HOME"] = str(Path(tmpdir) / ".config")
            command = (
                [_installed_posetag_gen_tags()]
                if _installed_posetag_gen_tags() is not None
                else [sys.executable, "-m", "posetag.cli.gen_tags"]
            )
            result = subprocess.run(
                command
                + [
                    "--project_root",
                    str(project_root),
                    "--tag-size-mm",
                    "40",
                    "--ids",
                    "1-4",
                    "--dpi",
                    "150",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            resolved_root = project_root.resolve()
            self.assertTrue((resolved_root / "boards").is_dir())
            self.assertTrue((resolved_root / "boards" / "patterns").is_dir())
            self.assertEqual(len(list((resolved_root / "boards" / "patterns").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
