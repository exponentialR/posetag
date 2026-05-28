from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from posetag.workflows.charuco_setup import (
    DEFAULT_DICTIONARY,
    OUTPUT_FORMAT_CHOICES,
    OUTPUT_FORMAT_PNG,
    CharucoBoardSetupConfig,
    CharucoBoardSetupError,
    default_output_dir,
    dictionary_choices,
    generate_charuco_board_setup,
)


class WorkflowCharucoSetupTests(unittest.TestCase):
    def test_valid_setup_generates_outputs_under_project_calib_boards(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with _isolated_project_config(tmpdir):
                result = generate_charuco_board_setup(
                    CharucoBoardSetupConfig(
                        project_root=project_root,
                        squares_x=3,
                        squares_y=4,
                        square_length_mm=20.0,
                        marker_length_mm=14.0,
                        dictionary_name="7X7_50",
                        paper="A4",
                        orientation="portrait",
                        dpi=80,
                    )
                )

            expected_dir = project_root.resolve() / "calib" / "boards"
            self.assertEqual(result.out_dir.resolve(), expected_dir)
            self.assertEqual(result.png.parent.resolve(), expected_dir)
            self.assertEqual(result.yaml.parent.resolve(), expected_dir)
            self.assertTrue(result.png.exists())
            self.assertTrue(result.yaml.exists())
            self.assertIn(result.png, result.output_paths)
            self.assertIn(result.yaml, result.output_paths)

            metadata = yaml.safe_load(result.yaml.read_text(encoding="utf-8"))
            self.assertEqual(metadata["squares_x"], 3)
            self.assertEqual(metadata["squares_y"], 4)
            self.assertEqual(metadata["square_length_mm"], 20.0)
            self.assertEqual(metadata["marker_length_mm"], 14.0)
            self.assertEqual(metadata["dictionary"], "7X7_50")
            self.assertEqual(metadata["paper"], "A4")
            self.assertEqual(metadata["png"], result.png.name)
            self.assertIn("Actual Size", metadata["notes"])

            if result.pdf is None:
                self.assertNotIn("pdf", metadata)
            else:
                self.assertTrue(result.pdf.exists())
                self.assertIn(result.pdf, result.output_paths)
                self.assertEqual(metadata["pdf"], result.pdf.name)

    def test_explicit_output_folder_overrides_project_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            out_dir = base / "printed-board"
            with _isolated_project_config(tmpdir):
                result = generate_charuco_board_setup(
                    CharucoBoardSetupConfig(
                        project_root=project_root,
                        out_dir=out_dir,
                        squares_x=3,
                        squares_y=4,
                        square_length_mm=20.0,
                        marker_length_mm=14.0,
                        dpi=80,
                    )
                )

            self.assertEqual(result.out_dir, out_dir)
            self.assertEqual(result.png.parent, out_dir)
            self.assertTrue(result.png.exists())
            self.assertFalse((project_root / "calib" / "boards").exists())

    def test_png_only_output_format_suppresses_pdf_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "printed-board"
            with _isolated_project_config(tmpdir):
                result = generate_charuco_board_setup(
                    CharucoBoardSetupConfig(
                        project_root=Path(tmpdir) / "project",
                        out_dir=out_dir,
                        output_format=OUTPUT_FORMAT_PNG,
                        squares_x=3,
                        squares_y=4,
                        square_length_mm=20.0,
                        marker_length_mm=14.0,
                        dpi=80,
                    )
                )

            self.assertTrue(result.png.exists())
            self.assertTrue(result.yaml.exists())
            self.assertIsNone(result.pdf)
            self.assertFalse(result.requested_pdf)
            self.assertEqual(sorted(out_dir.glob("*.pdf")), [])

    def test_invalid_values_fail_clearly_before_output_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            with _isolated_project_config(tmpdir):
                with self.assertRaises(CharucoBoardSetupError) as ctx:
                    generate_charuco_board_setup(
                        CharucoBoardSetupConfig(
                            project_root=Path(tmpdir) / "project",
                            out_dir=out_dir,
                            square_length_mm=20.0,
                            marker_length_mm=20.0,
                            dpi=80,
                        )
                    )

            self.assertIn("smaller than --square-length-mm", str(ctx.exception))
            self.assertFalse(out_dir.exists())

    def test_invalid_orientation_fails_before_project_dirs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with _isolated_project_config(tmpdir):
                with self.assertRaises(CharucoBoardSetupError) as ctx:
                    generate_charuco_board_setup(
                        CharucoBoardSetupConfig(
                            project_root=project_root,
                            orientation="diagonal",
                        )
                    )

            self.assertIn("Orientation must be one of", str(ctx.exception))
            self.assertFalse(project_root.exists())

    def test_invalid_output_format_fails_before_project_dirs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with _isolated_project_config(tmpdir):
                with self.assertRaises(CharucoBoardSetupError) as ctx:
                    generate_charuco_board_setup(
                        CharucoBoardSetupConfig(
                            project_root=project_root,
                            output_format="SVG",
                        )
                    )

            self.assertIn("Output format must be one of", str(ctx.exception))
            self.assertFalse(project_root.exists())

    def test_default_output_dir_is_display_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            output_dir = default_output_dir(project_root)

            self.assertEqual(output_dir, project_root / "calib" / "boards")
            self.assertFalse(project_root.exists())

    def test_dictionary_choices_keep_default_first_when_supported(self) -> None:
        choices = dictionary_choices()

        self.assertIn(DEFAULT_DICTIONARY, choices)
        self.assertEqual(choices[0], DEFAULT_DICTIONARY)

    def test_output_format_choices_include_pdf_and_png_only(self) -> None:
        self.assertIn("PNG + PDF", OUTPUT_FORMAT_CHOICES)
        self.assertIn(OUTPUT_FORMAT_PNG, OUTPUT_FORMAT_CHOICES)


def _isolated_project_config(tmpdir: str):
    base = Path(tmpdir)
    return patch.dict(
        os.environ,
        {
            "HOME": str(base / "home"),
            "XDG_CONFIG_HOME": str(base / ".config"),
        },
        clear=False,
    )


if __name__ == "__main__":
    unittest.main()
