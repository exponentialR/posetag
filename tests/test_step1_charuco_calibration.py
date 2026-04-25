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

import numpy as np
import yaml

from posetag.cli import charuco as charuco_cli
from posetag.pipelines.charuco_calibration import (
    CharucoCalibrationError,
    build_calibration_yaml,
    get_dictionary,
    prepare_project_io,
    write_calibration_yaml,
)


class CharucoCalibrationStep1Tests(unittest.TestCase):
    def test_cli_help_resolves(self) -> None:
        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                charuco_cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Colour ChArUco calibration", stdout.getvalue())
        self.assertIn("--source", stdout.getvalue())

    def test_console_script_target_help_resolves_after_install(self) -> None:
        code = (
            "from posetag.cli.charuco import main\n"
            "try:\n"
            "    main(['--help'])\n"
            "except SystemExit as exc:\n"
            "    raise SystemExit(exc.code)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Colour ChArUco calibration", result.stdout)

    def test_dictionary_parsing_accepts_valid_name(self) -> None:
        dictionary = get_dictionary("7X7_50")

        self.assertIsNotNone(dictionary)

    def test_dictionary_parsing_rejects_invalid_name_clearly(self) -> None:
        with self.assertRaises(CharucoCalibrationError) as ctx:
            get_dictionary("not-a-dictionary")

        self.assertIn("Unsupported ArUco dictionary", str(ctx.exception))
        self.assertIn("7X7_50", str(ctx.exception))

    def test_missing_video_for_video_source_fails_clearly(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            charuco_cli.main(["--source", "video"])

        self.assertIn("--video path is required", str(ctx.exception))

    def test_realsense_source_without_dependency_fails_clearly(self) -> None:
        with patch("utils.charuco_calibrate.rs", None):
            with self.assertRaises(SystemExit) as ctx:
                charuco_cli.main(["--source", "realsense"])

        self.assertIn("pyrealsense2 is not available", str(ctx.exception))

    def test_project_io_preparation_creates_calibration_layout(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }

            with patch.dict(os.environ, env, clear=False):
                project_io = prepare_project_io(
                    project_root=project_root,
                    timestamp="2026-04-25T12-00-00Z",
                )

            self.assertTrue((project_root / "calib").is_dir())
            self.assertTrue((project_root / "calib" / "images" / "set_01").is_dir())
            self.assertTrue(
                (project_root / "calib" / "runs" / "2026-04-25T12-00-00Z").is_dir()
            )
            self.assertEqual(
                project_io.out_yaml.resolve(),
                (project_root / "calib" / "calib_color.yaml").resolve(),
            )

    def test_explicit_out_overrides_default_calibration_yaml_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            out_yaml = base / "custom" / "camera.yaml"
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }

            with patch.dict(os.environ, env, clear=False):
                project_io = prepare_project_io(
                    project_root=project_root,
                    out=out_yaml,
                    timestamp="2026-04-25T12-00-00Z",
                )

            self.assertEqual(project_io.out_yaml, out_yaml)
            self.assertTrue(out_yaml.parent.is_dir())

    def test_calibration_yaml_schema_writes_and_round_trips(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calib_color.yaml"
            camera_matrix = np.array(
                [
                    [600.0, 0.0, 320.0],
                    [0.0, 610.0, 240.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            distortion = np.array([[-0.1, 0.02, 0.001, -0.002, 0.0]])
            data = build_calibration_yaml(
                image_width=640,
                image_height=480,
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion,
                reproj_rms=0.123,
                notes="ChArUco 3x5, square=50.0mm, marker=37.0mm, dict=7X7_50",
            )

            write_calibration_yaml(path, data)
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

            for key in (
                "image_width",
                "image_height",
                "camera_matrix",
                "distortion_coefficients",
                "reproj_rms",
                "model",
                "notes",
            ):
                self.assertIn(key, loaded)
            self.assertEqual(loaded["image_width"], 640)
            self.assertEqual(loaded["image_height"], 480)
            self.assertEqual(loaded["model"], "plumb_bob")
            self.assertEqual(loaded["camera_matrix"]["fx"], 600.0)
            self.assertEqual(loaded["distortion_coefficients"]["k1"], -0.1)


if __name__ == "__main__":
    unittest.main()
