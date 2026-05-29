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
from utils import charuco_calibrate
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

    def test_module_invocation_help_resolves_for_gui_launch(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-m", "posetag.cli.charuco", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Colour ChArUco calibration", result.stdout)

    def test_direct_legacy_script_help_resolves_from_checkout(self) -> None:
        result = subprocess.run(
            [sys.executable, "utils/charuco_calibrate.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Colour ChArUco calibration", result.stdout)

    def test_direct_script_fallback_adds_repo_root_and_src(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        repo_src = repo_root / "src"
        original_path = sys.path[:]

        try:
            sys.path[:] = [
                path
                for path in sys.path
                if path not in {str(repo_root), str(repo_src)}
            ]

            charuco_calibrate._ensure_checkout_import_paths()

            self.assertIn(str(repo_root), sys.path)
            self.assertIn(str(repo_src), sys.path)
            self.assertLess(sys.path.index(str(repo_src)), sys.path.index(str(repo_root)))
        finally:
            sys.path[:] = original_path

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

    def test_missing_video_path_fails_before_project_artifacts(self) -> None:
        class FakeClosedVideoCapture:
            def __init__(self, _path: str) -> None:
                pass

            def isOpened(self) -> bool:
                return False

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }

            with patch.dict(os.environ, env, clear=False):
                with patch("utils.charuco_calibrate.cv2.VideoCapture", FakeClosedVideoCapture):
                    with self.assertRaises(SystemExit) as ctx:
                        charuco_cli.main(
                            [
                                "--project_root",
                                str(project_root),
                                "--source",
                                "video",
                                "--video",
                                str(base / "missing.mp4"),
                            ]
                        )

            self.assertIn("Could not open video", str(ctx.exception))
            self.assertFalse((project_root / "calib").exists())

    def test_video_eof_exits_capture_loop_clearly(self) -> None:
        class FakeVideoCapture:
            def __init__(self, _path: str) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def read(self):
                return False, None

            def release(self) -> None:
                self.released = True

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }
            stdout = StringIO()

            with patch.dict(os.environ, env, clear=False):
                with patch("utils.charuco_calibrate.cv2.VideoCapture", FakeVideoCapture):
                    with patch("utils.charuco_calibrate.cv2.destroyAllWindows"):
                        with self.assertRaises(SystemExit) as ctx:
                            with redirect_stdout(stdout):
                                charuco_cli.main(
                                    [
                                        "--project_root",
                                        str(project_root),
                                        "--source",
                                        "video",
                                        "--video",
                                        str(base / "empty.mp4"),
                                    ]
                                )

            self.assertIn("video ended", stdout.getvalue())
            self.assertIn("Need at least", str(ctx.exception))

    def test_q_exits_cleanly_without_traceback_or_solve(self) -> None:
        class FakeVideoCapture:
            def __init__(self, _path: str) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def read(self):
                frame = np.zeros((20, 20, 3), dtype=np.uint8)
                return True, frame

            def release(self) -> None:
                self.released = True

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }
            stdout = StringIO()

            with patch.dict(os.environ, env, clear=False):
                with patch("utils.charuco_calibrate.cv2.VideoCapture", FakeVideoCapture):
                    with patch("utils.charuco_calibrate.cv2.imshow"):
                        with patch("utils.charuco_calibrate.cv2.waitKey", return_value=ord("q")):
                            with patch("utils.charuco_calibrate.cv2.destroyAllWindows"):
                                with redirect_stdout(stdout):
                                    result = charuco_cli.main(
                                        [
                                            "--project_root",
                                            str(project_root),
                                            "--source",
                                            "video",
                                            "--video",
                                            str(base / "sample.mp4"),
                                        ]
                                    )

            self.assertEqual(result, 0)
            self.assertIn("quit requested", stdout.getvalue())
            self.assertFalse((project_root / "calib" / "calib_color.yaml").exists())

    def test_realsense_source_without_dependency_fails_clearly(self) -> None:
        with patch("utils.charuco_calibrate.rs", None):
            with self.assertRaises(SystemExit) as ctx:
                charuco_cli.main(["--source", "realsense"])

        self.assertIn("pyrealsense2 is not available", str(ctx.exception))

    def test_min_corners_default_matches_solver_threshold(self) -> None:
        args = charuco_calibrate.parse_args([])

        self.assertEqual(args.min_corners, 4)

    def test_guided_capture_arguments_validate_to_grid_shape(self) -> None:
        args = charuco_calibrate.parse_args(
            [
                "--coverage-grid",
                "4x5",
                "--samples-per-cell",
                "2",
                "--guided-auto-cooldown",
                "3",
                "--no-guided-auto",
            ]
        )

        charuco_calibrate._validate_guided_capture_args(args)

        self.assertEqual(args.coverage_grid_shape, (4, 5))
        self.assertEqual(args.samples_per_cell, 2)
        self.assertEqual(args.guided_auto_cooldown, 3)
        self.assertFalse(args.guided_auto)

    def test_invalid_guided_capture_args_fail_before_project_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            env = {
                "HOME": str(base / "home"),
                "XDG_CONFIG_HOME": str(base / ".config"),
            }

            with patch.dict(os.environ, env, clear=False):
                with self.assertRaises(SystemExit) as ctx:
                    charuco_cli.main(
                        [
                            "--project_root",
                            str(project_root),
                            "--coverage-grid",
                            "0x3",
                        ]
                    )

            self.assertIn("--coverage-grid values must be positive", str(ctx.exception))
            self.assertFalse((project_root / "calib").exists())

    def test_guided_minimum_samples_reflect_grid_and_samples_per_cell(self) -> None:
        args = charuco_calibrate.parse_args(
            [
                "--coverage-grid",
                "4x4",
                "--samples-per-cell",
                "2",
                "--min-samples",
                "10",
            ]
        )
        charuco_calibrate._validate_guided_capture_args(args)

        self.assertEqual(charuco_calibrate._minimum_required_samples(args), 32)

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
