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

import make_board as legacy_make_board
from posetag.cli import make_board as make_board_cli
from posetag.pipelines.make_board import (
    build_board_yaml,
    load_registry,
    prepare_project_paths,
    resolve_calibration_path,
    save_registry,
    update_registry_entries,
    write_board_yaml,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_calibration(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "image_width": 640,
                "image_height": 480,
                "camera_matrix": {
                    "fx": 600.0,
                    "fy": 610.0,
                    "cx": 320.0,
                    "cy": 240.0,
                },
                "distortion_coefficients": {
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                    "k3": 0.0,
                },
            },
            handle,
        )


class FakeDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect(self, *args, **kwargs):
        return []


class MakeBoardStep2Tests(unittest.TestCase):
    def isolated_env(self, base: Path) -> dict[str, str]:
        return {
            "HOME": str(base / "home"),
            "XDG_CONFIG_HOME": str(base / ".config"),
        }

    def test_cli_help_resolves(self) -> None:
        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                make_board_cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Interactive AprilTag board builder", stdout.getvalue())
        self.assertIn("--source", stdout.getvalue())

    def test_console_script_target_help_resolves_after_install(self) -> None:
        code = (
            "from posetag.cli.make_board import main\n"
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
        self.assertIn("Interactive AprilTag board builder", result.stdout)

    def test_direct_legacy_script_help_resolves_from_checkout(self) -> None:
        result = subprocess.run(
            [sys.executable, "src/make_board.py", "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Interactive AprilTag board builder", result.stdout)

    def test_direct_legacy_script_help_resolves_outside_checkout_cwd(self) -> None:
        with TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "src" / "make_board.py"), "--help"],
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Interactive AprilTag board builder", result.stdout)

    def test_missing_video_for_video_source_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with self.assertRaises(SystemExit) as ctx:
                make_board_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--object_name",
                        "connection_plate_white_sideA",
                        "--source",
                        "video",
                    ]
                )

            self.assertIn("--video path is required", str(ctx.exception))
            self.assertFalse((project_root / "boards").exists())

    def test_unreadable_video_path_fails_before_project_artifacts(self) -> None:
        class FakeClosedVideoCapture:
            def __init__(self, _path: str) -> None:
                pass

            def isOpened(self) -> bool:
                return False

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            video = base / "empty.mp4"
            video.touch()
            _write_calibration(project_root / "calib" / "calib_color.yaml")

            with patch.dict(os.environ, self.isolated_env(base), clear=False):
                with patch("make_board._load_detector_class", return_value=FakeDetector):
                    with patch("make_board.cv2.VideoCapture", FakeClosedVideoCapture):
                        with self.assertRaises(SystemExit) as ctx:
                            make_board_cli.main(
                                [
                                    "--project_root",
                                    str(project_root),
                                    "--object_name",
                                    "connection_plate_white_sideA",
                                    "--source",
                                    "video",
                                    "--video",
                                    str(video),
                                ]
                            )

            self.assertIn("Could not open video", str(ctx.exception))
            self.assertFalse((project_root / "boards").exists())

    def test_realsense_source_without_dependency_fails_clearly(self) -> None:
        with patch("make_board.rs", None):
            with self.assertRaises(SystemExit) as ctx:
                make_board_cli.main(
                    [
                        "--object_name",
                        "connection_plate_white_sideA",
                        "--source",
                        "realsense",
                    ]
                )

        self.assertIn("pyrealsense2 is not available", str(ctx.exception))

    def test_missing_calibration_yaml_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with self.assertRaises(SystemExit) as ctx:
                make_board_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--object_name",
                        "connection_plate_white_sideA",
                    ]
                )

            self.assertIn("Calibration YAML not found", str(ctx.exception))
            self.assertFalse((project_root / "boards").exists())

    def test_malformed_calibration_yaml_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib = project_root / "calib" / "calib_color.yaml"
            calib.parent.mkdir(parents=True)
            calib.write_text("not_camera_matrix: true\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as ctx:
                make_board_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--object_name",
                        "connection_plate_white_sideA",
                    ]
                )

            self.assertIn("Malformed calibration YAML", str(ctx.exception))
            self.assertFalse((project_root / "boards").exists())

    def test_calibration_resolution_finds_project_calib_color_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            expected = project_root / "calib" / "calib_color.yaml"
            _write_calibration(expected)

            resolved = resolve_calibration_path(project_root, "calib_color.yaml")

            self.assertEqual(resolved.resolve(), expected.resolve())

    def test_project_path_preparation_creates_expected_board_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            with patch.dict(os.environ, self.isolated_env(base), clear=False):
                paths = prepare_project_paths(
                    project_root=project_root,
                    object_name="connection_plate_white_sideA",
                )

            self.assertTrue((project_root / "boards").is_dir())
            self.assertEqual(paths.boards_dir.resolve(), (project_root / "boards").resolve())
            self.assertEqual(
                paths.board_yaml_path.resolve(),
                (project_root / "boards" / "connection_plate_white_sideA.yaml").resolve(),
            )
            self.assertEqual(
                paths.registry_path.resolve(),
                (project_root / "boards" / "tag_registry.yaml").resolve(),
            )

    def test_explicit_output_overrides_are_respected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            out_dir = base / "custom_boards"
            registry = base / "registry" / "tags.yaml"
            shots_dir = base / "audit_shots"

            with patch.dict(os.environ, self.isolated_env(base), clear=False):
                paths = prepare_project_paths(
                    project_root=project_root,
                    object_name="connection_plate_white_sideA",
                    out_dir=out_dir,
                    registry=registry,
                    shots_dir=shots_dir,
                    save_shot=True,
                )

            self.assertEqual(paths.boards_dir, out_dir)
            self.assertEqual(paths.board_yaml_path, out_dir / "connection_plate_white_sideA.yaml")
            self.assertEqual(paths.registry_path, registry)
            self.assertEqual(paths.shots_dir, shots_dir)
            self.assertTrue(out_dir.is_dir())
            self.assertTrue(registry.parent.is_dir())
            self.assertTrue(shots_dir.is_dir())

    def test_board_yaml_schema_generation_round_trips(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "board.yaml"
            board = build_board_yaml(
                object_name="connection_plate_white_sideA",
                family="tag36h11",
                tag_size_mm=80.0,
                origin_id=52,
                entries=[
                    {"id": 53, "cx": 0.1, "cy": -0.2, "yaw_deg": 180.0},
                    {"id": 52, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
                ],
            )

            write_board_yaml(path, board)
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

            for key in ("object", "family", "tag_size_m", "origin_id", "tags", "notes"):
                self.assertIn(key, loaded)
            self.assertEqual(loaded["object"], "connection_plate_white_sideA")
            self.assertEqual(loaded["tag_size_m"], 0.08)
            self.assertEqual([entry["id"] for entry in loaded["tags"]], [52, 53])
            for entry in loaded["tags"]:
                for key in ("id", "cx", "cy", "yaw_deg"):
                    self.assertIn(key, entry)

    def test_tag_registry_creation_and_update(self) -> None:
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "boards" / "tag_registry.yaml"
            board_path = Path(tmpdir) / "boards" / "connection_plate_white_sideA.yaml"
            registry = load_registry(registry_path)

            update = update_registry_entries(
                registry,
                [{"id": 52}, {"id": 53}],
                "connection_plate_white_sideA",
                board_path,
            )
            save_registry(registry_path, registry)
            loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

            self.assertEqual(update.updated, 2)
            self.assertEqual(update.conflicts, ())
            self.assertEqual(loaded["version"], 1)
            self.assertEqual(
                loaded["tags"]["52"],
                {"object": "connection_plate_white_sideA", "yaml": str(board_path)},
            )
            self.assertIn("updated", loaded)

    def test_tag_registry_conflict_does_not_silently_overwrite(self) -> None:
        registry = {
            "version": 1,
            "updated": None,
            "tags": {
                "52": {"object": "old_face", "yaml": "/tmp/old_face.yaml"},
            },
        }

        update = update_registry_entries(
            registry,
            [{"id": 52}, {"id": 53}],
            "connection_plate_white_sideA",
            "/tmp/connection_plate_white_sideA.yaml",
        )

        self.assertEqual(update.updated, 1)
        self.assertEqual(len(update.conflicts), 1)
        self.assertEqual(registry["tags"]["52"]["yaml"], "/tmp/old_face.yaml")
        self.assertEqual(
            registry["tags"]["53"]["yaml"],
            "/tmp/connection_plate_white_sideA.yaml",
        )

    def test_esc_exits_cleanly_without_writing_board_yaml(self) -> None:
        class FakeVideoCapture:
            def __init__(self, _path: str) -> None:
                pass

            def isOpened(self) -> bool:
                return True

            def read(self):
                return True, np.zeros((20, 20, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            video = base / "sample.mp4"
            video.touch()
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            stdout = StringIO()

            with patch.dict(os.environ, self.isolated_env(base), clear=False):
                with patch("make_board._load_detector_class", return_value=FakeDetector):
                    with patch("make_board.cv2.VideoCapture", FakeVideoCapture):
                        with patch("make_board.cv2.imshow"):
                            with patch("make_board.cv2.waitKey", return_value=27):
                                with patch("make_board.cv2.destroyAllWindows"):
                                    with redirect_stdout(stdout):
                                        result = make_board_cli.main(
                                            [
                                                "--project_root",
                                                str(project_root),
                                                "--object_name",
                                                "connection_plate_white_sideA",
                                                "--source",
                                                "video",
                                                "--video",
                                                str(video),
                                            ]
                                        )

            self.assertEqual(result, 0)
            self.assertIn("quit requested", stdout.getvalue())
            self.assertFalse((project_root / "boards" / "connection_plate_white_sideA.yaml").exists())

    def test_video_eof_exits_without_hanging_or_writing_board_yaml(self) -> None:
        class FakeVideoCapture:
            def __init__(self, _path: str) -> None:
                pass

            def isOpened(self) -> bool:
                return True

            def read(self):
                return False, None

            def release(self) -> None:
                pass

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            video = base / "sample.mp4"
            video.touch()
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            stdout = StringIO()

            with patch.dict(os.environ, self.isolated_env(base), clear=False):
                with patch("make_board._load_detector_class", return_value=FakeDetector):
                    with patch("make_board.cv2.VideoCapture", FakeVideoCapture):
                        with patch("make_board.cv2.destroyAllWindows"):
                            with redirect_stdout(stdout):
                                result = make_board_cli.main(
                                    [
                                        "--project_root",
                                        str(project_root),
                                        "--object_name",
                                        "connection_plate_white_sideA",
                                        "--source",
                                        "video",
                                        "--video",
                                        str(video),
                                    ]
                                )

            self.assertEqual(result, 0)
            self.assertIn("video ended", stdout.getvalue())
            self.assertFalse((project_root / "boards" / "connection_plate_white_sideA.yaml").exists())


if __name__ == "__main__":
    unittest.main()
