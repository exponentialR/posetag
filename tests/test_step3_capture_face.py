from __future__ import annotations

import csv
import json
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

import capture_face as legacy_capture_face
from posetag.cli import capture_face as capture_face_cli
from posetag.pipelines.capture_face import (
    append_capture_manifest,
    build_capture_metadata,
    build_shot_paths,
    load_capture_registry,
    load_registered_faces,
    prepare_capture_paths,
    resolve_capture_calibration_path,
    select_initial_faces,
    validate_capture_metadata_schema,
)
from posetag.pipelines.make_board import build_board_yaml, write_board_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_env(base: Path) -> dict[str, str]:
    return {
        "HOME": str(base / "home"),
        "XDG_CONFIG_HOME": str(base / ".config"),
    }


def _write_calibration(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
            }
        ),
        encoding="utf-8",
    )


def _write_board_and_registry(project_root: Path) -> tuple[Path, Path]:
    board_path = project_root / "boards" / "connection_plate_white_sideA.yaml"
    registry_path = project_root / "boards" / "tag_registry.yaml"
    board = build_board_yaml(
        object_name="connection_plate_white_sideA",
        family="tag36h11",
        tag_size_mm=80.0,
        origin_id=52,
        entries=[
            {"id": 52, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
            {"id": 53, "cx": 0.1, "cy": -0.2, "yaw_deg": 180.0},
        ],
    )
    write_board_yaml(board_path, board)
    registry = {
        "version": 1,
        "updated": "2026-05-29T12:00:00Z",
        "tags": {
            "52": {"object": "connection_plate_white_sideA", "yaml": str(board_path)},
            "53": {"object": "connection_plate_white_sideA", "yaml": str(board_path)},
        },
    }
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return board_path, registry_path


class FakeDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect(self, *args, **kwargs):
        return []


class OneTagDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect(self, *args, **kwargs):
        return [FakeDetection(52)]


class FakeDetection:
    def __init__(self, tag_id: int) -> None:
        self.tag_id = tag_id
        self.corners = np.array(
            [[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]],
            dtype=float,
        )


class CaptureFaceStep3Tests(unittest.TestCase):
    def test_cli_help_resolves(self) -> None:
        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                capture_face_cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Capture wide shots per face", stdout.getvalue())
        self.assertIn("--source", stdout.getvalue())

    def test_console_script_target_help_resolves_after_install(self) -> None:
        code = (
            "from posetag.cli.capture_face import main\n"
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
        self.assertIn("Capture wide shots per face", result.stdout)

    def test_missing_video_for_video_source_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "video",
                        "--object_name",
                        "connection_plate_white",
                    ]
                )

            self.assertIn("--video path is required", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_unreadable_video_path_fails_before_capture_artifacts(self) -> None:
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
            _write_board_and_registry(project_root)

            with patch.dict(os.environ, _isolated_env(base), clear=False):
                with patch("capture_face.load_apriltag_detector_class", return_value=FakeDetector):
                    with patch.object(
                        legacy_capture_face.capture_source.cv2,
                        "VideoCapture",
                        FakeClosedVideoCapture,
                    ):
                        with self.assertRaises(SystemExit) as ctx:
                            capture_face_cli.main(
                                [
                                    "--project_root",
                                    str(project_root),
                                    "--source",
                                    "video",
                                    "--video",
                                    str(video),
                                    "--object_name",
                                    "connection_plate_white",
                                ]
                            )

            self.assertIn("Could not open video", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())
            self.assertFalse((project_root / "logs").exists())

    def test_realsense_source_without_dependency_fails_clearly(self) -> None:
        with patch.object(legacy_capture_face.capture_source, "rs", None):
            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(["--source", "realsense"])

        self.assertIn("pyrealsense2 is not available", str(ctx.exception))

    def test_missing_calibration_yaml_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "opencv",
                        "--object_name",
                        "connection_plate_white",
                    ]
                )

            self.assertIn("Calibration YAML not found", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_malformed_calibration_yaml_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib = project_root / "calib" / "calib_color.yaml"
            calib.parent.mkdir(parents=True)
            calib.write_text("not_camera_matrix: true\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "opencv",
                        "--object_name",
                        "connection_plate_white",
                    ]
                )

            self.assertIn("Malformed calibration YAML", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_missing_registry_yaml_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_calibration(project_root / "calib" / "calib_color.yaml")

            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "opencv",
                        "--object_name",
                        "connection_plate_white",
                    ]
                )

            self.assertIn("Tag registry YAML not found", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_malformed_registry_yaml_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            registry = project_root / "boards" / "tag_registry.yaml"
            registry.parent.mkdir(parents=True)
            registry.write_text("tags: []\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "opencv",
                        "--object_name",
                        "connection_plate_white",
                    ]
                )

            self.assertIn("Malformed tag registry YAML", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_missing_board_yaml_reference_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            registry = project_root / "boards" / "tag_registry.yaml"
            registry.parent.mkdir(parents=True)
            registry.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "tags": {
                            "52": {
                                "object": "connection_plate_white_sideA",
                                "yaml": str(project_root / "boards" / "missing.yaml"),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "opencv",
                        "--object_name",
                        "connection_plate_white",
                    ]
                )

            self.assertIn("Referenced board YAML", str(ctx.exception))
            self.assertIn("was not found", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_unknown_object_selection_fails_before_opening_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            _write_board_and_registry(project_root)

            with self.assertRaises(SystemExit) as ctx:
                capture_face_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--source",
                        "opencv",
                        "--object_name",
                        "unknown_object",
                    ]
                )

            self.assertIn("No registered face found", str(ctx.exception))
            self.assertFalse((project_root / "shots").exists())

    def test_registry_and_selection_helpers_resolve_faces(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            board_path, registry_path = _write_board_and_registry(project_root)

            registry = load_capture_registry(registry_path)
            faces = load_registered_faces(
                registry,
                project_root=project_root,
                registry_path=registry_path,
            )
            selection = select_initial_faces("connection_plate_white", faces)

            self.assertIsNotNone(selection)
            self.assertEqual(selection.object_base, "connection_plate_white")
            self.assertTrue(selection.auto_side)
            self.assertEqual(selection.faces[0]["yaml"], str(board_path))
            self.assertEqual(selection.faces[0]["tag_ids"], (52, 53))

    def test_path_metadata_and_manifest_helpers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib = project_root / "calib" / "calib_color.yaml"
            _write_calibration(calib)
            resolved_calib = resolve_capture_calibration_path(project_root, "calib_color.yaml")
            self.assertEqual(resolved_calib.resolve(), calib.resolve())

            paths = prepare_capture_paths(
                project_root=project_root,
                calib_path=calib,
                layout="by_object_side",
            )
            shot_paths = build_shot_paths(
                layout="by_object_side",
                out_dir=paths.out_dir,
                object_full="connection_plate_white_sideA",
                timestamp="20260529_120000",
            )
            resolved_project = project_root.resolve()
            self.assertEqual(
                shot_paths.raw_path,
                resolved_project
                / "shots"
                / "connection_plate_white"
                / "sideA"
                / "connection_plate_white_sideA_20260529_120000_raw.png",
            )

            metadata = build_capture_metadata(
                face={
                    "yaml": str(project_root / "boards" / "connection_plate_white_sideA.yaml"),
                    "object": "connection_plate_white_sideA",
                    "tag_ids": (52, 53),
                },
                shot_paths=shot_paths,
                detected_tag_ids={52},
                validation_ok=True,
                auto_face=True,
                frame_shape=(480, 640, 3),
                camera_params=(600.0, 610.0, 320.0, 240.0),
                timestamp="20260529_120000",
            )

            self.assertEqual(validate_capture_metadata_schema(metadata), ())
            append_capture_manifest(paths.manifest_path, metadata)

            with paths.manifest_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["object_full"], "connection_plate_white_sideA")
            self.assertEqual(rows[0]["detected_ids"], "52")
            self.assertEqual(rows[0]["expected_ids"], "52 53")

    def test_video_eof_exits_cleanly_without_capture(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            video = base / "input.mp4"
            video.touch()
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            _write_board_and_registry(project_root)

            stopped = {"called": False}

            def fake_open_source(*args, **kwargs):
                def read_frame():
                    return None

                def stop():
                    stopped["called"] = True

                return read_frame, stop, {"kind": "video"}

            with patch.dict(os.environ, _isolated_env(base), clear=False):
                with patch("capture_face.load_apriltag_detector_class", return_value=FakeDetector):
                    with patch.object(legacy_capture_face.capture_source, "open_source", fake_open_source):
                        with patch("capture_face.cv2.namedWindow"):
                            with patch("capture_face.cv2.resizeWindow"):
                                with patch("capture_face.cv2.destroyAllWindows"):
                                    result = capture_face_cli.main(
                                        [
                                            "--project_root",
                                            str(project_root),
                                            "--source",
                                            "video",
                                            "--video",
                                            str(video),
                                            "--object_name",
                                            "connection_plate_white",
                                        ]
                                    )

            self.assertEqual(result, 0)
            self.assertTrue(stopped["called"])
            self.assertFalse((project_root / "shots" / "manifest.csv").exists())

    def test_escape_exits_cleanly_without_capture(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            video = base / "input.mp4"
            video.touch()
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            _write_board_and_registry(project_root)
            frame = np.zeros((40, 60, 3), dtype=np.uint8)

            def fake_open_source(*args, **kwargs):
                return lambda: frame.copy(), lambda: None, {"kind": "video"}

            with patch.dict(os.environ, _isolated_env(base), clear=False):
                with patch("capture_face.load_apriltag_detector_class", return_value=FakeDetector):
                    with patch.object(legacy_capture_face.capture_source, "open_source", fake_open_source):
                        with patch("capture_face.cv2.namedWindow"):
                            with patch("capture_face.cv2.resizeWindow"):
                                with patch("capture_face.cv2.imshow"):
                                    with patch("capture_face.cv2.waitKeyEx", return_value=27):
                                        with patch("capture_face.cv2.destroyAllWindows"):
                                            result = capture_face_cli.main(
                                                [
                                                    "--project_root",
                                                    str(project_root),
                                                    "--source",
                                                    "video",
                                                    "--video",
                                                    str(video),
                                                    "--object_name",
                                                    "connection_plate_white",
                                                ]
                                            )

            self.assertEqual(result, 0)
            self.assertFalse((project_root / "shots" / "manifest.csv").exists())

    def test_synthetic_enter_save_writes_images_metadata_and_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            video = base / "input.mp4"
            video.touch()
            _write_calibration(project_root / "calib" / "calib_color.yaml")
            _write_board_and_registry(project_root)
            frame = np.full((40, 60, 3), 80, dtype=np.uint8)
            frames = [frame.copy(), None]

            def fake_open_source(*args, **kwargs):
                def read_frame():
                    return frames.pop(0)

                return read_frame, lambda: None, {"kind": "video"}

            with patch.dict(os.environ, _isolated_env(base), clear=False):
                with patch("capture_face.load_apriltag_detector_class", return_value=OneTagDetector):
                    with patch.object(legacy_capture_face.capture_source, "open_source", fake_open_source):
                        with patch("capture_face.capture_timestamp", return_value="20260529_120000"):
                            with patch("capture_face.cv2.namedWindow"):
                                with patch("capture_face.cv2.resizeWindow"):
                                    with patch("capture_face.cv2.imshow"):
                                        with patch("capture_face.cv2.waitKeyEx", return_value=13):
                                            with patch("capture_face.cv2.destroyAllWindows"):
                                                result = capture_face_cli.main(
                                                    [
                                                        "--project_root",
                                                        str(project_root),
                                                        "--source",
                                                        "video",
                                                        "--video",
                                                        str(video),
                                                        "--object_name",
                                                        "connection_plate_white",
                                                    ]
                                                )

            self.assertEqual(result, 0)
            resolved_project = project_root.resolve()
            shot_dir = resolved_project / "shots" / "connection_plate_white" / "sideA"
            raw_path = shot_dir / "connection_plate_white_sideA_20260529_120000_raw.png"
            ann_path = shot_dir / "connection_plate_white_sideA_20260529_120000_ann.png"
            meta_path = shot_dir / "connection_plate_white_sideA_20260529_120000_meta.json"
            manifest_path = resolved_project / "shots" / "manifest.csv"

            self.assertTrue(raw_path.is_file())
            self.assertTrue(ann_path.is_file())
            self.assertTrue(meta_path.is_file())
            self.assertTrue(manifest_path.is_file())

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_capture_metadata_schema(metadata), ())
            self.assertEqual(metadata["object_full"], "connection_plate_white_sideA")
            self.assertEqual(metadata["expected_tag_ids"], [52, 53])
            self.assertEqual(metadata["detected_tag_ids"], [52])
            self.assertTrue(metadata["validation_ok"])

            with manifest_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["path_meta"], str(meta_path))


if __name__ == "__main__":
    unittest.main()
