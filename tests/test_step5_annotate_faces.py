from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

import annotate_shots
from posetag.cli import annotate as annotate_cli


OBJECT = "connection_plate_white"
SIDE = "A"
FACE_KEY = f"{OBJECT}_side{SIDE}"


def _isolated_env(base: Path, project_root: Path) -> dict[str, str]:
    return {
        "HOME": str(base / "home"),
        "XDG_CONFIG_HOME": str(base / ".config"),
        "POSETAG_PROJECT": str(project_root),
    }


def _write_board(project_root: Path) -> Path:
    board_path = project_root / "boards" / f"{FACE_KEY}.yaml"
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_text(
        yaml.safe_dump(
            {
                "object": FACE_KEY,
                "family": "tag36h11",
                "tag_size_m": 0.08,
                "origin_id": 52,
                "tags": [
                    {"id": 52, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
                    {"id": 53, "cx": 0.12, "cy": 0.0, "yaw_deg": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return board_path


def _write_keypoints(project_root: Path) -> Path:
    keypoints_path = project_root / "objects" / OBJECT / "keypoints.json"
    keypoints_path.parent.mkdir(parents=True, exist_ok=True)
    keypoints_path.write_text(
        json.dumps(
            {
                "object": OBJECT,
                "units_to_m": 1.0,
                "points": {
                    "p0": [0.0, 0.0, 0.0],
                    "p1": [0.1, 0.0, 0.0],
                    "p2": [0.1, 0.1, 0.0],
                    "p3": [0.0, 0.1, 0.0],
                },
                "faces": {FACE_KEY: ["p0", "p1", "p2", "p3"]},
            }
        ),
        encoding="utf-8",
    )
    return keypoints_path


def _write_calibration(project_root: Path) -> Path:
    calib_path = project_root / "calib" / "calib_color.yaml"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(
        yaml.safe_dump(
            {
                "camera_matrix": {"fx": 600.0, "fy": 610.0, "cx": 320.0, "cy": 240.0},
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
    return calib_path


def _write_shot_fixture(project_root: Path) -> tuple[Path, Path, Path, Path]:
    board_path = _write_board(project_root)
    _write_keypoints(project_root)
    raw_path = project_root / "shots" / OBJECT / "sideA" / f"{FACE_KEY}_20260606_120000_raw.png"
    ann_path = raw_path.with_name(raw_path.name.replace("_raw.png", "_ann.png"))
    meta_path = raw_path.with_name(raw_path.name.replace("_raw.png", "_meta.json"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"synthetic raw placeholder")
    ann_path.write_bytes(b"synthetic annotation placeholder")
    meta = {
        "object_base": OBJECT,
        "object_full": FACE_KEY,
        "side": SIDE,
        "face_yaml": str(board_path),
        "expected_tag_ids": [52, 53],
        "detected_tag_ids": [52],
        "validation_ok": True,
        "auto_face": True,
        "image": {
            "path_raw": str(raw_path),
            "path_ann": str(ann_path),
            "path_meta": str(meta_path),
            "width": 640,
            "height": 480,
        },
        "camera": {"fx": 600.0, "fy": 610.0, "cx": 320.0, "cy": 240.0},
        "timestamp": "20260606_120000",
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    manifest_path = project_root / "shots" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "object_base",
                "object_full",
                "side",
                "face_yaml",
                "path_raw",
                "path_ann",
                "path_meta",
                "width",
                "height",
                "fx",
                "fy",
                "cx",
                "cy",
                "detected_ids",
                "expected_ids",
                "validation_ok",
                "auto_face",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "20260606_120000",
                "object_base": OBJECT,
                "object_full": FACE_KEY,
                "side": SIDE,
                "face_yaml": str(board_path),
                "path_raw": str(raw_path),
                "path_ann": str(ann_path),
                "path_meta": str(meta_path),
                "width": "640",
                "height": "480",
                "fx": "600.0",
                "fy": "610.0",
                "cx": "320.0",
                "cy": "240.0",
                "detected_ids": "52",
                "expected_ids": "52 53",
                "validation_ok": "True",
                "auto_face": "True",
            }
        )
    return manifest_path, raw_path, meta_path, board_path


def _manifest_row(manifest_path: Path) -> annotate_shots.ShotRow:
    return annotate_shots._read_manifest(manifest_path)[0]


class AnnotateFacesStep5Tests(unittest.TestCase):
    def test_cli_help_resolves(self) -> None:
        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                annotate_cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Annotate PoseTag faces", stdout.getvalue())
        self.assertIn("--batch", stdout.getvalue())

        result = subprocess.run(
            [sys.executable, "-m", "posetag.cli.annotate", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Annotate PoseTag faces", result.stdout)

    def test_output_record_schema_and_transform_composition(self) -> None:
        T_cam_board = np.eye(4)
        T_cam_board[:3, 3] = [0.25, -0.1, 1.5]
        expected_T_board_object = np.eye(4)
        expected_T_board_object[:3, 3] = [0.02, 0.03, 0.04]
        T_cam_object = T_cam_board @ expected_T_board_object

        record = annotate_shots.build_annotation_record(
            object_name=OBJECT,
            face_key=FACE_KEY,
            board_yaml_path="boards/connection_plate_white_sideA.yaml",
            shot_raw_path="shots/connection_plate_white/sideA/example_raw.png",
            rms_px=0.42,
            T_cam_board=T_cam_board,
            T_cam_object=T_cam_object,
            corner_mapping={
                "p0": (10.0, 20.0),
                "p1": (30.0, 20.0),
                "p2": (30.0, 40.0),
                "p3": (10.0, 40.0),
            },
            clicked_uv=[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
            face_corner_names=["p0", "p1", "p2", "p3"],
            pnp_fit={"rvec": np.zeros((3, 1)), "tvec": np.array([[0.0], [0.0], [1.0]])},
            tag_size_m=0.08,
            tag_scale_ratio=1.0,
            tag_scale_pairs=1,
            tag_scale_auto_corrected=False,
        )

        actual_T_board_object = np.array(record["T_board_object"]["matrix"], dtype=float)
        self.assertTrue(np.allclose(actual_T_board_object, expected_T_board_object))
        self.assertTrue(np.allclose(T_cam_board @ actual_T_board_object, T_cam_object))
        self.assertEqual(record["object"], OBJECT)
        self.assertEqual(record["face_key"], FACE_KEY)
        self.assertEqual(record["diagnostics"]["board_tag_size_mm"], 80.0)
        self.assertEqual(record["face_corner_names"], ["p0", "p1", "p2", "p3"])

    def test_annotation_browser_panels_render_queue_and_drag_guidance(self) -> None:
        items = [
            {
                "object": "column_white",
                "side": side,
                "face_key": f"column_white_{side}",
                "raw": Path(f"{side}_raw.png"),
                "board": f"column_white_{side}.yaml",
                "ann_ok": side == "front",
                "kp_ok": True,
                "board_ok": True,
                "rms": 0.35 if side == "front" else None,
            }
            for side in ("front", "back", "sideA", "sideB")
        ]
        panel = annotate_shots._render_list_faces(
            360,
            460,
            "Faces (latest per manifest)",
            items,
            1,
        )
        self.assertEqual(panel.shape, (360, 460, 3))
        self.assertGreater(np.count_nonzero(panel != panel[0, 0]), 1000)

        right = annotate_shots._right_column(
            Path("/tmp/project"),
            items[1],
            SimpleNamespace(
                pts_type="quad",
                check_tag_scale=False,
                auto_correct_scale=False,
                force=False,
            ),
            height=360,
            w=420,
        )
        self.assertEqual(right.shape, (360, 420, 3))
        self.assertGreater(np.count_nonzero(right != right[0, 0]), 1000)

        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        annotate_shots._draw_annotation_banner(
            canvas,
            ["Drag a rectangle around this face."],
        )
        self.assertGreater(int(canvas.sum()), 0)

    def test_face_manifest_sync_refreshes_from_yaml_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            side_a = project_root / "faces" / OBJECT / "sideA" / f"{FACE_KEY}_T_board_object.yaml"
            side_b = project_root / "faces" / OBJECT / "sideB" / f"{OBJECT}_sideB_T_board_object.yaml"
            for path, side in ((side_a, "A"), (side_b, "B")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    yaml.safe_dump(
                        {
                            "object": OBJECT,
                            "face_key": f"{OBJECT}_side{side}",
                            "board_yaml": f"boards/{OBJECT}_side{side}.yaml",
                            "image": f"shots/{OBJECT}/side{side}/example_raw.png",
                            "rms_px": 0.25 if side == "A" else 0.5,
                            "T_board_object": {"matrix": np.eye(4).tolist()},
                            "diagnostics": {"board_tag_size_mm": 80.0},
                        }
                    ),
                    encoding="utf-8",
                )

            annotate_shots._sync_face_manifest(project_root)
            with (project_root / "faces" / "face_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["object"], OBJECT)
            self.assertEqual(rows[0]["tag_size_m"], "0.080000")

            side_b.unlink()
            annotate_shots._sync_face_manifest(project_root)
            with (project_root / "faces" / "face_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["side"], "A")
            self.assertTrue(rows[0]["yaml_path"].endswith(f"{FACE_KEY}_T_board_object.yaml"))

    def test_preflight_and_batch_dry_run_validate_inputs_without_ui(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            manifest_path, raw_path, meta_path, board_path = _write_shot_fixture(project_root)
            calib_path = _write_calibration(project_root)
            row = _manifest_row(manifest_path)

            preflight = annotate_shots.preflight_annotation_shot(
                project_root,
                raw_path,
                row=row,
                calib="calib_color.yaml",
            )

            self.assertEqual(preflight.meta_path, meta_path)
            self.assertEqual(preflight.board_yaml_path, board_path)
            self.assertEqual(
                preflight.keypoints_path,
                (project_root / "objects" / OBJECT / "keypoints.json").resolve(),
            )
            self.assertEqual(preflight.calib_path, calib_path.resolve())
            self.assertTrue(np.allclose(preflight.K, [[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]]))

            with patch.dict(os.environ, _isolated_env(base, project_root), clear=False):
                annotate_cli.main(["--batch", "latest", "--manifest", str(manifest_path), "--dry-run"])

            self.assertFalse(
                (project_root / "faces" / OBJECT / "sideA" / f"{FACE_KEY}_T_board_object.yaml").exists()
            )

    def test_face_suffixed_shot_uses_base_object_keypoints(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            object_name = "column_white"
            face_key = "column_white_front"
            board_path = project_root / "boards" / f"{face_key}.yaml"
            board_path.parent.mkdir(parents=True, exist_ok=True)
            board_path.write_text(
                yaml.safe_dump(
                    {
                        "object": face_key,
                        "family": "tag36h11",
                        "tag_size_m": 0.08,
                        "origin_id": 10,
                        "tags": [
                            {"id": 10, "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
                            {"id": 11, "cx": 0.12, "cy": 0.0, "yaw_deg": 0.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            keypoints_path = project_root / "objects" / object_name / "keypoints.json"
            keypoints_path.parent.mkdir(parents=True, exist_ok=True)
            keypoints_path.write_text(
                json.dumps(
                    {
                        "object": object_name,
                        "units_to_m": 1.0,
                        "points": {
                            "p0": [0.0, 0.0, 0.0],
                            "p1": [0.1, 0.0, 0.0],
                            "p2": [0.1, 0.1, 0.0],
                            "p3": [0.0, 0.1, 0.0],
                        },
                        "faces": {face_key: ["p0", "p1", "p2", "p3"]},
                    }
                ),
                encoding="utf-8",
            )
            raw_path = project_root / "shots" / face_key / "UNRESOLVED" / f"{face_key}_20260606_121500_raw.png"
            ann_path = raw_path.with_name(raw_path.name.replace("_raw.png", "_ann.png"))
            meta_path = raw_path.with_name(raw_path.name.replace("_raw.png", "_meta.json"))
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"synthetic raw placeholder")
            ann_path.write_bytes(b"synthetic annotation placeholder")
            meta_path.write_text(
                json.dumps(
                    {
                        "object_base": face_key,
                        "object_full": face_key,
                        "side": "UNRESOLVED",
                        "face_yaml": str(board_path),
                        "image": {
                            "path_raw": str(raw_path),
                            "path_ann": str(ann_path),
                            "path_meta": str(meta_path),
                            "width": 640,
                            "height": 480,
                        },
                        "camera": {"fx": 600.0, "fy": 610.0, "cx": 320.0, "cy": 240.0},
                        "timestamp": "20260606_121500",
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = project_root / "shots" / "manifest.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "object_base",
                        "object_full",
                        "side",
                        "face_yaml",
                        "path_raw",
                        "path_ann",
                        "path_meta",
                        "width",
                        "height",
                        "fx",
                        "fy",
                        "cx",
                        "cy",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "20260606_121500",
                        "object_base": face_key,
                        "object_full": face_key,
                        "side": "UNRESOLVED",
                        "face_yaml": str(board_path),
                        "path_raw": str(raw_path),
                        "path_ann": str(ann_path),
                        "path_meta": str(meta_path),
                        "width": "640",
                        "height": "480",
                        "fx": "600.0",
                        "fy": "610.0",
                        "cx": "320.0",
                        "cy": "240.0",
                    }
                )

            row = _manifest_row(manifest_path)
            preflight = annotate_shots.preflight_annotation_shot(project_root, raw_path, row=row)

            self.assertEqual(preflight.object_base, object_name)
            self.assertEqual(preflight.side, "front")
            self.assertEqual(preflight.face_key, face_key)
            self.assertEqual(preflight.keypoints_path, keypoints_path.resolve())
            self.assertEqual(
                annotate_shots._out_yaml_path(
                    project_root,
                    preflight.object_base,
                    preflight.side,
                    face_key=preflight.face_key,
                ),
                project_root / "faces" / object_name / "front" / f"{face_key}_T_board_object.yaml",
            )

            with patch.dict(os.environ, _isolated_env(base, project_root), clear=False):
                annotate_cli.main(["--batch", "latest", "--manifest", str(manifest_path), "--dry-run"])

    def test_preflight_missing_inputs_fail_clearly(self) -> None:
        cases = (
            ("missing shot", lambda raw, meta, board, root: raw.unlink(), "Shot image not found"),
            (
                "missing annotated shot",
                lambda raw, meta, board, root: raw.with_name(raw.name.replace("_raw.png", "_ann.png")).unlink(),
                "Captured annotated shot image not found",
            ),
            ("missing metadata", lambda raw, meta, board, root: meta.unlink(), "Shot metadata JSON not found"),
            ("missing board", lambda raw, meta, board, root: board.unlink(), "Board YAML not found"),
            (
                "malformed board",
                lambda raw, meta, board, root: board.write_text("[]\n", encoding="utf-8"),
                "Malformed board YAML",
            ),
            (
                "missing keypoints",
                lambda raw, meta, board, root: (root / "objects" / OBJECT / "keypoints.json").unlink(),
                "Keypoints JSON not found",
            ),
            (
                "malformed keypoints",
                lambda raw, meta, board, root: (root / "objects" / OBJECT / "keypoints.json").write_text(
                    json.dumps({"points": {"p0": [0, 0, 0]}, "faces": {}}),
                    encoding="utf-8",
                ),
                "Malformed keypoints JSON",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                with TemporaryDirectory() as tmpdir:
                    project_root = Path(tmpdir) / "project"
                    manifest_path, raw_path, meta_path, board_path = _write_shot_fixture(project_root)
                    mutate(raw_path, meta_path, board_path, project_root)
                    with self.assertRaises(SystemExit) as ctx:
                        annotate_shots.preflight_annotation_shot(
                            project_root,
                            raw_path,
                            row=_manifest_row(manifest_path),
                        )
                    self.assertIn(expected, str(ctx.exception))

    def test_missing_intrinsics_calibration_manifest_and_mode_fail_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            manifest_path, raw_path, meta_path, _board_path = _write_shot_fixture(project_root)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.pop("camera")
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for key in ("fx", "fy", "cx", "cy"):
                rows[0][key] = ""
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(SystemExit) as ctx:
                annotate_shots.preflight_annotation_shot(
                    project_root,
                    raw_path,
                    row=_manifest_row(manifest_path),
                )
            self.assertIn("Camera intrinsics not found", str(ctx.exception))

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            manifest_path, raw_path, _meta_path, _board_path = _write_shot_fixture(project_root)
            with self.assertRaises(SystemExit) as ctx:
                annotate_shots.preflight_annotation_shot(
                    project_root,
                    raw_path,
                    row=_manifest_row(manifest_path),
                    calib="missing.yaml",
                )
            self.assertIn("Calibration YAML not found", str(ctx.exception))

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            with patch.dict(os.environ, _isolated_env(base, project_root), clear=False):
                with self.assertRaises(SystemExit) as ctx:
                    annotate_cli.main(["--batch", "latest", "--dry-run"])
            self.assertIn("manifest.csv not found", str(ctx.exception))

        stderr = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(stderr):
                annotate_cli.main([])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("choose one mode", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
