from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import gen_keypoints
from posetag.workflows.mesh_keypoints import (
    MeshKeypointWorkflowError,
    default_mesh_path,
    expected_face_keys,
    expected_keypoints_path,
    infer_known_objects,
    inspect_keypoint_object_statuses,
    load_obj_vertex_preview,
    mesh_keypoint_command_for_object,
    resolve_object_identity,
    split_object_face,
    supported_mesh_extensions,
    validate_keypoint_payload,
    validate_mesh_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MeshKeypointWorkflowTests(unittest.TestCase):
    def test_generate_annotation_ready_keypoints_schema_and_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            mesh_path = _write_box_obj(
                project_root / "meshes" / "connection_plate_white.obj"
            )

            result = gen_keypoints.generate_keypoints_for_mesh(
                project_root,
                mesh_path,
                units_to_m=0.001,
            )

            keypoints_path = project_root / "objects" / "connection_plate_white" / "keypoints.json"
            config_path = project_root / "objects" / "connection_plate_white" / "object_config.yaml"
            self.assertEqual(Path(result["keypoints_path"]), keypoints_path)
            self.assertTrue(keypoints_path.is_file())
            self.assertTrue(config_path.is_file())

            payload = json.loads(keypoints_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_keypoint_payload(payload), ())
            self.assertEqual(payload["units_to_m"], 0.001)
            self.assertEqual(len(payload["points"]), 8)
            self.assertEqual(
                sorted(payload["faces"]),
                [
                    "connection_plate_white_sideA",
                    "connection_plate_white_sideB",
                    "connection_plate_white_sideC",
                    "connection_plate_white_sideD",
                ],
            )

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["mesh"]["path"], "meshes/connection_plate_white.obj")
            self.assertEqual(config["mesh"]["units_to_m"], 0.001)
            self.assertEqual(
                config["mesh"]["T_mesh_object"]["matrix"],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )

    def test_known_objects_are_inferred_from_boards_registry_and_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            side_a = _write_board(
                project_root,
                "connection_plate_white_sideA",
                (52, 53),
            )
            side_b = _write_board(
                project_root,
                "connection_plate_white_sideB",
                (62, 63),
            )
            _write_registry(
                project_root,
                {
                    "52": ("connection_plate_white_sideA", side_a),
                    "53": ("connection_plate_white_sideA", side_a),
                    "62": ("connection_plate_white_sideB", side_b),
                    "63": ("connection_plate_white_sideB", side_b),
                },
            )
            _write_manifest(project_root, "connection_plate_white", "A")
            mesh_path = _write_box_obj(
                project_root / "meshes" / "connection_plate_white.obj"
            )

            known = infer_known_objects(project_root)

            self.assertEqual([item.object_name for item in known], ["connection_plate_white"])
            self.assertEqual(known[0].faces, ("sideA", "sideB"))
            self.assertIn("boards/*.yaml", known[0].sources)
            self.assertIn("boards/tag_registry.yaml", known[0].sources)
            self.assertIn("shots/manifest.csv", known[0].sources)

            identity = resolve_object_identity(project_root, mesh_path)
            self.assertEqual(identity.object_name, "connection_plate_white")
            self.assertEqual(
                identity.output_path,
                expected_keypoints_path(project_root, "connection_plate_white"),
            )

    def test_common_face_suffixes_are_grouped_under_base_object(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_board(project_root, "column_white_front", (10, 11))
            _write_board(project_root, "column_white_back", (12, 13))
            _write_board(project_root, "column_white_sideA", (14, 15))
            _write_board(project_root, "column_white_sideB", (16, 17))

            self.assertEqual(
                split_object_face("column_white_front"),
                ("column_white", "front"),
            )
            self.assertEqual(
                split_object_face("column_white_back"),
                ("column_white", "back"),
            )
            self.assertEqual(
                split_object_face("column_white_sideA"),
                ("column_white", "sideA"),
            )
            self.assertEqual(
                split_object_face("connection_plate_white"),
                ("connection_plate_white", None),
            )

            known = infer_known_objects(project_root)

            self.assertEqual([item.object_name for item in known], ["column_white"])
            self.assertEqual(known[0].faces, ("back", "front", "sideA", "sideB"))

            mesh_path = _write_box_obj(project_root / "meshes" / "column_white.obj")
            result = gen_keypoints.generate_keypoints_for_mesh(
                project_root,
                mesh_path,
            )
            payload = json.loads(
                Path(result["keypoints_path"]).read_text(encoding="utf-8")
            )

            self.assertIn("column_white_front", payload["faces"])
            self.assertIn("column_white_back", payload["faces"])
            self.assertEqual(
                validate_keypoint_payload(
                    payload,
                    expected_faces=expected_face_keys("column_white", known[0].faces),
                ),
                (),
            )

    def test_object_statuses_expose_mesh_import_and_command_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_board(project_root, "column_white_front", (10, 11))

            statuses = inspect_keypoint_object_statuses(project_root)

            self.assertEqual(len(statuses), 1)
            self.assertEqual(statuses[0].object_name, "column_white")
            self.assertEqual(
                statuses[0].mesh_path,
                default_mesh_path(project_root, "column_white"),
            )
            self.assertEqual(
                statuses[0].keypoints_path,
                expected_keypoints_path(project_root, "column_white"),
            )
            self.assertFalse(statuses[0].mesh_exists)
            self.assertFalse(statuses[0].keypoints_exists)
            self.assertFalse(statuses[0].keypoints_valid)
            self.assertIn("meshes/column_white.obj", statuses[0].command_preview)
            self.assertIn("--object_name column_white", statuses[0].command_preview)
            self.assertEqual(
                statuses[0].command_preview,
                mesh_keypoint_command_for_object(project_root, "column_white"),
            )

    def test_obj_vertex_preview_reports_bounds_and_vertex_sample(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mesh_path = _write_box_obj(Path(tmpdir) / "part.obj")

            preview = load_obj_vertex_preview(mesh_path, max_vertices=3)

            self.assertEqual(preview.path, mesh_path)
            self.assertEqual(preview.vertex_count, 5)
            self.assertEqual(len(preview.sampled_vertices), 3)
            self.assertEqual(preview.bounds_min, (0.0, 0.0, 0.0))
            self.assertEqual(preview.bounds_max, (2.0, 3.0, 4.0))

    def test_explicit_object_mapping_allows_mesh_filename_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_board(project_root, "connection_plate_white_sideA", (52, 53))
            mesh_path = _write_box_obj(project_root / "meshes" / "unity_export_plate_v12.obj")

            with self.assertRaises(MeshKeypointWorkflowError) as ctx:
                resolve_object_identity(project_root, mesh_path)
            self.assertIn("--object_name", str(ctx.exception))

            identity = resolve_object_identity(
                project_root,
                mesh_path,
                object_name="connection_plate_white",
            )
            self.assertEqual(identity.object_name, "connection_plate_white")
            self.assertEqual(
                identity.output_path,
                project_root / "objects" / "connection_plate_white" / "keypoints.json",
            )

            gen_keypoints.generate_keypoints_for_mesh(
                project_root,
                mesh_path,
                object_name="connection_plate_white",
            )
            statuses = inspect_keypoint_object_statuses(project_root)
            self.assertEqual(len(statuses), 1)
            self.assertEqual(statuses[0].mesh_path, mesh_path)
            self.assertTrue(statuses[0].mesh_exists)
            self.assertTrue(statuses[0].keypoints_valid)
            self.assertIn("unity_export_plate_v12.obj", statuses[0].command_preview)

    def test_status_rejects_keypoints_missing_inferred_face(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            side_a = _write_board(project_root, "connection_plate_white_sideA", (52, 53))
            side_b = _write_board(project_root, "connection_plate_white_sideB", (62, 63))
            _write_registry(
                project_root,
                {
                    "52": ("connection_plate_white_sideA", side_a),
                    "62": ("connection_plate_white_sideB", side_b),
                },
            )
            mesh_path = _write_box_obj(
                project_root / "meshes" / "connection_plate_white.obj"
            )
            result = gen_keypoints.generate_keypoints_for_mesh(
                project_root,
                mesh_path,
            )
            keypoints_path = Path(result["keypoints_path"])
            payload = json.loads(keypoints_path.read_text(encoding="utf-8"))
            del payload["faces"]["connection_plate_white_sideB"]
            keypoints_path.write_text(json.dumps(payload), encoding="utf-8")

            statuses = inspect_keypoint_object_statuses(project_root)

            self.assertEqual(len(statuses), 1)
            self.assertFalse(statuses[0].keypoints_valid)
            self.assertTrue(
                any(
                    "connection_plate_white_sideB" in error
                    for error in statuses[0].keypoints_errors
                )
            )

    def test_non_finite_geometry_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            mesh_path = project_root / "meshes" / "bad.obj"
            mesh_path.parent.mkdir(parents=True)
            mesh_path.write_text(
                "\n".join(["v 0 0 0", "v nan 1 1", "v 1 1 1"]) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(MeshKeypointWorkflowError) as ctx:
                gen_keypoints.generate_keypoints_for_mesh(project_root, mesh_path)

            self.assertIn("non-finite", str(ctx.exception))

    def test_non_finite_existing_keypoint_payload_is_invalid(self) -> None:
        payload = {
            "units_to_m": float("inf"),
            "points": {
                "a": [0.0, 0.0, 0.0],
                "b": [1.0, 0.0, 0.0],
                "c": [1.0, float("nan"), 0.0],
                "d": [0.0, 1.0, 0.0],
            },
            "faces": {
                "part_sideA": ["a", "b", "c", "d"],
            },
        }

        errors = validate_keypoint_payload(
            payload,
            expected_faces=("part_sideA",),
        )

        self.assertTrue(any("units_to_m" in error for error in errors))
        self.assertTrue(
            any("finite numeric coordinates" in error for error in errors)
        )

    def test_mesh_extension_and_missing_mesh_validation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            mesh_dir = project_root / "meshes"
            mesh_dir.mkdir(parents=True)
            unsupported = mesh_dir / "part.ply"
            unsupported.write_text("ply\n", encoding="utf-8")

            self.assertEqual(supported_mesh_extensions(), (".obj",))
            with self.assertRaises(MeshKeypointWorkflowError) as ctx:
                validate_mesh_path(project_root, unsupported)
            self.assertIn("Unsupported mesh extension", str(ctx.exception))

            with self.assertRaises(MeshKeypointWorkflowError) as ctx:
                validate_mesh_path(project_root, "meshes/missing.obj")
            self.assertIn("was not found", str(ctx.exception))

    def test_existing_output_requires_explicit_policy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            mesh_path = _write_box_obj(project_root / "meshes" / "block.obj")

            first = gen_keypoints.generate_keypoints_for_mesh(project_root, mesh_path)
            self.assertTrue(Path(first["keypoints_path"]).is_file())

            with self.assertRaises(MeshKeypointWorkflowError) as ctx:
                gen_keypoints.generate_keypoints_for_mesh(project_root, mesh_path)
            self.assertIn("--force", str(ctx.exception))

            second = gen_keypoints.generate_keypoints_for_mesh(
                project_root,
                mesh_path,
                keep_both=True,
            )
            self.assertEqual(
                Path(second["keypoints_path"]).name,
                "keypoints_v2.json",
            )

            forced = gen_keypoints.generate_keypoints_for_mesh(
                project_root,
                mesh_path,
                force=True,
                units_to_m=2.0,
            )
            self.assertEqual(Path(forced["keypoints_path"]).name, "keypoints.json")
            payload = json.loads(Path(forced["keypoints_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["units_to_m"], 2.0)

    def test_command_help_works_without_opening_browser(self) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        pythonpath = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        if existing:
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        completed = subprocess.run(
            [sys.executable, "-m", "gen_keypoints", "--help"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("objects/<object>/keypoints.json", completed.stdout)
        self.assertIn("--object_name", completed.stdout)


def _write_box_obj(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 2 0 0",
                "v 0 3 0",
                "v 0 0 4",
                "v 2 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_board(project_root: Path, object_name: str, tag_ids: tuple[int, int]) -> Path:
    path = project_root / "boards" / f"{object_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "object": object_name,
                "family": "tag36h11",
                "tag_size_m": 0.08,
                "origin_id": tag_ids[0],
                "tags": [
                    {"id": tag_ids[0], "cx": 0.0, "cy": 0.0, "yaw_deg": 0.0},
                    {"id": tag_ids[1], "cx": 0.1, "cy": 0.0, "yaw_deg": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_registry(
    project_root: Path,
    rows: dict[str, tuple[str, Path]],
) -> Path:
    registry_path = project_root / "boards" / "tag_registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "updated": "2026-06-04T12:00:00Z",
                "tags": {
                    tag_id: {"object": object_name, "yaml": str(board_path)}
                    for tag_id, (object_name, board_path) in rows.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path


def _write_manifest(project_root: Path, object_base: str, side: str) -> Path:
    path = project_root / "shots" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object_base", "object_full", "side", "validation_ok"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "object_base": object_base,
                "object_full": f"{object_base}_side{side}",
                "side": side,
                "validation_ok": "1",
            }
        )
    return path


if __name__ == "__main__":
    unittest.main()
