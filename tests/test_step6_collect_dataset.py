from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from posetag.cli import collect as collect_cli
from posetag.pipelines.collect_dataset import (
    AutoCaptureConfig,
    AutoCaptureState,
    CollectDatasetError,
    Intrinsics,
    auto_capture_candidates_from_results,
    build_frame_annotation_record,
    build_session_metadata,
    compose_T_cam_object,
    evaluate_auto_capture,
    load_intrinsics_from_calibration,
    parse_auto_capture_grid,
    pose_delta,
    record_auto_capture_save,
    resolve_calibration_path,
    resolve_dataset_root,
    resolve_face_manifest_path,
    resolve_registry_path,
    validate_collection_inputs,
    validate_source_args,
)
from collect_gt_dataset import save_frame
from utils.collect_gt_utils import (
    _capture_key_requests_quit,
    _capture_status_panel,
    _dashboard_strip,
    _normalize_capture_key,
)


OBJECT = "connection_plate_white"
SIDE = "sideA"
FACE_KEY = f"{OBJECT}_{SIDE}"


def _write_calibration(project_root: Path) -> Path:
    calib_path = project_root / "calib" / "calib_color.yaml"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(
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
    return calib_path


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


def _relative_to_project(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _write_registry(project_root: Path, board_path: Path) -> Path:
    registry_path = project_root / "boards" / "tag_registry.yaml"
    board_rel = _relative_to_project(project_root, board_path)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "updated": "2026-06-07T12:00:00Z",
                "tags": {
                    "52": {"object": FACE_KEY, "yaml": board_rel},
                    "53": {"object": FACE_KEY, "yaml": board_rel},
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path


def _write_annotation(project_root: Path, board_path: Path, *, include_transform: bool = True) -> Path:
    annotation_path = project_root / "faces" / OBJECT / SIDE / f"{FACE_KEY}_T_board_object.yaml"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "object": OBJECT,
        "face_key": FACE_KEY,
        "board_yaml": _relative_to_project(project_root, board_path),
        "shot_raw": "shots/example_raw.png",
        "rms_px": 0.42,
    }
    if include_transform:
        T_board_object = np.eye(4)
        T_board_object[:3, 3] = [0.02, 0.03, 0.04]
        record["T_board_object"] = {"matrix": T_board_object.tolist()}
    annotation_path.write_text(yaml.safe_dump(record), encoding="utf-8")
    return annotation_path


def _write_face_manifest(project_root: Path, annotation_path: Path, board_path: Path) -> Path:
    manifest_path = project_root / "faces" / "face_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "object",
                "side",
                "face_key",
                "yaml_path",
                "board_yaml",
                "image",
                "rms_px",
                "tag_size_m",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "20260607_120000",
                "object": OBJECT,
                "side": SIDE,
                "face_key": FACE_KEY,
                "yaml_path": _relative_to_project(project_root, annotation_path),
                "board_yaml": _relative_to_project(project_root, board_path),
                "image": "shots/example_raw.png",
                "rms_px": "0.42",
                "tag_size_m": "0.08",
            }
        )
    return manifest_path


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


def _write_project_fixture(project_root: Path, *, keypoints: bool = False) -> tuple[Path, Path, Path, Path]:
    calib_path = _write_calibration(project_root)
    board_path = _write_board(project_root)
    registry_path = _write_registry(project_root, board_path)
    annotation_path = _write_annotation(project_root, board_path)
    manifest_path = _write_face_manifest(project_root, annotation_path, board_path)
    if keypoints:
        _write_keypoints(project_root)
    return calib_path, registry_path, manifest_path, annotation_path


def _intrinsics() -> Intrinsics:
    return Intrinsics(
        path=Path("calib/calib_color.yaml"),
        fx=600.0,
        fy=610.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        dist=np.zeros((1, 5), dtype=float),
    )


def _pose_result() -> dict[str, object]:
    T_cam_board = np.eye(4)
    T_cam_board[:3, 3] = [0.25, -0.1, 1.5]
    T_board_object = np.eye(4)
    T_board_object[:3, 3] = [0.02, 0.03, 0.04]
    T_cam_object = compose_T_cam_object(T_cam_board, T_board_object)
    return {
        "object": OBJECT,
        "face_key": FACE_KEY,
        "board_yaml": f"boards/{FACE_KEY}.yaml",
        "tag_used": 52,
        "num_tags_visible": 2,
        "score": 2500.0,
        "score_tags": 2,
        "score_area_px": 500,
        "bbox_xywh": [64.0, 48.0, 128.0, 96.0],
        "bbox_source": "face_kps",
        "T_cam_board": {"matrix": T_cam_board.tolist()},
        "T_board_object": {"matrix": T_board_object.tolist()},
        "T_cam_object": {"matrix": T_cam_object.tolist()},
        "diagnostics": {
            "tag_scale_ratio": 1.0,
            "tag_scale_pairs": 1,
            "tag_scale_mad": 0.0,
            "tag_scale_auto_corrected": False,
        },
    }


class ClosedCapture:
    def __init__(self, _path: str) -> None:
        pass

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


class CollectDatasetStep6Tests(unittest.TestCase):
    def test_cli_help_resolves(self) -> None:
        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                collect_cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Collect multi-object/multi-face GT dataset", stdout.getvalue())
        self.assertIn("--dry-run", stdout.getvalue())
        self.assertIn("opencv", stdout.getvalue())

        result = subprocess.run(
            [sys.executable, "-m", "posetag.cli.collect", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("posetag-collect", result.stdout)

    def test_input_preflight_validates_required_collection_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path, registry_path, manifest_path, _annotation_path = _write_project_fixture(project_root)

            summary = validate_collection_inputs(
                project_root=project_root,
                calib_path=calib_path,
                registry_path=registry_path,
                face_manifest_path=manifest_path,
            )

        self.assertEqual(summary.calibration_path, calib_path)
        self.assertEqual(summary.registry_path, registry_path)
        self.assertEqual(summary.face_manifest_path, manifest_path)
        self.assertEqual(len(summary.annotations), 1)
        self.assertEqual(summary.annotations[0].object_name, OBJECT)
        self.assertEqual(summary.annotations[0].face_key, FACE_KEY)
        self.assertEqual(summary.annotations[0].tag_ids, (52, 53))
        self.assertEqual(len(summary.missing_keypoints), 1)

    def test_dry_run_succeeds_without_creating_dataset_session(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_project_fixture(project_root, keypoints=True)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = collect_cli.main(
                    [
                        "--project_root",
                        str(project_root),
                        "--mode",
                        "opencv",
                        "--session",
                        "run01",
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Dataset collection dry run OK", stdout.getvalue())
            self.assertFalse((project_root / "datasets" / "run01").exists())

    def test_path_helpers_resolve_canonical_inputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_project_fixture(project_root)

            self.assertEqual(
                resolve_calibration_path(project_root, "calib_color.yaml"),
                project_root.resolve() / "calib" / "calib_color.yaml",
            )
            self.assertEqual(resolve_registry_path(project_root), project_root.resolve() / "boards" / "tag_registry.yaml")
            self.assertEqual(
                resolve_face_manifest_path(project_root),
                project_root.resolve() / "faces" / "face_manifest.csv",
            )
            self.assertEqual(resolve_dataset_root(project_root), project_root.resolve() / "datasets")

    def test_missing_inputs_fail_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path, registry_path, manifest_path, annotation_path = _write_project_fixture(project_root)

            calib_path.unlink()
            with self.assertRaises(CollectDatasetError) as missing_calib:
                load_intrinsics_from_calibration(calib_path)
            self.assertIn("Calibration YAML not found", str(missing_calib.exception))

            calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib_path.write_text("not_camera_matrix: true\n", encoding="utf-8")
            with self.assertRaises(CollectDatasetError) as malformed_calib:
                load_intrinsics_from_calibration(calib_path)
            self.assertIn("Malformed calibration YAML", str(malformed_calib.exception))

            _write_calibration(project_root)
            registry_path.unlink()
            with self.assertRaises(CollectDatasetError) as missing_registry:
                validate_collection_inputs(
                    project_root=project_root,
                    calib_path=calib_path,
                    registry_path=registry_path,
                    face_manifest_path=manifest_path,
                )
            self.assertIn("Tag registry not found", str(missing_registry.exception))

            registry_path = _write_registry(project_root, project_root / "boards" / f"{FACE_KEY}.yaml")
            manifest_path.unlink()
            with self.assertRaises(CollectDatasetError) as missing_manifest:
                validate_collection_inputs(
                    project_root=project_root,
                    calib_path=calib_path,
                    registry_path=registry_path,
                    face_manifest_path=manifest_path,
                )
            self.assertIn("Face manifest not found", str(missing_manifest.exception))

            board_path = project_root / "boards" / f"{FACE_KEY}.yaml"
            board_path.unlink()
            manifest_path = _write_face_manifest(project_root, annotation_path, board_path)
            with self.assertRaises(CollectDatasetError) as missing_board:
                validate_collection_inputs(
                    project_root=project_root,
                    calib_path=calib_path,
                    registry_path=registry_path,
                    face_manifest_path=manifest_path,
                )
            self.assertIn("Board YAML not found", str(missing_board.exception))

    def test_missing_annotation_transform_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            calib_path = _write_calibration(project_root)
            board_path = _write_board(project_root)
            registry_path = _write_registry(project_root, board_path)
            annotation_path = _write_annotation(project_root, board_path, include_transform=False)
            manifest_path = _write_face_manifest(project_root, annotation_path, board_path)

            with self.assertRaises(CollectDatasetError) as ctx:
                validate_collection_inputs(
                    project_root=project_root,
                    calib_path=calib_path,
                    registry_path=registry_path,
                    face_manifest_path=manifest_path,
                )

        self.assertIn("T_board_object.matrix", str(ctx.exception))

    def test_source_argument_failures_are_clear(self) -> None:
        with TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "empty.mp4"
            video_path.touch()

            with self.assertRaises(CollectDatasetError) as missing_video:
                validate_source_args(mode="video", video=None, bag=None, realsense_module=object())
            self.assertIn("--video is required", str(missing_video.exception))

            with self.assertRaises(CollectDatasetError) as unreadable_video:
                validate_source_args(
                    mode="video",
                    video=video_path,
                    bag=None,
                    realsense_module=object(),
                    video_capture_factory=ClosedCapture,
                )
            self.assertIn("Could not open video", str(unreadable_video.exception))

            with self.assertRaises(CollectDatasetError) as missing_bag:
                validate_source_args(mode="bag", video=None, bag=None, realsense_module=object())
            self.assertIn("--bag is required", str(missing_bag.exception))

            with self.assertRaises(CollectDatasetError) as missing_realsense:
                validate_source_args(mode="live", video=None, bag=None, realsense_module=None)
            self.assertIn("pyrealsense2 is not available", str(missing_realsense.exception))

    def test_pose_schema_serializes_explicit_transform_convention(self) -> None:
        result = _pose_result()
        record = build_frame_annotation_record(
            dataset_name="run01",
            frame_index=7,
            timestamp=123.456,
            tag_family="tag36h11",
            image_filename="Run_20260607-120000_rgb_frame_000007.png",
            annotated_image="Run_20260607-120000_annotated_frame_000007.png",
            reproj_image="Run_20260607-120000_reproj_frame_000007.png",
            depth=None,
            intrinsics=_intrinsics(),
            results={FACE_KEY: result},
            capture_metadata={
                "mode": "smart_auto",
                "reason": "first_sample",
                "trigger_object": OBJECT,
            },
        )

        obj = record["objects"][0]
        T_cam_board = np.array(obj["transforms"]["T_cam_board"]["matrix"], dtype=float)
        T_board_object = np.array(obj["transforms"]["T_board_object"]["matrix"], dtype=float)
        T_cam_object = np.array(obj["transforms"]["T_cam_object"]["matrix"], dtype=float)

        self.assertEqual(record["pose_convention"]["composition"], "T_cam_object = T_cam_board @ T_board_object")
        self.assertTrue(np.allclose(T_cam_board @ T_board_object, T_cam_object))
        self.assertEqual(obj["object_name"], OBJECT)
        self.assertEqual(obj["selected_face"], FACE_KEY)
        self.assertEqual(obj["6DOF_pose"]["translation_units"], "meters")
        self.assertEqual(obj["6DOF_pose"]["rotation_quaternion_order"], "x, y, z, w")
        self.assertEqual(obj["6DOF_pose"]["orientation_order"], "roll, pitch, yaw")
        self.assertEqual(obj["6DOF_pose"]["orientation_units"], "degrees")
        self.assertEqual(obj["quality"]["num_tags_visible"], 2)
        self.assertEqual(record["camera_extrinsics"]["rotation"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(record["capture"]["mode"], "smart_auto")
        self.assertEqual(record["capture"]["reason"], "first_sample")

    def test_pose_delta_reports_translation_and_rotation_thresholds(self) -> None:
        T_a = np.eye(4)
        T_b = np.eye(4)
        T_b[:3, 3] = [0.1, 0.0, 0.0]
        theta = np.deg2rad(90.0)
        T_b[:3, :3] = [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]

        translation_delta, rotation_delta = pose_delta(T_a, T_b)

        self.assertAlmostEqual(translation_delta, 0.1)
        self.assertAlmostEqual(rotation_delta, 90.0)

    def test_smart_auto_capture_waits_for_stability_and_useful_views(self) -> None:
        config = AutoCaptureConfig(
            enabled=True,
            stable_frames=3,
            cooldown_sec=1.0,
            min_tags_visible=1,
            min_bbox_area_px=1000.0,
            min_translation_delta_m=0.02,
            min_rotation_delta_deg=5.0,
            grid_rows=2,
            grid_cols=2,
            distance_bin_m=0.25,
        )
        state = AutoCaptureState()
        result = _pose_result()
        candidates = auto_capture_candidates_from_results({FACE_KEY: result})

        first = evaluate_auto_capture(
            candidates,
            state,
            config,
            frame_width=640,
            frame_height=480,
            timestamp=1.0,
        )
        second = evaluate_auto_capture(
            candidates,
            state,
            config,
            frame_width=640,
            frame_height=480,
            timestamp=1.1,
        )
        third = evaluate_auto_capture(
            candidates,
            state,
            config,
            frame_width=640,
            frame_height=480,
            timestamp=1.2,
        )

        self.assertFalse(first.should_save)
        self.assertEqual(first.status, "waiting_for_stability")
        self.assertFalse(second.should_save)
        self.assertTrue(third.should_save)
        self.assertEqual(third.reason, "first_sample")

        record_auto_capture_save(
            state,
            candidates[0],
            config,
            frame_width=640,
            frame_height=480,
            timestamp=1.2,
        )
        cooldown = evaluate_auto_capture(
            candidates,
            state,
            config,
            frame_width=640,
            frame_height=480,
            timestamp=1.5,
        )
        duplicate = evaluate_auto_capture(
            candidates,
            state,
            config,
            frame_width=640,
            frame_height=480,
            timestamp=3.0,
        )

        self.assertFalse(cooldown.should_save)
        self.assertEqual(cooldown.status, "cooldown")
        self.assertFalse(duplicate.should_save)
        self.assertEqual(duplicate.status, "waiting_for_new_view")

        moved_result = _pose_result()
        moved_result["bbox_xywh"] = [450.0, 320.0, 128.0, 96.0]
        moved_candidates = auto_capture_candidates_from_results({FACE_KEY: moved_result})
        coverage = evaluate_auto_capture(
            moved_candidates,
            state,
            config,
            frame_width=640,
            frame_height=480,
            timestamp=3.1,
        )

        self.assertTrue(coverage.should_save)
        self.assertEqual(coverage.reason, "coverage_cell")

    def test_smart_auto_capture_rejects_low_quality_candidates(self) -> None:
        config = AutoCaptureConfig(
            enabled=True,
            stable_frames=1,
            min_tags_visible=2,
            min_bbox_area_px=2000.0,
            max_tag_scale_error=0.05,
        )
        result = _pose_result()
        result["num_tags_visible"] = 1
        result["bbox_xywh"] = [10.0, 20.0, 10.0, 10.0]
        result["diagnostics"]["tag_scale_ratio"] = 1.20
        result["diagnostics"]["tag_scale_pairs"] = 1
        decision = evaluate_auto_capture(
            auto_capture_candidates_from_results({FACE_KEY: result}),
            AutoCaptureState(),
            config,
            frame_width=640,
            frame_height=480,
            timestamp=1.0,
        )

        self.assertFalse(decision.should_save)
        self.assertEqual(decision.status, "waiting_for_quality")
        self.assertTrue(any("visible tags" in reason for reason in decision.rejected_reasons))
        self.assertTrue(any("tag scale error" in reason for reason in decision.rejected_reasons))

    def test_auto_capture_grid_parser_validates_rows_and_columns(self) -> None:
        self.assertEqual(parse_auto_capture_grid("4x5"), (4, 5))
        with self.assertRaises(CollectDatasetError):
            parse_auto_capture_grid("4")

    def test_pose_schema_rejects_transform_composition_drift(self) -> None:
        result = _pose_result()
        bad_T = np.array(result["T_cam_object"]["matrix"], dtype=float)
        bad_T[0, 3] += 1.0
        result["T_cam_object"] = {"matrix": bad_T.tolist()}

        with self.assertRaises(CollectDatasetError) as ctx:
            build_frame_annotation_record(
                dataset_name="run01",
                frame_index=0,
                timestamp=1.0,
                tag_family="tag36h11",
                image_filename="frame.png",
                annotated_image=None,
                reproj_image=None,
                depth=None,
                intrinsics=_intrinsics(),
                results={FACE_KEY: result},
            )

        self.assertIn("Pose composition mismatch", str(ctx.exception))

    def test_save_frame_rejects_empty_pose_results_without_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "datasets" / "run01"

            with self.assertRaises(CollectDatasetError) as ctx:
                save_frame(
                    out_dir=out_dir,
                    idx=0,
                    ts=123.456,
                    bgr_raw=np.zeros((16, 16, 3), dtype=np.uint8),
                    depth=None,
                    intr=_intrinsics(),
                    family="tag36h11",
                    results={},
                    reproj_img=None,
                    save_depth=False,
                    dataset_name="run01",
                )

            self.assertIn("without at least one pose-labelled object", str(ctx.exception))
            self.assertFalse(out_dir.exists())

    def test_capture_review_dashboard_renders_structured_status_panel(self) -> None:
        left = np.full((240, 320, 3), 180, dtype=np.uint8)
        mid = np.full((240, 320, 3), 96, dtype=np.uint8)
        panel = _capture_status_panel(
            session="run01",
            faces_visible=1,
            saved_count=2,
            object_summaries=[
                {
                    "object": OBJECT,
                    "face_key": FACE_KEY,
                    "translation_m": (0.25, -0.10, 1.50),
                    "distance_m": 1.53,
                    "rpy_deg": (1.0, 2.0, 3.0),
                    "tag_used": 52,
                    "bbox_source": "face_kps",
                    "tag_scale_ratio": 1.0,
                    "tag_scale_pairs": 1,
                }
            ],
            capture_message="Frame #002 captured [OK]",
            width=420,
            height=320,
        )
        strip = _dashboard_strip(
            left,
            mid,
            panel,
            tile_h=320,
            tile_w=360,
            right_w=420,
        )

        self.assertEqual(panel.shape, (320, 420, 3))
        self.assertEqual(strip.shape, (320, 1160, 3))
        self.assertGreater(np.count_nonzero(panel != panel[0, 0]), 1000)
        self.assertGreater(np.count_nonzero(strip != strip[0, 0]), 1000)

    def test_capture_review_quit_keys_close_cleanly(self) -> None:
        for key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
            with self.subTest(key=key):
                self.assertTrue(_capture_key_requests_quit(key))
                self.assertTrue(_capture_key_requests_quit(key | 0x100000))

        self.assertFalse(_capture_key_requests_quit(ord("p")))
        self.assertFalse(_capture_key_requests_quit(ord("h")))
        self.assertFalse(_capture_key_requests_quit(-1))
        self.assertEqual(_normalize_capture_key(27 | 0x100000), 27)

    def test_session_metadata_records_source_and_pose_convention(self) -> None:
        meta = build_session_metadata(
            session_name="run01",
            project_root=Path("/tmp/project"),
            dataset_root=Path("/tmp/project/datasets"),
            session_dir=Path("/tmp/project/datasets/run01"),
            intrinsics=_intrinsics(),
            tag_family="tag36h11",
            camera_kind="opencv_webcam",
            mode="opencv",
            tag_scale_controls={"check_tag_scale": False, "auto_correct_scale": False, "scale_tol": 0.02},
            faces_index={FACE_KEY: {"object": OBJECT, "board_yaml": f"boards/{FACE_KEY}.yaml"}},
        )

        self.assertEqual(meta["source"]["mode"], "opencv")
        self.assertEqual(meta["camera"]["kind"], "opencv_webcam")
        self.assertEqual(meta["pose_convention"]["translation_units"], "meters")
        self.assertEqual(meta["objects_available"], [OBJECT])


if __name__ == "__main__":
    unittest.main()
