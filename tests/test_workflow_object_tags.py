from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from posetag.pipelines.charuco_calibration import (
    build_calibration_yaml,
    write_calibration_yaml,
)
from posetag.workflows.object_tags import (
    ID_MODE_RANGE,
    PAPER_CUSTOM,
    ObjectTagGenerationConfig,
    ObjectTagGenerationError,
    build_object_tag_generation_arguments,
    build_object_tag_generation_command,
    default_output_dir,
    generate_object_tags,
    inspect_object_tag_generation,
)
from posetag.workflows.status import WorkflowStatus, inspect_project


class WorkflowObjectTagTests(unittest.TestCase):
    def test_readiness_requires_stage2_calibration_by_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            readiness = inspect_object_tag_generation(
                ObjectTagGenerationConfig(
                    project_root=project_root,
                    tag_size_mm=40.0,
                    ids="1-4",
                    dpi=80,
                )
            )

        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.command_preview, "")
        self.assertIn("Stage 2 camera calibration", readiness.errors[0])
        self.assertEqual(
            readiness.expected_output_dir,
            default_output_dir(project_root),
        )

    def test_list_generation_writes_default_project_patterns_and_refreshes_status(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            config = ObjectTagGenerationConfig(
                project_root=project_root,
                tag_size_mm=40.0,
                ids="1-3",
                dpi=80,
            )

            readiness = inspect_object_tag_generation(config)
            self.assertTrue(readiness.ready, readiness.errors)
            self.assertIn("posetag-gen-tags", readiness.command_preview)
            self.assertIn("--ids 1-3", readiness.command_preview)

            with _isolated_project_config(tmpdir):
                result = generate_object_tags(config)

            expected_dir = project_root.resolve() / "boards" / "patterns"
            self.assertEqual(result.out_dir.resolve(), expected_dir)
            self.assertEqual(result.parsed_ids, (1, 2, 3))
            self.assertEqual(len(result.png_paths), 1)
            self.assertTrue(result.png_paths[0].exists())
            self.assertIn(result.png_paths[0], result.output_paths)

            stage3 = {
                stage.stage_id: stage for stage in inspect_project(project_root)
            }[3]
            self.assertEqual(stage3.status, WorkflowStatus.COMPLETE)
            checked_paths = {Path(path).resolve() for path in stage3.checked_paths}
            self.assertIn(result.png_paths[0].resolve(), checked_paths)

    def test_range_count_builds_existing_id_start_end_arguments(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            project_root = base / "project"
            out_dir = base / "printable-tags"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            config = ObjectTagGenerationConfig(
                project_root=project_root,
                tag_size_mm=35.0,
                id_mode=ID_MODE_RANGE,
                id_start=10,
                id_count=3,
                out_dir=out_dir,
                dpi=80,
            )

            readiness = inspect_object_tag_generation(config)
            args = build_object_tag_generation_arguments(config)
            command = build_object_tag_generation_command(config)

            self.assertTrue(readiness.ready, readiness.errors)
            self.assertEqual(readiness.parsed_ids, (10, 11, 12))
            self.assertIn("--id_start", args)
            self.assertIn("10", args)
            self.assertIn("--id_end", args)
            self.assertIn("12", args)
            self.assertIn("--out_dir", args)
            self.assertIn(str(out_dir), command)
            self.assertTrue(
                any("Project status checks" in item for item in readiness.warnings)
            )

            with _isolated_project_config(tmpdir):
                result = generate_object_tags(config)

            self.assertEqual(result.out_dir, out_dir)
            self.assertEqual(result.parsed_ids, (10, 11, 12))
            self.assertTrue(result.png_paths[0].exists())

    def test_custom_paper_uses_existing_paper_mm_argument(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            config = ObjectTagGenerationConfig(
                project_root=project_root,
                tag_size_mm=20.0,
                ids="1",
                paper=PAPER_CUSTOM,
                paper_mm="80x80",
                dpi=80,
            )

            readiness = inspect_object_tag_generation(config)
            args = build_object_tag_generation_arguments(config)

            self.assertTrue(readiness.ready, readiness.errors)
            self.assertIn("--paper-mm", args)
            self.assertIn("80x80", args)

    def test_invalid_inputs_fail_without_running_generator(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            _write_valid_calibration(project_root / "calib" / "calib_color.yaml")
            config = ObjectTagGenerationConfig(
                project_root=project_root,
                family="tag25h9",
                tag_size_mm=40.0,
                ids="1",
                dpi=80,
            )

            readiness = inspect_object_tag_generation(config)
            self.assertFalse(readiness.ready)
            self.assertIn("tag36h11", readiness.errors[0])

            with self.assertRaises(ObjectTagGenerationError) as ctx:
                generate_object_tags(config)
            self.assertIn("tag36h11", str(ctx.exception))


def _write_valid_calibration(path: Path) -> None:
    camera_matrix = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 610.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    data = build_calibration_yaml(
        image_width=640,
        image_height=480,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        reproj_rms=0.12,
        notes="Synthetic calibration for object-tag workflow tests.",
    )
    write_calibration_yaml(path, data)


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
