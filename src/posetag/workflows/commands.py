"""Read-only command previews for PoseTag workflow stages."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable, Union

from posetag.workflows.mesh_keypoints import (
    inspect_keypoint_object_statuses,
)


_COMMAND_TEMPLATES = {
    "project_setup": (
        "posetag",
        "project",
        "new",
        "--root",
        "{project_root}",
    ),
    "generate_charuco_board": (
        "posetag-gen-charuco",
        "--project_root",
        "{project_root}",
        "--squares-x",
        "3",
        "--squares-y",
        "5",
        "--square-length-mm",
        "50",
        "--marker-length-mm",
        "37",
        "--dict",
        "7X7_50",
        "--paper",
        "A4",
        "--dpi",
        "300",
    ),
    "generate_tags": (
        "posetag-gen-tags",
        "--project_root",
        "{project_root}",
        "--tag-size-mm",
        "40",
        "--ids",
        "1-4",
    ),
    "generate_object_tags": (
        "posetag-gen-tags",
        "--project_root",
        "{project_root}",
        "--tag-size-mm",
        "40",
        "--ids",
        "1-4",
    ),
    "calibrate_camera": (
        "posetag-calib-charuco",
        "--project_root",
        "{project_root}",
        "--source",
        "opencv",
        "--cam",
        "0",
    ),
    "build_boards": (
        "posetag-make-board",
        "--project_root",
        "{project_root}",
        "--source",
        "opencv",
        "--cam",
        "0",
        "--object_name",
        "OBJECT_FACE",
        "--calib",
        "{project_root}/calib/calib_color.yaml",
        "--family",
        "tag36h11",
        "--tag_size_mm",
        "40",
    ),
    "build_object_boards": (
        "posetag-make-board",
        "--project_root",
        "{project_root}",
        "--source",
        "opencv",
        "--cam",
        "0",
        "--object_name",
        "OBJECT_FACE",
        "--calib",
        "{project_root}/calib/calib_color.yaml",
        "--family",
        "tag36h11",
        "--tag_size_mm",
        "40",
    ),
    "capture_face_shots": (
        "posetag-capture-face",
        "--project_root",
        "{project_root}",
        "--source",
        "opencv",
        "--cam",
        "0",
        "--object_name",
        "OBJECT_NAME",
    ),
    "generate_mesh_keypoints": (
        "posetag-gen-keypoints",
        "--project_root",
        "{project_root}",
        "--mesh",
        "{project_root}/meshes/OBJECT.obj",
        "--object_name",
        "OBJECT_NAME",
    ),
    "annotate_faces": (
        "env",
        "POSETAG_PROJECT={project_root}",
        "posetag-annotate",
        "--browse",
    ),
    "collect_dataset": (
        "env",
        "POSETAG_PROJECT={project_root}",
        "posetag-collect",
        "--mode",
        "live",
        "--session",
        "SESSION",
        "--calib",
        "calib_color.yaml",
    ),
}


def command_preview(stage_key: str, project_root: Union[Path, str]) -> str:
    """Return a copyable command preview for a workflow stage, if known."""

    if stage_key == "generate_mesh_keypoints":
        return _mesh_keypoints_command_preview(project_root)

    template = _COMMAND_TEMPLATES.get(stage_key)
    if template is None:
        return ""

    root = str(Path(project_root).expanduser())
    parts = (
        _format_part(part, root)
        for part in template
    )
    return _join_command(parts)


def _mesh_keypoints_command_preview(project_root: Union[Path, str]) -> str:
    root_path = Path(project_root).expanduser()
    statuses = inspect_keypoint_object_statuses(root_path)

    selected = next(
        (
            status
            for status in statuses
            if not status.keypoints_valid
        ),
        statuses[0] if statuses else None,
    )
    if selected is None:
        return _join_template("generate_mesh_keypoints", root_path)

    return selected.command_preview


def _join_template(stage_key: str, project_root: Path) -> str:
    template = _COMMAND_TEMPLATES.get(stage_key)
    if template is None:
        return ""
    root = str(project_root)
    return _join_command(_format_part(part, root) for part in template)


def _format_part(part: str, project_root: str) -> str:
    return part.replace("{project_root}", project_root)


def _join_command(parts: Iterable[str]) -> str:
    return shlex.join(tuple(parts))
