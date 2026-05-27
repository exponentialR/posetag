"""Read-only command previews for PoseTag workflow stages."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable, Union


_COMMAND_TEMPLATES = {
    "generate_tags": (
        "posetag-gen-tags",
        "--project_root",
        "{project_root}",
        "--tag-size-mm",
        "TAG_SIZE_MM",
        "--ids",
        "TAG_IDS",
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
        "--object_name",
        "OBJECT_FACE",
        "--calib",
        "calib_color.yaml",
    ),
    "capture_face_shots": (
        "posetag-capture-face",
        "--project_root",
        "{project_root}",
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

    template = _COMMAND_TEMPLATES.get(stage_key)
    if template is None:
        return ""

    root = str(Path(project_root).expanduser())
    parts = (
        _format_part(part, root)
        for part in template
    )
    return _join_command(parts)


def _format_part(part: str, project_root: str) -> str:
    return part.replace("{project_root}", project_root)


def _join_command(parts: Iterable[str]) -> str:
    return shlex.join(tuple(parts))
