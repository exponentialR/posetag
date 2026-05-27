"""Read-only workflow status inspection for PoseTag project directories.

The helpers in this module are GUI-independent.  They report whether the
validated PoseTag workflow artifacts for Steps 0-2 are present and readable,
while leaving camera capture, calibration, board construction, annotation, and
pose-estimation algorithms in their existing pipeline modules.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import yaml

from posetag.pipelines.make_board import (
    MakeBoardError,
    load_calibration_yaml,
    load_registry,
)


class WorkflowStatus(str, Enum):
    """Workflow-stage status values consumed by frontends."""

    COMPLETE = "complete"
    MISSING = "missing"
    NEEDS_ATTENTION = "needs_attention"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class StageSummary:
    """Structured status for one PoseTag workflow stage."""

    stage_id: int
    key: str
    name: str
    status: WorkflowStatus
    message: str
    checked_paths: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for GUI or CLI renderers."""

        data = asdict(self)
        data["status"] = self.status.value
        data["checked_paths"] = list(self.checked_paths)
        data["warnings"] = list(self.warnings)
        data["errors"] = list(self.errors)
        return data


_STAGE_NAMES = {
    0: ("generate_tags", "Generate AprilTag Sheets"),
    1: ("calibrate_camera", "Calibrate Camera"),
    2: ("build_boards", "Build Boards"),
    3: ("capture_face_shots", "Capture Face Shots"),
    4: ("annotate_faces", "Annotate Faces"),
    5: ("collect_dataset", "Collect Dataset"),
    6: ("review_export", "Review and Export"),
}

_BOARD_REQUIRED_FIELDS = ("object", "family", "tag_size_m", "origin_id", "tags", "notes")
_BOARD_TAG_REQUIRED_FIELDS = ("id", "cx", "cy", "yaw_deg")


def inspect_project(project_root: Union[Path, str]) -> list[StageSummary]:
    """Inspect a PoseTag project directory and return stage status summaries.

    The function is intentionally read-only: it does not call project-root
    resolvers, create directories, launch commands, import GUI frameworks, or
    open cameras.
    """

    root = Path(project_root).expanduser()
    summaries = [
        inspect_step0(root),
        inspect_step1(root),
        inspect_step2(root),
    ]
    summaries.extend(
        _placeholder_summaries(
            step2_complete=summaries[2].status == WorkflowStatus.COMPLETE
        )
    )
    return summaries


def inspect_step0(project_root: Union[Path, str]) -> StageSummary:
    """Inspect Step 0 AprilTag sheet outputs."""

    root = Path(project_root).expanduser()
    patterns_dir = root / "boards" / "patterns"
    png_paths = sorted(path for path in patterns_dir.glob("*.png") if path.is_file())
    checked_paths = _path_tuple([patterns_dir, *png_paths])

    if png_paths:
        count = len(png_paths)
        return _stage(
            stage_id=0,
            status=WorkflowStatus.COMPLETE,
            message=f"Found {count} AprilTag sheet PNG file{'s' if count != 1 else ''}.",
            checked_paths=checked_paths,
            next_action="Proceed to Step 1 camera calibration.",
        )

    return _stage(
        stage_id=0,
        status=WorkflowStatus.MISSING,
        message="No generated AprilTag sheet PNG files were found.",
        checked_paths=checked_paths,
        next_action="Run posetag-gen-tags to write sheets under boards/patterns/.",
    )


def inspect_step1(project_root: Union[Path, str]) -> StageSummary:
    """Inspect Step 1 colour-camera calibration output."""

    root = Path(project_root).expanduser()
    calib_path = root / "calib" / "calib_color.yaml"
    checked_paths = _path_tuple([calib_path])

    if not calib_path.exists():
        return _stage(
            stage_id=1,
            status=WorkflowStatus.MISSING,
            message="Colour-camera calibration YAML was not found.",
            checked_paths=checked_paths,
            next_action="Run posetag-calib-charuco to create calib/calib_color.yaml.",
        )

    try:
        load_calibration_yaml(calib_path)
    except MakeBoardError as exc:
        return _stage(
            stage_id=1,
            status=WorkflowStatus.NEEDS_ATTENTION,
            message="Colour-camera calibration YAML exists but did not pass schema checks.",
            checked_paths=checked_paths,
            errors=(str(exc),),
            next_action="Regenerate or repair calib/calib_color.yaml before building boards.",
        )

    return _stage(
        stage_id=1,
        status=WorkflowStatus.COMPLETE,
        message="Colour-camera calibration YAML exists and passed schema checks.",
        checked_paths=checked_paths,
        next_action="Proceed to Step 2 board building.",
    )


def inspect_step2(project_root: Union[Path, str]) -> StageSummary:
    """Inspect Step 2 board YAML files and tag registry output."""

    root = Path(project_root).expanduser()
    registry_path = root / "boards" / "tag_registry.yaml"
    checked_paths: list[Path] = [registry_path]

    if not registry_path.exists():
        return _stage(
            stage_id=2,
            status=WorkflowStatus.MISSING,
            message="Tag registry YAML was not found.",
            checked_paths=_path_tuple(checked_paths),
            next_action="Run posetag-make-board after completing camera calibration.",
        )

    try:
        registry = load_registry(registry_path)
    except MakeBoardError as exc:
        return _stage(
            stage_id=2,
            status=WorkflowStatus.NEEDS_ATTENTION,
            message="Tag registry YAML exists but did not pass schema checks.",
            checked_paths=_path_tuple(checked_paths),
            errors=(str(exc),),
            next_action="Repair boards/tag_registry.yaml or rerun posetag-make-board.",
        )

    tags = registry.get("tags", {})
    if not tags:
        return _stage(
            stage_id=2,
            status=WorkflowStatus.NEEDS_ATTENTION,
            message="Tag registry YAML has no tag mappings.",
            checked_paths=_path_tuple(checked_paths),
            errors=("boards/tag_registry.yaml must contain at least one tag mapping.",),
            next_action="Run posetag-make-board and select at least one visible tag.",
        )

    warnings: list[str] = []
    errors: list[str] = []
    valid_board_paths: set[Path] = set()

    for tag_id, entry in _sorted_registry_entries(tags):
        if not isinstance(entry, Mapping):
            errors.append(f"Registry entry {tag_id!r} must be a mapping.")
            continue

        yaml_value = entry.get("yaml")
        if not isinstance(yaml_value, str) or not yaml_value.strip():
            errors.append(f"Registry entry {tag_id!r} is missing a board YAML path.")
            continue

        board_path = _resolve_registry_board_path(root, registry_path, yaml_value)
        checked_paths.append(board_path)
        if not board_path.exists():
            errors.append(
                f"Referenced board YAML for tag {tag_id!r} was not found: {board_path}"
            )
            continue

        board_errors = _validate_board_yaml(board_path)
        if board_errors:
            errors.extend(f"{board_path}: {message}" for message in board_errors)
            continue

        valid_board_paths.add(board_path)

    if errors:
        if valid_board_paths:
            warnings.append(
                f"Found {len(valid_board_paths)} valid referenced board YAML file"
                f"{'s' if len(valid_board_paths) != 1 else ''}, but the registry also has errors."
            )
        return _stage(
            stage_id=2,
            status=WorkflowStatus.NEEDS_ATTENTION,
            message="Tag registry or referenced board YAML files need attention.",
            checked_paths=_path_tuple(checked_paths),
            warnings=tuple(warnings),
            errors=tuple(errors),
            next_action=(
                "Repair the registry references or rerun posetag-make-board "
                "for the affected board."
            ),
        )

    if not valid_board_paths:
        return _stage(
            stage_id=2,
            status=WorkflowStatus.NEEDS_ATTENTION,
            message="No referenced board YAML files passed schema checks.",
            checked_paths=_path_tuple(checked_paths),
            errors=("At least one referenced board YAML must exist and match the Step 2 schema.",),
            next_action="Run posetag-make-board to create a board YAML and registry mapping.",
        )

    count = len(valid_board_paths)
    return _stage(
        stage_id=2,
        status=WorkflowStatus.COMPLETE,
        message=(
            f"Tag registry maps tags to {count} valid board YAML "
            f"file{'s' if count != 1 else ''}."
        ),
        checked_paths=_path_tuple(checked_paths),
        next_action="Proceed to Step 3 face-shot capture.",
    )


def _placeholder_summaries(step2_complete: bool) -> list[StageSummary]:
    if step2_complete:
        step3 = _stage(
            stage_id=3,
            status=WorkflowStatus.MISSING,
            message="Face-shot capture status validation is not implemented yet.",
            checked_paths=(),
            next_action="Run the Step 3 capture workflow after confirming board coverage.",
        )
    else:
        step3 = _stage(
            stage_id=3,
            status=WorkflowStatus.NOT_APPLICABLE,
            message="Face-shot capture depends on a valid tag registry and board YAML.",
            checked_paths=(),
            next_action="Complete Step 2 before checking Step 3.",
        )

    return [
        step3,
        _stage(
            stage_id=4,
            status=WorkflowStatus.NOT_APPLICABLE,
            message="Face annotation depends on captured face shots.",
            checked_paths=(),
            next_action="Complete Step 3 before checking Step 4.",
        ),
        _stage(
            stage_id=5,
            status=WorkflowStatus.NOT_APPLICABLE,
            message="Dataset collection depends on calibration, boards, and annotations.",
            checked_paths=(),
            next_action="Complete Steps 1-4 before checking Step 5.",
        ),
        _stage(
            stage_id=6,
            status=WorkflowStatus.NOT_APPLICABLE,
            message="Review and export depends on generated dataset outputs.",
            checked_paths=(),
            next_action="Complete Step 5 before checking Step 6.",
        ),
    ]


def _validate_board_yaml(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        return (f"Malformed board YAML: {exc}",)
    except OSError as exc:
        return (f"Could not read board YAML: {exc}",)

    if not isinstance(data, Mapping):
        return ("Board YAML must contain a mapping.",)

    errors: list[str] = []
    for field in _BOARD_REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Board YAML is missing required field {field!r}.")

    if errors:
        return tuple(errors)

    for field in ("object", "family", "notes"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            errors.append(f"Board YAML field {field!r} must be a non-empty string.")

    _require_number(data, "tag_size_m", errors)
    _require_int(data, "origin_id", errors)

    tags = data.get("tags")
    if not isinstance(tags, list) or not tags:
        errors.append("Board YAML field 'tags' must be a non-empty list.")
    else:
        for index, tag in enumerate(tags):
            if not isinstance(tag, Mapping):
                errors.append(f"Board YAML tag entry {index} must be a mapping.")
                continue
            for field in _BOARD_TAG_REQUIRED_FIELDS:
                if field not in tag:
                    errors.append(
                        f"Board YAML tag entry {index} is missing field {field!r}."
                    )
            _require_int(tag, "id", errors, prefix=f"Board YAML tag entry {index}")
            for field in ("cx", "cy", "yaw_deg"):
                _require_number(tag, field, errors, prefix=f"Board YAML tag entry {index}")

    return tuple(errors)


def _resolve_registry_board_path(
    project_root: Path,
    registry_path: Path,
    yaml_value: str,
) -> Path:
    raw = Path(yaml_value).expanduser()
    if raw.is_absolute():
        return raw

    candidates = [
        project_root / raw,
        registry_path.parent / raw,
    ]
    if raw.parts and raw.parts[0] == project_root.name:
        candidates.append(project_root.parent / raw)
    candidates.append(raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _sorted_registry_entries(tags: Mapping[Any, Any]) -> Iterable[tuple[str, Any]]:
    return sorted(
        ((str(tag_id), entry) for tag_id, entry in tags.items()),
        key=lambda item: item[0],
    )


def _stage(
    stage_id: int,
    status: WorkflowStatus,
    message: str,
    checked_paths: Iterable[Union[Path, str]],
    next_action: str,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> StageSummary:
    key, name = _STAGE_NAMES[stage_id]
    return StageSummary(
        stage_id=stage_id,
        key=key,
        name=name,
        status=status,
        message=message,
        checked_paths=tuple(str(path) for path in checked_paths),
        warnings=warnings,
        errors=errors,
        next_action=next_action,
    )


def _path_tuple(paths: Iterable[Union[Path, str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(path) for path in paths))


def _require_number(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
    prefix: Optional[str] = None,
) -> None:
    label = f"{prefix} field {field!r}" if prefix else f"Board YAML field {field!r}"
    try:
        value = float(data[field])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{label} must be numeric.")
        return
    if not math.isfinite(value):
        errors.append(f"{label} must be finite.")


def _require_int(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
    prefix: Optional[str] = None,
) -> None:
    label = f"{prefix} field {field!r}" if prefix else f"Board YAML field {field!r}"
    try:
        value = data[field]
        int(value)
    except (KeyError, TypeError, ValueError):
        errors.append(f"{label} must be an integer.")
        return
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        errors.append(f"{label} must be an integer.")
