"""Reusable helpers for PoseTag Step 2 board building.

The interactive capture loop remains in the legacy top-level ``make_board``
module for now.  The helpers here cover validation, path preparation, board
YAML construction, calibration loading, and tag-registry updates so Step 2 can
be tested without camera hardware.
"""

from __future__ import annotations

import datetime
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

try:
    from posetag.utils.project_config import ensure_project_dirs, resolve_project_root
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from posetag.utils.project_config import ensure_project_dirs, resolve_project_root


BOARD_NOTES = "Board frame = origin tag centre; x,y follow origin tag axes; z ≈ 0."


class MakeBoardError(ValueError):
    """User-facing validation error for Step 2 board building."""


@dataclass(frozen=True)
class CalibrationData:
    """Loaded camera intrinsics needed by pupil-apriltags pose estimation."""

    path: Path
    camera_params: tuple[float, float, float, float]
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray


@dataclass(frozen=True)
class MakeBoardPaths:
    """Resolved Step 2 project and output paths."""

    project_root: Path
    boards_dir: Path
    shots_dir: Path
    registry_path: Path
    board_yaml_path: Path


@dataclass(frozen=True)
class RegistryConflict:
    """A tag registry conflict that was not overwritten."""

    tag_id: str
    existing_yaml: str
    requested_yaml: str


@dataclass(frozen=True)
class RegistryUpdate:
    """Summary of registry updates applied in memory."""

    updated: int
    conflicts: tuple[RegistryConflict, ...]


def _atomic_write_yaml(path: Union[Path, str], data: Mapping[str, Any]) -> None:
    """Write YAML atomically to avoid partial registry/board files."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="._tmp_yaml_", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(data), handle, sort_keys=False)
        shutil.move(str(tmp), str(target))
    finally:
        if tmp.exists():
            tmp.unlink()


def preview_project_root(project_root: Optional[Union[Path, str]]) -> Path:
    """Resolve a project root for read-only validation.

    An explicit ``--project_root`` is normalized without creating directories.
    When no root is supplied, PoseTag's existing project resolver is used so
    default behavior remains compatible with earlier workflow steps.
    """

    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return Path(resolve_project_root(None))


def resolve_calibration_path(
    project_root: Union[Path, str],
    calib: Optional[Union[Path, str]] = "calib_color.yaml",
) -> Path:
    """Resolve the calibration YAML used by Step 2.

    Relative calibration names are project-aware: PoseTag first checks
    ``<project_root>/calib/<name>`` and then falls back to the literal relative
    path for compatibility with older script usage.
    """

    raw = Path(calib or "calib_color.yaml").expanduser()
    if raw.is_absolute():
        return raw

    project_candidate = Path(project_root).expanduser().resolve() / "calib" / raw
    if project_candidate.exists() or not raw.exists():
        return project_candidate
    return raw


def load_calibration_yaml(path: Union[Path, str]) -> CalibrationData:
    """Load and validate the Step 1 colour-camera calibration YAML."""

    target = Path(path)
    if not target.exists():
        raise MakeBoardError(
            f"Calibration YAML not found: {target}. Run Step 1 to create calib_color.yaml first."
        )

    try:
        with target.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise MakeBoardError(f"Malformed calibration YAML: {target}: {exc}") from exc
    except OSError as exc:
        raise MakeBoardError(f"Could not read calibration YAML: {target}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise MakeBoardError(f"Malformed calibration YAML: {target} must contain a mapping.")

    camera_matrix = data.get("camera_matrix")
    if not isinstance(camera_matrix, Mapping):
        raise MakeBoardError(
            f"Malformed calibration YAML: {target} is missing camera_matrix fields."
        )

    missing = [name for name in ("fx", "fy", "cx", "cy") if name not in camera_matrix]
    if missing:
        joined = ", ".join(f"camera_matrix.{name}" for name in missing)
        raise MakeBoardError(f"Malformed calibration YAML: {target} is missing {joined}.")

    try:
        fx = float(camera_matrix["fx"])
        fy = float(camera_matrix["fy"])
        cx = float(camera_matrix["cx"])
        cy = float(camera_matrix["cy"])
    except (TypeError, ValueError) as exc:
        raise MakeBoardError(
            f"Malformed calibration YAML: {target} camera_matrix values must be numeric."
        ) from exc

    distortion = data.get("distortion_coefficients") or {}
    if not isinstance(distortion, Mapping):
        raise MakeBoardError(
            f"Malformed calibration YAML: {target} distortion_coefficients must be a mapping."
        )
    distortion_values: list[float] = []
    for name in ("k1", "k2", "p1", "p2", "k3"):
        try:
            distortion_values.append(float(distortion.get(name, 0.0)))
        except (TypeError, ValueError) as exc:
            raise MakeBoardError(
                f"Malformed calibration YAML: {target} "
                f"distortion_coefficients.{name} must be numeric."
            ) from exc

    camera_matrix_array = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], float)
    distortion_array = np.array([distortion_values], float)
    return CalibrationData(
        path=target,
        camera_params=(fx, fy, cx, cy),
        camera_matrix=camera_matrix_array,
        distortion_coefficients=distortion_array,
    )


def validate_source_args(source: str, video: Optional[str], realsense_module: Any) -> None:
    """Validate source-specific arguments before opening hardware or writing files."""

    if source == "video":
        if not video:
            raise MakeBoardError("--video path is required when --source=video")
        if not Path(video).expanduser().exists():
            raise MakeBoardError(f"Could not open video: {video}")
    if source == "realsense" and realsense_module is None:
        raise MakeBoardError(
            "pyrealsense2 is not available; install PoseTag with the 'realsense' "
            "extra or use --source opencv|video"
        )


def prepare_project_paths(
    project_root: Optional[Union[Path, str]],
    object_name: str,
    out_dir: Optional[Union[Path, str]] = None,
    shots_dir: Optional[Union[Path, str]] = None,
    registry: Optional[Union[Path, str]] = None,
    save_shot: bool = False,
) -> MakeBoardPaths:
    """Resolve and create the Step 2 output layout."""

    pr = Path(resolve_project_root(project_root))
    ensure_project_dirs(pr)

    boards_dir = Path(out_dir).expanduser() if out_dir else pr / "boards"
    resolved_shots_dir = Path(shots_dir).expanduser() if shots_dir else boards_dir / "shots"
    registry_path = Path(registry).expanduser() if registry else boards_dir / "tag_registry.yaml"
    board_yaml_path = boards_dir / f"{object_name}.yaml"

    boards_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if save_shot:
        resolved_shots_dir.mkdir(parents=True, exist_ok=True)

    return MakeBoardPaths(
        project_root=pr,
        boards_dir=boards_dir,
        shots_dir=resolved_shots_dir,
        registry_path=registry_path,
        board_yaml_path=board_yaml_path,
    )


def build_board_yaml(
    object_name: str,
    family: str,
    tag_size_mm: float,
    origin_id: int,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the public board YAML schema from computed tag entries."""

    tag_size_m = float(tag_size_mm) / 1000.0
    tags: list[dict[str, Any]] = []
    for entry in entries:
        tags.append(
            {
                "id": int(entry["id"]),
                "cx": float(entry["cx"]),
                "cy": float(entry["cy"]),
                "yaw_deg": float(entry["yaw_deg"]),
            }
        )

    tags.sort(key=lambda item: item["id"])
    return {
        "object": object_name,
        "family": family,
        "tag_size_m": tag_size_m,
        "origin_id": int(origin_id),
        "tags": tags,
        "notes": BOARD_NOTES,
    }


def write_board_yaml(path: Union[Path, str], board: Mapping[str, Any]) -> None:
    """Write a board YAML file."""

    _atomic_write_yaml(Path(path), board)


def load_registry(path: Union[Path, str]) -> dict[str, Any]:
    """Load the tag registry, returning an empty versioned registry if absent."""

    target = Path(path)
    if not target.exists():
        return {"version": 1, "updated": None, "tags": {}}
    try:
        with target.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise MakeBoardError(f"Malformed tag registry YAML: {target}: {exc}") from exc
    if not isinstance(registry, dict):
        raise MakeBoardError(f"Malformed tag registry YAML: {target} must contain a mapping.")
    registry.setdefault("version", 1)
    tags = registry.setdefault("tags", {})
    if not isinstance(tags, dict):
        raise MakeBoardError(f"Malformed tag registry YAML: {target} tags must be a mapping.")
    return registry


def update_registry_entries(
    registry: dict[str, Any],
    entries: Iterable[Mapping[str, Any]],
    object_name: str,
    board_yaml_path: Union[Path, str],
) -> RegistryUpdate:
    """Update registry mappings without overwriting conflicting tag IDs."""

    tags = registry.setdefault("tags", {})
    requested_yaml = str(Path(board_yaml_path))
    updated = 0
    conflicts: list[RegistryConflict] = []

    for entry in entries:
        tag_id = str(int(entry["id"]))
        current = tags.get(tag_id)
        if current is None or current.get("yaml") == requested_yaml:
            tags[tag_id] = {"object": object_name, "yaml": requested_yaml}
            updated += 1
            continue

        conflicts.append(
            RegistryConflict(
                tag_id=tag_id,
                existing_yaml=str(current.get("yaml")),
                requested_yaml=requested_yaml,
            )
        )

    return RegistryUpdate(updated=updated, conflicts=tuple(conflicts))


def save_registry(path: Union[Path, str], registry: Mapping[str, Any]) -> None:
    """Write the tag registry with an updated UTC timestamp."""

    data = dict(registry)
    data["updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_yaml(Path(path), data)
