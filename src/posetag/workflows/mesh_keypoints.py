"""Workflow helpers for PoseTag mesh-keypoint generation.

The interactive mesh browser still lives in the legacy top-level
``gen_keypoints`` module.  These helpers keep the mesh-keypoint
object-geometry contract testable without opening OpenCV windows or requiring
camera hardware.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import yaml

SUPPORTED_ANNOTATION_MESH_EXTENSIONS = (".obj",)
DEFAULT_KEYPOINTS_FILENAME = "keypoints.json"
IDENTITY_T_MESH_OBJECT = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

_SIDE_RE = re.compile(r"^(?P<base>.+?)_side(?P<side>[A-Za-z][A-Za-z0-9_-]*)$")
_FACE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<face>[A-Za-z][A-Za-z0-9-]*)$")
_COMMON_FACE_SUFFIXES = frozenset(
    {
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
        "upper",
        "lower",
    }
)


class MeshKeypointWorkflowError(ValueError):
    """User-facing validation error for the mesh-keypoint workflow."""


@dataclass(frozen=True)
class KnownObject:
    """Object identity carried forward from earlier PoseTag workflow outputs."""

    object_name: str
    faces: tuple[str, ...]
    sources: tuple[str, ...]
    board_paths: tuple[Path, ...]
    captured_faces: tuple[str, ...]


@dataclass(frozen=True)
class ObjectIdentityResolution:
    """Resolved object identity for one mesh-keypoint generation request."""

    object_name: str
    mesh_path: Path
    output_path: Path
    inferred_from_mesh: bool
    known_objects: tuple[KnownObject, ...]


@dataclass(frozen=True)
class MeshKeypointObjectStatus:
    """Per-object mesh/keypoint status for the dashboard and tests."""

    object_name: str
    faces: tuple[str, ...]
    sources: tuple[str, ...]
    mesh_path: Path
    keypoints_path: Path
    mesh_exists: bool
    keypoints_exists: bool
    keypoints_valid: bool
    keypoints_errors: tuple[str, ...]
    command_preview: str


@dataclass(frozen=True)
class MeshVertexPreview:
    """Small, display-only sample of OBJ vertices for dashboard previews."""

    path: Path
    vertex_count: int
    sampled_vertices: tuple[tuple[float, float, float], ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


@dataclass(frozen=True)
class MeshKeypointGenerationResult:
    """Summary of one generated annotation-ready keypoint file."""

    object_name: str
    mesh_path: Path
    keypoints_path: Path
    object_config_path: Optional[Path]


@dataclass(frozen=True)
class MeshKeypointRemovalResult:
    """Summary of removed Stage 6 object-geometry artifacts."""

    object_name: str
    removed_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]


def supported_mesh_extensions() -> tuple[str, ...]:
    """Return mesh suffixes supported by the annotation-ready generator."""

    return SUPPORTED_ANNOTATION_MESH_EXTENSIONS


def expected_keypoints_path(
    project_root: Union[Path, str],
    object_name: str,
) -> Path:
    """Return the annotation-ready keypoint JSON path for an object."""

    return (
        Path(project_root).expanduser()
        / "objects"
        / validate_object_name(object_name)
        / DEFAULT_KEYPOINTS_FILENAME
    )


def default_mesh_path(project_root: Union[Path, str], object_name: str) -> Path:
    """Return the canonical OBJ mesh path for a PoseTag object."""

    return (
        Path(project_root).expanduser()
        / "meshes"
        / f"{validate_object_name(object_name)}.obj"
    )


def mesh_keypoint_command_for_object(
    project_root: Union[Path, str],
    object_name: str,
    *,
    mesh_path: Optional[Union[Path, str]] = None,
) -> str:
    """Return a copyable keypoint-generation command for one object."""

    root = Path(project_root).expanduser()
    name = validate_object_name(object_name)
    mesh = (
        Path(mesh_path).expanduser()
        if mesh_path is not None
        else default_mesh_path(root, name)
    )
    return shlex.join(
        (
            "posetag-gen-keypoints",
            "--project_root",
            str(root),
            "--mesh",
            str(_mesh_cli_argument(root, mesh)),
            "--object_name",
            name,
        )
    )


def resolve_mesh_path(project_root: Union[Path, str], mesh: Union[Path, str]) -> Path:
    """Resolve a mesh path relative to the project root when needed."""

    root = Path(project_root).expanduser()
    raw = Path(mesh).expanduser()
    return raw if raw.is_absolute() else root / raw


def _mesh_cli_argument(project_root: Path, mesh_path: Path) -> Path:
    """Return a CLI mesh argument compatible with ``--project_root``."""

    try:
        root_abs = project_root.resolve(strict=False)
        mesh_abs = mesh_path.resolve(strict=False)
        return mesh_abs.relative_to(root_abs)
    except ValueError:
        return mesh_abs
    except OSError:
        return mesh_path


def validate_mesh_path(project_root: Union[Path, str], mesh: Union[Path, str]) -> Path:
    """Resolve and validate a mesh path for annotation-ready keypoint output."""

    path = resolve_mesh_path(project_root, mesh)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_ANNOTATION_MESH_EXTENSIONS:
        joined = ", ".join(SUPPORTED_ANNOTATION_MESH_EXTENSIONS)
        raise MeshKeypointWorkflowError(
            f"Unsupported mesh extension {suffix!r} for {path}. "
            f"The annotation-ready keypoint generator currently supports: {joined}."
        )
    if not path.exists():
        raise MeshKeypointWorkflowError(f"Mesh file was not found: {path}")
    if not path.is_file():
        raise MeshKeypointWorkflowError(f"Mesh path is not a file: {path}")
    return path


def load_obj_vertex_preview(
    mesh_path: Union[Path, str],
    *,
    max_vertices: int = 5000,
) -> MeshVertexPreview:
    """Load a display-only OBJ vertex sample and full vertex bounds."""

    path = Path(mesh_path).expanduser()
    if path.suffix.lower() not in SUPPORTED_ANNOTATION_MESH_EXTENSIONS:
        raise MeshKeypointWorkflowError(
            f"Unsupported mesh extension {path.suffix!r} for preview."
        )
    if not path.is_file():
        raise MeshKeypointWorkflowError(f"Mesh file was not found: {path}")
    if max_vertices <= 0:
        raise MeshKeypointWorkflowError("max_vertices must be positive.")

    sampled: list[tuple[float, float, float]] = []
    bounds_min: Optional[list[float]] = None
    bounds_max: Optional[list[float]] = None
    vertex_count = 0

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    vertex = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                except ValueError:
                    continue
                if not all(_is_finite_number(value) for value in vertex):
                    raise MeshKeypointWorkflowError(
                        f"OBJ mesh contains non-finite vertex coordinates: {path}"
                    )
                vertex_count += 1
                if bounds_min is None or bounds_max is None:
                    bounds_min = [vertex[0], vertex[1], vertex[2]]
                    bounds_max = [vertex[0], vertex[1], vertex[2]]
                else:
                    for axis, value in enumerate(vertex):
                        bounds_min[axis] = min(bounds_min[axis], value)
                        bounds_max[axis] = max(bounds_max[axis], value)
                if len(sampled) < max_vertices:
                    sampled.append(vertex)
    except OSError as exc:
        raise MeshKeypointWorkflowError(f"Could not read OBJ mesh: {exc}") from exc

    if vertex_count == 0 or bounds_min is None or bounds_max is None:
        raise MeshKeypointWorkflowError(f"No OBJ vertices found in {path}")

    return MeshVertexPreview(
        path=path,
        vertex_count=vertex_count,
        sampled_vertices=tuple(sampled),
        bounds_min=tuple(bounds_min),
        bounds_max=tuple(bounds_max),
    )


def validate_object_name(object_name: str) -> str:
    """Validate a PoseTag object identity used under ``objects/<object>/``."""

    name = str(object_name).strip()
    if not name:
        raise MeshKeypointWorkflowError("Object name must not be empty.")
    if Path(name).name != name or "\\" in name:
        raise MeshKeypointWorkflowError(
            "Object name must be a plain PoseTag object identity, not a path."
        )
    return name


def split_object_face(name: str) -> tuple[str, Optional[str]]:
    """Split PoseTag object-face labels into base object and face label."""

    raw = str(name).strip()
    match = _SIDE_RE.match(raw)
    if match:
        side = match.group("side")
        return match.group("base"), f"side{side}"

    match = _FACE_SUFFIX_RE.match(raw)
    if match:
        face = match.group("face").lower()
        if face in _COMMON_FACE_SUFFIXES:
            return match.group("base"), face

    return raw, None


def infer_known_objects(project_root: Union[Path, str]) -> tuple[KnownObject, ...]:
    """Infer PoseTag object identities from boards, registry, and shots."""

    root = Path(project_root).expanduser()
    accum: dict[str, dict[str, set]] = {}

    def touch(
        full_name: str,
        *,
        source: str,
        board_path: Optional[Path] = None,
        captured: bool = False,
    ) -> None:
        if not str(full_name).strip():
            return
        base, face = split_object_face(str(full_name))
        if not base:
            return
        rec = accum.setdefault(
            base,
            {
                "faces": set(),
                "sources": set(),
                "board_paths": set(),
                "captured_faces": set(),
            },
        )
        rec["sources"].add(source)
        if face:
            rec["faces"].add(face)
            if captured:
                rec["captured_faces"].add(f"{base}_{face}")
        if board_path is not None:
            rec["board_paths"].add(board_path)

    for board_path in _iter_board_yaml_paths(root):
        data = _safe_load_yaml_mapping(board_path)
        if not data:
            continue
        object_name = data.get("object")
        tags = data.get("tags")
        if isinstance(object_name, str) and isinstance(tags, list):
            touch(object_name, source="boards/*.yaml", board_path=board_path)

    registry_path = root / "boards" / "tag_registry.yaml"
    registry = _safe_load_yaml_mapping(registry_path)
    tags = registry.get("tags") if registry else None
    if isinstance(tags, Mapping):
        for entry in tags.values():
            if not isinstance(entry, Mapping):
                continue
            object_name = entry.get("object")
            yaml_value = entry.get("yaml")
            board_path = None
            if isinstance(yaml_value, str) and yaml_value.strip():
                board_path = _resolve_registry_board_path(root, registry_path, yaml_value)
            if isinstance(object_name, str):
                touch(
                    object_name,
                    source="boards/tag_registry.yaml",
                    board_path=board_path,
                )

    manifest_path = root / "shots" / "manifest.csv"
    if manifest_path.exists():
        for row in _iter_manifest_rows(manifest_path):
            object_base = str(row.get("object_base", "")).strip()
            object_full = str(row.get("object_full", "")).strip()
            side = str(row.get("side", "")).strip()
            if object_base and side:
                side_label = side if side.startswith("side") else f"side{side}"
                object_full = object_full or f"{object_base}_{side_label}"
            if object_full:
                touch(object_full, source="shots/manifest.csv", captured=True)
            elif object_base:
                touch(object_base, source="shots/manifest.csv", captured=True)

    out: list[KnownObject] = []
    for name, rec in sorted(accum.items()):
        out.append(
            KnownObject(
                object_name=name,
                faces=tuple(sorted(rec["faces"])),
                sources=tuple(sorted(rec["sources"])),
                board_paths=tuple(sorted(Path(p) for p in rec["board_paths"])),
                captured_faces=tuple(sorted(rec["captured_faces"])),
            )
        )
    return tuple(out)


def inspect_keypoint_object_statuses(
    project_root: Union[Path, str],
) -> tuple[MeshKeypointObjectStatus, ...]:
    """Return expected mesh/keypoint status for inferred PoseTag objects."""

    root = Path(project_root).expanduser()
    statuses: list[MeshKeypointObjectStatus] = []
    for item in infer_known_objects(root):
        keypoints_path = expected_keypoints_path(root, item.object_name)
        mesh_path = _configured_mesh_path(root, item.object_name)
        keypoints_exists = keypoints_path.exists()
        keypoints_errors = (
            _read_keypoint_errors(
                keypoints_path,
                expected_faces=expected_face_keys(item.object_name, item.faces),
            )
            if keypoints_exists
            else ()
        )
        statuses.append(
            MeshKeypointObjectStatus(
                object_name=item.object_name,
                faces=item.faces,
                sources=item.sources,
                mesh_path=mesh_path,
                keypoints_path=keypoints_path,
                mesh_exists=mesh_path.is_file(),
                keypoints_exists=keypoints_exists,
                keypoints_valid=keypoints_exists and not keypoints_errors,
                keypoints_errors=keypoints_errors,
                command_preview=mesh_keypoint_command_for_object(
                    root,
                    item.object_name,
                    mesh_path=mesh_path,
                ),
            )
        )
    return tuple(statuses)


def mesh_keypoint_generation_candidates(
    project_root: Union[Path, str],
) -> tuple[MeshKeypointObjectStatus, ...]:
    """Return staged objects that can generate missing keypoints now."""

    return tuple(
        status
        for status in inspect_keypoint_object_statuses(project_root)
        if status.mesh_exists and not status.keypoints_exists
    )


def generate_missing_mesh_keypoints(
    project_root: Union[Path, str],
    *,
    units_to_m: float = 1.0,
) -> tuple[MeshKeypointGenerationResult, ...]:
    """Generate keypoints for staged meshes whose canonical JSON is absent.

    Existing ``keypoints.json`` files are never overwritten here. Invalid
    existing keypoint files should be repaired deliberately or regenerated via
    ``posetag-gen-keypoints --force`` so unit/schema changes remain explicit.
    """

    root = Path(project_root).expanduser()
    if not _is_finite_number(units_to_m) or float(units_to_m) <= 0.0:
        raise MeshKeypointWorkflowError("units_to_m must be finite and positive.")

    statuses = inspect_keypoint_object_statuses(root)
    candidates = tuple(
        status
        for status in statuses
        if status.mesh_exists and not status.keypoints_exists
    )
    if not candidates:
        invalid_existing = tuple(
            status.object_name
            for status in statuses
            if status.keypoints_exists and not status.keypoints_valid
        )
        if invalid_existing:
            names = ", ".join(invalid_existing)
            raise MeshKeypointWorkflowError(
                "Existing keypoints.json file(s) need attention and will not "
                f"be overwritten automatically: {names}. Repair them or rerun "
                "posetag-gen-keypoints with --force after confirming units."
            )
        missing_meshes = tuple(
            status.object_name
            for status in statuses
            if not status.mesh_exists and not status.keypoints_valid
        )
        if missing_meshes:
            joined = ", ".join(missing_meshes)
            raise MeshKeypointWorkflowError(
                "No keypoints can be generated until OBJ meshes are staged "
                f"for: {joined}."
            )
        return ()

    results: list[MeshKeypointGenerationResult] = []
    for status in candidates:
        results.append(_generate_keypoints_for_status(root, status, units_to_m))
    return tuple(results)


def generate_mesh_keypoints_for_object(
    project_root: Union[Path, str],
    object_name: str,
    *,
    units_to_m: float = 1.0,
) -> Optional[MeshKeypointGenerationResult]:
    """Generate keypoints for one staged object when its JSON is absent."""

    root = Path(project_root).expanduser()
    if not _is_finite_number(units_to_m) or float(units_to_m) <= 0.0:
        raise MeshKeypointWorkflowError("units_to_m must be finite and positive.")

    name = validate_object_name(object_name)
    status = next(
        (
            item
            for item in inspect_keypoint_object_statuses(root)
            if item.object_name == name
        ),
        None,
    )
    if status is None:
        raise MeshKeypointWorkflowError(f"Object is not inferred by Stage 6: {name}")
    if status.keypoints_exists:
        if status.keypoints_valid:
            return None
        raise MeshKeypointWorkflowError(
            f"Existing keypoints.json needs attention and will not be overwritten "
            f"automatically: {status.keypoints_path}. Repair it or rerun "
            "posetag-gen-keypoints with --force after confirming units."
        )
    if not status.mesh_exists:
        raise MeshKeypointWorkflowError(
            f"No staged OBJ mesh is available for {name}: {status.mesh_path}"
        )
    return _generate_keypoints_for_status(root, status, units_to_m)


def remove_mesh_keypoint_object_artifacts(
    project_root: Union[Path, str],
    object_name: str,
) -> MeshKeypointRemovalResult:
    """Remove Stage 6 mesh/keypoint artifacts for one inferred object.

    This resets object geometry only. It does not remove board definitions,
    face shots, tag registry entries, or any upstream source that inferred the
    object identity.
    """

    root = Path(project_root).expanduser()
    name = validate_object_name(object_name)
    status = next(
        (
            item
            for item in inspect_keypoint_object_statuses(root)
            if item.object_name == name
        ),
        None,
    )
    if status is None:
        raise MeshKeypointWorkflowError(f"Object is not inferred by Stage 6: {name}")

    object_dir = root / "objects" / name
    config_path = object_dir / "object_config.yaml"
    candidates = (
        status.keypoints_path,
        config_path,
        status.mesh_path,
    )
    removed: list[Path] = []
    skipped: list[Path] = []
    project_root_abs = root.resolve(strict=False)

    for path in dict.fromkeys(candidates):
        if not path.exists():
            continue
        if path == status.mesh_path and not _is_within(path, project_root_abs):
            skipped.append(path)
            continue
        if not path.is_file():
            skipped.append(path)
            continue
        try:
            path.unlink()
        except OSError as exc:
            raise MeshKeypointWorkflowError(
                f"Could not remove Stage 6 artifact {path}: {exc}"
            ) from exc
        removed.append(path)

    for directory in (object_dir, root / "objects", root / "meshes"):
        try:
            directory.rmdir()
        except OSError:
            pass

    return MeshKeypointRemovalResult(
        object_name=name,
        removed_paths=tuple(removed),
        skipped_paths=tuple(skipped),
    )


def _generate_keypoints_for_status(
    root: Path,
    status: MeshKeypointObjectStatus,
    units_to_m: float,
) -> MeshKeypointGenerationResult:
    from gen_keypoints import generate_keypoints_for_mesh

    mesh_path = _mesh_generation_path(root, status.mesh_path)
    generated = generate_keypoints_for_mesh(
        root,
        mesh_path,
        object_name=status.object_name,
        units_to_m=float(units_to_m),
    )
    object_config = generated.get("object_config_path")
    return MeshKeypointGenerationResult(
        object_name=str(generated["object_name"]),
        mesh_path=Path(generated["mesh_path"]),
        keypoints_path=Path(generated["keypoints_path"]),
        object_config_path=(Path(object_config) if object_config else None),
    )


def _mesh_generation_path(project_root: Path, mesh_path: Path) -> Path:
    """Return an existing mesh path without prefixing project root twice."""

    path = Path(mesh_path).expanduser()
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    candidate = project_root / path
    return candidate.resolve() if candidate.is_file() else path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent)
    except ValueError:
        return False
    return True


def _configured_mesh_path(root: Path, object_name: str) -> Path:
    default_path = default_mesh_path(root, object_name)
    config_path = root / "objects" / validate_object_name(object_name) / "object_config.yaml"
    config = _safe_load_yaml_mapping(config_path)
    mesh = config.get("mesh")
    if not isinstance(mesh, Mapping):
        return default_path
    mesh_path = mesh.get("path")
    if not isinstance(mesh_path, str) or not mesh_path.strip():
        return default_path
    raw = Path(mesh_path).expanduser()
    return raw if raw.is_absolute() else root / raw


def resolve_object_identity(
    project_root: Union[Path, str],
    mesh_path: Union[Path, str],
    *,
    object_name: Optional[str] = None,
) -> ObjectIdentityResolution:
    """Resolve the PoseTag object identity for a mesh.

    Mesh stems are accepted automatically only when they exactly match a known
    base object, or when no earlier PoseTag object identities are present yet.
    If prior workflow outputs identify objects and the mesh stem differs, the
    caller must pass ``object_name`` explicitly.
    """

    root = Path(project_root).expanduser()
    mesh = validate_mesh_path(root, mesh_path)
    known = infer_known_objects(root)
    known_names = {item.object_name for item in known}

    if object_name is not None:
        resolved_name = validate_object_name(object_name)
        inferred = False
    elif mesh.stem in known_names or not known_names:
        resolved_name = validate_object_name(mesh.stem)
        inferred = True
    else:
        joined = ", ".join(sorted(known_names))
        raise MeshKeypointWorkflowError(
            f"Mesh filename {mesh.name!r} does not match a known PoseTag object. "
            f"Known objects: {joined}. Pass --object_name to map this mesh "
            "explicitly."
        )

    return ObjectIdentityResolution(
        object_name=resolved_name,
        mesh_path=mesh,
        output_path=expected_keypoints_path(root, resolved_name),
        inferred_from_mesh=inferred,
        known_objects=known,
    )


def choose_keypoints_output_path(
    keypoints_path: Union[Path, str],
    *,
    force: bool = False,
    keep_both: bool = False,
) -> Path:
    """Return the path to write, enforcing explicit overwrite policy."""

    path = Path(keypoints_path)
    if force and keep_both:
        raise MeshKeypointWorkflowError("--force and --keep-both cannot be combined.")
    if not path.exists():
        return path
    if force:
        return path
    if keep_both:
        index = 2
        while True:
            candidate = path.with_name(f"{path.stem}_v{index}{path.suffix}")
            if not candidate.exists():
                return candidate
            index += 1
    raise MeshKeypointWorkflowError(
        f"Keypoint output already exists: {path}. "
        "Pass --force to overwrite or --keep-both to write a versioned copy."
    )


def validate_keypoint_payload(
    payload: Mapping[str, object],
    *,
    expected_faces: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return schema errors for ``objects/<object>/keypoints.json``."""

    errors: list[str] = []
    units = payload.get("units_to_m")
    if not _is_finite_number(units) or float(units) <= 0:
        errors.append(
            "keypoints.json field 'units_to_m' must be a finite positive number."
        )

    points = payload.get("points")
    if not isinstance(points, Mapping) or not points:
        errors.append("keypoints.json field 'points' must be a non-empty mapping.")
        points = {}
    else:
        for name, xyz in points.items():
            if not isinstance(name, str) or not name:
                errors.append("keypoint names must be non-empty strings.")
            if not _is_xyz(xyz):
                errors.append(
                    f"keypoint {name!r} must contain three finite numeric coordinates."
                )

    faces = payload.get("faces")
    if not isinstance(faces, Mapping) or not faces:
        errors.append("keypoints.json field 'faces' must be a non-empty mapping.")
    else:
        point_names = set(points.keys()) if isinstance(points, Mapping) else set()
        for face_name, corners in faces.items():
            if not isinstance(face_name, str) or not face_name:
                errors.append("face names must be non-empty strings.")
            if (
                not isinstance(corners, list)
                or len(corners) != 4
                or any(not isinstance(item, str) for item in corners)
            ):
                errors.append(f"face {face_name!r} must list exactly four keypoint names.")
                continue
            missing = [name for name in corners if name not in point_names]
            if missing:
                errors.append(
                    f"face {face_name!r} references missing keypoint(s): "
                    + ", ".join(missing)
                )
        expected = tuple(dict.fromkeys(str(face) for face in expected_faces if str(face)))
        missing_faces = [face for face in expected if face not in faces]
        if missing_faces:
            errors.append(
                "keypoints.json field 'faces' is missing expected face(s): "
                + ", ".join(missing_faces)
            )

    return tuple(errors)


def expected_face_keys(
    object_name: str,
    faces: Iterable[str],
) -> tuple[str, ...]:
    """Return full keypoint face keys expected for an inferred object."""

    name = validate_object_name(object_name)
    keys: list[str] = []
    for face in faces:
        label = str(face).strip()
        if not label:
            continue
        key = label if label.startswith(f"{name}_") else f"{name}_{label}"
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def object_config_payload(
    project_root: Union[Path, str],
    mesh_path: Union[Path, str],
    *,
    units_to_m: float,
    facespec: Iterable[str],
) -> dict[str, object]:
    """Build ``object_config.yaml`` metadata for a generated keypoint file."""

    root = Path(project_root).expanduser()
    mesh = Path(mesh_path).expanduser()
    return {
        "mesh": {
            "path": _relative_display_path(root, mesh),
            "units_to_m": float(units_to_m),
            "T_mesh_object": {
                "matrix": [list(row) for row in IDENTITY_T_MESH_OBJECT],
            },
        },
        "notes": "Sides auto-mapped: A,B,C,D = " + ", ".join(facespec),
    }


def format_known_objects(objects: Iterable[KnownObject]) -> str:
    """Return a readable inventory of known objects and faces."""

    lines: list[str] = []
    for item in objects:
        faces = ", ".join(item.faces) if item.faces else "(no explicit faces)"
        sources = ", ".join(item.sources) if item.sources else "(unknown source)"
        lines.append(f"{item.object_name}: faces={faces}; sources={sources}")
    return "\n".join(lines) if lines else "No known objects found yet."


def _iter_board_yaml_paths(root: Path) -> Iterable[Path]:
    boards_dir = root / "boards"
    if not boards_dir.exists():
        return ()
    skipped = {"tag_registry.yaml", "board_building_queue.yaml"}
    return tuple(
        sorted(
            path
            for path in boards_dir.glob("*.yaml")
            if path.name not in skipped and path.is_file()
        )
    )


def _safe_load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_registry_board_path(root: Path, registry_path: Path, yaml_value: str) -> Path:
    raw = Path(yaml_value).expanduser()
    if raw.is_absolute():
        return raw
    candidates = [root / raw, registry_path.parent / raw, raw]
    if raw.parts and raw.parts[0] == root.name:
        candidates.insert(2, root.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _iter_manifest_rows(path: Path) -> Iterable[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    except OSError:
        return


def _read_keypoint_errors(
    path: Path,
    *,
    expected_faces: Iterable[str] = (),
) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return (f"Could not read keypoints JSON: {exc}",)
    if not isinstance(payload, Mapping):
        return ("keypoints.json must contain a mapping.",)
    return validate_keypoint_payload(payload, expected_faces=expected_faces)


def _is_xyz(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    return all(_is_finite_number(item) for item in value)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _relative_display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
