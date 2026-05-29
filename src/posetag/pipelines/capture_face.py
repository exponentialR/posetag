"""Reusable helpers for PoseTag Step 3 face-shot capture.

The interactive OpenCV viewer remains in the legacy top-level
``capture_face`` module for now.  The helpers here cover validation, path
resolution, registered-face loading, metadata construction, and manifest
writing so Step 3 can be tested without camera hardware.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import yaml

from posetag.pipelines.make_board import (
    MakeBoardError,
    load_calibration_yaml,
    preview_project_root,
)

try:
    from posetag.utils.project_config import ensure_project_dirs, resolve_project_root
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from posetag.utils.project_config import ensure_project_dirs, resolve_project_root


SIDE_RE = re.compile(r"^(?P<base>.+?)_side(?P<side>[A-Za-z]+)$")

CAPTURE_MANIFEST_FIELDS = (
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
)


class CaptureFaceError(ValueError):
    """User-facing validation error for Step 3 face-shot capture."""


@dataclass(frozen=True)
class CaptureCalibration:
    """Loaded colour-camera intrinsics used for capture metadata."""

    path: Path
    camera_params: tuple[float, float, float, float]


@dataclass(frozen=True)
class CaptureFacePaths:
    """Resolved Step 3 project and output paths."""

    project_root: Path
    calib_path: Path
    registry_path: Path
    out_dir: Path
    manifest_path: Path
    log_path: Path
    raw_dir: Optional[Path] = None
    ann_dir: Optional[Path] = None
    meta_dir: Optional[Path] = None


@dataclass(frozen=True)
class CaptureSelection:
    """Registered faces selected from an optional CLI object name."""

    object_base: str
    faces: tuple[dict[str, Any], ...]
    auto_side: bool


@dataclass(frozen=True)
class ShotPaths:
    """Output paths for one saved face-shot capture."""

    raw_path: Path
    ann_path: Path
    meta_path: Path
    object_base: str
    side_code: str
    file_stub: str


def preview_capture_project_root(project_root: Optional[Union[Path, str]]) -> Path:
    """Resolve the project root for validation without creating explicit roots."""

    return preview_project_root(project_root)


def activate_capture_project_root(project_root: Union[Path, str]) -> Path:
    """Register and create the project layout once validation has passed."""

    root = Path(resolve_project_root(project_root))
    ensure_project_dirs(root)
    return root


def resolve_under_project(
    project_root: Union[Path, str],
    value: Optional[Union[Path, str]],
    default: Union[Path, str],
) -> Path:
    """Resolve a path under the project root unless it is already absolute."""

    root = Path(project_root).expanduser().resolve()
    raw = Path(default if value is None else value).expanduser()
    return raw if raw.is_absolute() else root / raw


def resolve_capture_calibration_path(
    project_root: Union[Path, str],
    calib: Optional[Union[Path, str]] = "calib_color.yaml",
) -> Path:
    """Resolve Step 3 calibration input using the legacy project-aware order."""

    root = Path(project_root).expanduser().resolve()
    raw = Path(calib or "calib_color.yaml").expanduser()
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw

    candidates = (root / "calib" / raw, root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def prepare_capture_paths(
    *,
    project_root: Union[Path, str],
    calib_path: Union[Path, str],
    registry: Optional[Union[Path, str]] = None,
    out_dir: Optional[Union[Path, str]] = None,
    manifest: Optional[Union[Path, str]] = None,
    log_file: Optional[Union[Path, str]] = None,
    layout: str = "by_object_side",
    raw_dir: Optional[Union[Path, str]] = None,
    ann_dir: Optional[Union[Path, str]] = None,
    meta_dir: Optional[Union[Path, str]] = None,
) -> CaptureFacePaths:
    """Resolve Step 3 input/output paths without creating directories."""

    root = Path(project_root).expanduser().resolve()
    resolved_out_dir = resolve_under_project(root, out_dir, "shots")
    registry_path = resolve_under_project(root, registry, "boards/tag_registry.yaml")
    manifest_path = resolve_under_project(root, manifest, "shots/manifest.csv")

    logs_dir = root / "logs"
    if log_file is None:
        log_path = logs_dir / "capture_face.log"
    else:
        raw_log = Path(log_file).expanduser()
        log_path = raw_log if raw_log.is_absolute() else logs_dir / raw_log.name

    if layout == "split_type":
        resolved_raw_dir = resolve_under_project(root, raw_dir, resolved_out_dir / "images")
        resolved_ann_dir = resolve_under_project(root, ann_dir, resolved_out_dir / "ann")
        resolved_meta_dir = resolve_under_project(root, meta_dir, resolved_out_dir / "meta")
    else:
        resolved_raw_dir = resolved_ann_dir = resolved_meta_dir = None

    return CaptureFacePaths(
        project_root=root,
        calib_path=Path(calib_path).expanduser(),
        registry_path=registry_path,
        out_dir=resolved_out_dir,
        manifest_path=manifest_path,
        log_path=log_path,
        raw_dir=resolved_raw_dir,
        ann_dir=resolved_ann_dir,
        meta_dir=resolved_meta_dir,
    )


def create_capture_output_dirs(paths: CaptureFacePaths, *, layout: str) -> None:
    """Create the standard project and Step 3 output directories."""

    ensure_project_dirs(paths.project_root)
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if layout == "split_type":
        for directory in (paths.raw_dir, paths.ann_dir, paths.meta_dir):
            if directory is None:
                raise CaptureFaceError("split_type layout requires raw, annotated, and metadata directories.")
            directory.mkdir(parents=True, exist_ok=True)
    else:
        paths.out_dir.mkdir(parents=True, exist_ok=True)


def load_capture_calibration(path: Union[Path, str]) -> CaptureCalibration:
    """Load and validate the Step 1 colour-camera calibration YAML."""

    try:
        calibration = load_calibration_yaml(path)
    except MakeBoardError as exc:
        raise CaptureFaceError(str(exc)) from exc
    return CaptureCalibration(
        path=Path(path).expanduser(),
        camera_params=calibration.camera_params,
    )


def validate_capture_source_args(
    source: str,
    video: Optional[Union[Path, str]],
    realsense_module: Any,
) -> None:
    """Validate source-specific arguments before opening hardware."""

    if source == "video":
        if not video:
            raise CaptureFaceError("--video path is required when --source=video")
        if not Path(video).expanduser().exists():
            raise CaptureFaceError(f"Could not open video: {video}")
    if source == "realsense" and realsense_module is None:
        raise CaptureFaceError(
            "pyrealsense2 is not available; install PoseTag with the 'realsense' "
            "extra or use --source opencv|video"
        )


def load_apriltag_detector_class() -> Any:
    """Load the optional pupil-apriltags detector class."""

    try:
        from pupil_apriltags import Detector
    except Exception as exc:
        raise CaptureFaceError(
            "pupil-apriltags is not available; install PoseTag with the "
            "'apriltags' extra or run `python -m pip install pupil-apriltags`."
        ) from exc
    return Detector


def create_apriltag_detector(family: str, detector_class: Optional[Any] = None) -> Any:
    """Create the AprilTag detector used by face-shot capture."""

    Detector = detector_class or load_apriltag_detector_class()
    try:
        return Detector(
            families=family,
            nthreads=4,
            quad_decimate=1.0,
            refine_edges=True,
        )
    except Exception as exc:
        raise CaptureFaceError(
            f"Could not initialize AprilTag detector for family '{family}': {exc}"
        ) from exc


def load_capture_registry(path: Union[Path, str]) -> dict[str, Any]:
    """Load and validate the Step 2 tag registry YAML."""

    target = Path(path).expanduser()
    if not target.exists():
        raise CaptureFaceError(
            f"Tag registry YAML not found: {target}. Run Step 2 with posetag-make-board first."
        )
    try:
        with target.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise CaptureFaceError(f"Malformed tag registry YAML: {target}: {exc}") from exc
    except OSError as exc:
        raise CaptureFaceError(f"Could not read tag registry YAML: {target}: {exc}") from exc

    if not isinstance(registry, Mapping):
        raise CaptureFaceError(f"Malformed tag registry YAML: {target} must contain a mapping.")
    tags = registry.get("tags")
    if not isinstance(tags, Mapping):
        raise CaptureFaceError(f"Malformed tag registry YAML: {target} tags must be a mapping.")
    if not tags:
        raise CaptureFaceError(
            f"Tag registry YAML has no tag mappings: {target}. Run Step 2 with posetag-make-board first."
        )
    return dict(registry)


def load_registered_faces(
    registry: Mapping[str, Any],
    *,
    project_root: Union[Path, str],
    registry_path: Union[Path, str],
) -> tuple[dict[str, Any], ...]:
    """Resolve registry entries into readable registered face records."""

    root = Path(project_root).expanduser().resolve()
    reg_path = Path(registry_path).expanduser()
    tags = registry.get("tags")
    if not isinstance(tags, Mapping) or not tags:
        raise CaptureFaceError("Tag registry YAML must contain at least one tag mapping.")

    board_cache: dict[Path, tuple[str, frozenset[int]]] = {}
    grouped: dict[tuple[Path, str], set[int]] = {}

    for raw_tag_id, raw_entry in sorted(tags.items(), key=lambda item: str(item[0])):
        tag_id = _parse_int(raw_tag_id)
        if tag_id is None:
            raise CaptureFaceError(f"Registry tag key {raw_tag_id!r} must be an integer tag ID.")
        if not isinstance(raw_entry, Mapping):
            raise CaptureFaceError(f"Registry entry {raw_tag_id!r} must be a mapping.")

        object_name = raw_entry.get("object")
        if not isinstance(object_name, str) or not object_name.strip():
            raise CaptureFaceError(f"Registry entry {raw_tag_id!r} is missing an object name.")

        yaml_value = raw_entry.get("yaml")
        if not isinstance(yaml_value, str) or not yaml_value.strip():
            raise CaptureFaceError(f"Registry entry {raw_tag_id!r} is missing a board YAML path.")

        board_path = resolve_registry_board_path(root, reg_path, yaml_value)
        if not board_path.exists():
            raise CaptureFaceError(
                f"Referenced board YAML for tag {raw_tag_id!r} was not found: {board_path}"
            )

        cached = board_cache.get(board_path)
        if cached is None:
            board_object, board_tag_ids = load_board_face_schema(board_path)
            board_cache[board_path] = (board_object, board_tag_ids)
        else:
            board_object, board_tag_ids = cached

        if object_name != board_object:
            raise CaptureFaceError(
                f"Registry entry {raw_tag_id!r} maps object {object_name!r}, "
                f"but {board_path} declares object {board_object!r}."
            )
        if tag_id not in board_tag_ids:
            raise CaptureFaceError(
                f"Registry tag {raw_tag_id!r} is not present in referenced board YAML: {board_path}"
            )

        grouped.setdefault((board_path, object_name), set()).add(tag_id)

    faces = [
        {
            "yaml": str(board_path),
            "object": object_name,
            "tag_ids": tuple(sorted(tag_ids)),
        }
        for (board_path, object_name), tag_ids in grouped.items()
    ]
    faces.sort(key=lambda face: (str(face["object"]), Path(str(face["yaml"])).name))
    if not faces:
        raise CaptureFaceError("Tag registry YAML did not resolve to any readable board faces.")
    return tuple(faces)


def resolve_registry_board_path(
    project_root: Union[Path, str],
    registry_path: Union[Path, str],
    yaml_value: str,
) -> Path:
    """Resolve a board YAML reference stored in the tag registry."""

    root = Path(project_root).expanduser().resolve()
    registry_parent = Path(registry_path).expanduser().parent
    raw = Path(yaml_value).expanduser()
    if raw.is_absolute():
        return raw

    candidates = [
        root / raw,
        registry_parent / raw,
    ]
    if raw.parts and raw.parts[0] == root.name:
        candidates.append(root.parent / raw)
    candidates.append(raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_board_face_schema(path: Union[Path, str]) -> tuple[str, frozenset[int]]:
    """Load the object name and tag IDs from a Step 2 board YAML."""

    target = Path(path).expanduser()
    try:
        with target.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CaptureFaceError(f"Malformed board YAML: {target}: {exc}") from exc
    except OSError as exc:
        raise CaptureFaceError(f"Could not read board YAML: {target}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise CaptureFaceError(f"Malformed board YAML: {target} must contain a mapping.")
    object_name = data.get("object")
    if not isinstance(object_name, str) or not object_name.strip():
        raise CaptureFaceError(f"Malformed board YAML: {target} is missing object.")
    tags = data.get("tags")
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)) or not tags:
        raise CaptureFaceError(f"Malformed board YAML: {target} tags must be a non-empty list.")

    tag_ids: set[int] = set()
    for index, tag in enumerate(tags):
        if not isinstance(tag, Mapping):
            raise CaptureFaceError(f"Malformed board YAML: {target} tag entry {index} must be a mapping.")
        tag_id = _parse_int(tag.get("id"))
        if tag_id is None:
            raise CaptureFaceError(
                f"Malformed board YAML: {target} tag entry {index} id must be an integer."
            )
        tag_ids.add(tag_id)
    return object_name, frozenset(tag_ids)


def parse_base_and_side(name: str) -> tuple[str, Optional[str]]:
    """Split ``object_sideA`` style names into base object and side label."""

    match = SIDE_RE.match(name)
    if match:
        return match.group("base"), match.group("side")
    return name, None


def unique_bases(faces: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return stable base object names from registered face records."""

    bases = {
        parse_base_and_side(str(face.get("object", "")))[0]
        for face in faces
        if str(face.get("object", "")).strip()
    }
    return sorted(bases)


def faces_for_base(base: str, faces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return registered faces for one base object."""

    selected = [
        dict(face)
        for face in faces
        if parse_base_and_side(str(face.get("object", "")))[0] == base
    ]
    selected.sort(key=lambda face: (str(face["object"]), Path(str(face["yaml"])).name))
    return selected


def faces_for_object(object_name: str, faces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return registered faces for one full object/face name."""

    selected = [dict(face) for face in faces if str(face.get("object")) == object_name]
    selected.sort(key=lambda face: (str(face["object"]), Path(str(face["yaml"])).name))
    return selected


def select_initial_faces(
    object_name: Optional[str],
    faces: Sequence[Mapping[str, Any]],
) -> Optional[CaptureSelection]:
    """Resolve an optional CLI object selection against registered faces."""

    if object_name is None:
        if not faces:
            raise CaptureFaceError("No registered object faces were found in the tag registry.")
        return None

    requested = object_name.strip()
    if not requested:
        raise CaptureFaceError("--object_name must not be empty when supplied.")

    base, side = parse_base_and_side(requested)
    if side is None:
        selected = faces_for_base(base, faces)
        auto_side = True
    else:
        selected = faces_for_object(requested, faces)
        auto_side = False

    if not selected:
        raise CaptureFaceError(
            f"No registered face found for --object_name {requested!r}. "
            "Run Step 2 with posetag-make-board and confirm boards/tag_registry.yaml."
        )
    return CaptureSelection(
        object_base=base,
        faces=tuple(selected),
        auto_side=auto_side,
    )


def capture_timestamp() -> str:
    """Return the timestamp format used in Step 3 output filenames."""

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_shot_paths(
    *,
    layout: str,
    out_dir: Union[Path, str],
    object_full: str,
    timestamp: str,
    raw_dir: Optional[Union[Path, str]] = None,
    ann_dir: Optional[Union[Path, str]] = None,
    meta_dir: Optional[Union[Path, str]] = None,
) -> ShotPaths:
    """Build deterministic raw/annotated/meta paths for one saved shot."""

    base_name, side = parse_base_and_side(object_full)
    side_code = (side or "unresolved").upper()

    if layout == "flat":
        save_root = Path(out_dir)
        file_stub = f"{object_full}_{timestamp}"
        raw_path = save_root / f"{file_stub}_raw.png"
        ann_path = save_root / f"{file_stub}_ann.png"
        meta_path = save_root / f"{file_stub}_meta.json"
    elif layout == "by_object":
        save_root = Path(out_dir) / base_name
        file_stub = f"{base_name}_side{side_code}_{timestamp}"
        raw_path = save_root / f"{file_stub}_raw.png"
        ann_path = save_root / f"{file_stub}_ann.png"
        meta_path = save_root / f"{file_stub}_meta.json"
    elif layout == "by_object_side":
        save_root = Path(out_dir) / base_name / f"side{side_code}"
        file_stub = f"{base_name}_side{side_code}_{timestamp}"
        raw_path = save_root / f"{file_stub}_raw.png"
        ann_path = save_root / f"{file_stub}_ann.png"
        meta_path = save_root / f"{file_stub}_meta.json"
    elif layout == "split_type":
        if raw_dir is None or ann_dir is None or meta_dir is None:
            raise CaptureFaceError("split_type layout requires raw_dir, ann_dir, and meta_dir.")
        file_stub = f"{base_name}_side{side_code}_{timestamp}"
        raw_path = Path(raw_dir) / f"{file_stub}_raw.png"
        ann_path = Path(ann_dir) / f"{file_stub}_ann.png"
        meta_path = Path(meta_dir) / f"{file_stub}_meta.json"
    else:
        raise CaptureFaceError("Capture layout must be flat, by_object, by_object_side, or split_type.")

    return ShotPaths(
        raw_path=raw_path,
        ann_path=ann_path,
        meta_path=meta_path,
        object_base=base_name,
        side_code=side_code,
        file_stub=file_stub,
    )


def build_capture_metadata(
    *,
    face: Mapping[str, Any],
    shot_paths: ShotPaths,
    detected_tag_ids: Iterable[int],
    validation_ok: bool,
    auto_face: bool,
    frame_shape: Sequence[int],
    camera_params: Sequence[float],
    timestamp: str,
) -> dict[str, Any]:
    """Build the public per-shot metadata JSON schema."""

    fx, fy, cx, cy = [float(value) for value in camera_params]
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    expected_ids = sorted({int(tag_id) for tag_id in face.get("tag_ids", [])})
    detected_ids = sorted({int(tag_id) for tag_id in detected_tag_ids})

    return {
        "object_base": shot_paths.object_base,
        "object_full": str(face.get("object")),
        "side": shot_paths.side_code,
        "face_yaml": str(face.get("yaml")) if face.get("yaml") else None,
        "expected_tag_ids": expected_ids,
        "detected_tag_ids": detected_ids,
        "validation_ok": bool(validation_ok),
        "auto_face": bool(auto_face),
        "image": {
            "path_raw": str(shot_paths.raw_path),
            "path_ann": str(shot_paths.ann_path),
            "path_meta": str(shot_paths.meta_path),
            "width": width,
            "height": height,
        },
        "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "timestamp": timestamp,
    }


def write_capture_outputs(
    *,
    raw_frame: Any,
    annotated_frame: Any,
    metadata: Mapping[str, Any],
    shot_paths: ShotPaths,
    manifest_path: Union[Path, str],
    image_writer: Callable[[str, Any], bool],
) -> None:
    """Write raw/annotated images, metadata JSON, and one manifest row."""

    for path, image in (
        (shot_paths.raw_path, raw_frame),
        (shot_paths.ann_path, annotated_frame),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not image_writer(str(path), image):
            raise CaptureFaceError(f"Could not write capture image: {path}")

    shot_paths.meta_path.parent.mkdir(parents=True, exist_ok=True)
    with shot_paths.meta_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2)
        handle.write("\n")

    append_capture_manifest(manifest_path, metadata)


def append_capture_manifest(
    manifest_path: Union[Path, str],
    metadata: Mapping[str, Any],
) -> None:
    """Append one face-shot metadata row to the Step 3 CSV manifest."""

    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = metadata.get("image", {})
    camera = metadata.get("camera", {})
    row = {
        "timestamp": metadata.get("timestamp", ""),
        "object_base": metadata.get("object_base", ""),
        "object_full": metadata.get("object_full", ""),
        "side": metadata.get("side", ""),
        "face_yaml": metadata.get("face_yaml", "") or "",
        "path_raw": image.get("path_raw", ""),
        "path_ann": image.get("path_ann", ""),
        "path_meta": image.get("path_meta", ""),
        "width": image.get("width", 0),
        "height": image.get("height", 0),
        "fx": camera.get("fx", 0.0),
        "fy": camera.get("fy", 0.0),
        "cx": camera.get("cx", 0.0),
        "cy": camera.get("cy", 0.0),
        "detected_ids": " ".join(map(str, metadata.get("detected_tag_ids", []))),
        "expected_ids": " ".join(map(str, metadata.get("expected_tag_ids", []))),
        "validation_ok": int(bool(metadata.get("validation_ok", False))),
        "auto_face": int(bool(metadata.get("auto_face", False))),
    }

    write_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAPTURE_MANIFEST_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def validate_capture_metadata_schema(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Return schema errors for one Step 3 metadata JSON mapping."""

    errors: list[str] = []
    for field in (
        "object_base",
        "object_full",
        "side",
        "face_yaml",
        "expected_tag_ids",
        "detected_tag_ids",
        "validation_ok",
        "auto_face",
        "image",
        "camera",
        "timestamp",
    ):
        if field not in metadata:
            errors.append(f"Capture metadata is missing field {field!r}.")

    for field in ("object_base", "object_full", "side", "timestamp"):
        if field in metadata and (
            not isinstance(metadata.get(field), str) or not str(metadata[field]).strip()
        ):
            errors.append(f"Capture metadata field {field!r} must be a non-empty string.")

    for field in ("expected_tag_ids", "detected_tag_ids"):
        value = metadata.get(field)
        if not isinstance(value, list) or any(_parse_int(item) is None for item in value):
            errors.append(f"Capture metadata field {field!r} must be a list of integer tag IDs.")

    for field in ("validation_ok", "auto_face"):
        if field in metadata and not isinstance(metadata.get(field), bool):
            errors.append(f"Capture metadata field {field!r} must be boolean.")

    image = metadata.get("image")
    if not isinstance(image, Mapping):
        errors.append("Capture metadata field 'image' must be a mapping.")
    else:
        for field in ("path_raw", "path_ann", "path_meta"):
            if not isinstance(image.get(field), str) or not image[field].strip():
                errors.append(f"Capture metadata image field {field!r} must be a non-empty string.")
        for field in ("width", "height"):
            parsed = _parse_int(image.get(field))
            if parsed is None or parsed <= 0:
                errors.append(f"Capture metadata image field {field!r} must be a positive integer.")

    camera = metadata.get("camera")
    if not isinstance(camera, Mapping):
        errors.append("Capture metadata field 'camera' must be a mapping.")
    else:
        for field in ("fx", "fy", "cx", "cy"):
            value = camera.get(field)
            if isinstance(value, bool):
                errors.append(f"Capture metadata camera field {field!r} must be numeric.")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"Capture metadata camera field {field!r} must be numeric.")
                continue
            if not math.isfinite(numeric):
                errors.append(f"Capture metadata camera field {field!r} must be finite.")

    return tuple(errors)


def _parse_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return parsed
