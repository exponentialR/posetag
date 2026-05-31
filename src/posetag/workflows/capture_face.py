"""GUI-independent helpers for guided face-shot capture.

This module validates Stage 5 dashboard values, builds copyable
``posetag-capture-face`` commands, and inspects saved face-shot coverage.
Camera capture, AprilTag detection, overlays, metadata writing, and manifest
updates remain in the existing ``posetag-capture-face`` workflow.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from posetag.pipelines.capture_face import (
    CaptureFaceError,
    CaptureFacePaths,
    load_capture_calibration,
    load_capture_registry,
    load_registered_faces,
    parse_base_and_side,
    prepare_capture_paths,
    resolve_capture_calibration_path,
    select_initial_faces,
    unique_bases,
    validate_capture_metadata_schema,
    validate_capture_source_args,
)


SOURCE_OPENCV = "opencv"
SOURCE_REALSENSE = "realsense"
SOURCE_VIDEO = "video"
CAPTURE_SOURCE_CHOICES = (SOURCE_OPENCV, SOURCE_REALSENSE, SOURCE_VIDEO)
CAPTURE_SOURCE_LABELS = {
    SOURCE_OPENCV: "webcam/OpenCV",
    SOURCE_REALSENSE: "RealSense",
    SOURCE_VIDEO: "video",
}
LAYOUT_CHOICES = ("flat", "by_object", "by_object_side", "split_type")
DEFAULT_FAMILY = "tag36h11"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_MIN_EXPECTED = 1
DEFAULT_PANEL_WIDTH = 420
DEFAULT_RECENT_WIDTH = 320
CAPTURE_FACE_GUIDANCE = (
    "Run Capture Face starts the existing posetag-capture-face workflow. Use "
    "the OpenCV preview to select an object/face, press ENTER to save, and "
    "press q or ESC to quit cleanly without writing a new shot."
)
CAPTURE_FACE_PROCESS_NOT_STARTED = "not_started"
CAPTURE_FACE_PROCESS_RUNNING = "running"
CAPTURE_FACE_PROCESS_FINISHED = "finished"
CAPTURE_FACE_PROCESS_FAILED_CANCELLED = "failed_cancelled"


class CaptureFaceWorkflowError(ValueError):
    """User-facing validation error for guided face-shot capture."""


@dataclass(frozen=True)
class CaptureFaceConfig:
    """User-facing face-shot capture command state."""

    project_root: Union[Path, str]
    object_name: str = ""
    family: str = DEFAULT_FAMILY
    calibration_path: Optional[Union[Path, str]] = None
    registry_path: Optional[Union[Path, str]] = None
    source: str = SOURCE_OPENCV
    camera_index: int = DEFAULT_CAMERA_INDEX
    video_path: Optional[Union[Path, str]] = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    min_expected: int = DEFAULT_MIN_EXPECTED
    layout: str = "by_object_side"
    out_dir: Optional[Union[Path, str]] = None
    manifest_path: Optional[Union[Path, str]] = None
    raw_dir: Optional[Union[Path, str]] = None
    ann_dir: Optional[Union[Path, str]] = None
    meta_dir: Optional[Union[Path, str]] = None
    panel_width: int = DEFAULT_PANEL_WIDTH
    recent_width: int = DEFAULT_RECENT_WIDTH
    log_file: Optional[Union[Path, str]] = None


@dataclass(frozen=True)
class CaptureFaceOutputStatus:
    """Expected Stage 5 paths and current face-shot coverage state."""

    registry_path: Path
    manifest_path: Path
    out_dir: Path
    checked_paths: tuple[Path, ...]
    registered_faces: tuple[str, ...]
    covered_faces: tuple[str, ...]
    missing_faces: tuple[str, ...]
    valid_shot_count: int
    invalid_shot_count: int
    manifest_exists: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def registered_face_count(self) -> int:
        """Return the number of board faces expected by the registry."""

        return len(self.registered_faces)

    @property
    def covered_face_count(self) -> int:
        """Return the number of registered faces with at least one valid shot."""

        return len(self.covered_faces)

    @property
    def complete(self) -> bool:
        """Return true when every registered face has a valid saved shot."""

        return (
            not self.errors
            and self.manifest_exists
            and bool(self.registered_faces)
            and not self.missing_faces
        )


@dataclass(frozen=True)
class CaptureFaceReadiness:
    """Display-ready readiness state for guided face-shot capture."""

    ready: bool
    command_preview: str
    calibration_path: Path
    paths: CaptureFacePaths
    output_status: CaptureFaceOutputStatus
    registered_bases: tuple[str, ...]
    registered_faces: tuple[str, ...]
    selected_faces: tuple[str, ...]
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def expected_manifest(self) -> Path:
        """Return the manifest path expected from the current form values."""

        return self.paths.manifest_path


@dataclass(frozen=True)
class CaptureFaceLaunchSpec:
    """Editable-install-safe process launch details for face capture."""

    program: str
    arguments: tuple[str, ...]
    display_command: str
    expected_manifest: Path


@dataclass(frozen=True)
class CaptureFaceProcessState:
    """GUI-independent face-shot capture child-process state."""

    state: str
    label: str
    message: str
    running: bool = False
    success: bool = False
    expected_manifest: Optional[Path] = None
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class _ValidatedCaptureFaceInputs:
    object_name: str
    family: str
    source: str
    camera_index: int
    video_path: Optional[Path]
    width: int
    height: int
    fps: int
    min_expected: int
    layout: str
    panel_width: int
    recent_width: int


def default_calibration_path(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 5 calibration YAML path without creating it."""

    return Path(project_root).expanduser() / "calib" / "calib_color.yaml"


def default_registry_path(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 5 tag-registry path without creating it."""

    return Path(project_root).expanduser() / "boards" / "tag_registry.yaml"


def default_shots_dir(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 5 shots directory without creating it."""

    return Path(project_root).expanduser() / "shots"


def default_manifest_path(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 5 manifest path without creating it."""

    return default_shots_dir(project_root) / "manifest.csv"


def expected_calibration_path(config: CaptureFaceConfig) -> Path:
    """Return the calibration YAML path used by the current form values."""

    root = Path(config.project_root).expanduser()
    raw = _optional_path(config.calibration_path)
    return resolve_capture_calibration_path(root, raw or "calib_color.yaml")


def expected_capture_paths(config: CaptureFaceConfig) -> CaptureFacePaths:
    """Return resolved Stage 5 paths without creating directories."""

    calibration_path = expected_calibration_path(config)
    return prepare_capture_paths(
        project_root=Path(config.project_root).expanduser(),
        calib_path=calibration_path,
        registry=config.registry_path,
        out_dir=config.out_dir,
        manifest=config.manifest_path,
        log_file=config.log_file,
        layout=config.layout,
        raw_dir=config.raw_dir,
        ann_dir=config.ann_dir,
        meta_dir=config.meta_dir,
    )


def inspect_capture_face_outputs(
    project_root: Union[Path, str],
    *,
    registry_path: Optional[Union[Path, str]] = None,
    manifest_path: Optional[Union[Path, str]] = None,
    out_dir: Optional[Union[Path, str]] = None,
) -> CaptureFaceOutputStatus:
    """Inspect saved Stage 5 outputs and registry coverage."""

    root = Path(project_root).expanduser()
    paths = prepare_capture_paths(
        project_root=root,
        calib_path=default_calibration_path(root),
        registry=registry_path,
        manifest=manifest_path,
        out_dir=out_dir,
    )
    checked_paths: list[Path] = [paths.registry_path, paths.manifest_path, paths.out_dir]
    warnings: list[str] = []
    errors: list[str] = []

    face_records: tuple[dict[str, Any], ...] = ()
    try:
        registry = load_capture_registry(paths.registry_path)
        face_records = load_registered_faces(
            registry,
            project_root=paths.project_root,
            registry_path=paths.registry_path,
        )
    except CaptureFaceError as exc:
        errors.append(str(exc))

    face_keys = {
        _face_key(face): _face_label(face)
        for face in face_records
    }
    registered_faces = tuple(sorted(face_keys.values()))
    covered_keys: set[tuple[str, str]] = set()
    valid_shot_count = 0
    invalid_shot_count = 0

    if not paths.manifest_path.exists():
        return CaptureFaceOutputStatus(
            registry_path=paths.registry_path,
            manifest_path=paths.manifest_path,
            out_dir=paths.out_dir,
            checked_paths=tuple(dict.fromkeys(checked_paths)),
            registered_faces=registered_faces,
            covered_faces=(),
            missing_faces=registered_faces,
            valid_shot_count=0,
            invalid_shot_count=0,
            manifest_exists=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    try:
        rows = _read_manifest_rows(paths.manifest_path)
    except CaptureFaceWorkflowError as exc:
        return CaptureFaceOutputStatus(
            registry_path=paths.registry_path,
            manifest_path=paths.manifest_path,
            out_dir=paths.out_dir,
            checked_paths=tuple(dict.fromkeys(checked_paths)),
            registered_faces=registered_faces,
            covered_faces=(),
            missing_faces=registered_faces,
            valid_shot_count=0,
            invalid_shot_count=0,
            manifest_exists=True,
            errors=tuple([*errors, str(exc)]),
            warnings=tuple(warnings),
        )

    for row_index, row in enumerate(rows, start=2):
        metadata_path = _resolve_output_path(root, row.get("path_meta", ""))
        if metadata_path is None:
            invalid_shot_count += 1
            warnings.append(f"Manifest row {row_index} is missing path_meta.")
            continue
        checked_paths.append(metadata_path)
        valid, row_warnings, key, shot_paths = _inspect_metadata_row(
            metadata_path,
            root=root,
            face_keys=face_keys,
            row_index=row_index,
        )
        checked_paths.extend(shot_paths)
        if not valid or key is None:
            invalid_shot_count += 1
            warnings.extend(row_warnings)
            continue
        valid_shot_count += 1
        covered_keys.add(key)
        warnings.extend(row_warnings)

    covered_faces = tuple(sorted(face_keys[key] for key in covered_keys))
    missing_faces = tuple(
        sorted(label for key, label in face_keys.items() if key not in covered_keys)
    )

    return CaptureFaceOutputStatus(
        registry_path=paths.registry_path,
        manifest_path=paths.manifest_path,
        out_dir=paths.out_dir,
        checked_paths=tuple(dict.fromkeys(checked_paths)),
        registered_faces=registered_faces,
        covered_faces=covered_faces,
        missing_faces=missing_faces,
        valid_shot_count=valid_shot_count,
        invalid_shot_count=invalid_shot_count,
        manifest_exists=True,
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def inspect_capture_face_readiness(
    config: CaptureFaceConfig,
    *,
    realsense_available: Optional[bool] = None,
) -> CaptureFaceReadiness:
    """Return command, readiness, registered faces, and output state."""

    root = Path(config.project_root).expanduser()
    calibration_path = expected_calibration_path(config)
    paths = expected_capture_paths(config)
    outputs = inspect_capture_face_outputs(
        root,
        registry_path=paths.registry_path,
        manifest_path=paths.manifest_path,
        out_dir=paths.out_dir,
    )
    checked_paths = [
        calibration_path,
        paths.registry_path,
        paths.manifest_path,
        paths.out_dir,
    ]
    if paths.raw_dir is not None:
        checked_paths.append(paths.raw_dir)
    if paths.ann_dir is not None:
        checked_paths.append(paths.ann_dir)
    if paths.meta_dir is not None:
        checked_paths.append(paths.meta_dir)

    warnings: list[str] = list(outputs.warnings)
    errors: list[str] = []
    validated: Optional[_ValidatedCaptureFaceInputs] = None
    face_records: tuple[dict[str, Any], ...] = ()
    selected_faces: tuple[str, ...] = ()
    registered_bases: tuple[str, ...] = ()
    registered_faces: tuple[str, ...] = ()

    try:
        validated = _validated_inputs(config)
    except CaptureFaceWorkflowError as exc:
        errors.append(str(exc))

    try:
        load_capture_calibration(calibration_path)
    except CaptureFaceError as exc:
        errors.append(str(exc))

    try:
        registry = load_capture_registry(paths.registry_path)
        face_records = load_registered_faces(
            registry,
            project_root=paths.project_root,
            registry_path=paths.registry_path,
        )
        registered_bases = tuple(unique_bases(face_records))
        registered_faces = tuple(_face_label(face) for face in face_records)
    except CaptureFaceError as exc:
        errors.append(str(exc))

    if validated is not None:
        try:
            selection = select_initial_faces(validated.object_name, face_records)
            selected_faces = (
                tuple(_face_label(face) for face in selection.faces)
                if selection is not None
                else ()
            )
        except CaptureFaceError as exc:
            errors.append(str(exc))
        try:
            validate_capture_source_args(
                validated.source,
                str(validated.video_path) if validated.video_path is not None else None,
                _realsense_module_token(realsense_available),
            )
        except CaptureFaceError as exc:
            errors.append(str(exc))

    default_registry = default_registry_path(root)
    if paths.registry_path != default_registry:
        warnings.append(
            "Project status checks look for the Stage 5 registry at "
            f"{default_registry}. Custom registry paths can run capture, but "
            "they may not mark Stage 5 complete."
        )

    default_manifest = default_manifest_path(root)
    if paths.manifest_path != default_manifest:
        warnings.append(
            "Project status checks look for face-shot coverage at "
            f"{default_manifest}. Custom manifests can run capture, but they "
            "may not mark Stage 5 complete."
        )

    default_calib = default_calibration_path(root)
    if calibration_path != default_calib and not default_calib.exists():
        warnings.append(
            "Project status checks look for camera calibration at "
            f"{default_calib}. Custom calibration paths can run capture, but "
            "Stage 2 may still appear incomplete."
        )

    command_preview = ""
    if not errors:
        try:
            command_preview = build_capture_face_command(config)
        except CaptureFaceWorkflowError as exc:
            errors.append(str(exc))

    return CaptureFaceReadiness(
        ready=not errors,
        command_preview=command_preview,
        calibration_path=calibration_path,
        paths=paths,
        output_status=outputs,
        registered_bases=registered_bases,
        registered_faces=registered_faces,
        selected_faces=selected_faces,
        checked_paths=tuple(dict.fromkeys(checked_paths)),
        warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(dict.fromkeys(errors)),
    )


def build_capture_face_command(config: CaptureFaceConfig) -> str:
    """Build a copyable ``posetag-capture-face`` command from GUI state."""

    return shlex.join(("posetag-capture-face", *build_capture_face_arguments(config)))


def build_capture_face_launch(
    config: CaptureFaceConfig,
    *,
    python_executable: Optional[Union[Path, str]] = None,
) -> CaptureFaceLaunchSpec:
    """Build process launch details for the existing face-shot workflow."""

    arguments = build_capture_face_arguments(config)
    executable = str(python_executable) if python_executable else sys.executable
    paths = expected_capture_paths(config)
    return CaptureFaceLaunchSpec(
        program=executable,
        arguments=("-m", "posetag.cli.capture_face", *arguments),
        display_command=build_capture_face_command(config),
        expected_manifest=paths.manifest_path,
    )


def build_capture_face_arguments(config: CaptureFaceConfig) -> tuple[str, ...]:
    """Build argv items for ``posetag-capture-face`` without a shell."""

    validated = _validated_inputs(config)
    root = Path(config.project_root).expanduser()
    calibration_path = expected_calibration_path(config)
    paths = expected_capture_paths(config)

    parts: list[str] = [
        "--project_root",
        str(root),
        "--source",
        validated.source,
    ]
    if validated.source == SOURCE_OPENCV:
        parts.extend(["--cam", str(validated.camera_index)])
        parts.extend(_capture_size_arguments(validated))
    elif validated.source == SOURCE_REALSENSE:
        parts.extend(_capture_size_arguments(validated))
    elif validated.source == SOURCE_VIDEO:
        assert validated.video_path is not None
        parts.extend(["--video", str(validated.video_path)])

    parts.extend(
        [
            "--object_name",
            validated.object_name,
            "--calib",
            str(calibration_path),
            "--registry",
            str(paths.registry_path),
            "--family",
            validated.family,
            "--min_expected",
            str(validated.min_expected),
            "--layout",
            validated.layout,
            "--manifest",
            str(paths.manifest_path),
            "--panel_w",
            str(validated.panel_width),
            "--recent_w",
            str(validated.recent_width),
        ]
    )

    out_dir = _optional_path(config.out_dir)
    if out_dir is not None:
        parts.extend(["--out_dir", str(out_dir)])
    if validated.layout == "split_type":
        if paths.raw_dir is not None:
            parts.extend(["--raw_dir", str(paths.raw_dir)])
        if paths.ann_dir is not None:
            parts.extend(["--ann_dir", str(paths.ann_dir)])
        if paths.meta_dir is not None:
            parts.extend(["--meta_dir", str(paths.meta_dir)])
    log_file = _optional_path(config.log_file)
    if log_file is not None:
        parts.extend(["--log_file", str(log_file)])
    return tuple(parts)


def capture_face_process_not_started(
    expected_manifest: Optional[Union[Path, str]] = None,
) -> CaptureFaceProcessState:
    """Return the initial face-shot capture process state."""

    return CaptureFaceProcessState(
        state=CAPTURE_FACE_PROCESS_NOT_STARTED,
        label="not started",
        message="Face-shot capture has not been launched from this dashboard session.",
        expected_manifest=_optional_path(expected_manifest),
    )


def capture_face_process_running(
    expected_manifest: Union[Path, str],
) -> CaptureFaceProcessState:
    """Return the active face-shot capture process state."""

    return CaptureFaceProcessState(
        state=CAPTURE_FACE_PROCESS_RUNNING,
        label="running",
        message=(
            "Face-shot capture is running in the existing OpenCV workflow. "
            "Press ENTER in the preview to save a shot and q or ESC to quit."
        ),
        running=True,
        expected_manifest=Path(expected_manifest).expanduser(),
    )


def capture_face_process_failed(
    message: str,
    *,
    expected_manifest: Optional[Union[Path, str]] = None,
    exit_code: Optional[int] = None,
) -> CaptureFaceProcessState:
    """Return a failed/cancelled face-shot capture process state."""

    return CaptureFaceProcessState(
        state=CAPTURE_FACE_PROCESS_FAILED_CANCELLED,
        label="failed/cancelled",
        message=message,
        expected_manifest=_optional_path(expected_manifest),
        exit_code=exit_code,
    )


def summarize_capture_face_process_result(
    *,
    exit_code: int,
    crashed: bool = False,
    expected_manifest: Optional[Union[Path, str]] = None,
    previous_manifest_mtime_ns: Optional[int] = None,
    require_output_update: bool = False,
) -> CaptureFaceProcessState:
    """Summarize process completion from the expected manifest output."""

    manifest = _optional_path(expected_manifest)
    manifest_exists = bool(manifest and manifest.exists())
    manifest_updated = (
        not require_output_update
        or (
            manifest is not None
            and _path_mtime_ns(manifest) != previous_manifest_mtime_ns
        )
    )

    if exit_code == 0 and not crashed and manifest_exists and manifest_updated:
        return CaptureFaceProcessState(
            state=CAPTURE_FACE_PROCESS_FINISHED,
            label="finished",
            message=(
                "Face-shot capture exited cleanly and the expected manifest "
                "was updated."
            ),
            success=True,
            expected_manifest=manifest,
            exit_code=exit_code,
        )

    if crashed:
        message = "Face-shot capture process crashed or was cancelled."
    elif exit_code != 0:
        message = f"Face-shot capture process exited with code {exit_code}."
    elif not manifest_exists:
        message = (
            "Face-shot capture exited cleanly without writing the expected "
            "manifest. This usually means the user quit before saving."
        )
    else:
        message = (
            "Face-shot capture exited cleanly, but the manifest was not "
            "updated during this launch."
        )

    return capture_face_process_failed(
        message,
        expected_manifest=manifest,
        exit_code=exit_code,
    )


def normalize_source(source: str) -> str:
    """Normalize a user-facing face-shot source value."""

    normalized = str(source).strip().lower()
    aliases = {
        "webcam": SOURCE_OPENCV,
        "webcam/opencv": SOURCE_OPENCV,
        "opencv webcam": SOURCE_OPENCV,
        "opencv": SOURCE_OPENCV,
        "realsense": SOURCE_REALSENSE,
        "intel realsense": SOURCE_REALSENSE,
        "video": SOURCE_VIDEO,
        "video file": SOURCE_VIDEO,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise CaptureFaceWorkflowError(
            "Face-shot source must be one of: "
            + ", ".join(CAPTURE_SOURCE_CHOICES)
            + "."
        ) from exc


def realsense_dependency_available() -> bool:
    """Return whether ``pyrealsense2`` is importable in this environment."""

    return importlib.util.find_spec("pyrealsense2") is not None


def _validated_inputs(config: CaptureFaceConfig) -> _ValidatedCaptureFaceInputs:
    errors: list[str] = []
    object_name = str(config.object_name).strip()
    if not object_name:
        errors.append("Choose a registered object or face before capture.")
    elif Path(object_name).name != object_name or "\\" in object_name:
        errors.append("Object or face selection must be a registered name, not a path.")

    family = str(config.family).strip()
    if not family:
        errors.append("AprilTag family must not be empty.")

    try:
        source = normalize_source(config.source)
    except CaptureFaceWorkflowError as exc:
        source = SOURCE_OPENCV
        errors.append(str(exc))

    camera_index = _parse_int(config.camera_index, "Camera index", errors)
    width = _parse_int(config.width, "Capture width", errors)
    height = _parse_int(config.height, "Capture height", errors)
    fps = _parse_int(config.fps, "Capture FPS", errors)
    min_expected = _parse_int(config.min_expected, "Minimum expected tags", errors)
    panel_width = _parse_int(config.panel_width, "Info panel width", errors)
    recent_width = _parse_int(config.recent_width, "Recent-shot panel width", errors)
    layout = str(config.layout).strip()
    if layout not in LAYOUT_CHOICES:
        errors.append(
            "Capture layout must be one of: " + ", ".join(LAYOUT_CHOICES) + "."
        )

    video_path = _optional_path(config.video_path)
    if camera_index is not None and camera_index < 0:
        errors.append("Camera index must be zero or greater.")
    for label, value in (
        ("Capture width", width),
        ("Capture height", height),
        ("Capture FPS", fps),
        ("Minimum expected tags", min_expected),
    ):
        if value is not None and value <= 0:
            errors.append(f"{label} must be positive.")
    for label, value in (
        ("Info panel width", panel_width),
        ("Recent-shot panel width", recent_width),
    ):
        if value is not None and value < 0:
            errors.append(f"{label} must be zero or greater.")

    if errors:
        raise CaptureFaceWorkflowError(" ".join(errors))

    assert camera_index is not None
    assert width is not None
    assert height is not None
    assert fps is not None
    assert min_expected is not None
    assert panel_width is not None
    assert recent_width is not None
    return _ValidatedCaptureFaceInputs(
        object_name=object_name,
        family=family,
        source=source,
        camera_index=camera_index,
        video_path=video_path,
        width=width,
        height=height,
        fps=fps,
        min_expected=min_expected,
        layout=layout,
        panel_width=panel_width,
        recent_width=recent_width,
    )


def _inspect_metadata_row(
    metadata_path: Path,
    *,
    root: Path,
    face_keys: Mapping[tuple[str, str], str],
    row_index: int,
) -> tuple[bool, list[str], Optional[tuple[str, str]], tuple[Path, ...]]:
    warnings: list[str] = []
    checked_paths: list[Path] = []
    if not metadata_path.exists():
        return (
            False,
            [f"Manifest row {row_index} metadata JSON was not found: {metadata_path}"],
            None,
            (),
        )

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            False,
            [f"Manifest row {row_index} metadata JSON is malformed: {metadata_path}: {exc}"],
            None,
            (),
        )
    except OSError as exc:
        return (
            False,
            [f"Could not read manifest row {row_index} metadata JSON: {metadata_path}: {exc}"],
            None,
            (),
        )

    if not isinstance(data, Mapping):
        return (
            False,
            [f"Manifest row {row_index} metadata JSON must contain a mapping: {metadata_path}"],
            None,
            (),
        )

    schema_errors = validate_capture_metadata_schema(data)
    if schema_errors:
        return (
            False,
            [f"Manifest row {row_index}: {message}" for message in schema_errors],
            None,
            (),
        )

    image = data.get("image", {})
    raw_path = _resolve_output_path(root, str(image.get("path_raw", "")))
    ann_path = _resolve_output_path(root, str(image.get("path_ann", "")))
    for label, path in (("raw image", raw_path), ("annotated image", ann_path)):
        if path is None:
            warnings.append(f"Manifest row {row_index} is missing {label} path.")
            continue
        checked_paths.append(path)
        if not path.exists():
            warnings.append(
                f"Manifest row {row_index} {label} was not found: {path}"
            )

    face_yaml = _resolve_output_path(root, str(data.get("face_yaml", "")))
    if face_yaml is None:
        warnings.append(f"Manifest row {row_index} is missing face_yaml.")
        return False, warnings, None, tuple(checked_paths)
    checked_paths.append(face_yaml)
    object_full = str(data.get("object_full", "")).strip()
    key = _normal_key(object_full, face_yaml)
    if key not in face_keys:
        warnings.append(
            f"Manifest row {row_index} references an unregistered face: "
            f"{object_full} ({face_yaml})."
        )
        return False, warnings, None, tuple(checked_paths)

    expected = [
        item
        for item in (
            _parse_int_or_none(raw_item)
            for raw_item in data.get("expected_tag_ids", [])
        )
        if item is not None
    ]
    expected.sort()
    registered = _registered_ids_for_warning(face_yaml)
    if registered and expected and sorted(registered) != expected:
        warnings.append(
            f"Manifest row {row_index} expected tag IDs do not match "
            f"{face_yaml}."
        )

    if not bool(data.get("validation_ok")):
        warnings.append(
            f"Manifest row {row_index} did not validate expected face tags and "
            "does not count toward coverage."
        )
        return False, warnings, key, tuple(checked_paths)
    if any(not path.exists() for path in checked_paths[:2]):
        return False, warnings, key, tuple(checked_paths)
    return True, warnings, key, tuple(checked_paths)


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise CaptureFaceWorkflowError(
                    f"Face-shot manifest has no header: {path}"
                )
            return [dict(row) for row in reader]
    except csv.Error as exc:
        raise CaptureFaceWorkflowError(
            f"Malformed face-shot manifest CSV: {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise CaptureFaceWorkflowError(
            f"Could not read face-shot manifest CSV: {path}: {exc}"
        ) from exc


def _registered_ids_for_warning(face_yaml: Path) -> tuple[int, ...]:
    try:
        from posetag.pipelines.capture_face import load_board_face_schema

        _object_name, tag_ids = load_board_face_schema(face_yaml)
    except Exception:
        return ()
    return tuple(sorted(tag_ids))


def _face_key(face: Mapping[str, Any]) -> tuple[str, str]:
    return _normal_key(str(face.get("object", "")), Path(str(face.get("yaml", ""))))


def _normal_key(object_name: str, yaml_path: Path) -> tuple[str, str]:
    try:
        normalized_path = str(yaml_path.expanduser().resolve())
    except OSError:
        normalized_path = str(yaml_path.expanduser())
    return (str(object_name).strip(), normalized_path)


def _face_label(face: Mapping[str, Any]) -> str:
    object_name = str(face.get("object", "")).strip()
    if object_name:
        return object_name
    yaml_path = Path(str(face.get("yaml", "")))
    return yaml_path.stem or str(yaml_path)


def _resolve_output_path(root: Path, value: str) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else root / path


def _capture_size_arguments(validated: _ValidatedCaptureFaceInputs) -> list[str]:
    return [
        "--width",
        str(validated.width),
        "--height",
        str(validated.height),
        "--fps",
        str(validated.fps),
    ]


def _parse_int(value: object, label: str, errors: list[str]) -> Optional[int]:
    if isinstance(value, bool):
        errors.append(f"{label} must be an integer.")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an integer.")
        return None
    return parsed


def _parse_int_or_none(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return parsed


def _optional_path(value: Optional[Union[Path, str]]) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _realsense_module_token(available: Optional[bool]) -> object:
    if available is None:
        return object() if realsense_dependency_available() else None
    return object() if available else None


def _path_mtime_ns(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None
