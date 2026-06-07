"""GUI-independent helpers for launching dataset collection.

The interactive capture, AprilTag detection, pose estimation, review UI, and
dataset writing remain in ``posetag-collect``.  This module only validates the
dashboard-facing launch values, builds shell-free process arguments, and
summarizes child-process state.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from posetag.pipelines.collect_dataset import (
    CollectDatasetError,
    CollectionInputSummary,
    preview_project_root,
    resolve_calibration_path,
    resolve_dataset_root,
    resolve_face_manifest_path,
    resolve_registry_path,
    validate_collection_inputs,
    validate_source_args,
)


SOURCE_OPENCV = "opencv"
SOURCE_LIVE = "live"
SOURCE_VIDEO = "video"
SOURCE_BAG = "bag"
COLLECT_SOURCE_CHOICES = (SOURCE_OPENCV, SOURCE_LIVE, SOURCE_VIDEO, SOURCE_BAG)
COLLECT_SOURCE_LABELS = {
    SOURCE_OPENCV: "webcam/OpenCV",
    SOURCE_LIVE: "RealSense live",
    SOURCE_VIDEO: "video",
    SOURCE_BAG: "RealSense bag",
}

DEFAULT_SESSION_NAME = "run01"
DEFAULT_FAMILY = "tag36h11"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FPS = 30
COLLECT_PROCESS_NOT_STARTED = "not_started"
COLLECT_PROCESS_RUNNING = "running"
COLLECT_PROCESS_FINISHED = "finished"
COLLECT_PROCESS_FAILED_CANCELLED = "failed_cancelled"


@dataclass(frozen=True)
class DatasetCollectionConfig:
    """User-facing ``posetag-collect`` launch state."""

    project_root: Union[Path, str]
    session: str = DEFAULT_SESSION_NAME
    mode: str = SOURCE_OPENCV
    camera_index: int = DEFAULT_CAMERA_INDEX
    video_path: Optional[Union[Path, str]] = None
    bag_path: Optional[Union[Path, str]] = None
    calib_path: Optional[Union[Path, str]] = None
    registry_path: Optional[Union[Path, str]] = None
    face_manifest_path: Optional[Union[Path, str]] = None
    dataset_root: Optional[Union[Path, str]] = None
    family: str = DEFAULT_FAMILY
    continuous: bool = False
    auto_capture: bool = False
    auto_stable_frames: int = 5
    auto_cooldown_sec: float = 1.0
    max_frames: int = 0
    width: int = 0
    height: int = 0
    fps: int = DEFAULT_FPS
    rs_width: int = 0
    rs_height: int = 0
    rs_fps: int = DEFAULT_FPS
    save_depth: bool = False
    allow_face_scan: bool = False
    check_tag_scale: bool = False
    auto_correct_scale: bool = False
    scale_tol: float = 0.02
    axes: str = "both"
    bbox_mode: str = "auto"


@dataclass(frozen=True)
class DatasetCollectionReadiness:
    """Display-ready preflight state for dataset collection."""

    ready: bool
    command_preview: str
    dry_run_command_preview: str
    project_root: Path
    calibration_path: Path
    registry_path: Path
    face_manifest_path: Path
    dataset_root: Path
    expected_session_dir: Path
    input_summary: CollectionInputSummary | None
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetCollectionLaunch:
    """Editable-install-safe launch details for ``posetag-collect``."""

    program: str
    arguments: tuple[str, ...]
    display_command: str
    project_root: Path
    expected_session_dir: Path
    dry_run: bool


@dataclass(frozen=True)
class DatasetCollectionProcessState:
    """GUI-independent dataset-collection child-process state."""

    state: str
    label: str
    message: str
    running: bool = False
    success: bool = False
    expected_session_dir: Path | None = None
    exit_code: int | None = None
    dry_run: bool = False


def inspect_dataset_collection_readiness(
    config: DatasetCollectionConfig,
) -> DatasetCollectionReadiness:
    """Validate collection inputs and return display-ready launch state."""

    root = preview_project_root(config.project_root)
    calibration_path = resolve_calibration_path(root, config.calib_path)
    registry_path = resolve_registry_path(root, config.registry_path)
    face_manifest_path = resolve_face_manifest_path(root, config.face_manifest_path)
    dataset_root = resolve_dataset_root(root, config.dataset_root)
    errors: list[str] = []
    try:
        session = normalize_session_name(config.session)
    except ValueError as exc:
        session = DEFAULT_SESSION_NAME
        errors.append(str(exc))
    expected_session_dir = dataset_root / session

    checked_paths: list[Path] = [
        calibration_path,
        registry_path,
        face_manifest_path,
        dataset_root,
        expected_session_dir,
    ]
    warnings: list[str] = []
    input_summary: CollectionInputSummary | None = None
    try:
        mode = normalize_source(config.mode)
    except ValueError as exc:
        mode = SOURCE_OPENCV
        errors.append(str(exc))

    try:
        validate_source_args(
            mode=mode,
            video=config.video_path,
            bag=config.bag_path,
            realsense_module=_realsense_module_marker(),
            video_capture_factory=None,
        )
    except CollectDatasetError as exc:
        errors.append(str(exc))

    try:
        input_summary = validate_collection_inputs(
            project_root=root,
            calib_path=calibration_path,
            registry_path=registry_path,
            face_manifest_path=face_manifest_path,
            allow_face_scan=config.allow_face_scan,
        )
    except CollectDatasetError as exc:
        errors.append(str(exc))
    else:
        checked_paths.extend(annotation.yaml_path for annotation in input_summary.annotations)
        checked_paths.extend(annotation.board_yaml for annotation in input_summary.annotations)
        checked_paths.extend(
            path for path in input_summary.missing_keypoints if path is not None
        )
        if input_summary.missing_keypoints:
            warnings.append(
                "Optional object keypoints are missing for bbox/review behavior; "
                "collection can still use tag-based fallback boxes."
            )

    command_preview = ""
    dry_run_command_preview = ""
    if not errors:
        try:
            command_preview = build_dataset_collection_command(config, dry_run=False)
            dry_run_command_preview = build_dataset_collection_command(
                config,
                dry_run=True,
            )
        except ValueError as exc:
            errors.append(str(exc))

    return DatasetCollectionReadiness(
        ready=not errors,
        command_preview=command_preview,
        dry_run_command_preview=dry_run_command_preview,
        project_root=root,
        calibration_path=calibration_path,
        registry_path=registry_path,
        face_manifest_path=face_manifest_path,
        dataset_root=dataset_root,
        expected_session_dir=expected_session_dir,
        input_summary=input_summary,
        checked_paths=tuple(dict.fromkeys(checked_paths)),
        warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(dict.fromkeys(errors)),
    )


def build_dataset_collection_launch(
    config: DatasetCollectionConfig,
    *,
    dry_run: bool = False,
    python_executable: Optional[Union[Path, str]] = None,
) -> DatasetCollectionLaunch:
    """Build process launch details for the existing collection workflow."""

    executable = str(python_executable) if python_executable else sys.executable
    arguments = build_dataset_collection_arguments(config, dry_run=dry_run)
    root = preview_project_root(config.project_root)
    dataset_root = resolve_dataset_root(root, config.dataset_root)
    expected_session_dir = dataset_root / normalize_session_name(config.session)
    return DatasetCollectionLaunch(
        program=executable,
        arguments=("-m", "posetag.cli.collect", *arguments),
        display_command=build_dataset_collection_command(config, dry_run=dry_run),
        project_root=root,
        expected_session_dir=expected_session_dir,
        dry_run=bool(dry_run),
    )


def build_dataset_collection_command(
    config: DatasetCollectionConfig,
    *,
    dry_run: bool = False,
) -> str:
    """Build a copyable ``posetag-collect`` command from GUI state."""

    root = preview_project_root(config.project_root)
    parts = (
        "env",
        f"POSETAG_PROJECT={root}",
        "posetag-collect",
        *build_dataset_collection_arguments(config, dry_run=dry_run),
    )
    return shlex.join(parts)


def build_dataset_collection_arguments(
    config: DatasetCollectionConfig,
    *,
    dry_run: bool = False,
) -> tuple[str, ...]:
    """Build argv items for ``posetag-collect`` without a shell."""

    root = preview_project_root(config.project_root)
    mode = normalize_source(config.mode)
    session = normalize_session_name(config.session)
    family = str(config.family or DEFAULT_FAMILY).strip() or DEFAULT_FAMILY
    if bool(config.continuous) and bool(config.auto_capture):
        raise ValueError("Continuous capture and smart auto-capture cannot both be enabled.")
    if family != DEFAULT_FAMILY:
        family_arg = family
    else:
        family_arg = DEFAULT_FAMILY

    parts: list[str] = [
        "--project_root",
        str(root),
        "--mode",
        mode,
        "--session",
        session,
        "--calib",
        str(config.calib_path or "calib_color.yaml"),
    ]
    if config.registry_path:
        parts.extend(["--registry", str(Path(config.registry_path).expanduser())])
    if config.face_manifest_path:
        parts.extend(
            ["--face_manifest", str(Path(config.face_manifest_path).expanduser())]
        )
    if config.dataset_root:
        parts.extend(["--dataset_root", str(Path(config.dataset_root).expanduser())])

    if mode == SOURCE_OPENCV:
        parts.extend(["--cam", str(int(config.camera_index))])
        if int(config.width) > 0:
            parts.extend(["--width", str(int(config.width))])
        if int(config.height) > 0:
            parts.extend(["--height", str(int(config.height))])
        if int(config.fps) != DEFAULT_FPS:
            parts.extend(["--fps", str(int(config.fps))])
    elif mode == SOURCE_VIDEO:
        if not config.video_path:
            raise ValueError("--video is required when dataset source is video.")
        parts.extend(["--video", str(Path(config.video_path).expanduser())])
    elif mode == SOURCE_BAG:
        if not config.bag_path:
            raise ValueError("--bag is required when dataset source is bag.")
        parts.extend(["--bag", str(Path(config.bag_path).expanduser())])
    elif mode == SOURCE_LIVE:
        if int(config.rs_width) > 0:
            parts.extend(["--rs_w", str(int(config.rs_width))])
        if int(config.rs_height) > 0:
            parts.extend(["--rs_h", str(int(config.rs_height))])
        if int(config.rs_fps) != DEFAULT_FPS:
            parts.extend(["--rs_fps", str(int(config.rs_fps))])

    if family_arg:
        parts.extend(["--family", family_arg])
    if bool(config.continuous):
        parts.append("--continuous")
    if bool(config.auto_capture):
        parts.append("--auto-capture")
        if int(config.auto_stable_frames) != 5:
            parts.extend(["--auto-stable-frames", str(int(config.auto_stable_frames))])
        if float(config.auto_cooldown_sec) != 1.0:
            parts.extend(["--auto-cooldown-sec", f"{float(config.auto_cooldown_sec):g}"])
    if int(config.max_frames) > 0:
        parts.extend(["--max_frames", str(int(config.max_frames))])
    if bool(config.save_depth):
        parts.append("--save_depth")
    if bool(config.allow_face_scan):
        parts.append("--allow-face-scan")
    if bool(config.check_tag_scale):
        parts.append("--check-tag-scale")
    if bool(config.auto_correct_scale):
        parts.append("--auto-correct-scale")
    if float(config.scale_tol) != 0.02:
        parts.extend(["--scale-tol", f"{float(config.scale_tol):g}"])
    if str(config.axes) != "both":
        parts.extend(["--axes", str(config.axes)])
    if str(config.bbox_mode) != "auto":
        parts.extend(["--bbox-mode", str(config.bbox_mode)])
    if dry_run:
        parts.append("--dry-run")
    return tuple(parts)


def collect_dataset_process_not_started(
    expected_session_dir: Optional[Union[Path, str]] = None,
) -> DatasetCollectionProcessState:
    """Return the initial dataset-collection process state."""

    path = _optional_path(expected_session_dir)
    return DatasetCollectionProcessState(
        state=COLLECT_PROCESS_NOT_STARTED,
        label="not started",
        message=(
            "Dataset collection has not been launched from this dashboard "
            "session."
        ),
        expected_session_dir=path,
    )


def collect_dataset_process_running(
    expected_session_dir: Union[Path, str],
    *,
    dry_run: bool = False,
) -> DatasetCollectionProcessState:
    """Return the active dataset-collection process state."""

    if dry_run:
        message = (
            "Dataset collection dry-run is validating inputs without opening "
            "a camera or writing dataset outputs."
        )
    else:
        message = (
            "Dataset collection is running in the existing OpenCV/RealSense "
            "workflow. Use the review window to accept or force-save frames; "
            "q, Q, ESC, x, or the window close button exits cleanly."
        )
    return DatasetCollectionProcessState(
        state=COLLECT_PROCESS_RUNNING,
        label="running",
        message=message,
        running=True,
        expected_session_dir=Path(expected_session_dir).expanduser(),
        dry_run=bool(dry_run),
    )


def collect_dataset_process_failed(
    message: str,
    *,
    expected_session_dir: Optional[Union[Path, str]] = None,
    exit_code: Optional[int] = None,
    dry_run: bool = False,
) -> DatasetCollectionProcessState:
    """Return a failed/cancelled collection process state."""

    return DatasetCollectionProcessState(
        state=COLLECT_PROCESS_FAILED_CANCELLED,
        label="failed/cancelled",
        message=message,
        expected_session_dir=_optional_path(expected_session_dir),
        exit_code=exit_code,
        dry_run=bool(dry_run),
    )


def summarize_dataset_collection_process_result(
    *,
    exit_code: int,
    crashed: bool = False,
    expected_session_dir: Optional[Union[Path, str]] = None,
    previous_session_yaml_mtime_ns: Optional[int] = None,
    require_output_update: bool = False,
    dry_run: bool = False,
) -> DatasetCollectionProcessState:
    """Summarize process completion from dry-run or expected session output."""

    session_dir = _optional_path(expected_session_dir)
    session_yaml = session_dir / "session.yaml" if session_dir is not None else None
    session_yaml_exists = bool(session_yaml and session_yaml.exists())
    session_yaml_updated = (
        not require_output_update
        or (
            session_yaml is not None
            and _path_mtime_ns(session_yaml) != previous_session_yaml_mtime_ns
        )
    )

    if crashed:
        return collect_dataset_process_failed(
            "Dataset collection process crashed or was cancelled.",
            expected_session_dir=session_dir,
            exit_code=exit_code,
            dry_run=dry_run,
        )
    if exit_code != 0:
        return collect_dataset_process_failed(
            f"Dataset collection process exited with code {exit_code}.",
            expected_session_dir=session_dir,
            exit_code=exit_code,
            dry_run=dry_run,
        )
    if dry_run:
        return DatasetCollectionProcessState(
            state=COLLECT_PROCESS_FINISHED,
            label="finished",
            message=(
                "Dataset collection dry-run completed. Inputs passed preflight; "
                "run collection to create a pose-labelled session."
            ),
            success=True,
            expected_session_dir=session_dir,
            exit_code=exit_code,
            dry_run=True,
        )
    if session_yaml_exists and session_yaml_updated:
        return DatasetCollectionProcessState(
            state=COLLECT_PROCESS_FINISHED,
            label="finished",
            message=(
                "Dataset collection exited cleanly. Stage 8 status was "
                "refreshed from the saved session outputs."
            ),
            success=True,
            expected_session_dir=session_dir,
            exit_code=exit_code,
        )

    if not session_yaml_exists:
        message = (
            "Dataset collection exited cleanly without writing session.yaml. "
            "This usually means collection stopped before outputs were saved."
        )
    else:
        message = (
            "Dataset collection exited cleanly, but the expected session "
            "metadata was not updated during this launch."
        )
    return collect_dataset_process_failed(
        message,
        expected_session_dir=session_dir,
        exit_code=exit_code,
    )


def normalize_source(source: str) -> str:
    """Normalize a user-facing dataset source value."""

    normalized = str(source or "").strip().lower().replace("_", "-")
    aliases = {
        "opencv": SOURCE_OPENCV,
        "webcam": SOURCE_OPENCV,
        "webcam/opencv": SOURCE_OPENCV,
        "camera": SOURCE_OPENCV,
        "live": SOURCE_LIVE,
        "realsense": SOURCE_LIVE,
        "realsense live": SOURCE_LIVE,
        "video": SOURCE_VIDEO,
        "file": SOURCE_VIDEO,
        "bag": SOURCE_BAG,
        "realsense bag": SOURCE_BAG,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(COLLECT_SOURCE_CHOICES)
        raise ValueError(f"Unsupported dataset source {source!r}. Use {supported}.") from exc


def normalize_session_name(session: str) -> str:
    """Validate and normalize a dataset session folder name."""

    value = str(session or "").strip()
    if not value:
        raise ValueError("Dataset session name is required.")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError("Dataset session must be a simple folder name.")
    return value


def _realsense_module_marker() -> object | None:
    try:
        return object() if importlib.util.find_spec("pyrealsense2") else None
    except (ImportError, ValueError):
        return None


def _optional_path(value: Optional[Union[Path, str]]) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _path_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None
