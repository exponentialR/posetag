"""GUI-independent readiness helpers for guided camera calibration.

The helpers in this module prepare users to run ``posetag-calib-charuco``
without moving camera capture, ChArUco detection, calibration solving, or YAML
writing into GUI code.
"""

from __future__ import annotations

import importlib.util
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import yaml

from posetag.pipelines.charuco_calibration import (
    CharucoCalibrationError,
    get_dictionary,
    validate_capture_args,
)
from posetag.pipelines.make_board import MakeBoardError, load_calibration_yaml
from posetag.workflows.calibration_guidance import parse_grid_shape
from posetag.workflows.charuco_setup import (
    DEFAULT_DICTIONARY,
    DEFAULT_MARKER_LENGTH_MM,
    DEFAULT_SQUARE_LENGTH_MM,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
)
from posetag.workflows.status import WorkflowStatus, inspect_camera_calibration


SOURCE_OPENCV = "opencv"
SOURCE_REALSENSE = "realsense"
SOURCE_VIDEO = "video"
CALIBRATION_SOURCE_CHOICES = (SOURCE_OPENCV, SOURCE_REALSENSE, SOURCE_VIDEO)
CALIBRATION_SOURCE_LABELS = {
    SOURCE_OPENCV: "webcam/OpenCV",
    SOURCE_REALSENSE: "RealSense",
    SOURCE_VIDEO: "video",
}
DEFAULT_CAMERA_INDEX = 0
DEFAULT_COVERAGE_GRID = "3x3"
DEFAULT_SAMPLES_PER_CELL = 1
DEFAULT_GUIDED_AUTO_COOLDOWN = 8
CALIBRATION_PROCESS_NOT_STARTED = "not_started"
CALIBRATION_PROCESS_RUNNING = "running"
CALIBRATION_PROCESS_FINISHED = "finished"
CALIBRATION_PROCESS_FAILED_CANCELLED = "failed_cancelled"


class CameraCalibrationFlowError(ValueError):
    """User-facing calibration guidance error."""


@dataclass(frozen=True)
class CharucoCalibrationMetadata:
    """Board parameters read from a generated ChArUco metadata YAML."""

    path: Path
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    dictionary_name: str


@dataclass(frozen=True)
class CharucoMetadataInspection:
    """Readiness state for generated ChArUco board metadata."""

    metadata_paths: tuple[Path, ...]
    selected_path: Optional[Path]
    metadata: Optional[CharucoCalibrationMetadata]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def exists(self) -> bool:
        """Return true when at least one metadata file was found."""

        return bool(self.metadata_paths)


@dataclass(frozen=True)
class CameraCalibrationConfig:
    """User-facing camera calibration command state."""

    project_root: Union[Path, str]
    source: str = SOURCE_OPENCV
    camera_index: int = DEFAULT_CAMERA_INDEX
    video_path: Optional[Union[Path, str]] = None
    squares_x: int = DEFAULT_SQUARES_X
    squares_y: int = DEFAULT_SQUARES_Y
    square_length_mm: float = DEFAULT_SQUARE_LENGTH_MM
    marker_length_mm: float = DEFAULT_MARKER_LENGTH_MM
    dictionary_name: str = DEFAULT_DICTIONARY
    coverage_grid: str = DEFAULT_COVERAGE_GRID
    samples_per_cell: int = DEFAULT_SAMPLES_PER_CELL
    guided_auto: bool = True
    guided_auto_cooldown: int = DEFAULT_GUIDED_AUTO_COOLDOWN


@dataclass(frozen=True)
class CameraCalibrationReadiness:
    """Display-ready readiness result for the guided calibration flow."""

    ready: bool
    command_preview: str
    expected_output: Path
    output_exists: bool
    metadata_path: Optional[Path]
    metadata: Optional[CharucoCalibrationMetadata]
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class CameraCalibrationLaunchSpec:
    """Editable-install-safe process launch details for calibration."""

    program: str
    arguments: tuple[str, ...]
    display_command: str
    expected_output: Path


@dataclass(frozen=True)
class CameraCalibrationProcessState:
    """GUI-independent calibration child-process state."""

    state: str
    label: str
    message: str
    running: bool = False
    success: bool = False
    expected_output: Optional[Path] = None
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class CameraCalibrationOutputSummary:
    """Parsed status for an existing calibration YAML artifact."""

    path: Path
    exists: bool
    valid: bool
    message: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    model: Optional[str] = None
    reproj_rms: Optional[float] = None
    camera_params: Optional[tuple[float, float, float, float]] = None
    distortion_coefficients: tuple[tuple[str, float], ...] = ()
    latest_run_dir: Optional[Path] = None


def generated_charuco_metadata_paths(project_root: Union[Path, str]) -> tuple[Path, ...]:
    """Return generated ChArUco metadata paths without creating directories."""

    boards_dir = Path(project_root).expanduser() / "calib" / "boards"
    return tuple(
        sorted(path for path in boards_dir.glob("charuco_*.yaml") if path.is_file())
    )


def inspect_charuco_metadata(
    project_root: Union[Path, str],
) -> CharucoMetadataInspection:
    """Inspect generated ChArUco board metadata under a project root."""

    root = Path(project_root).expanduser()
    paths = generated_charuco_metadata_paths(root)
    if not paths:
        return CharucoMetadataInspection(
            metadata_paths=(),
            selected_path=None,
            metadata=None,
        )

    selected = _latest_path(paths)
    try:
        metadata = load_charuco_metadata(selected)
    except CameraCalibrationFlowError as exc:
        return CharucoMetadataInspection(
            metadata_paths=paths,
            selected_path=selected,
            metadata=None,
            errors=(str(exc),),
        )

    warnings: tuple[str, ...] = ()
    if len(paths) > 1:
        warnings = (
            f"Found {len(paths)} ChArUco metadata files; using {selected.name}.",
        )

    return CharucoMetadataInspection(
        metadata_paths=paths,
        selected_path=selected,
        metadata=metadata,
        warnings=warnings,
    )


def load_charuco_metadata(path: Union[Path, str]) -> CharucoCalibrationMetadata:
    """Load and validate the board fields needed by calibration."""

    metadata_path = Path(path).expanduser()
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CameraCalibrationFlowError(
            f"Malformed ChArUco metadata YAML: {exc}"
        ) from exc
    except OSError as exc:
        raise CameraCalibrationFlowError(
            f"Could not read ChArUco metadata: {exc}"
        ) from exc

    if not isinstance(data, Mapping):
        raise CameraCalibrationFlowError(
            "ChArUco metadata YAML must contain a mapping."
        )

    errors: list[str] = []
    squares_x = _require_int(data, "squares_x", errors)
    squares_y = _require_int(data, "squares_y", errors)
    square_length_mm = _require_float(data, "square_length_mm", errors)
    marker_length_mm = _require_float(data, "marker_length_mm", errors)
    dictionary_name = _require_string(data, "dictionary", errors)

    if squares_x is not None and squares_x < 2:
        errors.append("ChArUco metadata field 'squares_x' must be at least 2.")
    if squares_y is not None and squares_y < 2:
        errors.append("ChArUco metadata field 'squares_y' must be at least 2.")
    if square_length_mm is not None and square_length_mm <= 0:
        errors.append(
            "ChArUco metadata field 'square_length_mm' must be positive."
        )
    if marker_length_mm is not None and marker_length_mm <= 0:
        errors.append(
            "ChArUco metadata field 'marker_length_mm' must be positive."
        )
    if (
        square_length_mm is not None
        and marker_length_mm is not None
        and marker_length_mm >= square_length_mm
    ):
        errors.append(
            "ChArUco metadata marker length must be smaller than square length."
        )

    if errors:
        raise CameraCalibrationFlowError(" ".join(errors))

    assert squares_x is not None
    assert squares_y is not None
    assert square_length_mm is not None
    assert marker_length_mm is not None
    assert dictionary_name is not None
    return CharucoCalibrationMetadata(
        path=metadata_path,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        dictionary_name=dictionary_name,
    )


def camera_calibration_config_from_project(
    project_root: Union[Path, str],
    *,
    source: str = SOURCE_OPENCV,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    video_path: Optional[Union[Path, str]] = None,
) -> CameraCalibrationConfig:
    """Build a calibration config, autofilling board fields from metadata."""

    metadata = inspect_charuco_metadata(project_root).metadata
    if metadata is None:
        return CameraCalibrationConfig(
            project_root=project_root,
            source=source,
            camera_index=camera_index,
            video_path=video_path,
        )
    return CameraCalibrationConfig(
        project_root=project_root,
        source=source,
        camera_index=camera_index,
        video_path=video_path,
        squares_x=metadata.squares_x,
        squares_y=metadata.squares_y,
        square_length_mm=metadata.square_length_mm,
        marker_length_mm=metadata.marker_length_mm,
        dictionary_name=metadata.dictionary_name,
    )


def inspect_camera_calibration_readiness(
    config: CameraCalibrationConfig,
    *,
    realsense_available: Optional[bool] = None,
) -> CameraCalibrationReadiness:
    """Return command and readiness state for guided camera calibration."""

    root = Path(config.project_root).expanduser()
    metadata_state = inspect_charuco_metadata(root)
    expected_output = root / "calib" / "calib_color.yaml"
    checked_paths = _path_tuple(
        [
            root / "calib" / "boards",
            *metadata_state.metadata_paths,
            expected_output,
        ]
    )
    warnings = list(metadata_state.warnings)
    errors = list(metadata_state.errors)

    if not metadata_state.metadata_paths:
        errors.append(
            "Missing ChArUco board metadata. Generate the board first so the "
            "calibration parameters match the printed target."
        )

    if not expected_output.exists():
        warnings.append(
            f"Calibration output is missing: {expected_output}"
        )

    try:
        source = normalize_source(config.source)
    except CameraCalibrationFlowError as exc:
        errors.append(str(exc))
        source = SOURCE_OPENCV
    try:
        validate_capture_args(
            source,
            _video_arg(config.video_path),
            _realsense_module_token(realsense_available),
        )
    except CharucoCalibrationError as exc:
        errors.append(str(exc))
    errors.extend(_camera_calibration_board_errors(config))
    errors.extend(_camera_calibration_guidance_errors(config))

    command_preview = ""
    if not errors:
        try:
            command_preview = build_camera_calibration_command(config)
        except CameraCalibrationFlowError as exc:
            errors.append(str(exc))

    return CameraCalibrationReadiness(
        ready=not errors,
        command_preview=command_preview,
        expected_output=expected_output,
        output_exists=expected_output.exists(),
        metadata_path=metadata_state.selected_path,
        metadata=metadata_state.metadata,
        checked_paths=checked_paths,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def build_camera_calibration_command(config: CameraCalibrationConfig) -> str:
    """Build a copyable ``posetag-calib-charuco`` command from GUI state."""

    return shlex.join(
        ("posetag-calib-charuco", *build_camera_calibration_arguments(config))
    )


def build_camera_calibration_launch(
    config: CameraCalibrationConfig,
    *,
    python_executable: Optional[Union[Path, str]] = None,
) -> CameraCalibrationLaunchSpec:
    """Build process launch details for the existing calibration workflow."""

    arguments = build_camera_calibration_arguments(config)
    executable = str(python_executable) if python_executable else sys.executable
    return CameraCalibrationLaunchSpec(
        program=executable,
        arguments=("-m", "posetag.cli.charuco", *arguments),
        display_command=build_camera_calibration_command(config),
        expected_output=Path(config.project_root).expanduser()
        / "calib"
        / "calib_color.yaml",
    )


def build_camera_calibration_arguments(
    config: CameraCalibrationConfig,
) -> tuple[str, ...]:
    """Build argv items for ``posetag-calib-charuco`` without a shell."""

    source = normalize_source(config.source)
    video = _video_arg(config.video_path)
    if source == SOURCE_VIDEO and video is None:
        raise CameraCalibrationFlowError(
            "--video path is required when --source=video"
        )
    _validate_camera_calibration_board(config)
    coverage_grid, samples_per_cell, guided_auto_cooldown = (
        _validate_camera_calibration_guidance(config)
    )

    parts: list[str] = [
        "--project_root",
        str(Path(config.project_root).expanduser()),
        "--source",
        source,
    ]
    if source == SOURCE_OPENCV:
        parts.extend(["--cam", str(int(config.camera_index))])
    elif source == SOURCE_VIDEO:
        assert video is not None
        parts.extend(["--video", video])

    parts.extend(
        [
            "--squares-x",
            str(int(config.squares_x)),
            "--squares-y",
            str(int(config.squares_y)),
            "--square-length-mm",
            _format_number(config.square_length_mm),
            "--marker-length-mm",
            _format_number(config.marker_length_mm),
            "--dict",
            str(config.dictionary_name).strip(),
            "--coverage-grid",
            coverage_grid,
            "--samples-per-cell",
            str(samples_per_cell),
            "--guided-auto-cooldown",
            str(guided_auto_cooldown),
        ]
    )
    if not bool(config.guided_auto):
        parts.append("--no-guided-auto")
    return tuple(parts)


def calibration_process_not_started(
    expected_output: Optional[Union[Path, str]] = None,
) -> CameraCalibrationProcessState:
    """Return the initial calibration process state."""

    return CameraCalibrationProcessState(
        state=CALIBRATION_PROCESS_NOT_STARTED,
        label="not started",
        message="Calibration has not been launched from this dashboard session.",
        expected_output=_optional_path(expected_output),
    )


def calibration_process_running(
    expected_output: Union[Path, str],
) -> CameraCalibrationProcessState:
    """Return the active calibration process state."""

    target = Path(expected_output).expanduser()
    return CameraCalibrationProcessState(
        state=CALIBRATION_PROCESS_RUNNING,
        label="running",
        message=(
            "Calibration is running in the existing OpenCV window. Guided "
            "auto-capture saves useful samples; SPACE manually adds a sample, "
            "ENTER solves, and q quits."
        ),
        running=True,
        expected_output=target,
    )


def calibration_process_failed(
    message: str,
    *,
    expected_output: Optional[Union[Path, str]] = None,
    exit_code: Optional[int] = None,
) -> CameraCalibrationProcessState:
    """Return a failed/cancelled calibration process state."""

    return CameraCalibrationProcessState(
        state=CALIBRATION_PROCESS_FAILED_CANCELLED,
        label="failed/cancelled",
        message=message,
        expected_output=_optional_path(expected_output),
        exit_code=exit_code,
    )


def summarize_camera_calibration_process_result(
    project_root: Union[Path, str],
    *,
    exit_code: int,
    crashed: bool = False,
    expected_output: Optional[Union[Path, str]] = None,
    previous_output_mtime_ns: Optional[int] = None,
    require_output_update: bool = False,
) -> CameraCalibrationProcessState:
    """Summarize process completion using the calibration YAML status checks."""

    root = Path(project_root).expanduser()
    target = _optional_path(expected_output) or root / "calib" / "calib_color.yaml"
    stage = inspect_camera_calibration(root)
    output_exists = target.exists()
    output_updated = (
        not require_output_update
        or _path_mtime_ns(target) != previous_output_mtime_ns
    )
    if (
        exit_code == 0
        and not crashed
        and output_exists
        and output_updated
        and stage.status == WorkflowStatus.COMPLETE
    ):
        return CameraCalibrationProcessState(
            state=CALIBRATION_PROCESS_FINISHED,
            label="finished",
            message=(
                "Calibration finished and calib/calib_color.yaml passed the "
                "current status checks."
            ),
            success=True,
            expected_output=target,
            exit_code=exit_code,
        )

    if crashed:
        message = "Calibration process crashed or was cancelled."
    elif exit_code != 0:
        message = f"Calibration process exited with code {exit_code}."
    elif output_exists and not output_updated:
        message = (
            "Calibration process exited, but calib/calib_color.yaml was not "
            "updated during this launch."
        )
    else:
        message = (
            "Calibration process exited, but calib/calib_color.yaml is missing "
            "or did not pass the current status checks."
        )

    detail = _process_status_detail(stage)
    if detail:
        message = f"{message} {detail}"
    return calibration_process_failed(
        message,
        expected_output=target,
        exit_code=exit_code,
    )


def inspect_camera_calibration_output(
    project_root: Union[Path, str],
    *,
    expected_output: Optional[Union[Path, str]] = None,
) -> CameraCalibrationOutputSummary:
    """Return parsed calibration result fields for GUI/result summaries."""

    root = Path(project_root).expanduser()
    target = _optional_path(expected_output) or root / "calib" / "calib_color.yaml"
    if not target.exists():
        return CameraCalibrationOutputSummary(
            path=target,
            exists=False,
            valid=False,
            message=f"Calibration output is missing: {target}",
            latest_run_dir=_latest_calibration_run_dir(root),
        )

    try:
        data = _load_yaml_mapping(target)
        calibration = load_calibration_yaml(target)
    except (CameraCalibrationFlowError, MakeBoardError) as exc:
        return CameraCalibrationOutputSummary(
            path=target,
            exists=True,
            valid=False,
            message=str(exc),
            latest_run_dir=_latest_calibration_run_dir(root),
        )

    stage = inspect_camera_calibration(root)
    valid = stage.status == WorkflowStatus.COMPLETE
    message = (
        stage.message
        if valid
        else (_process_status_detail(stage) or stage.message)
    )
    return CameraCalibrationOutputSummary(
        path=target,
        exists=True,
        valid=valid,
        message=message,
        image_width=_optional_int(data.get("image_width")),
        image_height=_optional_int(data.get("image_height")),
        model=_optional_string(data.get("model")),
        reproj_rms=_optional_float(data.get("reproj_rms")),
        camera_params=tuple(float(value) for value in calibration.camera_params),
        distortion_coefficients=_distortion_coefficients(data),
        latest_run_dir=_latest_calibration_run_dir(root),
    )


def read_camera_calibration_yaml_text(path: Union[Path, str]) -> str:
    """Read calibration YAML text for a raw artifact view."""

    target = Path(path).expanduser()
    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        raise CameraCalibrationFlowError(
            f"Could not read calibration YAML: {exc}"
        ) from exc


def normalize_source(source: str) -> str:
    """Normalize a user-facing calibration source value."""

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
        raise CameraCalibrationFlowError(
            "Calibration source must be one of: "
            + ", ".join(CALIBRATION_SOURCE_CHOICES)
            + "."
        ) from exc


def realsense_dependency_available() -> bool:
    """Return whether ``pyrealsense2`` is importable in the current environment."""

    return importlib.util.find_spec("pyrealsense2") is not None


def _latest_path(paths: tuple[Path, ...]) -> Path:
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _latest_calibration_run_dir(project_root: Path) -> Optional[Path]:
    runs_dir = project_root / "calib" / "runs"
    if not runs_dir.is_dir():
        return None
    run_dirs = tuple(path for path in runs_dir.iterdir() if path.is_dir())
    if not run_dirs:
        return None
    return _latest_path(run_dirs)


def _path_tuple(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _optional_path(path: Optional[Union[Path, str]]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser()


def _path_mtime_ns(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _process_status_detail(stage: Any) -> str:
    details = tuple(stage.errors or stage.warnings or (stage.message,))
    return " ".join(str(detail) for detail in details if str(detail).strip())


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CameraCalibrationFlowError(
            f"Malformed calibration YAML: {exc}"
        ) from exc
    except OSError as exc:
        raise CameraCalibrationFlowError(
            f"Could not read calibration YAML: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise CameraCalibrationFlowError("Calibration YAML must contain a mapping.")
    return data


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _optional_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _distortion_coefficients(data: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    distortion = data.get("distortion_coefficients")
    if not isinstance(distortion, Mapping):
        return ()
    values: list[tuple[str, float]] = []
    for name in ("k1", "k2", "p1", "p2", "k3"):
        parsed = _optional_float(distortion.get(name))
        if parsed is not None:
            values.append((name, parsed))
    return tuple(values)


def _realsense_module_token(realsense_available: Optional[bool]) -> Any:
    available = (
        realsense_dependency_available()
        if realsense_available is None
        else bool(realsense_available)
    )
    return object() if available else None


def _video_arg(video_path: Optional[Union[Path, str]]) -> Optional[str]:
    if video_path is None:
        return None
    text = str(video_path).strip()
    if not text:
        return None
    return str(Path(text).expanduser())


def _validate_camera_calibration_board(config: CameraCalibrationConfig) -> None:
    errors = _camera_calibration_board_errors(config)
    if errors:
        raise CameraCalibrationFlowError(" ".join(errors))


def _validate_camera_calibration_guidance(
    config: CameraCalibrationConfig,
) -> tuple[str, int, int]:
    errors = _camera_calibration_guidance_errors(config)
    if errors:
        raise CameraCalibrationFlowError(" ".join(errors))
    rows, cols = parse_grid_shape(config.coverage_grid)
    return (
        f"{rows}x{cols}",
        int(config.samples_per_cell),
        int(config.guided_auto_cooldown),
    )


def _camera_calibration_board_errors(
    config: CameraCalibrationConfig,
) -> list[str]:
    errors: list[str] = []
    squares_x = _coerce_int(config.squares_x, "--squares-x", errors)
    squares_y = _coerce_int(config.squares_y, "--squares-y", errors)
    square_length_mm = _coerce_float(
        config.square_length_mm,
        "--square-length-mm",
        errors,
    )
    marker_length_mm = _coerce_float(
        config.marker_length_mm,
        "--marker-length-mm",
        errors,
    )
    dictionary_name = str(config.dictionary_name).strip()

    if squares_x is not None and squares_x < 2:
        errors.append("--squares-x must be at least 2.")
    if squares_y is not None and squares_y < 2:
        errors.append("--squares-y must be at least 2.")
    if square_length_mm is not None and square_length_mm <= 0:
        errors.append("--square-length-mm must be a positive millimetre value.")
    if marker_length_mm is not None and marker_length_mm <= 0:
        errors.append("--marker-length-mm must be a positive millimetre value.")
    if (
        square_length_mm is not None
        and marker_length_mm is not None
        and marker_length_mm >= square_length_mm
    ):
        errors.append("--marker-length-mm must be smaller than --square-length-mm.")
    if not dictionary_name:
        errors.append("--dict must be a non-empty ArUco dictionary name.")
    else:
        try:
            get_dictionary(dictionary_name)
        except CharucoCalibrationError as exc:
            errors.append(str(exc))

    return errors


def _camera_calibration_guidance_errors(
    config: CameraCalibrationConfig,
) -> list[str]:
    errors: list[str] = []
    try:
        parse_grid_shape(config.coverage_grid)
    except ValueError as exc:
        errors.append(str(exc))

    samples_per_cell = _coerce_int(
        config.samples_per_cell,
        "--samples-per-cell",
        errors,
    )
    guided_auto_cooldown = _coerce_int(
        config.guided_auto_cooldown,
        "--guided-auto-cooldown",
        errors,
    )
    if samples_per_cell is not None and samples_per_cell < 1:
        errors.append("--samples-per-cell must be at least 1.")
    if guided_auto_cooldown is not None and guided_auto_cooldown < 0:
        errors.append("--guided-auto-cooldown must be zero or greater.")
    return errors


def _coerce_int(value: Any, label: str, errors: list[str]) -> Optional[int]:
    if isinstance(value, bool):
        errors.append(f"{label} must be an integer.")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an integer.")
        return None
    if isinstance(value, float) and not value.is_integer():
        errors.append(f"{label} must be an integer.")
        return None
    return parsed


def _coerce_float(value: Any, label: str, errors: list[str]) -> Optional[float]:
    if isinstance(value, bool):
        errors.append(f"{label} must be numeric.")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be numeric.")
        return None
    if not math.isfinite(parsed):
        errors.append(f"{label} must be finite.")
        return None
    return parsed


def _require_int(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
) -> Optional[int]:
    value = data.get(field)
    if isinstance(value, bool):
        errors.append(f"ChArUco metadata field {field!r} must be an integer.")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"ChArUco metadata field {field!r} must be an integer.")
        return None
    if isinstance(value, float) and not value.is_integer():
        errors.append(f"ChArUco metadata field {field!r} must be an integer.")
        return None
    return parsed


def _require_float(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
) -> Optional[float]:
    value = data.get(field)
    if isinstance(value, bool):
        errors.append(f"ChArUco metadata field {field!r} must be numeric.")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"ChArUco metadata field {field!r} must be numeric.")
        return None
    if not math.isfinite(parsed):
        errors.append(f"ChArUco metadata field {field!r} must be finite.")
        return None
    return parsed


def _require_string(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
) -> Optional[str]:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(
            f"ChArUco metadata field {field!r} must be a non-empty string."
        )
        return None
    return value.strip()


def _format_number(value: float) -> str:
    return f"{float(value):g}"
