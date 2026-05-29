"""GUI-independent readiness helpers for guided camera calibration.

The helpers in this module prepare users to run ``posetag-calib-charuco``
without moving camera capture, ChArUco detection, calibration solving, or YAML
writing into GUI code.
"""

from __future__ import annotations

import importlib.util
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import yaml

from posetag.pipelines.charuco_calibration import (
    CharucoCalibrationError,
    get_dictionary,
    validate_capture_args,
)
from posetag.workflows.charuco_setup import (
    DEFAULT_DICTIONARY,
    DEFAULT_MARKER_LENGTH_MM,
    DEFAULT_SQUARE_LENGTH_MM,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
)


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

    source = normalize_source(config.source)
    video = _video_arg(config.video_path)
    if source == SOURCE_VIDEO and video is None:
        raise CameraCalibrationFlowError(
            "--video path is required when --source=video"
        )
    _validate_camera_calibration_board(config)

    parts: list[str] = [
        "posetag-calib-charuco",
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
        ]
    )
    return shlex.join(parts)


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


def _path_tuple(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


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
