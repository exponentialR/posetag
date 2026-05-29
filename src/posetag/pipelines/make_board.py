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
from math import atan2, degrees
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

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


@dataclass(frozen=True)
class NonplanarTag:
    """A selected tag whose recovered z offset exceeds the planarity threshold."""

    tag_id: int
    z_offset_m: float
    threshold_m: float


@dataclass(frozen=True)
class BoardLayoutResult:
    """Computed board-frame tag entries plus planarity diagnostics."""

    entries: tuple[dict[str, float], ...]
    nonplanar_tags: tuple[NonplanarTag, ...]


@dataclass(frozen=True)
class SavedBoardDefinition:
    """Paths and registry update from writing a board definition."""

    board_yaml_path: Path
    registry_path: Path
    registry_update: RegistryUpdate
    board: Mapping[str, Any]


class FrameSource:
    """Small wrapper around a camera/video read/stop pair."""

    def __init__(
        self,
        read: Callable[[], Optional[np.ndarray]],
        stop: Callable[[], None],
        *,
        source: str,
    ) -> None:
        self._read = read
        self._stop = stop
        self.source = source
        self.closed = False

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or ``None`` when no frame is available."""

        return self._read()

    def stop(self) -> None:
        """Release the underlying capture source once."""

        if self.closed:
            return
        self.closed = True
        self._stop()


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


def open_frame_source(
    *,
    source: str,
    camera_index: int = 0,
    video_path: Optional[Union[Path, str]] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    realsense_module: Any = None,
) -> FrameSource:
    """Open a BGR frame source for board building.

    The CLI and GUI use this helper so camera/video source handling stays in
    the board-building pipeline rather than in presentation code.
    """

    import cv2

    if source == "realsense":
        rs_module = realsense_module
        if rs_module is None:
            try:
                import pyrealsense2 as rs_module  # type: ignore[no-redef]
            except Exception as exc:
                raise MakeBoardError(
                    "pyrealsense2 is not available; install PoseTag with the "
                    "'realsense' extra or use --source opencv|video"
                ) from exc
        pipe, cfg = rs_module.pipeline(), rs_module.config()
        cfg.enable_stream(
            rs_module.stream.color,
            int(width),
            int(height),
            rs_module.format.bgr8,
            int(fps),
        )
        pipe.start(cfg)

        def _read_realsense() -> Optional[np.ndarray]:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            return None if not color else np.asanyarray(color.get_data())

        return FrameSource(_read_realsense, pipe.stop, source=source)

    if source == "opencv":
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            raise MakeBoardError(f"Could not open camera index {camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))

        def _read_opencv() -> Optional[np.ndarray]:
            ok, frame = cap.read()
            return frame if ok else None

        return FrameSource(_read_opencv, cap.release, source=source)

    if source == "video":
        if video_path is None:
            raise MakeBoardError("--video path is required when --source=video")
        expanded = Path(video_path).expanduser()
        cap = cv2.VideoCapture(str(expanded))
        if not cap.isOpened():
            raise MakeBoardError(f"Could not open video: {video_path}")

        def _read_video() -> Optional[np.ndarray]:
            ok, frame = cap.read()
            return frame if ok else None

        return FrameSource(_read_video, cap.release, source=source)

    raise MakeBoardError("Board-building source must be opencv, realsense, or video.")


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


def load_detector_class() -> Any:
    """Load the optional pupil-apriltags detector class."""

    try:
        from pupil_apriltags import Detector
    except Exception as exc:
        raise MakeBoardError(
            "pupil-apriltags is not available; install PoseTag with the "
            "'apriltags' extra or run `python -m pip install pupil-apriltags`."
        ) from exc
    return Detector


def create_apriltag_detector(family: str, detector_class: Optional[Any] = None) -> Any:
    """Create the AprilTag detector used by board-building workflows."""

    Detector = detector_class or load_detector_class()
    try:
        return Detector(
            families=family,
            nthreads=4,
            quad_decimate=1.0,
            refine_edges=True,
        )
    except Exception as exc:
        raise MakeBoardError(
            f"Could not initialize AprilTag detector for family '{family}': {exc}"
        ) from exc


def detect_frame_tags(
    frame_bgr: np.ndarray,
    detector: Any,
    *,
    calibration: Optional[CalibrationData] = None,
    tag_size_m: Optional[float] = None,
    estimate_pose: bool = False,
) -> tuple[Any, ...]:
    """Detect AprilTags in a BGR frame using the existing detector API."""

    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if estimate_pose:
        if calibration is None or tag_size_m is None:
            raise MakeBoardError("Calibration and tag size are required for pose estimation.")
        return tuple(
            detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=calibration.camera_params,
                tag_size=float(tag_size_m),
            )
        )
    return tuple(detector.detect(gray, estimate_tag_pose=False))


def annotate_detections(
    frame_bgr: np.ndarray,
    detections: Sequence[Any],
    *,
    guidance_lines: Sequence[str] = (),
) -> np.ndarray:
    """Return a BGR preview frame with detected tag outlines and IDs."""

    import cv2

    annotated = frame_bgr.copy()
    for detection in detections:
        corners = getattr(detection, "corners", None)
        if corners is None:
            continue
        pts = np.asarray(corners).astype(int)
        cv2.polylines(annotated, [pts], True, (0, 170, 0), 2)
        cv2.putText(
            annotated,
            str(int(getattr(detection, "tag_id"))),
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 220),
            2,
        )
    if guidance_lines:
        _draw_guidance_overlay(annotated, guidance_lines)
    return annotated


def _draw_guidance_overlay(
    frame_bgr: np.ndarray,
    guidance_lines: Sequence[str],
) -> None:
    """Draw wrapped, readable guidance text directly onto the preview frame."""

    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    margin = 12
    padding_x = 10
    padding_y = 8
    line_gap = 7
    height, width = frame_bgr.shape[:2]
    if height <= margin * 2 or width <= margin * 2:
        return
    panel_width = min(width - (margin * 2), max(260, int(width * 0.56)))
    text_width = max(40, panel_width - (padding_x * 2))

    wrapped: list[str] = []
    for line in guidance_lines:
        wrapped.extend(
            _wrap_overlay_text(
                str(line),
                max_width_px=text_width,
                font=font,
                scale=scale,
                thickness=thickness,
            )
        )
    if not wrapped:
        return

    text_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
    panel_height = (padding_y * 2) + (len(wrapped) * text_height)
    panel_height += max(0, len(wrapped) - 1) * line_gap
    panel_height = min(panel_height, height - (margin * 2))

    x0 = margin
    y0 = margin
    x1 = min(width - margin, x0 + panel_width)
    y1 = min(height - margin, y0 + panel_height)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (22, 32, 42), -1)
    cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (70, 156, 190), 1)

    baseline_y = y0 + padding_y + text_height
    max_text_y = y1 - padding_y
    for line in wrapped:
        if baseline_y > max_text_y:
            break
        cv2.putText(
            frame_bgr,
            line,
            (x0 + padding_x, baseline_y),
            font,
            scale,
            (245, 250, 252),
            thickness,
            lineType=cv2.LINE_AA,
        )
        baseline_y += text_height + line_gap


def _wrap_overlay_text(
    text: str,
    *,
    max_width_px: int,
    font: int,
    scale: float,
    thickness: int,
) -> list[str]:
    """Wrap OpenCV overlay text to fit inside the camera frame."""

    import cv2

    words = str(text).split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width_px:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from rotation and translation."""

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def inv_se3(transform: np.ndarray) -> np.ndarray:
    """Invert a homogeneous SE(3) transform."""

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def board_entries_from_detections(
    detections: Sequence[Any],
    selected_ids: Sequence[int],
    origin_id: int,
    *,
    z_threshold_m: float = 0.01,
) -> BoardLayoutResult:
    """Compute board-frame tag entries from pose-estimated detections."""

    selected = tuple(int(tag_id) for tag_id in selected_ids)
    if not selected:
        raise MakeBoardError("Need at least one selected tag ID.")
    if int(origin_id) not in selected:
        raise MakeBoardError("Origin must be one of the selected IDs.")

    det_map = {int(getattr(det, "tag_id")): det for det in detections}
    transforms: dict[int, np.ndarray] = {}
    for tag_id in selected:
        detection = det_map.get(tag_id)
        if detection is None:
            raise MakeBoardError(f"Selected tag ID {tag_id} was not detected.")
        pose_r = getattr(detection, "pose_R", None)
        pose_t = getattr(detection, "pose_t", None)
        if pose_r is None or pose_t is None:
            raise MakeBoardError(f"Missing pose for selected tag ID {tag_id}.")
        transforms[tag_id] = se3(
            np.asarray(pose_r, dtype=float),
            np.asarray(pose_t, dtype=float).reshape(3),
        )

    origin_from_camera = inv_se3(transforms[int(origin_id)])
    entries: list[dict[str, float]] = []
    nonplanar: list[NonplanarTag] = []
    for tag_id in selected:
        origin_from_tag = origin_from_camera @ transforms[tag_id]
        translation = origin_from_tag[:3, 3]
        rotation = origin_from_tag[:3, :3]
        yaw_deg = degrees(atan2(rotation[1, 0], rotation[0, 0]))
        z_offset = float(translation[2])
        if abs(z_offset) > float(z_threshold_m):
            nonplanar.append(
                NonplanarTag(
                    tag_id=tag_id,
                    z_offset_m=z_offset,
                    threshold_m=float(z_threshold_m),
                )
            )
        entries.append(
            {
                "id": int(tag_id),
                "cx": float(translation[0]),
                "cy": float(translation[1]),
                "yaw_deg": float(yaw_deg),
            }
        )

    entries.sort(key=lambda item: item["id"])
    return BoardLayoutResult(
        entries=tuple(entries),
        nonplanar_tags=tuple(nonplanar),
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
    requested_entries = tuple(entries)
    requested_tag_ids = {str(int(entry["id"])) for entry in requested_entries}
    updated = 0
    conflicts: list[RegistryConflict] = []

    for tag_id, current in tuple(tags.items()):
        if (
            isinstance(current, Mapping)
            and current.get("yaml") == requested_yaml
            and str(tag_id) not in requested_tag_ids
        ):
            del tags[tag_id]

    for entry in requested_entries:
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


def save_board_definition(
    paths: MakeBoardPaths,
    *,
    object_name: str,
    family: str,
    tag_size_mm: float,
    origin_id: int,
    entries: Sequence[Mapping[str, Any]],
) -> SavedBoardDefinition:
    """Write board YAML and update the tag registry using existing schemas."""

    board = build_board_yaml(
        object_name=object_name,
        family=family,
        tag_size_mm=tag_size_mm,
        origin_id=origin_id,
        entries=entries,
    )
    write_board_yaml(paths.board_yaml_path, board)
    registry = load_registry(paths.registry_path)
    registry_update = update_registry_entries(
        registry,
        board["tags"],
        object_name,
        paths.board_yaml_path,
    )
    save_registry(paths.registry_path, registry)
    return SavedBoardDefinition(
        board_yaml_path=paths.board_yaml_path,
        registry_path=paths.registry_path,
        registry_update=registry_update,
        board=board,
    )
