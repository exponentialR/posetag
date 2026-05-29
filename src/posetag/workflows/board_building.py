"""GUI-independent helpers for guided object board building.

This module validates Stage 4 dashboard values and builds copyable
``posetag-make-board`` commands. AprilTag detection, pose estimation,
board-frame construction, board YAML writing, and registry updates remain in
the existing ``posetag-make-board`` workflow.
"""

from __future__ import annotations

import importlib.util
import datetime
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml

from posetag.pipelines.make_board import (
    FrameSource,
    MakeBoardError,
    annotate_detections,
    board_entries_from_detections,
    create_apriltag_detector,
    detect_frame_tags,
    load_calibration_yaml,
    open_frame_source,
    prepare_project_paths,
    resolve_calibration_path,
    save_board_definition,
    validate_source_args,
)
from posetag.workflows.status import WorkflowStatus, inspect_object_tags


SOURCE_OPENCV = "opencv"
SOURCE_REALSENSE = "realsense"
SOURCE_VIDEO = "video"
BOARD_SOURCE_CHOICES = (SOURCE_OPENCV, SOURCE_REALSENSE, SOURCE_VIDEO)
BOARD_SOURCE_LABELS = {
    SOURCE_OPENCV: "webcam/OpenCV",
    SOURCE_REALSENSE: "RealSense",
    SOURCE_VIDEO: "video",
}
DEFAULT_OBJECT_LABEL = "connection_plate_white"
DEFAULT_SIDE_LABEL = "sideA"
DEFAULT_SIDE_LABELS = ("sideA", "sideB", "sideC", "sideD", "sideE", "sideF")
DEFAULT_OBJECT_NAME = "connection_plate_white_sideA"
DEFAULT_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_MM = 40.0
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_Z_THRESHOLD_M = 0.01
BOARD_BATCH_DRAFT_FILENAME = "board_building_queue.yaml"
BOARD_BATCH_DRAFT_KIND = "posetag_board_building_queue"
BOARD_PROCESS_NOT_STARTED = "not_started"
BOARD_PROCESS_RUNNING = "running"
BOARD_PROCESS_FINISHED = "finished"
BOARD_PROCESS_FAILED_CANCELLED = "failed_cancelled"
BOARD_BUILDING_GUIDANCE = (
    "Run Board Builder starts the existing posetag-make-board workflow. In the "
    "OpenCV preview, ENTER captures the current frame and ESC quits cleanly "
    "without writing board YAML. After capture, send the selected tag IDs and "
    "origin tag through the prompt response box."
)
BOARD_NATIVE_CAPTURE_GUIDANCE = (
    "Guided Capture uses the same board-building detector, board-frame math, "
    "YAML schema, and tag-registry writer as posetag-make-board. Hold the "
    "object steady until the visible tag IDs stabilize, then confirm the IDs "
    "and origin tag in the dialog."
)


class BoardBuildingFlowError(ValueError):
    """User-facing validation error for guided board building."""


@dataclass(frozen=True)
class BoardBuildingConfig:
    """User-facing object board-building command state."""

    project_root: Union[Path, str]
    object_name: str = DEFAULT_OBJECT_NAME
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM
    family: str = DEFAULT_FAMILY
    calibration_path: Optional[Union[Path, str]] = None
    source: str = SOURCE_OPENCV
    camera_index: int = DEFAULT_CAMERA_INDEX
    video_path: Optional[Union[Path, str]] = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    out_dir: Optional[Union[Path, str]] = None
    registry_path: Optional[Union[Path, str]] = None
    save_shot: bool = False
    shots_dir: Optional[Union[Path, str]] = None
    z_threshold_m: float = DEFAULT_Z_THRESHOLD_M
    allow_nonplanar: bool = False
    require_object_tags: bool = True


@dataclass(frozen=True)
class BoardBuildingOutputStatus:
    """Expected Stage 4 output paths and current existence state."""

    board_yaml_path: Path
    registry_path: Path
    shots_dir: Path
    board_yaml_exists: bool
    registry_exists: bool
    shots_dir_exists: bool
    checked_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BoardBuildingReadiness:
    """Display-ready readiness state for guided board building."""

    ready: bool
    command_preview: str
    calibration_path: Path
    output_status: BoardBuildingOutputStatus
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def expected_board_yaml(self) -> Path:
        """Return the board YAML path expected from the current form values."""

        return self.output_status.board_yaml_path

    @property
    def expected_registry(self) -> Path:
        """Return the tag registry path expected from the current form values."""

        return self.output_status.registry_path

    @property
    def expected_shots_dir(self) -> Path:
        """Return the audit-shot directory expected from the current form values."""

        return self.output_status.shots_dir


@dataclass(frozen=True)
class BoardBuildingLaunchSpec:
    """Editable-install-safe process launch details for board building."""

    program: str
    arguments: tuple[str, ...]
    display_command: str
    expected_board_yaml: Path
    expected_registry: Path


@dataclass(frozen=True)
class BoardBuildingProcessState:
    """GUI-independent board-building child-process state."""

    state: str
    label: str
    message: str
    running: bool = False
    success: bool = False
    expected_board_yaml: Optional[Path] = None
    expected_registry: Optional[Path] = None
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class NativeBoardCaptureObservation:
    """One native guided board-capture observation."""

    frame_bgr: Optional[Any]
    annotated_frame_bgr: Optional[Any]
    detections: tuple[Any, ...]
    detected_ids: tuple[int, ...]
    pose_ready_ids: tuple[int, ...]
    guidance: str
    ready_to_capture: bool
    stable_frames: int
    source_exhausted: bool = False


@dataclass(frozen=True)
class NativeBoardCaptureResult:
    """Outputs from saving a native guided board capture."""

    board_yaml_path: Path
    registry_path: Path
    selected_ids: tuple[int, ...]
    origin_id: int
    registry_updated: int
    registry_conflicts: tuple[Any, ...]
    nonplanar_warnings: tuple[Any, ...]
    audit_shot_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class BoardBatchRow:
    """User-facing object/instance/sides row for batch board capture."""

    object_label: str
    instances: str = ""
    sides: str = DEFAULT_SIDE_LABEL
    tag_size_mm: Optional[float] = None


@dataclass(frozen=True)
class BoardBatchItem:
    """One concrete board definition to capture in a batch."""

    object_label: str
    instance_label: str
    side_label: str
    object_name: str
    tag_size_mm: Optional[float] = None


@dataclass(frozen=True)
class BoardBatchDraft:
    """Persisted Stage 4 queue and capture settings before capture starts."""

    path: Path
    rows: tuple[BoardBatchRow, ...]
    family: str = DEFAULT_FAMILY
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM
    source: str = SOURCE_OPENCV
    camera_index: int = DEFAULT_CAMERA_INDEX
    video_path: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    calibration_path: str = ""
    out_dir: str = ""
    registry_path: str = ""
    save_shot: bool = False
    shots_dir: str = ""
    z_threshold_m: float = DEFAULT_Z_THRESHOLD_M
    allow_nonplanar: bool = False


@dataclass(frozen=True)
class _ValidatedBoardBuildingInputs:
    object_name: str
    family: str
    tag_size_mm: float
    source: str
    camera_index: int
    video_path: Optional[Path]
    width: int
    height: int
    fps: int
    z_threshold_m: float


class NativeBoardCaptureSession:
    """GUI-independent runtime for native guided board capture."""

    def __init__(
        self,
        config: BoardBuildingConfig,
        *,
        detector: Optional[Any] = None,
        frame_source: Optional[FrameSource] = None,
        calibration: Optional[Any] = None,
        min_tags: int = 2,
        stable_frames_required: int = 8,
    ) -> None:
        self.config = config
        self.validated = _validated_inputs(config)
        self.min_tags = max(1, int(min_tags))
        self.stable_frames_required = max(1, int(stable_frames_required))
        self.calibration = calibration or load_calibration_yaml(
            expected_calibration_path(config)
        )
        self.detector = detector or create_apriltag_detector(self.validated.family)
        self.frame_source = frame_source or open_frame_source(
            source=self.validated.source,
            camera_index=self.validated.camera_index,
            video_path=self.validated.video_path,
            width=self.validated.width,
            height=self.validated.height,
            fps=self.validated.fps,
        )
        self.tag_size_m = self.validated.tag_size_mm / 1000.0
        self._last_pose_ids: tuple[int, ...] = ()
        self._stable_frames = 0
        self._last_observation: Optional[NativeBoardCaptureObservation] = None
        self._captured_observation: Optional[NativeBoardCaptureObservation] = None

    def read_observation(self) -> NativeBoardCaptureObservation:
        """Read, detect, annotate, and score the next frame."""

        frame = self.frame_source.read()
        if frame is None:
            exhausted = self.validated.source == SOURCE_VIDEO
            guidance = (
                "Video ended before a stable board view was captured."
                if exhausted
                else "Waiting for a camera frame."
            )
            observation = NativeBoardCaptureObservation(
                frame_bgr=None,
                annotated_frame_bgr=None,
                detections=(),
                detected_ids=(),
                pose_ready_ids=(),
                guidance=guidance,
                ready_to_capture=False,
                stable_frames=0,
                source_exhausted=exhausted,
            )
            self._last_observation = observation
            return observation

        detections = detect_frame_tags(
            frame,
            self.detector,
            calibration=self.calibration,
            tag_size_m=self.tag_size_m,
            estimate_pose=True,
        )
        detected_ids = tuple(sorted({int(getattr(det, "tag_id")) for det in detections}))
        pose_ready_ids = tuple(
            sorted(
                {
                    int(getattr(det, "tag_id"))
                    for det in detections
                    if getattr(det, "pose_R", None) is not None
                    and getattr(det, "pose_t", None) is not None
                }
            )
        )

        if len(pose_ready_ids) < self.min_tags:
            self._stable_frames = 0
            self._last_pose_ids = pose_ready_ids
            guidance = (
                f"Need at least {self.min_tags} pose-estimated tags. "
                "Move the object so more attached tags are visible."
            )
        else:
            if pose_ready_ids == self._last_pose_ids:
                self._stable_frames += 1
            else:
                self._last_pose_ids = pose_ready_ids
                self._stable_frames = 1
            if self._stable_frames >= self.stable_frames_required:
                guidance = "Stable tag layout detected. Ready to capture."
            else:
                guidance = (
                    "Hold still while PoseTag verifies a stable board view "
                    f"({self._stable_frames}/{self.stable_frames_required})."
                )

        id_summary = _format_id_tuple(detected_ids) if detected_ids else "none"
        annotated = annotate_detections(
            frame,
            detections,
            guidance_lines=(guidance, f"Detected IDs: {id_summary}"),
        )
        observation = NativeBoardCaptureObservation(
            frame_bgr=frame,
            annotated_frame_bgr=annotated,
            detections=tuple(detections),
            detected_ids=detected_ids,
            pose_ready_ids=pose_ready_ids,
            guidance=guidance,
            ready_to_capture=self._stable_frames >= self.stable_frames_required
            and len(pose_ready_ids) >= self.min_tags,
            stable_frames=self._stable_frames,
        )
        self._last_observation = observation
        return observation

    def capture_current(
        self,
        observation: Optional[NativeBoardCaptureObservation] = None,
    ) -> NativeBoardCaptureObservation:
        """Freeze the latest observation for ID/origin confirmation."""

        selected = observation or self._last_observation
        if selected is None or not selected.pose_ready_ids:
            raise MakeBoardError("No pose-estimated tags are available to capture.")
        self._captured_observation = selected
        return selected

    @property
    def captured_observation(self) -> Optional[NativeBoardCaptureObservation]:
        """Return the frozen capture observation, if any."""

        return self._captured_observation

    def clear_capture(self) -> None:
        """Discard the frozen observation and continue live capture."""

        self._captured_observation = None

    def set_capture_config(self, config: BoardBuildingConfig) -> None:
        """Update per-board capture parameters without reopening the frame source."""

        previous_calibration = expected_calibration_path(self.config)
        next_calibration = expected_calibration_path(config)
        validated = _validated_inputs(config)
        if validated.family != self.validated.family:
            self.detector = create_apriltag_detector(validated.family)
        if next_calibration != previous_calibration:
            self.calibration = load_calibration_yaml(next_calibration)
        self.config = config
        self.validated = validated
        self.tag_size_m = validated.tag_size_mm / 1000.0
        self._last_pose_ids = ()
        self._stable_frames = 0
        self.clear_capture()

    def save_capture(
        self,
        *,
        selected_ids: Sequence[int],
        origin_id: int,
        config: Optional[BoardBuildingConfig] = None,
    ) -> NativeBoardCaptureResult:
        """Save the frozen capture as board YAML and registry entries."""

        observation = self._captured_observation or self._last_observation
        if observation is None:
            raise MakeBoardError("Capture a stable board view before saving.")
        save_config = config or self.config
        save_validated = _validated_inputs(save_config)
        layout = board_entries_from_detections(
            observation.detections,
            selected_ids,
            origin_id,
            z_threshold_m=save_validated.z_threshold_m,
        )
        if layout.nonplanar_tags and not save_config.allow_nonplanar:
            first = layout.nonplanar_tags[0]
            raise MakeBoardError(
                "Captured tags are not planar enough for this board definition: "
                f"tag {first.tag_id} has z offset {first.z_offset_m:.3f} m. "
                "Re-capture a flatter view or enable Allow non-planar warning."
            )

        paths = prepare_project_paths(
            project_root=save_config.project_root,
            object_name=save_validated.object_name,
            out_dir=save_config.out_dir,
            shots_dir=save_config.shots_dir,
            registry=save_config.registry_path,
            save_shot=save_config.save_shot,
        )
        saved = save_board_definition(
            paths,
            object_name=save_validated.object_name,
            family=save_validated.family,
            tag_size_mm=save_validated.tag_size_mm,
            origin_id=origin_id,
            entries=layout.entries,
        )
        audit_paths = self._save_audit_shot(
            paths.shots_dir,
            observation,
            save_config=save_config,
            object_name=save_validated.object_name,
        )
        return NativeBoardCaptureResult(
            board_yaml_path=saved.board_yaml_path,
            registry_path=saved.registry_path,
            selected_ids=tuple(int(tag_id) for tag_id in selected_ids),
            origin_id=int(origin_id),
            registry_updated=saved.registry_update.updated,
            registry_conflicts=saved.registry_update.conflicts,
            nonplanar_warnings=layout.nonplanar_tags,
            audit_shot_paths=audit_paths,
        )

    def close(self) -> None:
        """Release the frame source."""

        self.frame_source.stop()

    def _save_audit_shot(
        self,
        shots_dir: Path,
        observation: NativeBoardCaptureObservation,
        *,
        save_config: BoardBuildingConfig,
        object_name: str,
    ) -> tuple[Path, ...]:
        if not save_config.save_shot or observation.frame_bgr is None:
            return ()
        import cv2

        shots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = shots_dir / f"{object_name}_{timestamp}"
        raw_path = Path(f"{base}_raw.png")
        annotated_path = Path(f"{base}_ann.png")
        cv2.imwrite(str(raw_path), observation.frame_bgr)
        if observation.annotated_frame_bgr is not None:
            cv2.imwrite(str(annotated_path), observation.annotated_frame_bgr)
            return (raw_path, annotated_path)
        return (raw_path,)


def default_calibration_path(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 4 calibration YAML path without creating it."""

    return Path(project_root).expanduser() / "calib" / "calib_color.yaml"


def default_boards_dir(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 4 boards directory without creating it."""

    return Path(project_root).expanduser() / "boards"


def infer_latest_object_tag_size_mm(
    project_root: Union[Path, str],
) -> Optional[float]:
    """Infer the most recent Stage 3 printed tag size from pattern filenames."""

    patterns_dir = default_boards_dir(project_root) / "patterns"
    candidates: list[tuple[int, float]] = []
    for path in patterns_dir.glob("*mm_*"):
        if not path.is_file():
            continue
        match = re.search(r"_(\d+(?:\.\d+)?)mm_", path.name)
        if match is None:
            continue
        try:
            tag_size_mm = float(match.group(1))
        except ValueError:
            continue
        if tag_size_mm <= 0 or not math.isfinite(tag_size_mm):
            continue
        candidates.append((_path_mtime_ns(path) or 0, tag_size_mm))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def compose_board_object_name(
    object_label: str,
    side_label: Optional[str] = DEFAULT_SIDE_LABEL,
) -> str:
    """Compose the board-definition ID passed to ``posetag-make-board``.

    The existing CLI stores one board YAML per ``--object_name``. The GUI uses
    this helper to make the user-facing object and side/face labels explicit
    while preserving that single CLI value and output schema.
    """

    object_part = _clean_board_label_part(object_label, "Object label")
    side_part = _clean_board_label_part(
        side_label,
        "Side/face label",
        allow_empty=True,
    )
    return f"{object_part}_{side_part}" if side_part else object_part


def next_default_side_label(
    project_root: Union[Path, str],
    object_label: str,
    *,
    out_dir: Optional[Union[Path, str]] = None,
    side_labels: tuple[str, ...] = DEFAULT_SIDE_LABELS,
) -> str:
    """Return the first default side label without an existing board YAML."""

    boards_dir = _optional_path(out_dir) or default_boards_dir(project_root)
    for side_label in side_labels:
        board_name = compose_board_object_name(object_label, side_label)
        if not (boards_dir / f"{board_name}.yaml").exists():
            return side_label
    return f"side{chr(ord('A') + len(side_labels))}"


def build_board_batch_items(
    rows: Sequence[BoardBatchRow],
) -> tuple[BoardBatchItem, ...]:
    """Expand batch rows into concrete board-definition names."""

    items: list[BoardBatchItem] = []
    seen: set[str] = set()
    for row in rows:
        object_label = _clean_board_label_part(row.object_label, "Object label")
        instances = parse_instance_labels(row.instances)
        sides = parse_side_labels(row.sides)
        tag_size_mm = _normalize_batch_tag_size(row.tag_size_mm)
        for instance_label in instances:
            object_part = (
                compose_board_object_name(object_label, instance_label)
                if instance_label
                else object_label
            )
            for side_label in sides:
                object_name = compose_board_object_name(object_part, side_label)
                if object_name in seen:
                    raise BoardBuildingFlowError(
                        f"Duplicate board name in batch: {object_name}."
                    )
                seen.add(object_name)
                items.append(
                    BoardBatchItem(
                        object_label=object_label,
                        instance_label=instance_label,
                        side_label=side_label,
                        object_name=object_name,
                        tag_size_mm=tag_size_mm,
                    )
                )
    return tuple(items)


def parse_board_batch_rows(text: str) -> tuple[BoardBatchRow, ...]:
    """Parse ``object | instances | sides | tag_size_mm`` batch rows."""

    rows: list[BoardBatchRow] = []
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) not in {3, 4}:
            raise BoardBuildingFlowError(
                "Batch rows must use: object label | instances | sides/faces "
                "| optional tag size mm "
                f"(line {line_number})."
            )
        rows.append(
            BoardBatchRow(
                object_label=parts[0],
                instances=parts[1],
                sides=parts[2],
                tag_size_mm=_normalize_batch_tag_size(parts[3])
                if len(parts) == 4 and parts[3]
                else None,
            )
        )
    return tuple(rows)


def default_board_batch_draft_path(project_root: Union[Path, str]) -> Path:
    """Return the Stage 4 draft queue path without creating board YAML files."""

    return default_boards_dir(project_root) / BOARD_BATCH_DRAFT_FILENAME


def save_board_batch_draft(
    project_root: Union[Path, str],
    rows: Sequence[BoardBatchRow],
    config: BoardBuildingConfig,
) -> Path:
    """Persist queued board names/settings before any board capture starts."""

    normalized_rows = tuple(rows)
    build_board_batch_items(normalized_rows)
    path = default_board_batch_draft_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "kind": BOARD_BATCH_DRAFT_KIND,
        "updated": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {
            "family": str(config.family).strip() or DEFAULT_FAMILY,
            "tag_size_mm": float(config.tag_size_mm),
            "source": normalize_source(config.source),
            "camera_index": int(config.camera_index),
            "video_path": _path_text(config.video_path),
            "width": int(config.width),
            "height": int(config.height),
            "fps": int(config.fps),
            "calibration_path": _path_text(config.calibration_path),
            "out_dir": _path_text(config.out_dir),
            "registry_path": _path_text(config.registry_path),
            "save_shot": bool(config.save_shot),
            "shots_dir": _path_text(config.shots_dir),
            "z_threshold_m": float(config.z_threshold_m),
            "allow_nonplanar": bool(config.allow_nonplanar),
        },
        "rows": [_board_batch_row_to_mapping(row) for row in normalized_rows],
    }
    _atomic_write_draft_yaml(path, data)
    return path


def load_board_batch_draft(
    project_root: Union[Path, str],
) -> Optional[BoardBatchDraft]:
    """Load the persisted Stage 4 draft queue, if one exists."""

    path = default_board_batch_draft_path(project_root)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise BoardBuildingFlowError(f"Malformed board-building draft: {exc}") from exc
    except OSError as exc:
        raise BoardBuildingFlowError(
            f"Could not read board-building draft: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise BoardBuildingFlowError("Board-building draft must contain a mapping.")
    if data.get("kind") != BOARD_BATCH_DRAFT_KIND:
        raise BoardBuildingFlowError(
            f"Board-building draft has unexpected kind: {data.get('kind')!r}."
        )

    rows_data = data.get("rows", ())
    if not isinstance(rows_data, list):
        raise BoardBuildingFlowError("Board-building draft rows must be a list.")
    rows = tuple(_board_batch_row_from_mapping(row) for row in rows_data)
    build_board_batch_items(rows)

    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise BoardBuildingFlowError("Board-building draft settings must be a mapping.")

    return BoardBatchDraft(
        path=path,
        rows=rows,
        family=_setting_text(settings, "family", DEFAULT_FAMILY),
        tag_size_mm=_setting_float(settings, "tag_size_mm", DEFAULT_TAG_SIZE_MM),
        source=normalize_source(_setting_text(settings, "source", SOURCE_OPENCV)),
        camera_index=_setting_int(settings, "camera_index", DEFAULT_CAMERA_INDEX),
        video_path=_setting_text(settings, "video_path", ""),
        width=_setting_int(settings, "width", DEFAULT_WIDTH),
        height=_setting_int(settings, "height", DEFAULT_HEIGHT),
        fps=_setting_int(settings, "fps", DEFAULT_FPS),
        calibration_path=_setting_text(settings, "calibration_path", ""),
        out_dir=_setting_text(settings, "out_dir", ""),
        registry_path=_setting_text(settings, "registry_path", ""),
        save_shot=bool(settings.get("save_shot", False)),
        shots_dir=_setting_text(settings, "shots_dir", ""),
        z_threshold_m=_setting_float(
            settings,
            "z_threshold_m",
            DEFAULT_Z_THRESHOLD_M,
        ),
        allow_nonplanar=bool(settings.get("allow_nonplanar", False)),
    )


def delete_board_batch_draft(project_root: Union[Path, str]) -> bool:
    """Remove the persisted Stage 4 draft queue, if it exists."""

    path = default_board_batch_draft_path(project_root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise BoardBuildingFlowError(
            f"Could not remove board-building draft: {exc}"
        ) from exc
    return True


def parse_instance_labels(spec: str) -> tuple[str, ...]:
    """Parse an optional instance spec such as ``01-08`` or ``01,02``."""

    text = str(spec or "").strip()
    if not text:
        return ("",)
    return _parse_label_spec(text, "Instance")


def parse_side_labels(spec: str) -> tuple[str, ...]:
    """Parse side/face labels such as ``sideA, sideB`` or ``sideA-sideD``."""

    text = str(spec or "").strip()
    if not text:
        raise BoardBuildingFlowError("At least one side/face label is required.")
    return _parse_label_spec(text, "Side/face")


def _normalize_batch_tag_size(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BoardBuildingFlowError("Batch tag size must be numeric.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise BoardBuildingFlowError("Batch tag size must be a positive number.")
    return parsed


def expected_calibration_path(config: BoardBuildingConfig) -> Path:
    """Return the calibration YAML path used by the current form values."""

    root = Path(config.project_root).expanduser()
    raw = _optional_path(config.calibration_path)
    if raw is None:
        return default_calibration_path(root)
    return resolve_calibration_path(root, raw)


def expected_board_yaml_path(config: BoardBuildingConfig) -> Path:
    """Return the expected board YAML path without creating directories."""

    out_dir = _optional_path(config.out_dir) or default_boards_dir(config.project_root)
    object_name = str(config.object_name).strip() or "OBJECT_FACE"
    return out_dir / f"{object_name}.yaml"


def expected_registry_path(config: BoardBuildingConfig) -> Path:
    """Return the expected tag registry path without creating directories."""

    return (
        _optional_path(config.registry_path)
        or default_boards_dir(config.project_root) / "tag_registry.yaml"
    )


def expected_shots_dir(config: BoardBuildingConfig) -> Path:
    """Return the expected audit-shot directory without creating directories."""

    return _optional_path(config.shots_dir) or (
        (_optional_path(config.out_dir) or default_boards_dir(config.project_root))
        / "shots"
    )


def inspect_board_building_outputs(
    config: BoardBuildingConfig,
) -> BoardBuildingOutputStatus:
    """Inspect expected Stage 4 output paths without parsing or creating them."""

    board_yaml = expected_board_yaml_path(config)
    registry = expected_registry_path(config)
    shots = expected_shots_dir(config)
    checked_paths = [board_yaml, registry]
    if bool(config.save_shot):
        checked_paths.append(shots)
    return BoardBuildingOutputStatus(
        board_yaml_path=board_yaml,
        registry_path=registry,
        shots_dir=shots,
        board_yaml_exists=board_yaml.exists(),
        registry_exists=registry.exists(),
        shots_dir_exists=shots.is_dir(),
        checked_paths=tuple(dict.fromkeys(checked_paths)),
    )


def inspect_board_building(
    config: BoardBuildingConfig,
    *,
    realsense_available: Optional[bool] = None,
) -> BoardBuildingReadiness:
    """Return command, readiness, and output-path state for Stage 4."""

    root = Path(config.project_root).expanduser()
    calibration_path = expected_calibration_path(config)
    outputs = inspect_board_building_outputs(config)
    checked_paths = [calibration_path, root / "boards" / "patterns"]
    checked_paths.extend(outputs.checked_paths)
    warnings: list[str] = []
    errors: list[str] = []
    validated: Optional[_ValidatedBoardBuildingInputs] = None

    try:
        validated = _validated_inputs(config)
    except BoardBuildingFlowError as exc:
        errors.append(str(exc))

    try:
        load_calibration_yaml(calibration_path)
    except MakeBoardError as exc:
        errors.append(str(exc))

    object_tags = inspect_object_tags(root, calibration_complete=True)
    if object_tags.status != WorkflowStatus.COMPLETE:
        message = (
            "Complete Stage 3 object AprilTag generation before building "
            "object board definitions from the guided GUI flow."
        )
        if bool(config.require_object_tags):
            errors.append(message)
        else:
            warnings.append(
                message
                + " Continue only if the physical tags were generated and "
                "printed outside the default project folder."
            )

    default_registry = default_boards_dir(root) / "tag_registry.yaml"
    if outputs.registry_path != default_registry:
        warnings.append(
            "Project status checks look for the Stage 4 registry at "
            f"{default_registry}. Custom registry paths are supported by "
            "posetag-make-board, but they may not mark Stage 4 complete."
        )

    default_calib = default_calibration_path(root)
    if calibration_path != default_calib and not default_calib.exists():
        warnings.append(
            "Project status checks look for camera calibration at "
            f"{default_calib}. Custom calibration paths can run the command, "
            "but Stage 2 may still appear incomplete."
        )

    if validated is not None:
        try:
            validate_source_args(
                validated.source,
                str(validated.video_path) if validated.video_path is not None else None,
                _realsense_module_token(realsense_available),
            )
        except MakeBoardError as exc:
            errors.append(str(exc))

    command_preview = ""
    if not errors:
        try:
            command_preview = build_board_building_command(config)
        except BoardBuildingFlowError as exc:
            errors.append(str(exc))

    return BoardBuildingReadiness(
        ready=not errors,
        command_preview=command_preview,
        calibration_path=calibration_path,
        output_status=outputs,
        checked_paths=tuple(dict.fromkeys(checked_paths)),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def build_board_building_command(config: BoardBuildingConfig) -> str:
    """Build a copyable ``posetag-make-board`` command from GUI state."""

    return shlex.join(
        ("posetag-make-board", *build_board_building_arguments(config))
    )


def build_board_building_launch(
    config: BoardBuildingConfig,
    *,
    python_executable: Optional[Union[Path, str]] = None,
) -> BoardBuildingLaunchSpec:
    """Build process launch details for the existing board-building workflow."""

    arguments = build_board_building_arguments(config)
    executable = str(python_executable) if python_executable else sys.executable
    outputs = inspect_board_building_outputs(config)
    return BoardBuildingLaunchSpec(
        program=executable,
        arguments=("-m", "posetag.cli.make_board", *arguments),
        display_command=build_board_building_command(config),
        expected_board_yaml=outputs.board_yaml_path,
        expected_registry=outputs.registry_path,
    )


def build_board_building_arguments(config: BoardBuildingConfig) -> tuple[str, ...]:
    """Build argv items for ``posetag-make-board`` without a shell."""

    validated = _validated_inputs(config)
    root = Path(config.project_root).expanduser()
    calibration_path = expected_calibration_path(config)

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
            "--family",
            validated.family,
            "--tag_size_mm",
            _format_number(validated.tag_size_mm),
            "--z_thresh",
            _format_number(validated.z_threshold_m),
        ]
    )

    out_dir = _optional_path(config.out_dir)
    if out_dir is not None:
        parts.extend(["--out_dir", str(out_dir)])
    registry = _optional_path(config.registry_path)
    if registry is not None:
        parts.extend(["--registry", str(registry)])
    if bool(config.save_shot):
        parts.append("--save_shot")
        shots = _optional_path(config.shots_dir)
        if shots is not None:
            parts.extend(["--shots_dir", str(shots)])
    if bool(config.allow_nonplanar):
        parts.append("--allow_nonplanar")
    return tuple(parts)


def board_building_process_not_started(
    expected_board_yaml: Optional[Union[Path, str]] = None,
    expected_registry: Optional[Union[Path, str]] = None,
) -> BoardBuildingProcessState:
    """Return the initial board-building process state."""

    return BoardBuildingProcessState(
        state=BOARD_PROCESS_NOT_STARTED,
        label="not started",
        message="Board building has not been launched from this dashboard session.",
        expected_board_yaml=_optional_path(expected_board_yaml),
        expected_registry=_optional_path(expected_registry),
    )


def board_building_process_running(
    expected_board_yaml: Union[Path, str],
    expected_registry: Union[Path, str],
) -> BoardBuildingProcessState:
    """Return the active board-building process state."""

    return BoardBuildingProcessState(
        state=BOARD_PROCESS_RUNNING,
        label="running",
        message=(
            "Board building is running in the existing OpenCV workflow. Press "
            "ENTER in the preview to capture, then send selected tag IDs and "
            "the origin tag when prompted."
        ),
        running=True,
        expected_board_yaml=Path(expected_board_yaml).expanduser(),
        expected_registry=Path(expected_registry).expanduser(),
    )


def board_building_process_failed(
    message: str,
    *,
    expected_board_yaml: Optional[Union[Path, str]] = None,
    expected_registry: Optional[Union[Path, str]] = None,
    exit_code: Optional[int] = None,
) -> BoardBuildingProcessState:
    """Return a failed/cancelled board-building process state."""

    return BoardBuildingProcessState(
        state=BOARD_PROCESS_FAILED_CANCELLED,
        label="failed/cancelled",
        message=message,
        expected_board_yaml=_optional_path(expected_board_yaml),
        expected_registry=_optional_path(expected_registry),
        exit_code=exit_code,
    )


def summarize_board_building_process_result(
    *,
    exit_code: int,
    crashed: bool = False,
    expected_board_yaml: Optional[Union[Path, str]] = None,
    expected_registry: Optional[Union[Path, str]] = None,
    previous_board_mtime_ns: Optional[int] = None,
    previous_registry_mtime_ns: Optional[int] = None,
    require_output_update: bool = False,
) -> BoardBuildingProcessState:
    """Summarize process completion from expected board and registry outputs."""

    board_yaml = _optional_path(expected_board_yaml)
    registry = _optional_path(expected_registry)
    board_exists = bool(board_yaml and board_yaml.exists())
    registry_exists = bool(registry and registry.exists())
    board_updated = (
        not require_output_update
        or (board_yaml is not None and _path_mtime_ns(board_yaml) != previous_board_mtime_ns)
    )
    registry_updated = (
        not require_output_update
        or (registry is not None and _path_mtime_ns(registry) != previous_registry_mtime_ns)
    )

    if (
        exit_code == 0
        and not crashed
        and board_exists
        and registry_exists
        and board_updated
        and registry_updated
    ):
        return BoardBuildingProcessState(
            state=BOARD_PROCESS_FINISHED,
            label="finished",
            message=(
                "Board building finished and the expected board YAML plus "
                "tag registry outputs are present."
            ),
            success=True,
            expected_board_yaml=board_yaml,
            expected_registry=registry,
            exit_code=exit_code,
        )

    if crashed:
        message = "Board-building process crashed or was cancelled."
    elif exit_code != 0:
        message = f"Board-building process exited with code {exit_code}."
    elif not (board_exists and registry_exists):
        message = (
            "Board-building process exited, but the expected board YAML or "
            "tag registry output is missing."
        )
    else:
        message = (
            "Board-building process exited, but the expected board YAML or "
            "tag registry was not updated during this launch."
        )

    return board_building_process_failed(
        message,
        expected_board_yaml=board_yaml,
        expected_registry=registry,
        exit_code=exit_code,
    )


def normalize_source(source: str) -> str:
    """Normalize a user-facing board-building source value."""

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
        raise BoardBuildingFlowError(
            "Board-building source must be one of: "
            + ", ".join(BOARD_SOURCE_CHOICES)
            + "."
        ) from exc


def realsense_dependency_available() -> bool:
    """Return whether ``pyrealsense2`` is importable in this environment."""

    return importlib.util.find_spec("pyrealsense2") is not None


def _validated_inputs(config: BoardBuildingConfig) -> _ValidatedBoardBuildingInputs:
    errors: list[str] = []
    object_name = str(config.object_name).strip()
    if not object_name:
        errors.append("Board definition name must not be empty.")
    elif Path(object_name).name != object_name or "\\" in object_name:
        errors.append(
            "Board definition name must be a filename-safe name, not a path."
        )

    family = str(config.family).strip()
    if not family:
        errors.append("AprilTag family must not be empty.")

    tag_size_mm = _coerce_float(config.tag_size_mm, "Tag size", errors)
    if tag_size_mm is not None and tag_size_mm <= 0:
        errors.append("Tag size must be a positive millimetre value.")

    try:
        source = normalize_source(config.source)
    except BoardBuildingFlowError as exc:
        source = SOURCE_OPENCV
        errors.append(str(exc))

    camera_index = _coerce_int(config.camera_index, "Camera index", errors)
    if camera_index is not None and camera_index < 0:
        errors.append("Camera index must be zero or greater.")

    width = _coerce_int(config.width, "Capture width", errors)
    height = _coerce_int(config.height, "Capture height", errors)
    fps = _coerce_int(config.fps, "Capture FPS", errors)
    for label, value in (
        ("Capture width", width),
        ("Capture height", height),
        ("Capture FPS", fps),
    ):
        if value is not None and value <= 0:
            errors.append(f"{label} must be positive.")

    z_threshold_m = _coerce_float(
        config.z_threshold_m,
        "Planarity threshold",
        errors,
    )
    if z_threshold_m is not None and z_threshold_m <= 0:
        errors.append("Planarity threshold must be a positive metre value.")

    video_path = _optional_path(config.video_path)
    if source == SOURCE_VIDEO and video_path is None:
        errors.append("--video path is required when --source=video")

    if errors:
        raise BoardBuildingFlowError(" ".join(errors))

    assert camera_index is not None
    assert tag_size_mm is not None
    assert width is not None
    assert height is not None
    assert fps is not None
    assert z_threshold_m is not None
    return _ValidatedBoardBuildingInputs(
        object_name=object_name,
        family=family,
        tag_size_mm=tag_size_mm,
        source=source,
        camera_index=camera_index,
        video_path=video_path,
        width=width,
        height=height,
        fps=fps,
        z_threshold_m=z_threshold_m,
    )


def _capture_size_arguments(
    validated: _ValidatedBoardBuildingInputs,
) -> list[str]:
    return [
        "--width",
        str(validated.width),
        "--height",
        str(validated.height),
        "--fps",
        str(validated.fps),
    ]


def _realsense_module_token(realsense_available: Optional[bool]) -> Any:
    available = (
        realsense_dependency_available()
        if realsense_available is None
        else bool(realsense_available)
    )
    return object() if available else None


def _optional_path(path: Optional[Union[Path, str]]) -> Optional[Path]:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _path_text(path: Optional[Union[Path, str]]) -> str:
    if path is None:
        return ""
    return str(path).strip()


def _board_batch_row_to_mapping(row: BoardBatchRow) -> dict[str, Any]:
    data: dict[str, Any] = {
        "object_label": str(row.object_label),
        "instances": str(row.instances),
        "sides": str(row.sides),
    }
    if row.tag_size_mm is not None:
        data["tag_size_mm"] = float(row.tag_size_mm)
    return data


def _board_batch_row_from_mapping(data: object) -> BoardBatchRow:
    if not isinstance(data, dict):
        raise BoardBuildingFlowError("Board-building draft row must be a mapping.")
    return BoardBatchRow(
        object_label=_setting_text(data, "object_label", ""),
        instances=_setting_text(data, "instances", ""),
        sides=_setting_text(data, "sides", DEFAULT_SIDE_LABEL),
        tag_size_mm=(
            _normalize_batch_tag_size(data.get("tag_size_mm"))
            if data.get("tag_size_mm") not in (None, "")
            else None
        ),
    )


def _setting_text(
    settings: dict[str, Any],
    key: str,
    default: str,
) -> str:
    value = settings.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _setting_int(
    settings: dict[str, Any],
    key: str,
    default: int,
) -> int:
    errors: list[str] = []
    value = _coerce_int(settings.get(key, default), key, errors)
    if errors or value is None:
        raise BoardBuildingFlowError(errors[0] if errors else f"{key} is invalid.")
    return value


def _setting_float(
    settings: dict[str, Any],
    key: str,
    default: float,
) -> float:
    errors: list[str] = []
    value = _coerce_float(settings.get(key, default), key, errors)
    if errors or value is None:
        raise BoardBuildingFlowError(errors[0] if errors else f"{key} is invalid.")
    return value


def _atomic_write_draft_yaml(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _clean_board_label_part(
    value: Optional[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    text = str(value or "").strip()
    if not text:
        if allow_empty:
            return ""
        raise BoardBuildingFlowError(f"{label} must not be empty.")
    text = "_".join(text.split())
    if Path(text).name != text or "\\" in text:
        raise BoardBuildingFlowError(
            f"{label} must be a filename-safe label, not a path."
        )
    return text


def _parse_label_spec(spec: str, label: str) -> tuple[str, ...]:
    values: list[str] = []
    for token in str(spec).split(","):
        part = token.strip()
        if not part:
            continue
        expanded = _expand_label_range(part, label)
        values.extend(expanded)
    if not values:
        raise BoardBuildingFlowError(f"{label} labels must not be empty.")
    cleaned = tuple(
        _clean_board_label_part(value, f"{label} label") for value in values
    )
    if len(set(cleaned)) != len(cleaned):
        raise BoardBuildingFlowError(f"{label} labels must be unique.")
    return cleaned


def _expand_label_range(part: str, label: str) -> tuple[str, ...]:
    if "-" not in part:
        return (part,)
    start, end = [chunk.strip() for chunk in part.split("-", 1)]
    if not start or not end:
        raise BoardBuildingFlowError(f"Invalid {label.lower()} range: {part}.")
    if start.isdigit() and end.isdigit():
        first = int(start)
        last = int(end)
        if last < first:
            raise BoardBuildingFlowError(f"Invalid descending {label.lower()} range: {part}.")
        width = max(len(start), len(end))
        return tuple(f"{number:0{width}d}" for number in range(first, last + 1))

    start_prefix, start_suffix = _split_trailing_alpha(start)
    end_prefix, end_suffix = _split_trailing_alpha(end)
    if (
        start_prefix
        and start_prefix == end_prefix
        and len(start_suffix) == 1
        and len(end_suffix) == 1
    ):
        first = ord(start_suffix.upper())
        last = ord(end_suffix.upper())
        if last < first:
            raise BoardBuildingFlowError(f"Invalid descending {label.lower()} range: {part}.")
        return tuple(
            f"{start_prefix}{chr(code)}" for code in range(first, last + 1)
        )
    return (part,)


def _split_trailing_alpha(value: str) -> tuple[str, str]:
    if not value or not value[-1].isalpha():
        return value, ""
    return value[:-1], value[-1]


def _path_mtime_ns(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _coerce_int(value: object, label: str, errors: list[str]) -> Optional[int]:
    if isinstance(value, bool):
        errors.append(f"{label} must be an integer.")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an integer.")
        return None


def _coerce_float(value: object, label: str, errors: list[str]) -> Optional[float]:
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


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _format_id_tuple(values: Sequence[int]) -> str:
    return ", ".join(str(int(value)) for value in values)
