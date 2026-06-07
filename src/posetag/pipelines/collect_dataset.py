"""Pure helpers for the PoseTag dataset-collection workflow.

The interactive capture loop still lives in the legacy collector script for
now.  This module holds the small pieces that need hardware-free tests:
input preflight, source argument validation, session metadata, and pose-label
serialization.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

from utils.annotation_utils import load_board
from utils.project_config import resolve_project_root


DATASET_VERSION = "1.0"
POSE_COMPOSITION = "T_cam_object = T_cam_board @ T_board_object"
TRANSLATION_UNITS = "meters"
ANGLE_UNITS = "degrees"
QUATERNION_ORDER = "x, y, z, w"
RPY_ORDER = "roll, pitch, yaw"


class CollectDatasetError(RuntimeError):
    """Raised when dataset collection cannot start safely."""


@dataclass(frozen=True)
class Intrinsics:
    """Validated colour-camera intrinsics for collection."""

    path: Path
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray


@dataclass(frozen=True)
class AnnotationInput:
    """Validated face annotation used by collection."""

    object_name: str
    face_key: str
    yaml_path: Path
    board_yaml: Path
    tag_ids: tuple[int, ...]
    tag_size_m: float
    keypoints_path: Optional[Path]


@dataclass(frozen=True)
class CollectionInputSummary:
    """Summary returned by collection preflight."""

    project_root: Path
    calibration_path: Path
    registry_path: Path
    face_manifest_path: Path
    annotations: tuple[AnnotationInput, ...]
    missing_keypoints: tuple[Path, ...]


@dataclass(frozen=True)
class AutoCaptureConfig:
    """Quality gates for smart dataset auto-capture.

    This policy is deliberately independent of camera/GUI code.  It consumes
    already-estimated object poses and decides whether the current frame is
    stable, good enough, and useful enough to save.
    """

    enabled: bool = False
    stable_frames: int = 5
    cooldown_sec: float = 1.0
    min_tags_visible: int = 1
    min_bbox_area_px: float = 12000.0
    max_tag_scale_error: float = 0.08
    min_translation_delta_m: float = 0.04
    min_rotation_delta_deg: float = 8.0
    grid_rows: int = 3
    grid_cols: int = 3
    distance_bin_m: float = 0.25
    max_frames_per_object: int = 0


@dataclass(frozen=True)
class AutoCaptureCandidate:
    """One detected object candidate considered for smart auto-capture."""

    object_name: str
    face_key: str
    T_cam_object: Any
    bbox_xywh_px: tuple[float, float, float, float]
    num_tags_visible: int
    score: float = 0.0
    score_area_px: float = 0.0
    tag_scale_ratio: float = 1.0
    tag_scale_pairs: int = 0


@dataclass
class AutoCaptureTrack:
    """Per-object stability state for smart auto-capture."""

    last_face_key: str = ""
    last_T_cam_object: Optional[np.ndarray] = None
    stable_count: int = 0


@dataclass
class AutoCaptureState:
    """Mutable smart auto-capture state accumulated during one session."""

    tracks: dict[str, AutoCaptureTrack] = field(default_factory=dict)
    accepted_cells: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    accepted_distance_bins: dict[str, set[int]] = field(default_factory=dict)
    accepted_poses: dict[str, list[np.ndarray]] = field(default_factory=dict)
    accepted_counts: dict[str, int] = field(default_factory=dict)
    last_save_timestamp: Optional[float] = None


@dataclass(frozen=True)
class AutoCaptureDecision:
    """Decision returned by the smart auto-capture policy."""

    should_save: bool
    status: str
    message: str
    reason: str = ""
    trigger_object: Optional[str] = None
    trigger_face: Optional[str] = None
    stable_count: int = 0
    rejected_reasons: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe record for per-frame capture metadata."""

        return {
            "mode": "smart_auto",
            "status": self.status,
            "reason": self.reason,
            "trigger_object": self.trigger_object,
            "trigger_face": self.trigger_face,
            "stable_count": int(self.stable_count),
            "rejected_reasons": list(self.rejected_reasons),
        }


def preview_project_root(project_root: Optional[Union[Path, str]]) -> Path:
    """Resolve the project root without creating directories when explicit."""

    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return Path(resolve_project_root(None))


def resolve_under_project(
    project_root: Union[Path, str],
    value: Optional[Union[Path, str]],
    default: Union[Path, str],
) -> Path:
    """Resolve a project-relative path unless the provided path is absolute."""

    root = Path(project_root).expanduser().resolve()
    raw = Path(default if value is None else value).expanduser()
    return raw if raw.is_absolute() else root / raw


def resolve_calibration_path(
    project_root: Union[Path, str],
    calib: Optional[Union[Path, str]] = "calib_color.yaml",
) -> Path:
    """Resolve the required Step 1 colour calibration YAML.

    The canonical location is ``<project_root>/calib/calib_color.yaml``.
    Relative names are resolved there first, then under the project root, then
    as a literal relative path for compatibility with older script usage.
    """

    root = Path(project_root).expanduser().resolve()
    raw = Path(calib or "calib_color.yaml").expanduser()
    if raw.is_absolute():
        return raw

    candidates: list[Path] = []
    if len(raw.parts) > 1:
        candidates.append(root / raw)
    else:
        candidates.append(root / "calib" / raw)
        candidates.append(root / raw)
    candidates.append(raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_registry_path(
    project_root: Union[Path, str],
    registry: Optional[Union[Path, str]] = None,
) -> Path:
    """Resolve the required board tag registry."""

    return resolve_under_project(project_root, registry, "boards/tag_registry.yaml")


def resolve_face_manifest_path(
    project_root: Union[Path, str],
    face_manifest: Optional[Union[Path, str]] = None,
) -> Path:
    """Resolve the required face annotation manifest."""

    return resolve_under_project(project_root, face_manifest, "faces/face_manifest.csv")


def resolve_dataset_root(
    project_root: Union[Path, str],
    dataset_root: Optional[Union[Path, str]] = None,
) -> Path:
    """Resolve the root containing dataset sessions."""

    return resolve_under_project(project_root, dataset_root, "datasets")


def load_intrinsics_from_calibration(calib_path: Union[Path, str]) -> Intrinsics:
    """Load and validate the Step 1 colour-camera calibration YAML."""

    target = Path(calib_path).expanduser()
    if not target.exists():
        raise CollectDatasetError(
            f"Calibration YAML not found: {target}. Run Step 1 to create "
            "calib/calib_color.yaml first."
        )

    data = _load_yaml_mapping(target, "calibration YAML")
    camera_matrix = data.get("camera_matrix")
    if not isinstance(camera_matrix, Mapping):
        raise CollectDatasetError(
            f"Malformed calibration YAML: {target} is missing camera_matrix fields."
        )

    missing = [name for name in ("fx", "fy", "cx", "cy") if name not in camera_matrix]
    if missing:
        joined = ", ".join(f"camera_matrix.{name}" for name in missing)
        raise CollectDatasetError(f"Malformed calibration YAML: {target} is missing {joined}.")

    try:
        fx = float(camera_matrix["fx"])
        fy = float(camera_matrix["fy"])
        cx = float(camera_matrix["cx"])
        cy = float(camera_matrix["cy"])
    except (TypeError, ValueError) as exc:
        raise CollectDatasetError(
            f"Malformed calibration YAML: {target} camera_matrix values must be numeric."
        ) from exc
    if not all(np.isfinite(value) for value in (fx, fy, cx, cy)):
        raise CollectDatasetError(
            f"Malformed calibration YAML: {target} camera_matrix values must be finite."
        )

    width = _positive_int(data, "image_width", target)
    height = _positive_int(data, "image_height", target)

    distortion = data.get("distortion_coefficients") or {}
    if not isinstance(distortion, Mapping):
        raise CollectDatasetError(
            f"Malformed calibration YAML: {target} distortion_coefficients must be a mapping."
        )
    distortion_values: list[float] = []
    for name in ("k1", "k2", "p1", "p2", "k3"):
        try:
            distortion_values.append(float(distortion.get(name, 0.0)))
        except (TypeError, ValueError) as exc:
            raise CollectDatasetError(
                f"Malformed calibration YAML: {target} distortion_coefficients.{name} "
                "must be numeric."
            ) from exc
    dist = np.array([distortion_values], dtype=float)

    return Intrinsics(
        path=target,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        dist=dist,
    )


def validate_source_args(
    *,
    mode: str,
    video: Optional[Union[Path, str]],
    bag: Optional[Union[Path, str]],
    realsense_module: Any,
    video_capture_factory: Optional[Callable[[str], Any]] = None,
) -> None:
    """Validate source-specific arguments before opening hardware or outputs."""

    if mode == "video":
        if not video:
            raise CollectDatasetError("--video is required when --mode video")
        video_path = Path(video).expanduser()
        if not video_path.exists():
            raise CollectDatasetError(f"Could not open video: {video_path}")
        if video_capture_factory is not None:
            cap = video_capture_factory(str(video_path))
            try:
                is_open = bool(cap.isOpened()) if hasattr(cap, "isOpened") else True
            finally:
                if hasattr(cap, "release"):
                    cap.release()
            if not is_open:
                raise CollectDatasetError(f"Could not open video: {video_path}")

    if mode == "bag":
        if not bag:
            raise CollectDatasetError("--bag is required when --mode bag")
        bag_path = Path(bag).expanduser()
        if not bag_path.exists():
            raise CollectDatasetError(f"RealSense bag file not found: {bag_path}")

    if mode in {"live", "bag"} and realsense_module is None:
        raise CollectDatasetError(
            "pyrealsense2 is not available; install PoseTag with the 'realsense' "
            "extra or use --mode opencv|video."
        )


def validate_collection_inputs(
    *,
    project_root: Union[Path, str],
    calib_path: Union[Path, str],
    registry_path: Union[Path, str],
    face_manifest_path: Union[Path, str],
    allow_face_scan: bool = False,
) -> CollectionInputSummary:
    """Validate required collection inputs without opening cameras."""

    root = Path(project_root).expanduser().resolve()
    calibration = Path(calib_path).expanduser()
    registry = Path(registry_path).expanduser()
    manifest = Path(face_manifest_path).expanduser()

    # Validate calibration for clearer preflight failures even when caller
    # already loaded it.
    load_intrinsics_from_calibration(calibration)

    registry_index = load_tag_registry_index(registry, root)
    rows = _read_face_manifest_rows(manifest, root, allow_face_scan=allow_face_scan)

    annotations: list[AnnotationInput] = []
    missing_keypoints: set[Path] = set()
    errors: list[str] = []
    for row_number, row in rows:
        try:
            annotation = _validate_annotation_row(row, row_number, root, registry_index)
            annotations.append(annotation)
            if annotation.keypoints_path is None:
                missing_keypoints.add(root / "objects" / annotation.object_name / "keypoints.json")
        except CollectDatasetError as exc:
            errors.append(str(exc))

    if errors:
        raise CollectDatasetError("\n".join(errors))
    if not annotations:
        raise CollectDatasetError(f"No annotation YAMLs found from {manifest}.")

    return CollectionInputSummary(
        project_root=root,
        calibration_path=calibration,
        registry_path=registry,
        face_manifest_path=manifest,
        annotations=tuple(annotations),
        missing_keypoints=tuple(sorted(missing_keypoints)),
    )


def load_tag_registry_index(registry_path: Union[Path, str], project_root: Union[Path, str]) -> dict[int, Path]:
    """Load ``boards/tag_registry.yaml`` as ``tag_id -> board YAML``."""

    target = Path(registry_path).expanduser()
    if not target.exists():
        raise CollectDatasetError(
            f"Tag registry not found: {target}. Run Step 2 to create boards/tag_registry.yaml."
        )

    data = _load_yaml_mapping(target, "tag registry YAML")
    tags = data.get("tags")
    if not isinstance(tags, Mapping) or not tags:
        raise CollectDatasetError(f"Malformed tag registry YAML: {target} tags must be a non-empty mapping.")

    root = Path(project_root).expanduser().resolve()
    index: dict[int, Path] = {}
    for raw_tag_id, value in tags.items():
        try:
            tag_id = int(raw_tag_id)
        except (TypeError, ValueError) as exc:
            raise CollectDatasetError(
                f"Malformed tag registry YAML: {target} tag id {raw_tag_id!r} must be an integer."
            ) from exc
        if not isinstance(value, Mapping):
            raise CollectDatasetError(
                f"Malformed tag registry YAML: {target} tags.{raw_tag_id} must be a mapping."
            )
        yaml_value = value.get("yaml")
        if not yaml_value:
            raise CollectDatasetError(
                f"Malformed tag registry YAML: {target} tags.{raw_tag_id}.yaml is required."
            )
        board_path = Path(str(yaml_value)).expanduser()
        if not board_path.is_absolute():
            board_path = root / board_path
        index[tag_id] = board_path.resolve()
    return index


def compose_T_cam_object(T_cam_board: np.ndarray, T_board_object: np.ndarray) -> np.ndarray:
    """Compose PoseTag's runtime object pose convention."""

    cam_board = _matrix4(T_cam_board, "T_cam_board")
    board_object = _matrix4(T_board_object, "T_board_object")
    return cam_board @ board_object


def pose_convention_record() -> dict[str, str]:
    """Return the explicit transform and rotation convention metadata."""

    return {
        "composition": POSE_COMPOSITION,
        "matrix_shape": "4x4 homogeneous transform",
        "matrix_storage": "row-major nested lists",
        "point_convention": "column vectors",
        "T_cam_board": "maps board-frame points into the camera frame",
        "T_board_object": "maps object-frame points into the board frame",
        "T_cam_object": "maps object-frame points into the camera frame",
        "translation_units": TRANSLATION_UNITS,
        "quaternion_order": QUATERNION_ORDER,
        "rpy_order": RPY_ORDER,
        "angle_units": ANGLE_UNITS,
    }


def rpy_deg_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic roll, pitch, yaw angles in degrees."""

    rot = np.asarray(R, dtype=float)
    sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy >= 1e-6:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = math.atan2(-rot[1, 2], rot[1, 1])
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = 0.0
    return tuple(float(v) for v in np.degrees([roll, pitch, yaw]).tolist())


def quaternion_xyzw_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to quaternion order ``[x, y, z, w]``."""

    rot = np.asarray(R, dtype=float)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        index = int(np.argmax([rot[0, 0], rot[1, 1], rot[2, 2]]))
        if index == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm > 0.0 and np.isfinite(norm):
        quat /= norm
    return tuple(float(v) for v in quat.tolist())


def parse_auto_capture_grid(value: Union[str, tuple[int, int], list[int]]) -> tuple[int, int]:
    """Parse a smart auto-capture coverage grid such as ``3x3``."""

    if isinstance(value, str):
        raw = value.strip().lower().replace(" ", "")
        if "x" not in raw:
            raise CollectDatasetError("--auto-grid must look like ROWSxCOLS, for example 3x3.")
        row_s, col_s = raw.split("x", 1)
        try:
            rows, cols = int(row_s), int(col_s)
        except ValueError as exc:
            raise CollectDatasetError("--auto-grid rows and columns must be integers.") from exc
    else:
        try:
            rows, cols = int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise CollectDatasetError("--auto-grid must provide rows and columns.") from exc
    if rows <= 0 or cols <= 0:
        raise CollectDatasetError("--auto-grid rows and columns must be positive.")
    return rows, cols


def auto_capture_candidates_from_results(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[AutoCaptureCandidate, ...]:
    """Convert collector pose results into smart auto-capture candidates."""

    candidates: list[AutoCaptureCandidate] = []
    for result in results.values():
        bbox = result.get("bbox_xywh") or result.get("bbox_xywh_px")
        transform = result.get("T_cam_object")
        if isinstance(transform, Mapping):
            transform = transform.get("matrix")
        diagnostics = dict(result.get("diagnostics", {}))
        try:
            bbox_xywh = tuple(float(v) for v in bbox)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            bbox_xywh = (0.0, 0.0, 0.0, 0.0)
        if len(bbox_xywh) != 4:
            bbox_xywh = (0.0, 0.0, 0.0, 0.0)
        candidates.append(
            AutoCaptureCandidate(
                object_name=str(result.get("object", "")),
                face_key=str(result.get("face_key", "")),
                T_cam_object=transform,
                bbox_xywh_px=bbox_xywh,  # type: ignore[arg-type]
                num_tags_visible=int(result.get("num_tags_visible", 0)),
                score=float(result.get("score", 0.0)),
                score_area_px=float(result.get("score_area_px", 0.0)),
                tag_scale_ratio=float(diagnostics.get("tag_scale_ratio", 1.0)),
                tag_scale_pairs=int(diagnostics.get("tag_scale_pairs", 0)),
            )
        )
    return tuple(sorted(candidates, key=lambda c: c.score, reverse=True))


def evaluate_auto_capture(
    candidates: Sequence[AutoCaptureCandidate],
    state: AutoCaptureState,
    config: AutoCaptureConfig,
    *,
    frame_width: int,
    frame_height: int,
    timestamp: float,
) -> AutoCaptureDecision:
    """Evaluate smart auto-capture gates for one frame.

    The function mutates ``state`` by updating per-object stability tracks.  It
    does not mark frames as accepted; call ``record_auto_capture_save`` after a
    successful write.
    """

    if not config.enabled:
        return AutoCaptureDecision(False, "disabled", "Smart auto-capture is disabled.")
    if not candidates:
        return AutoCaptureDecision(
            False,
            "waiting_for_detection",
            "Auto: waiting for a registered board face.",
        )

    rows, cols = _coverage_grid(config)
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    quality_candidates: list[tuple[AutoCaptureCandidate, np.ndarray, int]] = []
    rejected: list[str] = []

    for candidate in candidates:
        matrix, reasons = _auto_candidate_quality_rejections(candidate, config)
        if reasons or matrix is None:
            rejected.extend(f"{candidate.object_name or '?'}: {reason}" for reason in reasons)
            _reset_auto_capture_track(state, candidate)
            continue
        stable_count = _update_auto_capture_track(state, candidate, matrix, config)
        quality_candidates.append((candidate, matrix, stable_count))

    if not quality_candidates:
        msg = "Auto: waiting for quality"
        if rejected:
            msg = f"{msg} ({rejected[0]})"
        return AutoCaptureDecision(
            False,
            "waiting_for_quality",
            msg,
            rejected_reasons=tuple(rejected[:6]),
        )

    stable_candidates = [
        (candidate, matrix, stable_count)
        for candidate, matrix, stable_count in quality_candidates
        if stable_count >= max(1, int(config.stable_frames))
    ]
    if not stable_candidates:
        best_count = max(stable_count for _, _, stable_count in quality_candidates)
        needed = max(1, int(config.stable_frames))
        return AutoCaptureDecision(
            False,
            "waiting_for_stability",
            f"Auto: stabilizing pose {best_count}/{needed}",
            stable_count=best_count,
            rejected_reasons=tuple(rejected[:6]),
        )

    if _auto_capture_cooling_down(state, config, timestamp):
        remaining = max(0.0, float(config.cooldown_sec) - (float(timestamp) - float(state.last_save_timestamp)))
        best = stable_candidates[0]
        return AutoCaptureDecision(
            False,
            "cooldown",
            f"Auto: cooldown {remaining:.1f}s",
            trigger_object=best[0].object_name,
            trigger_face=best[0].face_key,
            stable_count=best[2],
            rejected_reasons=tuple(rejected[:6]),
        )

    duplicate_reasons: list[str] = []
    for candidate, matrix, stable_count in stable_candidates:
        useful_reason, duplicate_reason = _auto_capture_usefulness(
            state,
            candidate,
            matrix,
            config,
            frame_width=width,
            frame_height=height,
            grid_rows=rows,
            grid_cols=cols,
        )
        if useful_reason:
            return AutoCaptureDecision(
                True,
                "ready",
                f"Auto: save {candidate.object_name} ({useful_reason})",
                reason=useful_reason,
                trigger_object=candidate.object_name,
                trigger_face=candidate.face_key,
                stable_count=stable_count,
                rejected_reasons=tuple(rejected[:6]),
            )
        duplicate_reasons.append(f"{candidate.object_name}: {duplicate_reason}")

    return AutoCaptureDecision(
        False,
        "waiting_for_new_view",
        "Auto: waiting for a new view",
        stable_count=max(stable_count for _, _, stable_count in stable_candidates),
        rejected_reasons=tuple((duplicate_reasons or rejected)[:6]),
    )


def record_auto_capture_save(
    state: AutoCaptureState,
    candidate: AutoCaptureCandidate,
    config: AutoCaptureConfig,
    *,
    frame_width: int,
    frame_height: int,
    timestamp: float,
) -> None:
    """Record that a smart auto-capture candidate was written to disk."""

    matrix = _matrix4(candidate.T_cam_object, "auto-capture T_cam_object")
    rows, cols = _coverage_grid(config)
    obj = candidate.object_name
    state.accepted_counts[obj] = int(state.accepted_counts.get(obj, 0)) + 1
    state.accepted_cells.setdefault(obj, set()).add(
        _bbox_grid_cell(candidate.bbox_xywh_px, frame_width, frame_height, rows, cols)
    )
    state.accepted_distance_bins.setdefault(obj, set()).add(
        _distance_bin(matrix, config)
    )
    state.accepted_poses.setdefault(obj, []).append(matrix.copy())
    state.last_save_timestamp = float(timestamp)


def _coverage_grid(config: AutoCaptureConfig) -> tuple[int, int]:
    rows = max(1, int(config.grid_rows))
    cols = max(1, int(config.grid_cols))
    return rows, cols


def _auto_candidate_quality_rejections(
    candidate: AutoCaptureCandidate,
    config: AutoCaptureConfig,
) -> tuple[Optional[np.ndarray], list[str]]:
    reasons: list[str] = []
    matrix: Optional[np.ndarray] = None
    if not candidate.object_name:
        reasons.append("missing object name")
    if not candidate.face_key:
        reasons.append("missing face key")
    try:
        matrix = _matrix4(candidate.T_cam_object, "auto-capture T_cam_object")
    except CollectDatasetError as exc:
        reasons.append(str(exc))
    x0, y0, w_px, h_px = candidate.bbox_xywh_px
    if not all(np.isfinite([x0, y0, w_px, h_px])):
        reasons.append("bbox is not finite")
    if w_px <= 0.0 or h_px <= 0.0:
        reasons.append("bbox is empty")
    if (w_px * h_px) < float(config.min_bbox_area_px):
        reasons.append(f"bbox area < {float(config.min_bbox_area_px):g}px^2")
    if int(candidate.num_tags_visible) < int(config.min_tags_visible):
        reasons.append(f"visible tags < {int(config.min_tags_visible)}")
    if int(candidate.tag_scale_pairs) > 0:
        scale_error = abs(float(candidate.tag_scale_ratio) - 1.0)
        if scale_error > float(config.max_tag_scale_error):
            reasons.append(
                f"tag scale error {scale_error:.3f} > {float(config.max_tag_scale_error):.3f}"
            )
    return matrix, reasons


def _reset_auto_capture_track(
    state: AutoCaptureState,
    candidate: AutoCaptureCandidate,
) -> None:
    if not candidate.object_name:
        return
    state.tracks[candidate.object_name] = AutoCaptureTrack(
        last_face_key=candidate.face_key,
        last_T_cam_object=None,
        stable_count=0,
    )


def _update_auto_capture_track(
    state: AutoCaptureState,
    candidate: AutoCaptureCandidate,
    matrix: np.ndarray,
    config: AutoCaptureConfig,
) -> int:
    track = state.tracks.get(candidate.object_name)
    stable_count = 1
    if (
        track is not None
        and track.last_T_cam_object is not None
        and track.last_face_key == candidate.face_key
    ):
        translation_delta, rotation_delta = pose_delta(
            track.last_T_cam_object,
            matrix,
        )
        if (
            translation_delta <= float(config.min_translation_delta_m)
            and rotation_delta <= float(config.min_rotation_delta_deg)
        ):
            stable_count = int(track.stable_count) + 1
    state.tracks[candidate.object_name] = AutoCaptureTrack(
        last_face_key=candidate.face_key,
        last_T_cam_object=matrix.copy(),
        stable_count=stable_count,
    )
    return stable_count


def _auto_capture_cooling_down(
    state: AutoCaptureState,
    config: AutoCaptureConfig,
    timestamp: float,
) -> bool:
    return (
        state.last_save_timestamp is not None
        and float(config.cooldown_sec) > 0.0
        and (float(timestamp) - float(state.last_save_timestamp)) < float(config.cooldown_sec)
    )


def _auto_capture_usefulness(
    state: AutoCaptureState,
    candidate: AutoCaptureCandidate,
    matrix: np.ndarray,
    config: AutoCaptureConfig,
    *,
    frame_width: int,
    frame_height: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[Optional[str], str]:
    obj = candidate.object_name
    count = int(state.accepted_counts.get(obj, 0))
    if int(config.max_frames_per_object) > 0 and count >= int(config.max_frames_per_object):
        return None, "per-object limit reached"
    if count == 0:
        return "first_sample", ""

    cell = _bbox_grid_cell(
        candidate.bbox_xywh_px,
        frame_width,
        frame_height,
        grid_rows,
        grid_cols,
    )
    if cell not in state.accepted_cells.get(obj, set()):
        return "coverage_cell", ""

    distance_bin = _distance_bin(matrix, config)
    if distance_bin not in state.accepted_distance_bins.get(obj, set()):
        return "distance_bin", ""

    previous_poses = state.accepted_poses.get(obj, [])
    if previous_poses:
        translation_delta, rotation_delta = pose_delta(previous_poses[-1], matrix)
        if (
            translation_delta >= float(config.min_translation_delta_m)
            or rotation_delta >= float(config.min_rotation_delta_deg)
        ):
            return "pose_delta", ""
    return None, "duplicate view"


def _bbox_grid_cell(
    bbox_xywh_px: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    x0, y0, w_px, h_px = [float(v) for v in bbox_xywh_px]
    cx = min(max((x0 + 0.5 * w_px) / max(1.0, float(frame_width)), 0.0), 0.999999)
    cy = min(max((y0 + 0.5 * h_px) / max(1.0, float(frame_height)), 0.0), 0.999999)
    return int(cy * max(1, rows)), int(cx * max(1, cols))


def _distance_bin(matrix: np.ndarray, config: AutoCaptureConfig) -> int:
    bin_size = max(1e-9, float(config.distance_bin_m))
    distance = float(np.linalg.norm(matrix[:3, 3]))
    return int(math.floor(distance / bin_size))


def pose_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return translation distance in meters and rotation angle in degrees."""

    a = _matrix4(T_a, "T_a")
    b = _matrix4(T_b, "T_b")
    translation_delta = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rel = a[:3, :3].T @ b[:3, :3]
    cos_theta = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) * 0.5))
    rotation_delta = float(math.degrees(math.acos(cos_theta)))
    return translation_delta, rotation_delta


def build_session_metadata(
    *,
    session_name: str,
    project_root: Union[Path, str],
    dataset_root: Union[Path, str],
    session_dir: Union[Path, str],
    intrinsics: Intrinsics,
    tag_family: str,
    camera_kind: str,
    mode: str,
    tag_scale_controls: Mapping[str, Any],
    faces_index: Mapping[str, Mapping[str, Any]],
    missing_keypoints: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Build session-wide metadata for a dataset collection run."""

    objects = sorted(
        {
            str(value.get("object"))
            for value in faces_index.values()
            if value.get("object")
        }
    )
    return {
        "version": DATASET_VERSION,
        "name": session_name,
        "project_root": str(Path(project_root).expanduser().resolve()),
        "paths": {
            "root": str(Path(dataset_root).expanduser()),
            "session_dir": str(Path(session_dir).expanduser()),
        },
        "source": {
            "mode": mode,
            "camera_kind": camera_kind,
        },
        "camera": {
            "kind": camera_kind,
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "distortion_coefficients": intrinsics.dist.reshape(-1).astype(float).tolist(),
            "calibration_yaml": str(intrinsics.path),
        },
        "tag_family": tag_family,
        "pose_convention": pose_convention_record(),
        "tag_scale_controls": dict(tag_scale_controls),
        "faces_index": dict(faces_index),
        "objects_available": objects,
        "optional_keypoints_missing": [str(path) for path in missing_keypoints],
        "objects_seen_counts": {},
        "frames_captured": 0,
    }


def build_frame_annotation_record(
    *,
    dataset_name: str,
    frame_index: int,
    timestamp: float,
    tag_family: str,
    image_filename: str,
    annotated_image: Optional[str],
    reproj_image: Optional[str],
    depth: Optional[str],
    intrinsics: Intrinsics,
    results: Mapping[str, Mapping[str, Any]],
    pts3d_by_object: Optional[Mapping[str, np.ndarray]] = None,
    capture_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Serialize one accepted dataset frame using PoseTag's pose schema."""

    camera_extrinsics = _camera_extrinsics_from_results(results)
    objects_out = [
        build_pose_object_record(result, intrinsics, pts3d_by_object=pts3d_by_object)
        for result in results.values()
    ]

    record = {
        "version": DATASET_VERSION,
        "metadata": {
            "dataset_name": dataset_name,
            "frame_index": int(frame_index),
            "timestamp": float(timestamp),
            "tag_family": tag_family,
        },
        "pose_convention": pose_convention_record(),
        "image_filename": image_filename,
        "objects": objects_out,
        "annotated_image": annotated_image,
        "reproj_image": reproj_image,
        "depth": depth,
        "camera_intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
        },
        "camera_extrinsics": camera_extrinsics,
    }
    if capture_metadata:
        record["capture"] = _json_safe(capture_metadata)
    return record


def build_pose_object_record(
    result: Mapping[str, Any],
    intrinsics: Intrinsics,
    *,
    pts3d_by_object: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """Serialize one selected object/face pose from an estimated result."""

    obj_name = str(result["object"])
    cls_id, cls_name = _map_class_id_and_name(obj_name)
    x0, y0, w_px, h_px = [float(v) for v in result["bbox_xywh"]]
    width = float(intrinsics.width)
    height = float(intrinsics.height)
    cx = (x0 + 0.5 * w_px) / width
    cy = (y0 + 0.5 * h_px) / height

    T_cam_board = _matrix4(result["T_cam_board"]["matrix"], "T_cam_board")
    T_board_object = _matrix4(result["T_board_object"]["matrix"], "T_board_object")
    T_cam_object = _matrix4(result["T_cam_object"]["matrix"], "T_cam_object")

    composed = compose_T_cam_object(T_cam_board, T_board_object)
    if not np.allclose(composed, T_cam_object, rtol=1e-9, atol=1e-9):
        raise CollectDatasetError(
            f"Pose composition mismatch for {obj_name}/{result.get('face_key')}: "
            f"{POSE_COMPOSITION} was not preserved."
        )

    R = T_cam_object[:3, :3]
    t = T_cam_object[:3, 3]
    roll, pitch, yaw = rpy_deg_from_matrix(R)
    q_xyzw = quaternion_xyzw_from_matrix(R)

    dimensions = None
    obb_corners_cam = None
    obb_corners_img = None
    keypoints_source = None
    if pts3d_by_object and obj_name in pts3d_by_object:
        points = np.asarray(pts3d_by_object[obj_name], dtype=float)
        if points.size > 0:
            keypoints_source = f"objects/{obj_name}/keypoints.json"
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            dims = maxs - mins
            dimensions = [float(dims[0]), float(dims[1]), float(dims[2])]
            corners_obj = np.array(
                [[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
                dtype=float,
            )
            corners_cam = (R @ corners_obj.T + t.reshape(3, 1)).T
            obb_corners_cam = corners_cam.tolist()
            obb_corners_img = _project_points(corners_obj.astype(np.float32), T_cam_object, intrinsics).tolist()

    diagnostics = dict(result.get("diagnostics", {}))
    quality = {
        "tag_used": int(result["tag_used"]),
        "num_tags_visible": int(result["num_tags_visible"]),
        "score": float(result["score"]),
        "score_tags": int(result.get("score_tags", result["num_tags_visible"])),
        "score_area_px": int(result.get("score_area_px", 0)),
        "bbox_source": result.get("bbox_source"),
        "tag_scale_ratio": float(diagnostics.get("tag_scale_ratio", 1.0)),
        "tag_scale_pairs": int(diagnostics.get("tag_scale_pairs", 0)),
        "tag_scale_mad": float(diagnostics.get("tag_scale_mad", 0.0)),
        "tag_scale_auto_corrected": bool(diagnostics.get("tag_scale_auto_corrected", False)),
    }

    pose = {
        "position": [float(t[0]), float(t[1]), float(t[2])],
        "orientation": [float(roll), float(pitch), float(yaw)],
        "rotation": [float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])],
        "translation_units": TRANSLATION_UNITS,
        "orientation_order": RPY_ORDER,
        "orientation_units": ANGLE_UNITS,
        "rotation_quaternion_order": QUATERNION_ORDER,
        **({"dimensions": dimensions} if dimensions is not None else {}),
        **({"obb_corners_cam": obb_corners_cam, "obb_corners_img": obb_corners_img}
           if obb_corners_cam is not None else {}),
    }

    return {
        "object_name": obj_name,
        "selected_face": str(result["face_key"]),
        "selected_board": str(result["board_yaml"]),
        "class_id": int(cls_id),
        "class_name": cls_name,
        "2D_center": [float(cx), float(cy)],
        "width": float(w_px / width),
        "height": float(h_px / height),
        "bbox_xywh_px": [float(x0), float(y0), float(w_px), float(h_px)],
        "bbox_source": result.get("bbox_source"),
        "keypoints_source": keypoints_source,
        "transforms": {
            "T_cam_board": {"matrix": T_cam_board.tolist()},
            "T_board_object": {"matrix": T_board_object.tolist()},
            "T_cam_object": {"matrix": T_cam_object.tolist()},
        },
        "quality": quality,
        "6DOF_pose": pose,
    }


def _camera_extrinsics_from_results(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    camera_pos = [0.0, 0.0, 0.0]
    cam_quat = [0.0, 0.0, 0.0, 1.0]
    cam_rpy = None
    reference: dict[str, Any] = {}

    if results:
        ref = max(results.values(), key=lambda r: (r.get("num_tags_visible", 0), r.get("score", 0.0)))
        T_cam_board = _matrix4(ref["T_cam_board"]["matrix"], "T_cam_board")
        T_board_cam = np.linalg.inv(T_cam_board)
        Rbc = T_board_cam[:3, :3]
        tbc = T_board_cam[:3, 3]
        camera_pos = [float(tbc[0]), float(tbc[1]), float(tbc[2])]
        cam_quat = [float(v) for v in quaternion_xyzw_from_matrix(Rbc)]
        cam_rpy = [float(v) for v in rpy_deg_from_matrix(Rbc)]
        reference = {
            "reference_face": str(ref.get("face_key", "")),
            "reference_board": str(ref.get("board_yaml", "")),
        }

    return {
        "position": camera_pos,
        "orientation": cam_quat,
        "rotation": cam_quat,
        **({"rpy_deg": cam_rpy} if cam_rpy is not None else {}),
        **reference,
        "position_units": TRANSLATION_UNITS,
        "rotation_quaternion_order": QUATERNION_ORDER,
        "rpy_order": RPY_ORDER,
        "rpy_units": ANGLE_UNITS,
    }


def _read_face_manifest_rows(
    manifest_path: Path,
    project_root: Path,
    *,
    allow_face_scan: bool,
) -> list[tuple[int, Mapping[str, str]]]:
    if not manifest_path.exists():
        if not allow_face_scan:
            raise CollectDatasetError(
                f"Face manifest not found: {manifest_path}. Run Step 5 annotation "
                "so faces/face_manifest.csv lists T_board_object YAMLs."
            )
        scanned: list[tuple[int, Mapping[str, str]]] = []
        for index, yaml_path in enumerate(sorted((project_root / "faces").glob("*/*/*_T_board_object.yaml")), start=1):
            scanned.append(
                (
                    index,
                    {
                        "yaml_path": str(yaml_path),
                        "board_yaml": "",
                        "object": "",
                        "face_key": "",
                    },
                )
            )
        return scanned

    try:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [(index, row) for index, row in enumerate(reader, start=2)]
    except OSError as exc:
        raise CollectDatasetError(f"Could not read face manifest: {manifest_path}: {exc}") from exc


def _validate_annotation_row(
    row: Mapping[str, str],
    row_number: int,
    project_root: Path,
    registry_index: Mapping[int, Path],
) -> AnnotationInput:
    yaml_value = str(row.get("yaml_path") or "").strip()
    if not yaml_value:
        raise CollectDatasetError(f"faces/face_manifest.csv row {row_number} is missing yaml_path.")
    yaml_path = _resolve_manifest_path(project_root, yaml_value)
    if not yaml_path.exists():
        raise CollectDatasetError(f"Annotation YAML not found: {yaml_path}")

    data = _load_yaml_mapping(yaml_path, "annotation YAML")
    object_name = str(data.get("object") or row.get("object") or "").strip()
    if not object_name:
        raise CollectDatasetError(f"Malformed annotation YAML: {yaml_path} is missing object.")
    face_key = str(data.get("face_key") or row.get("face_key") or yaml_path.stem.replace("_T_board_object", "")).strip()
    if not face_key:
        raise CollectDatasetError(f"Malformed annotation YAML: {yaml_path} is missing face_key.")

    t_board_object = data.get("T_board_object")
    if not isinstance(t_board_object, Mapping) or "matrix" not in t_board_object:
        raise CollectDatasetError(f"Malformed annotation YAML: {yaml_path} is missing T_board_object.matrix.")
    _matrix4(t_board_object["matrix"], f"{yaml_path} T_board_object.matrix")

    board_value = data.get("board_yaml") or row.get("board_yaml")
    if not board_value:
        raise CollectDatasetError(f"Malformed annotation YAML: {yaml_path} is missing board_yaml.")
    board_path = _resolve_manifest_path(project_root, str(board_value))
    if not board_path.exists():
        raise CollectDatasetError(f"Board YAML not found for annotation {yaml_path}: {board_path}")

    try:
        _origin_id, tag_size_m, t_board_tag = load_board(board_path)
    except Exception as exc:
        raise CollectDatasetError(f"Malformed board YAML for annotation {yaml_path}: {board_path}: {exc}") from exc

    resolved_board = board_path.resolve()
    for tag_id in t_board_tag:
        registered_board = registry_index.get(int(tag_id))
        if registered_board is None:
            raise CollectDatasetError(
                f"Tag registry is missing tag id {tag_id} from board YAML {board_path}."
            )
        if registered_board != resolved_board:
            raise CollectDatasetError(
                f"Tag registry maps tag id {tag_id} to {registered_board}, "
                f"but annotation {yaml_path} uses board {resolved_board}."
            )

    keypoints_path = project_root / "objects" / object_name / "keypoints.json"
    return AnnotationInput(
        object_name=object_name,
        face_key=face_key,
        yaml_path=yaml_path,
        board_yaml=resolved_board,
        tag_ids=tuple(sorted(int(tag_id) for tag_id in t_board_tag)),
        tag_size_m=float(tag_size_m),
        keypoints_path=keypoints_path if keypoints_path.exists() else None,
    )


def _resolve_manifest_path(project_root: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    return raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CollectDatasetError(f"Malformed {label}: {path}: {exc}") from exc
    except OSError as exc:
        raise CollectDatasetError(f"Could not read {label}: {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise CollectDatasetError(f"Malformed {label}: {path} must contain a mapping.")
    return data


def _positive_int(data: Mapping[str, Any], field: str, path: Path) -> int:
    if field not in data:
        raise CollectDatasetError(f"Malformed calibration YAML: {path} is missing {field}.")
    try:
        value = int(data[field])
    except (TypeError, ValueError) as exc:
        raise CollectDatasetError(f"Malformed calibration YAML: {path} {field} must be an integer.") from exc
    if value <= 0:
        raise CollectDatasetError(f"Malformed calibration YAML: {path} {field} must be positive.")
    return value


def _matrix4(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CollectDatasetError(f"Malformed transform: {name} must be numeric.") from exc
    if matrix.shape != (4, 4):
        raise CollectDatasetError(f"Malformed transform: {name} must be a 4x4 matrix.")
    if not np.all(np.isfinite(matrix)):
        raise CollectDatasetError(f"Malformed transform: {name} values must be finite.")
    return matrix


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _project_points(pts3d: np.ndarray, T_cam_obj: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    import cv2

    R = T_cam_obj[:3, :3]
    t = T_cam_obj[:3, 3].reshape(3, 1)
    K = np.array(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, t, K, intrinsics.dist)
    return uv.reshape(-1, 2)


def _map_class_id_and_name(object_name: str) -> tuple[int, str]:
    name = object_name.lower()
    if "connection_plate" in name or ("connection" in name and "plate" in name):
        return 2, "connection_plate"
    if "full_assembly" in name or ("full" in name and "assembly" in name):
        return 3, "full_assembly"
    if "column" in name:
        return 1, "column"
    return 0, object_name
