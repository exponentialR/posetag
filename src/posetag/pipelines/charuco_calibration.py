"""Reusable helpers for PoseTag Step 1 ChArUco camera calibration.

This module intentionally keeps camera capture and calibration solving in the
existing CLI implementation for now.  The helpers here cover validation,
project IO preparation, and YAML serialization so they can be tested without
camera hardware.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import yaml

from posetag.utils.project_config import ensure_project_dirs, resolve_project_root


class CharucoCalibrationError(ValueError):
    """User-facing validation error for ChArUco calibration setup."""


@dataclass(frozen=True)
class CalibrationProjectIO:
    """Resolved Step 1 project paths."""

    project_root: Path
    calib_dir: Path
    images_dir: Path
    run_dir: Path
    out_yaml: Path


def _aruco_module():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise CharucoCalibrationError(
            "Installed OpenCV build does not provide cv2.aruco; install an "
            "OpenCV build with ArUco/ChArUco support."
        )
    return aruco


def _dictionary_ids() -> dict[str, Optional[int]]:
    aruco = _aruco_module()
    return {
        "4X4_50": getattr(aruco, "DICT_4X4_50", None),
        "4X4_100": getattr(aruco, "DICT_4X4_100", None),
        "4X4_250": getattr(aruco, "DICT_4X4_250", None),
        "4X4_1000": getattr(aruco, "DICT_4X4_1000", None),
        "5X5_50": getattr(aruco, "DICT_5X5_50", None),
        "5X5_100": getattr(aruco, "DICT_5X5_100", None),
        "5X5_250": getattr(aruco, "DICT_5X5_250", None),
        "5X5_1000": getattr(aruco, "DICT_5X5_1000", None),
        "6X6_50": getattr(aruco, "DICT_6X6_50", None),
        "6X6_100": getattr(aruco, "DICT_6X6_100", None),
        "6X6_250": getattr(aruco, "DICT_6X6_250", None),
        "6X6_1000": getattr(aruco, "DICT_6X6_1000", None),
        "7X7_50": getattr(aruco, "DICT_7X7_50", None),
        "7X7_100": getattr(aruco, "DICT_7X7_100", None),
        "7X7_250": getattr(aruco, "DICT_7X7_250", None),
        "7X7_1000": getattr(aruco, "DICT_7X7_1000", None),
        "APRILTAG_36H11": getattr(aruco, "DICT_APRILTAG_36h11", None),
    }


def supported_dictionary_names() -> list[str]:
    """Return the ChArUco/Aruco dictionary names supported by this OpenCV build."""

    return sorted(name for name, value in _dictionary_ids().items() if value is not None)


def normalize_dictionary_name(name: str) -> str:
    """Normalize user-facing dictionary spelling without changing semantics."""

    return name.upper().replace("-", "_")


def get_dictionary(name: str):
    """Return an OpenCV ArUco dictionary object from a user-facing name."""

    normalized = normalize_dictionary_name(name)
    aruco = _aruco_module()
    dictionary_id = _dictionary_ids().get(normalized)
    if dictionary_id is None:
        supported = ", ".join(supported_dictionary_names())
        raise CharucoCalibrationError(
            f"Unsupported ArUco dictionary '{name}'. Supported dictionaries: {supported}"
        )
    return aruco.getPredefinedDictionary(dictionary_id)


def validate_capture_args(source: str, video: Optional[str], realsense_module: Any) -> None:
    """Validate source-specific arguments before opening hardware or writing files."""

    if source == "video" and not video:
        raise CharucoCalibrationError("--video path is required when --source=video")
    if source == "realsense" and realsense_module is None:
        raise CharucoCalibrationError(
            "pyrealsense2 is not available; install PoseTag with the 'realsense' "
            "extra or use --source opencv|video"
        )


def now_iso_utc() -> str:
    """Return the run-directory timestamp format used by Step 1."""

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def prepare_project_io(
    project_root: Optional[Union[Path, str]],
    out: Optional[Union[Path, str]] = None,
    timestamp: Optional[str] = None,
) -> CalibrationProjectIO:
    """Resolve the PoseTag project and create deterministic Step 1 output paths."""

    pr = Path(resolve_project_root(project_root))
    ensure_project_dirs(pr)

    calib_dir = pr / "calib"
    images_root = calib_dir / "images"
    runs_root = calib_dir / "runs"
    calib_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    existing_indices: list[int] = []
    for path in images_root.iterdir():
        if not path.is_dir() or not path.name.startswith("set_"):
            continue
        try:
            existing_indices.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    next_index = max(existing_indices, default=0) + 1
    images_dir = images_root / f"set_{next_index:02d}"
    images_dir.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / (timestamp or now_iso_utc())
    run_dir.mkdir(parents=True, exist_ok=True)

    out_yaml = Path(out).expanduser() if out else calib_dir / "calib_color.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    return CalibrationProjectIO(
        project_root=pr,
        calib_dir=calib_dir,
        images_dir=images_dir,
        run_dir=run_dir,
        out_yaml=out_yaml,
    )


def build_calibration_yaml(
    image_width: int,
    image_height: int,
    camera_matrix: Any,
    distortion_coefficients: Any,
    reproj_rms: float,
    notes: str,
    model: str = "plumb_bob",
) -> dict[str, Any]:
    """Build the public calibration YAML schema from solved intrinsics."""

    k = np.asarray(camera_matrix, dtype=float)
    dist = np.asarray(distortion_coefficients, dtype=float).reshape(1, -1)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    k1 = float(dist[0, 0]) if dist.shape[1] > 0 else 0.0
    k2 = float(dist[0, 1]) if dist.shape[1] > 1 else 0.0
    p1 = float(dist[0, 2]) if dist.shape[1] > 2 else 0.0
    p2 = float(dist[0, 3]) if dist.shape[1] > 3 else 0.0
    k3 = float(dist[0, 4]) if dist.shape[1] > 4 else 0.0

    return {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_matrix": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "data": k.tolist(),
        },
        "distortion_coefficients": {
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "k3": k3,
            "data": dist.tolist(),
        },
        "reproj_rms": float(reproj_rms),
        "model": model,
        "notes": notes,
    }


def write_calibration_yaml(path: Union[Path, str], data: dict[str, Any]) -> None:
    """Write calibration YAML using the public schema."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
