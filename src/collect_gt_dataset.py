# -*- coding: utf-8 -*-
"""
collect_gt_dataset.py
=====================
Collect a ground-truth dataset (multi-object, multi-face) from an OpenCV
webcam, Intel RealSense (live or .bag), or any OpenCV-readable video. For each
frame, estimate a pose for every face whose board has at least one visible
AprilTag. A review UI is provided, with optional continuous capture.

What's new vs. previous script
------------------------------
- Project layout via `resolve_project_root(...)`.
- Validates required inputs before capture: calib/calib_color.yaml,
  boards/tag_registry.yaml, faces/face_manifest.csv, board YAMLs, and
  annotation YAMLs containing T_board_object.
- Writes session metadata and per-frame annotations in a **blueprint-style schema**.
- Camera extrinsics now use `rotation` (quaternion [x,y,z,w]); optional `rpy_deg`
  is added for readability.
- Saves three visual outputs per accepted frame:
  * images/      = "raw" RGB with AprilTags covered (no overlays)
  * annotated/   = overlays (bboxes, labels, axes)
  * reproj/      = tag reprojection/debug view
- AprilTag scale verification (copied from annotate_shots.py):
  `--check-tag-scale` to report s, and `--auto-correct-scale` to divide
  T_cam_board translation by s when |s-1| > tol. Diagnostics recorded.
- Bounding box logic from 3D keypoints (face/object) with tag-based fallback.
- Rolling JSONL capture log per frame.

Project layout (inputs/outputs)
-------------------------------
<project_root>/
  faces/<object>/sideA|B|C|D/<face_key>_T_board_object.yaml  # inputs
  objects/<object>/keypoints.json                             # inputs
  datasets/<session>/
    images/
      Run_<YYYYMMDD-HHMMSS>_rgb_frame_<000000>.png            # raw (tags covered)
    annotated/
      Run_<YYYYMMDD-HHMMSS>_annotated_frame_<000000>.png      # UI overlays
    reproj/
      Run_<YYYYMMDD-HHMMSS>_reproj_frame_<000000>.png         # reprojection debug
    depth/
      Run_<YYYYMMDD-HHMMSS>_depth_frame_<000000>.npy          # optional
    annotations/
      Run_<YYYYMMDD-HHMMSS>_frame_<000000>.json               # per-frame GT (blueprint schema)

    session.yaml             # session-wide metadata for this run
    <session>.jsonl          # rolling capture log
  logs/collect_gt_dataset.log

Per-frame JSON schema (blueprint-style)
---------------------------------------
{
  "version": "1.0",
  "metadata": {
    "dataset_name": "<session>",
    "frame_index": <int>,
    "timestamp": <float>,                # seconds
    "tag_family": "<apriltag_family>"
  },
  "image_filename": "Run_<tag>_rgb_frame_<idx>.png",
  "annotated_image": "Run_<tag>_annotated_frame_<idx>.png",   # optional
  "reproj_image":    "Run_<tag>_reproj_frame_<idx>.png",      # optional
  "depth":           "Run_<tag>_depth_frame_<idx>.npy",       # optional

  "camera_intrinsics": {
    "fx": <float>, "fy": <float>, "cx": <float>, "cy": <float>,
    "width": <int>, "height": <int>
  },

  "camera_extrinsics": {
    "position": [tx, ty, tz],           # camera position in board frame (meters)
    "rotation": [x, y, z, w],           # quaternion, camera orientation in board frame
    "rpy_deg":  [roll, pitch, yaw]      # optional, degrees
  },

  "objects": [
    {
      "object_name": "<str>",
      "selected_face": "<face_key>",
      "selected_board": "<board_yaml>",
      "class_id": <int>,
      "class_name": "<str>",
      "2D_center": [cx, cy],            # normalized [0..1]
      "width": <float>, "height": <float>,  # normalized box size
      "transforms": {
        "T_cam_board": {"matrix": [[...]]},
        "T_board_object": {"matrix": [[...]]},
        "T_cam_object": {"matrix": [[...]]}
      },
      "quality": {...},
      "6DOF_pose": {
        "position":   [tx, ty, tz],     # camera->object translation (meters)
        "orientation":[roll, pitch, yaw],# degrees, for readability
        "rotation":   [x, y, z, w],     # quaternion (camera->object)
        "dimensions": [dx, dy, dz],     # optional, from object keypoints
        "obb_corners_cam": [[...]],     # optional, 8 corners in camera frame
        "obb_corners_img": [[...]]      # optional, 2D projections of corners
      }
    },
    ...
  ]
}

Notes on interpretation
-----------------------
- `position` fields are translations in meters.
- `orientation` lists (roll, pitch, yaw) in degrees; `rotation` is a quaternion [x,y,z,w].
- 2D centers and box sizes are normalized by image width/height.
- Class mapping is controlled by `_map_class_id_and_name(...)` (defaults):
    {1: "column", 2: "connection_plate", 3: "full_assembly"}.
- The raw image saved in `images/` is **tag-covered** to visually suppress AprilTags.

Usage examples
--------------
  # Preflight without opening a camera or writing dataset outputs
  python -m src.collect_gt_dataset --mode opencv --session run01 --dry-run

  # OpenCV webcam
  python -m src.collect_gt_dataset --mode opencv --cam 0 --session run01 --calib calib_color.yaml

  # RealSense live (RGB + depth if available)
  python -m src.collect_gt_dataset --mode live --session run02 --calib calib_color.yaml

  # RealSense playback from a .bag
  python -m src.collect_gt_dataset --mode bag --bag path/to/recording.bag --session run03 --calib calib_color.yaml

  # Any video readable by OpenCV
  python -m src.collect_gt_dataset --mode video --video sample.mp4 --session run04 --calib calib_color.yaml

Common options
--------------
  --project_root PATH            Project root (default resolver/env/config)
  --registry PATH                Tag registry (default boards/tag_registry.yaml)
  --face_manifest PATH           Face manifest (default faces/face_manifest.csv)
  --dry-run                      Validate inputs without capture/output writes
  --continuous                   Save every pose-labelled frame automatically
  --auto-capture                 Quality-gated smart auto-capture
  --auto-stable-frames N         Stable frames before smart save (default 5)
  --auto-cooldown-sec FLOAT      Delay between smart saves (default 1.0)
  --auto-min-tags N              Minimum visible tags for smart save (default 1)
  --auto-min-bbox-area FLOAT     Minimum smart-save bbox area in pixels
  --auto-max-tag-scale-error F   Max relative tag-scale error when available
  --auto-grid ROWSxCOLS          Image coverage grid for smart diversity
  --auto-distance-bin-m FLOAT    Distance-bin size for smart diversity
  --max_frames N                 Stop after N frames
  --rs_w/--rs_h/--rs_fps         Live RealSense stream parameters
  --save_depth                   Save aligned depth as .npy
  --axes {both,board,object,none}
  --bbox-mode {auto,face,object} Bbox from face/object keypoints or auto-select
  --bbox-tag-mult FLOAT          Fallback tag-square multiplier (default 5.0)
  --bbox-min-area INT            Minimum bbox area before fallback (default 12000)
  --bbox-pad-frac FLOAT          Pad bbox by fraction of its size (default 0.06)
  --check-tag-scale              Report tag scale ratio s
  --auto-correct-scale           Apply 1/s to T_cam_board translation if |s-1|>tol
  --scale-tol FLOAT              Relative tolerance (default 0.02)
  --tag-cover-margin-m FLOAT     Extra paper allowance around tag (default 0.010 m)
  --tag-sample-extra-m FLOAT     Ring thickness for background color sample (default 0.008 m)

Keys
----
  ENTER / y / s     accept / force-save frame
  r / n / BACKSPACE reject / skip frame
  ESC / q / Q       close capture window cleanly
  x / X             close capture window cleanly
  SPACE or p        pause/resume stream
  h                 toggle help panel
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2, yaml

log: logging.Logger

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import pyrealsense2 as rs
except Exception:
    rs = None  # optional

# --- repo utils ---
from utils.project_config import ensure_project_dirs
from utils.logger import init_project_logger
from utils.annotation_utils import (
    detect_tags, load_board, se3, inv_se3, load_keypoints_fuzzy
)
from utils.collect_gt_utils import rpy_from_R, _estimate_tag_scale, _show_dash, _ts_tag_from_epoch, \
    _capture_key_requests_quit, _capture_status_panel, _dominant_color_near_polygon, \
    _normalize_capture_key
from posetag.pipelines.collect_dataset import (
    AutoCaptureConfig,
    AutoCaptureState,
    CollectDatasetError,
    Intrinsics,
    auto_capture_candidates_from_results,
    build_frame_annotation_record,
    build_session_metadata,
    evaluate_auto_capture,
    load_intrinsics_from_calibration,
    parse_auto_capture_grid,
    preview_project_root,
    record_auto_capture_save,
    resolve_calibration_path,
    resolve_dataset_root,
    resolve_face_manifest_path,
    resolve_registry_path,
    validate_collection_inputs,
    validate_source_args,
)

try:
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
except Exception:
    pass

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# --------------------------- Camera / Sources --------------------------------
def load_intrinsics_from_calib(calib_path: Path) -> Intrinsics:
    """Compatibility wrapper for the validated collection calibration loader."""

    return load_intrinsics_from_calibration(calib_path)


class FrameSource:
    def start(self): pass

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:  # rgb, depth, ts_sec
        raise NotImplementedError

    def stop(self): pass

    def camera_kind(self) -> str: return "generic"


class VideoSource(FrameSource):
    def __init__(self, path: Path, fps_hint: float = 30.0):
        if not Path(path).expanduser().exists():
            raise RuntimeError(f"Could not open video: {path}")
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or fps_hint
        self.fps_delay = 1.0 / max(1e-6, fps)

    def read(self):
        ok, bgr = self.cap.read()
        if not ok: return None, None, time.time()
        time.sleep(self.fps_delay)
        return bgr, None, time.time()

    def stop(self): self.cap.release()


class OpenCVWebcamSource(FrameSource):
    def __init__(self, cam: int = 0, width: int = 0, height: int = 0, fps: int = 30):
        self.cam = int(cam)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.cam)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera index {self.cam}")
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None, None, time.time()
        return bgr, None, time.time()

    def stop(self):
        if self.cap is not None:
            self.cap.release()

    def camera_kind(self) -> str:
        return "opencv_webcam"


class RealSenseLive(FrameSource):
    def __init__(self, width=0, height=0, fps=30):
        if rs is None: raise RuntimeError("pyrealsense2 not available.")
        self.req_w, self.req_h, self.req_fps = int(width), int(height), int(fps)
        self.w = self.h = self.fps = 0
        self.pipe = None;
        self.align = None;
        self.depth_enabled = False;
        self.profile = None

    def _try_start(self, w, h, fps, with_depth=True):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        depth_ok = False
        if with_depth:
            try:
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps);
                depth_ok = True
            except Exception:
                try:
                    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps);
                    depth_ok = True
                except Exception:
                    depth_ok = False
        pipe = rs.pipeline();
        profile = pipe.start(cfg)
        return pipe, profile, depth_ok

    def start(self):
        ctx = rs.context()
        if len(ctx.devices) == 0: raise RuntimeError("No RealSense device found.")
        trials = []
        if self.req_w > 0 and self.req_h > 0:
            trials.append((self.req_w, self.req_h, self.req_fps))
        trials += [(640, 480, 30), (640, 360, 30)]
        last_err = None
        for (w, h, fps) in trials:
            try:
                pipe, profile, depth_ok = self._try_start(w, h, fps, with_depth=True)
                self.pipe, self.profile = pipe, profile
                self.depth_enabled = depth_ok
                self.w, self.h, self.fps = w, h, fps
                self.align = rs.align(rs.stream.color) if self.depth_enabled else None
                log.info("[RealSense] Started %s at %dx%d@%d",
                         "COLOR+DEPTH" if self.depth_enabled else "COLOR-ONLY", w, h, fps)
                for _ in range(10): _ = self.pipe.wait_for_frames()  # warmup
                return
            except Exception as e:
                last_err = e
                try:
                    if self.pipe: self.pipe.stop()
                except:
                    pass
                self.pipe = None
        raise RuntimeError(f"RealSense start failed: {last_err}")

    def read(self):
        frames = self.pipe.wait_for_frames()
        if self.depth_enabled and self.align is not None:
            frames = self.align.process(frames)
            d = frames.get_depth_frame()
        else:
            d = None
        c = frames.get_color_frame()
        if not c: return None, None, time.time()
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).copy() if d else None
        ts = c.get_timestamp() / 1000.0
        return bgr, depth, ts

    def stop(self):
        if self.pipe:
            try:
                self.pipe.stop()
            except:
                pass

    def camera_kind(self) -> str:
        return "intel_realsense"


class RealSenseBag(FrameSource):
    def __init__(self, bag_path: Path):
        if rs is None: raise RuntimeError("pyrealsense2 not available.")
        self.bag_path = bag_path
        self.pipe, self.align, self.profile = None, None, None

    def start(self):
        cfg = rs.config();
        cfg.enable_device_from_file(str(self.bag_path), repeat_playback=False)
        self.pipe = rs.pipeline();
        self.profile = self.pipe.start(cfg);
        self.align = rs.align(rs.stream.color)

    def read(self):
        try:
            frames = self.pipe.wait_for_frames()
        except Exception:
            return None, None, time.time()
        frames = self.align.process(frames)
        c = frames.get_color_frame();
        d = frames.get_depth_frame() if frames.get_depth_frame() else None
        if not c: return None, None, time.time()
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).copy() if d else None
        ts = c.get_timestamp() / 1000.0
        return bgr, depth, ts

    def stop(self):
        if self.pipe: self.pipe.stop()

    def camera_kind(self) -> str:
        return "intel_realsense"


# ------------------------------ Face registry --------------------------------

@dataclass
class FaceEntry:
    object_name: str
    face_key: str
    board_yaml: Path  # absolute
    T_board_object: np.ndarray  # 4x4
    origin_id: int
    tag_size_m: float
    T_board_tag: Dict[int, np.ndarray]  # tag_id -> 4x4


# --- Registry using manifest (fallback: scan faces/*) ------------------------
def _scan_faces_yaml(project_root: Path) -> List[Path]:
    return list((project_root / "faces").glob("*/*/*_T_board_object.yaml"))


def load_face_registry(
        project_root: Path,
        *,
        face_manifest_path: Optional[Path] = None,
        allow_face_scan: bool = False,
) -> Dict[str, FaceEntry]:
    """
    Build registry from faces/face_manifest.csv when present; otherwise scan faces/*.
    """
    reg: Dict[str, FaceEntry] = {}
    manifest_rows = _read_faces_manifest(project_root, face_manifest_path)

    yaml_paths: List[Tuple[str, Optional[float], Path, Optional[
        Path]]] = []  # (face_key, tag_size_m_hint, yaml_path, board_yaml_from_manifest)

    if manifest_rows:
        for r in manifest_rows:
            if not r.yaml_path.exists():
                log.warning("[faces-manifest] YAML missing on disk, skipping: %s", r.yaml_path)
                continue
            yaml_paths.append((r.face_key, r.tag_size_m, r.yaml_path, r.board_yaml))
    else:
        if not allow_face_scan:
            manifest_path = face_manifest_path or _faces_manifest_path(project_root)
            raise SystemExit(
                f"[!] Face manifest not found: {manifest_path}. Run posetag-annotate "
                "so faces/face_manifest.csv lists T_board_object YAMLs."
            )
        # Fallback scan
        for y in _scan_faces_yaml(project_root):
            face_key = y.stem.replace("_T_board_object", "")
            yaml_paths.append((face_key, None, y, None))

    if not yaml_paths:
        raise SystemExit("[!] No faces found. Run annotate_shots to create face YAMLs / manifest.")

    for face_key, tag_hint, ypath, board_yaml_from_manifest in yaml_paths:
        try:
            data = yaml.safe_load(open(ypath, "r"))
            obj = data["object"]
            face_key_file = data.get("face_key", face_key)
            if face_key_file != face_key:
                face_key = face_key_file  # prefer explicit

            # Board file: prefer YAML’s field; fallback to manifest-provided
            board_yaml = data.get("board_yaml")
            board_yaml = Path(board_yaml) if board_yaml else board_yaml_from_manifest
            if board_yaml is None:
                log.warning("[faces] %s has no board_yaml; skipping", ypath)
                continue
            if not Path(board_yaml).is_absolute():
                board_yaml = (project_root / board_yaml).resolve()

            T_bo = np.array(data["T_board_object"]["matrix"], float)

            # Truth from board file (also gives the tag dictionary)
            origin_id, tag_size_m_board, T_board_tag = load_board(board_yaml)

            # Optional consistency check with manifest hint
            if (tag_hint is not None) and (abs(float(tag_hint) - float(tag_size_m_board)) > 1e-9):
                log.warning("[faces] tag_size mismatch for %s: manifest=%.6f, board=%.6f (using board)",
                            face_key, float(tag_hint), float(tag_size_m_board))

            reg[face_key] = FaceEntry(
                object_name=obj,
                face_key=face_key,
                board_yaml=board_yaml,
                T_board_object=T_bo,
                origin_id=int(origin_id),
                tag_size_m=float(tag_size_m_board),
                T_board_tag=T_board_tag,
            )
        except Exception as e:
            log.warning("[faces] Skip bad YAML %s: %s", ypath, e)

    if not reg:
        raise SystemExit("[!] Found no valid faces after reading manifest/scan.")
    return reg


# --- faces/face_manifest.csv -------------------------------------------------
@dataclass
class FaceManifestRow:
    timestamp: str
    object: str
    side: str
    face_key: str
    yaml_path: Path
    board_yaml: Path
    image: Optional[str]
    rms_px: Optional[float]
    tag_size_m: Optional[float]


def _faces_manifest_path(project_root: Path, face_manifest_path: Optional[Path] = None) -> Path:
    if face_manifest_path is not None:
        return Path(face_manifest_path).expanduser()
    return project_root / "faces" / "face_manifest.csv"


def _read_faces_manifest(project_root: Path, face_manifest_path: Optional[Path] = None) -> List[FaceManifestRow]:
    p = _faces_manifest_path(project_root, face_manifest_path)
    rows: List[FaceManifestRow] = []
    if not p.exists():
        log.warning("[faces-manifest] Not found: %s (will fall back to scanning faces/*)", p)
        return rows

    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                obj = (d.get("object") or d.get("object_base") or "").strip()
                side = (d.get("side") or "").strip().upper()
                face_key = (d.get("face_key") or "").strip()
                yaml_path = (project_root / d["yaml_path"]).resolve() if not Path(
                    d["yaml_path"]).is_absolute() else Path(d["yaml_path"])
                board_yaml = (project_root / d["board_yaml"]).resolve() if not Path(
                    d["board_yaml"]).is_absolute() else Path(d["board_yaml"])
                rms_px = float(d["rms_px"]) if d.get("rms_px") not in (None, "", "None") else None
                tag_sz = float(d["tag_size_m"]) if d.get("tag_size_m") not in (None, "", "None") else None
                rows.append(FaceManifestRow(
                    timestamp=d.get("timestamp", ""),
                    object=obj, side=side, face_key=face_key,
                    yaml_path=yaml_path, board_yaml=board_yaml,
                    image=d.get("image") or None, rms_px=rms_px, tag_size_m=tag_sz
                ))
            except Exception as e:
                log.warning("[faces-manifest] Skip bad row: %s (%s)", d, e)
    return rows


# ------------------------------ Projections ----------------------------------
def project_points(pts3d: np.ndarray, T_cam_obj: np.ndarray, K: np.ndarray, dist: Optional[np.ndarray]) -> np.ndarray:
    R = T_cam_obj[:3, :3];
    t = T_cam_obj[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, t, K, dist if dist is not None else np.zeros((1, 5)))
    return uv.reshape(-1, 2)


def draw_axes(img: np.ndarray, T: np.ndarray, K: np.ndarray, dist: Optional[np.ndarray], scale: float = 0.05):
    origin = np.array([[0, 0, 0]], np.float32)
    axes = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]], np.float32)
    pts = np.vstack([origin, axes])
    uv = project_points(pts, T, K, dist)
    o = tuple(np.int32(uv[0]))
    x = tuple(np.int32(uv[1]));
    y = tuple(np.int32(uv[2]));
    z = tuple(np.int32(uv[3]))
    cv2.line(img, o, x, (0, 0, 255), 2);
    cv2.line(img, o, y, (0, 255, 0), 2);
    cv2.line(img, o, z, (255, 0, 0), 2)


# ---------------------------- Pose estimation per frame -----------------------

def estimate_poses_multi(
        bgr: np.ndarray,
        intr: Intrinsics,
        family: str,
        faces: Dict[str, FaceEntry],
        pts3d_by_face: Dict[str, np.ndarray],
        pts3d_by_object: Dict[str, np.ndarray],
        check_tag_scale: bool,
        auto_correct_scale: bool,
        scale_tol: float,
        axes_mode: str = "both",
        bbox_mode: str = "auto",
        bbox_tag_mult: float = 5.0,
        bbox_pad_frac: float = 0.06,
        bbox_min_area: int = 12000,
        tag_cover_margin_m: float = 0.003,  # ~1.5 cm of paper allowance
        tag_sample_extra_m: float = 0.003,  # color ring thickness outside allowance
) -> Tuple[np.ndarray, np.ndarray, Dict[str, dict], np.ndarray]:
    """
    Returns:
      anno_vis (left panel), reproj_vis (mid panel), results dict, and
      covered_vis (tag-cover-only image to save as the 'raw' image).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], float)
    dist = intr.dist if intr.dist is not None else np.zeros((1, 5), float)

    # detect once per unique tag size
    tag_sizes = sorted({f.tag_size_m for f in faces.values()})
    det_by_id: Dict[int, object] = {}
    for tag_size in tag_sizes:
        dets = detect_tags(gray, intr.fx, intr.fy, intr.cx, intr.cy, tag_size, family=family)
        for d in dets:
            det_by_id[int(d.tag_id)] = d

    covered = bgr.copy()  # <-- tag cover only
    anno = bgr.copy()  # <-- tag cover + overlays for UI
    reproj = bgr.copy()
    results: Dict[str, dict] = {}

    H, W = bgr.shape[:2]

    def _box_from_uv(uv: np.ndarray, W: int, H: int) -> Tuple[List[float], float]:
        x0, y0 = uv.min(axis=0);
        x1, y1 = uv.max(axis=0)
        x0 = max(0.0, min(W - 1.0, float(x0)));
        y0 = max(0.0, min(H - 1.0, float(y0)))
        x1 = max(0.0, min(W - 1.0, float(x1)));
        y1 = max(0.0, min(H - 1.0, float(y1)))
        w = max(0.0, x1 - x0);
        h = max(0.0, y1 - y0)
        return [x0, y0, w, h], (w * h)

    def _square3d(side_len_m: float) -> np.ndarray:
        hlf = 0.5 * float(side_len_m)
        return np.array([[-hlf, -hlf, 0],
                         [hlf, -hlf, 0],
                         [hlf, hlf, 0],
                         [-hlf, hlf, 0]], np.float32)

    def _poly_from_uv(uv: np.ndarray) -> np.ndarray:
        return uv.astype(np.int32).reshape(-1, 1, 2)

    def _project_square(T_cam_tag: np.ndarray, side_len_m: float) -> np.ndarray:
        return project_points(_square3d(side_len_m), T_cam_tag, K, dist)

    def _median_color_ring(img: np.ndarray, uv_inner: np.ndarray, uv_outer: np.ndarray) -> Tuple[int, int, int]:
        mask_outer = np.zeros(img.shape[:2], np.uint8)
        mask_inner = np.zeros_like(mask_outer)
        cv2.fillConvexPoly(mask_outer, _poly_from_uv(uv_outer), 255, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask_inner, _poly_from_uv(uv_inner), 255, lineType=cv2.LINE_AA)
        ring = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))
        ys, xs = np.where(ring > 0)
        if ys.size == 0:
            try:
                b, g, r = _dominant_color_near_polygon(img, uv_outer)
                return int(b), int(g), int(r)
            except Exception:
                return 128, 128, 128
        med = np.median(img[ys, xs].astype(np.uint8), axis=0)
        return int(med[0]), int(med[1]), int(med[2])

    for face_key, fe in faces.items():
        common = [tid for tid in fe.T_board_tag.keys() if tid in det_by_id]
        if not common and (fe.origin_id not in det_by_id):
            continue

        # pose from origin or best common
        if fe.origin_id in det_by_id and getattr(det_by_id[fe.origin_id], "pose_R", None) is not None:
            d_used = det_by_id[fe.origin_id]
            T_cam_tag_used = se3(d_used.pose_R.astype(float), d_used.pose_t.reshape(3).astype(float))
            T_cam_board = T_cam_tag_used
            tag_used = fe.origin_id
        else:
            tag_used = max(common, key=lambda tid: getattr(det_by_id[tid], "decision_margin", 0.0))
            d_used = det_by_id[tag_used]
            T_cam_tag_used = se3(d_used.pose_R.astype(float), d_used.pose_t.reshape(3).astype(float))
            T_cam_board = T_cam_tag_used @ inv_se3(fe.T_board_tag[tag_used])

        # ---- tag cover & sampling (paper allowance aware) ----
        tag_side = float(fe.tag_size_m)
        cover_side = tag_side + 2.0 * float(tag_cover_margin_m)
        sample_side = cover_side + 2.0 * float(tag_sample_extra_m)

        uv_cover = _project_square(T_cam_tag_used, cover_side)
        uv_sample = _project_square(T_cam_tag_used, sample_side)
        overlay_bgr = _median_color_ring(bgr, uv_cover, uv_sample)

        # paint cover on BOTH covered & anno buffers
        cv2.fillConvexPoly(covered, _poly_from_uv(uv_cover), overlay_bgr, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(anno, _poly_from_uv(uv_cover), overlay_bgr, lineType=cv2.LINE_AA)

        # draw id only on anno
        cxy = np.int32(uv_cover.mean(axis=0))
        cv2.putText(anno, str(tag_used), (int(cxy[0]) + 1, int(cxy[1]) - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(anno, str(tag_used), (int(cxy[0]) + 1, int(cxy[1]) - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_bgr, 1, cv2.LINE_AA)

        # cover other visible tags on the same rigid body
        for tid in common:
            if tid == tag_used:
                continue
            d = det_by_id[tid]
            if getattr(d, "pose_R", None) is None:
                continue
            T_cam_tag_i = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
            uv_cov_i = _project_square(T_cam_tag_i, cover_side)
            cv2.fillConvexPoly(covered, _poly_from_uv(uv_cov_i), overlay_bgr, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(anno, _poly_from_uv(uv_cov_i), overlay_bgr, lineType=cv2.LINE_AA)
            cxy_i = np.int32(uv_cov_i.mean(axis=0))
            cv2.putText(anno, str(tid), (int(cxy_i[0]) + 1, int(cxy_i[1]) - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(anno, str(tid), (int(cxy_i[0]) + 1, int(cxy_i[1]) - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_bgr, 1, cv2.LINE_AA)

        # optional scale check/correct
        s, n_pairs, mad = 1.0, 0, 0.0
        if check_tag_scale or auto_correct_scale:
            s, n_pairs, mad = _estimate_tag_scale(det_by_id, fe.T_board_tag)
            if abs(s - 1.0) > scale_tol:
                if auto_correct_scale:
                    T_cam_board[:3, 3] /= s
                    log.warning("[scale] tag_size mismatch ~%.1f%% -> applied 1/s to T_cam_board (face=%s)",
                                (s - 1.0) * 100.0, face_key)
                else:
                    log.warning("[scale] tag_size mismatch ~%.1f%% (face=%s) consider --auto-correct-scale",
                                (s - 1.0) * 100.0, face_key)

        T_cam_obj = T_cam_board @ fe.T_board_object

        # reprojection debug
        try:
            pts3d_dbg = np.array([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]], np.float32)
            uv_dbg = project_points(pts3d_dbg, T_cam_board, K, dist)
            for (u, v) in uv_dbg:
                cv2.circle(reproj, (int(u), int(v)), 3, (0, 0, 255), -1)
        except Exception:
            pass

        # bboxes (for anno/UI only)
        candidates = []
        if bbox_mode in ("face", "auto") and face_key in pts3d_by_face:
            uv_face = project_points(pts3d_by_face[face_key], T_cam_obj, K, dist)
            for (u, v) in uv_face:
                cv2.circle(anno, (int(u), int(v)), 2, (255, 255, 0), -1)
            box_face, area_face = _box_from_uv(uv_face, W, H)
            candidates.append(("face_kps", box_face, area_face))

        if bbox_mode in ("object", "auto") and fe.object_name in pts3d_by_object:
            uv_obj = project_points(pts3d_by_object[fe.object_name], T_cam_obj, K, dist)
            box_obj, area_obj = _box_from_uv(uv_obj, W, H)
            candidates.append(("object_kps", box_obj, area_obj))

        bbox_source, bbox_xywh, score_area = None, None, 0.0
        if candidates:
            bbox_source, bbox_xywh, score_area = max(candidates, key=lambda t: t[2])

        if (bbox_xywh is None) or (score_area < bbox_min_area):
            side_len_fb = float(fe.tag_size_m) * float(bbox_tag_mult)
            uv_tag_fb = _project_square(T_cam_tag_used, side_len_fb)
            bbox_xywh, score_area = _box_from_uv(uv_tag_fb, W, H)
            bbox_source = "fallback_tag"

        if bbox_xywh is not None:
            x0, y0, w, h = bbox_xywh
            padw = w * float(bbox_pad_frac);
            padh = h * float(bbox_pad_frac)
            x0 = max(0.0, x0 - padw);
            y0 = max(0.0, y0 - padh)
            x1 = min(W - 1.0, x0 + w + 2 * padw);
            y1 = min(H - 1.0, y0 + h + 2 * padh)
            bbox_xywh = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            score_area = bbox_xywh[2] * bbox_xywh[3]
            cv2.rectangle(anno, (int(x0), int(y0)),
                          (int(x1), int(y1)), (0, 255, 255), 2)

        # axes + label on anno only
        if axes_mode in ("both", "board"):  draw_axes(anno, T_cam_board, K, dist, scale=0.06)
        if axes_mode in ("both", "object"): draw_axes(anno, T_cam_obj, K, dist, scale=0.04)
        label = f"{fe.object_name}/{fe.face_key}"
        cv2.putText(anno, label, (12, 24 + 18 * (hash(face_key) % 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        num_tags = len(common) if common else (1 if fe.origin_id in det_by_id else 0)
        score = (num_tags * 1_000.0) + float(score_area)

        results[face_key] = {
            "object": fe.object_name,
            "face_key": fe.face_key,
            "board_yaml": str(fe.board_yaml),
            "tag_used": int(tag_used),
            "num_tags_visible": int(num_tags),
            "score": float(score),
            "score_tags": int(num_tags),
            "score_area_px": int(score_area),
            "bbox_xywh": bbox_xywh,
            "bbox_source": bbox_source,
            "T_cam_board": {"matrix": T_cam_board.tolist()},
            "T_board_object": {"matrix": fe.T_board_object.tolist()},
            "T_cam_object": {"matrix": T_cam_obj.tolist()},
            "overlay_bgr": [int(overlay_bgr[0]), int(overlay_bgr[1]), int(overlay_bgr[2])],
            "diagnostics": {
                "tag_scale_ratio": float(s),
                "tag_scale_pairs": int(n_pairs),
                "tag_scale_mad": float(mad),
                "tag_scale_auto_corrected": bool(auto_correct_scale and abs(s - 1.0) > scale_tol),
            },
        }

    return anno, reproj, results, covered


# ------------------------------- Saver ---------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_frame(
        out_dir: Path,
        idx: int,
        ts: float,
        bgr_raw: np.ndarray,  # <- RAW image (no overlays)
        depth: Optional[np.ndarray],
        intr: Intrinsics,
        family: str,
        results: Dict[str, dict],
        reproj_img: Optional[np.ndarray],
        save_depth: bool,
        dataset_name: str,
        pts3d_by_object: Optional[Dict[str, np.ndarray]] = None,
        anno_img: Optional[np.ndarray] = None,  # <- NEW: annotated image
        capture_metadata: Optional[dict] = None,
):
    if not results:
        raise CollectDatasetError(
            "Cannot save a dataset frame without at least one pose-labelled object."
        )

    # --- folder layout ---
    images_dir = out_dir / "images"  # raw rgb only
    reproj_dir = out_dir / "reproj"  # reprojection / tag debug
    annotated_dir = out_dir / "annotated"  # overlays (bboxes, tag covers, labels)
    depth_dir = out_dir / "depth"
    ann_dir = out_dir / "annotations"

    for p in (images_dir, reproj_dir, annotated_dir, depth_dir, ann_dir):
        p.mkdir(parents=True, exist_ok=True)

    # ---- naming ----
    tag = _ts_tag_from_epoch(ts)
    rgb_name = f"Run_{tag}_rgb_frame_{idx:06d}.png"
    reproj_name = f"Run_{tag}_reproj_frame_{idx:06d}.png"
    anno_img_name = f"Run_{tag}_annotated_frame_{idx:06d}.png"
    depth_name = f"Run_{tag}_depth_frame_{idx:06d}.npy"
    ann_name = f"Run_{tag}_frame_{idx:06d}.json"

    img_path = images_dir / rgb_name
    reproj_path = reproj_dir / reproj_name
    annoimg_path = annotated_dir / anno_img_name
    depth_path = depth_dir / depth_name
    ann_path = ann_dir / ann_name
    log_path = out_dir / f"{dataset_name}.jsonl"

    # ---- write imagery ----
    cv2.imwrite(str(img_path), bgr_raw)  # RAW saved here
    if anno_img is not None:
        cv2.imwrite(str(annoimg_path), anno_img)
    if reproj_img is not None:
        cv2.imwrite(str(reproj_path), reproj_img)
    if save_depth and depth is not None:
        np.save(str(depth_path), depth)

    record = build_frame_annotation_record(
        dataset_name=dataset_name,
        frame_index=idx,
        timestamp=ts,
        tag_family=family,
        image_filename=rgb_name,
        annotated_image=(annoimg_path.name if anno_img is not None else None),
        reproj_image=(reproj_path.name if reproj_img is not None else None),
        depth=(depth_path.name if (save_depth and depth is not None) else None),
        intrinsics=intr,
        results=results,
        pts3d_by_object=pts3d_by_object,
        capture_metadata=capture_metadata,
    )
    ann_path.write_text(json.dumps(record, indent=5))

    log_entry = {
        "idx": idx,
        "timestamp": float(ts),
        "image": img_path.name,
        "annotated_image": (annoimg_path.name if anno_img is not None else None),
        "reproj_image": (reproj_path.name if reproj_img is not None else None),
        "depth": (depth_path.name if (save_depth and depth is not None) else None),
        "tag_family": family,
        "objects": sorted({r["object"] for r in results.values()}),
    }
    if capture_metadata:
        log_entry["capture"] = capture_metadata
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# ------------------------------- Main loop -----------------------------------

class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Show defaults while preserving multiline help."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posetag-collect",
        description="Collect multi-object/multi-face GT dataset with review UI",
        formatter_class=_HelpFmt,
    )
    parser.add_argument("--project_root", type=Path, default=None,
                        help="Root for boards/faces/objects/datasets (default: resolver/env/config).")
    parser.add_argument("--mode", choices=["live", "opencv", "bag", "video"], required=True,
                        help="Frame source: RealSense live, OpenCV webcam, RealSense bag, or video file.")
    parser.add_argument("--bag", type=str, help="Path to RealSense .bag (for mode=bag)")
    parser.add_argument("--video", type=str, help="Path to a video file (for mode=video)")
    parser.add_argument("--session", required=True, help="Dataset session name (folder under datasets/)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Dataset root directory (default: <project_root>/datasets)")
    parser.add_argument("--calib", default="calib_color.yaml",
                        help="Calibration YAML (default resolves to calib/calib_color.yaml)")
    parser.add_argument("--registry", default=None,
                        help="Tag registry YAML (default: <project_root>/boards/tag_registry.yaml)")
    parser.add_argument("--face_manifest", default=None,
                        help="Face manifest CSV (default: <project_root>/faces/face_manifest.csv)")
    parser.add_argument("--allow-face-scan", action="store_true",
                        help="Legacy fallback: scan faces/**/*_T_board_object.yaml if face_manifest is missing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print a summary without opening a camera or writing outputs.")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--continuous", action="store_true",
                        help="Do not prompt per frame; save each frame that has at least one pose-labelled object.")
    parser.add_argument("--auto-capture", action="store_true",
                        help="Quality-gated smart auto-capture; saves stable, useful views automatically.")
    parser.add_argument("--auto-stable-frames", type=int, default=5,
                        help="Consecutive stable frames required before smart auto-capture can save.")
    parser.add_argument("--auto-cooldown-sec", type=float, default=1.0,
                        help="Minimum time between smart auto-capture saves.")
    parser.add_argument("--auto-min-tags", type=int, default=1,
                        help="Minimum visible tags for a smart auto-capture candidate.")
    parser.add_argument("--auto-min-bbox-area", type=float, default=12000.0,
                        help="Minimum candidate bbox area in pixels for smart auto-capture.")
    parser.add_argument("--auto-max-tag-scale-error", type=float, default=0.08,
                        help="Maximum allowed relative tag-scale error when scale diagnostics are available.")
    parser.add_argument("--auto-min-translation-delta-m", type=float, default=0.04,
                        help="Translation threshold used for stability and pose-diversity gating.")
    parser.add_argument("--auto-min-rotation-delta-deg", type=float, default=8.0,
                        help="Rotation threshold used for stability and pose-diversity gating.")
    parser.add_argument("--auto-grid", default="3x3",
                        help="Image coverage grid for smart auto-capture, formatted ROWSxCOLS.")
    parser.add_argument("--auto-distance-bin-m", type=float, default=0.25,
                        help="Distance-bin size used for smart auto-capture diversity.")
    parser.add_argument("--auto-max-frames-per-object", type=int, default=0,
                        help="Maximum smart auto-captured frames per object (0=unlimited).")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0=unlimited)")
    parser.add_argument("--cam", type=int, default=0, help="OpenCV camera index when --mode=opencv")
    parser.add_argument("--width", type=int, default=0, help="OpenCV webcam width; 0=calibration width")
    parser.add_argument("--height", type=int, default=0, help="OpenCV webcam height; 0=calibration height")
    parser.add_argument("--fps", type=int, default=30, help="OpenCV webcam FPS request")
    parser.add_argument("--rs_w", type=int, default=0, help="Color width; 0=auto")
    parser.add_argument("--rs_h", type=int, default=0, help="Color height; 0=auto")
    parser.add_argument("--rs_fps", type=int, default=30, help="FPS for live mode")
    parser.add_argument("--save_depth", action="store_true",
                        help="If present, save aligned depth as .npy per frame")
    parser.add_argument("--axes", choices=["both", "board", "object", "none"], default="both",
                        help="Which axes to draw on the left panel")
    # Tag scale options (parity with annotate_shots.py)
    parser.add_argument("--check-tag-scale", action="store_true",
                        help="Print/record scale ratio s from inter-tag distances")
    parser.add_argument("--auto-correct-scale", action="store_true",
                        help="If |s-1|>tol, divide T_cam_board translation by s")
    parser.add_argument("--scale-tol", type=float, default=0.02,
                        help="Relative tolerance (default 0.02 = 2%%)")
    parser.add_argument("--bbox-mode", choices=["auto", "face", "object"], default="auto",
                        help="Which 3D points drive the bbox: 'face' (4 corners), 'object' (all kps), or 'auto' (larger).")
    parser.add_argument("--bbox-tag-mult", type=float, default=5.0,
                        help="Multiplier for fallback tag-square (default 5.0; was hardcoded 3.0).")
    parser.add_argument("--bbox-min-area", type=int, default=12000,
                        help="If bbox area < this, use tag fallback (default 12000 px^2).")
    parser.add_argument("--bbox-pad-frac", type=float, default=0.06,
                        help="Pad final bbox by this fraction of its size (default 6%%).")
    parser.add_argument("--tag-cover-margin-m", type=float, default=0.010)
    parser.add_argument("--tag-sample-extra-m", type=float, default=0.008)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        if args.continuous and args.auto_capture:
            raise CollectDatasetError("--continuous and --auto-capture are mutually exclusive.")
        auto_grid_rows, auto_grid_cols = parse_auto_capture_grid(args.auto_grid)
        if args.auto_stable_frames <= 0:
            raise CollectDatasetError("--auto-stable-frames must be positive.")
        if args.auto_cooldown_sec < 0.0:
            raise CollectDatasetError("--auto-cooldown-sec must be non-negative.")
        if args.auto_min_tags <= 0:
            raise CollectDatasetError("--auto-min-tags must be positive.")
        if args.auto_min_bbox_area < 0.0:
            raise CollectDatasetError("--auto-min-bbox-area must be non-negative.")
        if args.auto_max_tag_scale_error < 0.0:
            raise CollectDatasetError("--auto-max-tag-scale-error must be non-negative.")
        if args.auto_min_translation_delta_m < 0.0:
            raise CollectDatasetError("--auto-min-translation-delta-m must be non-negative.")
        if args.auto_min_rotation_delta_deg < 0.0:
            raise CollectDatasetError("--auto-min-rotation-delta-deg must be non-negative.")
        if args.auto_distance_bin_m <= 0.0:
            raise CollectDatasetError("--auto-distance-bin-m must be positive.")
        if args.auto_max_frames_per_object < 0:
            raise CollectDatasetError("--auto-max-frames-per-object must be non-negative.")
        project_root = preview_project_root(args.project_root)
        calib_path = resolve_calibration_path(project_root, args.calib)
        intr = load_intrinsics_from_calib(calib_path)
        registry_path = resolve_registry_path(project_root, args.registry)
        face_manifest_path = resolve_face_manifest_path(project_root, args.face_manifest)
        datasets_root = resolve_dataset_root(project_root, args.dataset_root)

        validate_source_args(
            mode=args.mode,
            video=args.video,
            bag=args.bag,
            realsense_module=rs,
            video_capture_factory=cv2.VideoCapture if args.mode == "video" else None,
        )
        input_summary = validate_collection_inputs(
            project_root=project_root,
            calib_path=calib_path,
            registry_path=registry_path,
            face_manifest_path=face_manifest_path,
            allow_face_scan=args.allow_face_scan,
        )
    except CollectDatasetError as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print("Dataset collection dry run OK")
        print(f"  project_root: {project_root}")
        print(f"  calibration : {input_summary.calibration_path}")
        print(f"  registry    : {input_summary.registry_path}")
        print(f"  manifest    : {input_summary.face_manifest_path}")
        print(f"  annotations : {len(input_summary.annotations)}")
        print(f"  source mode : {args.mode}")
        if input_summary.missing_keypoints:
            print("  optional keypoints missing:")
            for path in input_summary.missing_keypoints:
                print(f"    - {path}")
        return 0

    ensure_project_dirs(project_root)

    # Logger
    global log
    log = init_project_logger(project_root / "logs" / "collect_gt_dataset.log",
                              level="INFO", console=True)

    # Face registry (from <project_root>/faces)
    faces = load_face_registry(
        project_root,
        face_manifest_path=face_manifest_path,
        allow_face_scan=args.allow_face_scan,
    )
    log.info("[i] Faces loaded: %d", len(faces))

    # Choose frame source
    if args.mode == "live":
        w = args.rs_w or intr.width
        h = args.rs_h or intr.height
        src = RealSenseLive(width=w, height=h, fps=args.rs_fps)
    elif args.mode == "opencv":
        w = args.width or intr.width
        h = args.height or intr.height
        src = OpenCVWebcamSource(cam=args.cam, width=w, height=h, fps=args.fps)
    elif args.mode == "bag":
        src = RealSenseBag(Path(args.bag).expanduser())
    else:
        src = VideoSource(Path(args.video).expanduser())

    # Start source
    src.start()
    cam_kind = src.camera_kind()

    # Output layout
    out_dir = datasets_root / args.session
    ensure_dir(datasets_root)
    ensure_dir(out_dir)

    cv2.namedWindow("GT Capture", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("GT Capture", 1640, 520)

    # Preload face keypoints for bbox projection
    pts3d_by_face: Dict[str, np.ndarray] = {}
    pts3d_by_object: Dict[str, np.ndarray] = {}
    _kpcache: Dict[str, tuple[dict, dict]] = {}
    for fk, fe in faces.items():
        try:
            if fe.object_name not in _kpcache:
                pts3d_dict, faces_map, _, _ = load_keypoints_fuzzy(fe.object_name, project_root)
                _kpcache[fe.object_name] = (pts3d_dict, faces_map)
                pts3d_by_object[fe.object_name] = np.vstack([v for v in pts3d_dict.values()]).astype(np.float32)
            pts3d_dict, faces_map = _kpcache[fe.object_name]
            if fk in faces_map:
                names = faces_map[fk]
                pts3d_by_face[fk] = np.vstack([pts3d_dict[n] for n in names]).astype(np.float32)
        except Exception as e:
            log.warning("[warn] keypoints for %s not fully available: %s", fk, e)

    auto_capture_config = AutoCaptureConfig(
        enabled=bool(args.auto_capture),
        stable_frames=int(args.auto_stable_frames),
        cooldown_sec=float(args.auto_cooldown_sec),
        min_tags_visible=int(args.auto_min_tags),
        min_bbox_area_px=float(args.auto_min_bbox_area),
        max_tag_scale_error=float(args.auto_max_tag_scale_error),
        min_translation_delta_m=float(args.auto_min_translation_delta_m),
        min_rotation_delta_deg=float(args.auto_min_rotation_delta_deg),
        grid_rows=int(auto_grid_rows),
        grid_cols=int(auto_grid_cols),
        distance_bin_m=float(args.auto_distance_bin_m),
        max_frames_per_object=int(args.auto_max_frames_per_object),
    )
    auto_capture_state = AutoCaptureState()

    # Session metadata scaffold
    faces_index = {fk: {"object": fe.object_name, "board_yaml": str(fe.board_yaml)} for fk, fe in faces.items()}
    session_meta = build_session_metadata(
        session_name=args.session,
        project_root=project_root,
        dataset_root=datasets_root,
        session_dir=out_dir,
        intrinsics=intr,
        tag_family=args.family,
        camera_kind=cam_kind,
        mode=args.mode,
        tag_scale_controls={
            "check_tag_scale": bool(args.check_tag_scale),
            "auto_correct_scale": bool(args.auto_correct_scale),
            "scale_tol": float(args.scale_tol),
        },
        faces_index=faces_index,
        missing_keypoints=input_summary.missing_keypoints,
    )
    session_meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    session_meta["capture_policy"] = {
        "mode": "smart_auto" if args.auto_capture else ("continuous" if args.continuous else "review"),
        "continuous": bool(args.continuous),
        "auto_capture": {
            "enabled": bool(auto_capture_config.enabled),
            "stable_frames": int(auto_capture_config.stable_frames),
            "cooldown_sec": float(auto_capture_config.cooldown_sec),
            "min_tags_visible": int(auto_capture_config.min_tags_visible),
            "min_bbox_area_px": float(auto_capture_config.min_bbox_area_px),
            "max_tag_scale_error": float(auto_capture_config.max_tag_scale_error),
            "min_translation_delta_m": float(auto_capture_config.min_translation_delta_m),
            "min_rotation_delta_deg": float(auto_capture_config.min_rotation_delta_deg),
            "grid": f"{int(auto_capture_config.grid_rows)}x{int(auto_capture_config.grid_cols)}",
            "distance_bin_m": float(auto_capture_config.distance_bin_m),
            "max_frames_per_object": int(auto_capture_config.max_frames_per_object),
        },
    }
    (out_dir / "session.yaml").write_text(yaml.safe_dump(session_meta, sort_keys=False))

    # Main loop
    idx = 0
    paused = False
    show_help = True
    last_anno = None;
    last_reproj = None
    last_bgr = None;
    last_depth = None;
    last_ts = None
    last_cov = None
    capture_msg, capture_msg_until = "", 0.0
    objects_seen_counts: Dict[str, int] = {}

    try:
        last_results = None
        while True:
            results: Dict[str, dict] = {}
            if not paused:
                bgr, depth, ts = src.read()
                if bgr is None:
                    log.info("[i] End of stream.")
                    break
                if (bgr.shape[1], bgr.shape[0]) != (intr.width, intr.height):
                    cv2.destroyAllWindows();
                    src.stop()
                    sys.exit(f"[!] Stream size {bgr.shape[1]}x{bgr.shape[0]} "
                             f"!= calibration {intr.width}x{intr.height}. "
                             f"Start with --rs_w {intr.width} --rs_h {intr.height} or fix the calibration.")
                # tag-covered, no overlays (what we save in images/)
                # estimate poses
                anno, reproj, results, covered = estimate_poses_multi(
                    bgr, intr, args.family, faces,
                    pts3d_by_face=pts3d_by_face,
                    pts3d_by_object=pts3d_by_object,
                    check_tag_scale=args.check_tag_scale,
                    auto_correct_scale=args.auto_correct_scale,
                    scale_tol=args.scale_tol,
                    axes_mode=args.axes,
                    bbox_mode=args.bbox_mode,
                    bbox_tag_mult=args.bbox_tag_mult,
                    bbox_pad_frac=args.bbox_pad_frac,
                    bbox_min_area=args.bbox_min_area,
                    tag_cover_margin_m=args.tag_cover_margin_m,  # <-- add
                    tag_sample_extra_m=args.tag_sample_extra_m,
                )
                last_anno, last_reproj, last_cov = anno, reproj, covered

                last_bgr, last_depth, last_ts = bgr, depth, ts
                last_results = results

            else:
                if last_anno is None:
                    H, W = intr.height, intr.width
                    last_anno = np.zeros((H, W, 3), np.uint8)
                if last_reproj is None:
                    last_reproj = np.zeros_like(last_anno)
                if last_results is not None:
                    results = last_results
                if last_cov is None:
                    last_cov = np.zeros_like(last_anno)

            # pick best per object (highest score)
            best_by_object: Dict[str, dict] = {}
            for fk, r in results.items():
                obj = r["object"]
                if (obj not in best_by_object) or (r["score"] > best_by_object[obj]["score"]):
                    best_by_object[obj] = r

            now = time.time()
            auto_candidates = auto_capture_candidates_from_results(best_by_object)
            auto_capture_decision = None
            auto_capture_message = ""
            if args.auto_capture and not paused:
                auto_capture_decision = evaluate_auto_capture(
                    auto_candidates,
                    auto_capture_state,
                    auto_capture_config,
                    frame_width=intr.width,
                    frame_height=intr.height,
                    timestamp=now,
                )
                auto_capture_message = auto_capture_decision.message
            elif args.auto_capture:
                auto_capture_message = "Auto: paused"

            object_summaries = []
            for obj, r in best_by_object.items():
                M = np.array(r["T_cam_object"]["matrix"], float)
                R, t = M[:3, :3], M[:3, 3]
                roll, pitch, yaw = rpy_from_R(R)
                diag = r.get("diagnostics", {})
                object_summaries.append(
                    {
                        "object": obj,
                        "face_key": r["face_key"],
                        "translation_m": tuple(float(v) for v in t),
                        "distance_m": float(np.linalg.norm(t)),
                        "rpy_deg": (float(roll), float(pitch), float(yaw)),
                        "tag_used": r.get("tag_used", "?"),
                        "bbox_source": r.get("bbox_source", "?"),
                        "tag_scale_ratio": diag.get("tag_scale_ratio", 1.0),
                        "tag_scale_pairs": diag.get("tag_scale_pairs", 0),
                        "tag_scale_auto_corrected": diag.get(
                            "tag_scale_auto_corrected",
                            False,
                        ),
                    }
                )

            right = _capture_status_panel(
                session=args.session,
                faces_visible=len(results),
                saved_count=idx,
                object_summaries=object_summaries,
                paused=paused,
                continuous=args.continuous,
                auto_capture_enabled=args.auto_capture,
                auto_capture_status=auto_capture_message,
                show_help=show_help,
                capture_message=capture_msg if capture_msg and now < capture_msg_until else "",
            )
            _show_dash(last_anno, last_reproj, right)

            # decide
            raw_key = cv2.waitKeyEx(1) if hasattr(cv2, "waitKeyEx") else cv2.waitKey(1)
            k = _normalize_capture_key(raw_key)
            try:
                if cv2.getWindowProperty("GT Capture", cv2.WND_PROP_VISIBLE) < 1:
                    log.info("[i] GT Capture window closed.")
                    break
            except cv2.error:
                log.info("[i] GT Capture window closed.")
                break
            if _capture_key_requests_quit(raw_key):
                log.info("[i] Quit requested from GT Capture window.")
                break
            if k == ord('h'): show_help = not show_help
            if k in (ord('p'), 32): paused = not paused  # pause/resume

            def _capture_now(capture_metadata: Optional[dict] = None) -> bool:
                nonlocal idx, capture_msg, capture_msg_until, objects_seen_counts
                if last_bgr is None or last_ts is None or last_cov is None:
                    return False
                if not best_by_object:
                    capture_msg = "No pose-labelled object visible; frame not saved"
                    capture_msg_until = time.time() + 2.0
                    return False
                save_frame(
                    out_dir, idx, last_ts,
                    bgr_raw=last_cov,  # <- was last_anno; now the pure frame
                    depth=last_depth,
                    intr=intr,
                    family=args.family,
                    results=best_by_object,
                    reproj_img=last_reproj,
                    save_depth=args.save_depth,
                    dataset_name=args.session,
                    pts3d_by_object=pts3d_by_object,
                    anno_img=last_anno,  # <- annotated goes to its own folder
                    capture_metadata=capture_metadata,
                )

                # update counters
                for r in best_by_object.values():
                    objects_seen_counts[r["object"]] = objects_seen_counts.get(r["object"], 0) + 1
                mode = (capture_metadata or {}).get("mode", "manual")
                reason = (capture_metadata or {}).get("reason")
                suffix = f" {reason}" if reason else ""
                capture_msg = f"Frame #{idx:03d} captured [{mode}]{suffix}"
                capture_msg_until = time.time() + 2.0
                idx += 1
                return True

            if k in (13, ord('y'), ord('s')):  # accept
                if _capture_now({"mode": "manual_review", "trigger": "key"}):
                    if args.auto_capture and auto_candidates:
                        try:
                            record_auto_capture_save(
                                auto_capture_state,
                                auto_candidates[0],
                                auto_capture_config,
                                frame_width=intr.width,
                                frame_height=intr.height,
                                timestamp=now,
                            )
                        except CollectDatasetError as exc:
                            log.warning("[auto-capture] manual save bookkeeping skipped: %s", exc)
            elif (
                args.auto_capture
                and not paused
                and auto_capture_decision is not None
                and auto_capture_decision.should_save
            ):
                if _capture_now(auto_capture_decision.metadata()):
                    for candidate in auto_candidates:
                        if (
                            candidate.object_name == auto_capture_decision.trigger_object
                            and candidate.face_key == auto_capture_decision.trigger_face
                        ):
                            record_auto_capture_save(
                                auto_capture_state,
                                candidate,
                                auto_capture_config,
                                frame_width=intr.width,
                                frame_height=intr.height,
                                timestamp=now,
                            )
                            break
            if args.continuous and not paused:
                _capture_now({"mode": "continuous"})

            if args.max_frames and idx >= args.max_frames:
                log.info("[i] Reached max_frames=%d.", args.max_frames)
                break

    except SystemExit as e:
        log.error("Exited: %s", e)
        raise
    except BaseException as e:
        log.error("[!] Error: %s", e, exc_info=True)
        raise
    finally:
        # finalize session metadata
        session_meta["frames_captured"] = int(idx)
        session_meta["objects_seen_counts"] = {k: int(v) for k, v in sorted(objects_seen_counts.items())}
        (out_dir / "session.yaml").write_text(yaml.safe_dump(session_meta, sort_keys=False))

        try:
            src.stop()
        except:
            pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
