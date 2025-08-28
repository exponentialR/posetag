# -*- coding: utf-8 -*-
"""
collect_gt_dataset.py
=====================
Collect a ground-truth dataset (multi-object, multi-face) from Intel RealSense
or a recorded video. For each frame, estimate pose for every face whose board
has at least one visible AprilTag. A review UI is provided, with optional
continuous capture.

What's new vs. previous script
------------------------------
- Uses the new project layout based on `resolve_project_root(...)`.
- Scans faces from <project_root>/faces/**/<face_key>_T_board_object.yaml.
- Session metadata is saved (version, name, camera kind, resolution, tag family,
  available objects/faces, created time, frames captured, objects_seen counts,
  and paths). Per-frame JSON + JSONL are also written.
- Tag-size verification copied from annotate_shots.py:
  --check-tag-scale to print the s ratio, and --auto-correct-scale to divide
  T_cam_board translation by s when |s-1| > tol. Diagnostics are recorded.

Project layout (inputs/outputs)
-------------------------------
<project_root>/
  faces/<object>/sideA|B|C|D/<face_key>_T_board_object.yaml   # inputs (from annotate_shots)
  objects/<object>/keypoints.json                              # inputs
  datasets/<session>/
    frames/
      000000_<ts>.png
      000000_<ts>_reproj.png
      000000_<ts>_depth.npy   (optional)
      000000_<ts>.json        # per-frame camera + poses
    session.yaml              # session-wide metadata (this run)
    <session>.jsonl           # rolling capture log
  logs/collect_gt_dataset.log # run log

Usage examples
--------------
  # RealSense live (RGB + depth if available)
  python -m src.collect_gt_dataset --mode live --session run01 --calib calib_color.yaml

  # RealSense playback from a .bag
  python -m src.collect_gt_dataset --mode bag --bag path/to/recording.bag --session run02 --calib calib_color.yaml

  # Any video readable by OpenCV
  python -m src.collect_gt_dataset --mode video --video sample.mp4 --session run03 --calib calib_color.yaml

Keys
----
  ENTER / y / s  -> accept & save frame
  r / n / BACKSPACE -> reject/skip frame
  ESC / q       -> abort current frame (continue stream)
  Q / X         -> quit-all immediately (stops stream + closes)
  SPACE or p    -> pause/resume stream (continuous mode keeps saving)
  h             -> toggle help panel
"""

from __future__ import annotations
import argparse, json, os, sys, time, math, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2, yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import pyrealsense2 as rs
except Exception:
    rs = None  # optional

# --- repo utils ---
from utils.project_config import resolve_project_root, ensure_project_dirs
from utils.logger import init_project_logger
from utils.annotation_utils import (
    detect_tags, load_board, se3, inv_se3, load_keypoints_fuzzy
)
from utils.gt_pose_utils import rotation_to_quat

try:
    from scipy.spatial.transform import Rotation as Rot
    def to_quat(R): return Rot.from_matrix(R).as_quat()  # [x,y,z,w]
except Exception:
    def to_quat(R): return rotation_to_quat(R)

try:
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
except Exception:
    pass

EXIT_QUIT_ALL = 99
DATASET_VERSION = "1.0"

log: logging.Logger  # set in main()

# --------------------------- Small math/helpers ------------------------------

def rpy_from_R(R: np.ndarray) -> tuple[float,float,float]:
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    return tuple(np.degrees([roll, pitch, yaw]).tolist())

def _estimate_tag_scale(det_by_id: Dict[int, object], T_board_tag: Dict[int, np.ndarray]) -> tuple[float,int,float]:
    """Median ratio of inter-tag distances (camera / board)."""
    ratios: List[float] = []
    ids = [k for k in det_by_id.keys() if k in T_board_tag]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pa = det_by_id[a].pose_t.reshape(3).astype(float)
            pb = det_by_id[b].pose_t.reshape(3).astype(float)
            d_cam = float(np.linalg.norm(pa - pb))
            ba = T_board_tag[a][:3, 3]; bb = T_board_tag[b][:3, 3]
            d_board = float(np.linalg.norm(ba - bb))
            if d_board > 1e-9:
                ratios.append(d_cam / d_board)
    if not ratios:
        return 1.0, 0, 0.0
    ratios = np.asarray(ratios, float)
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    return med, int(len(ratios)), mad

# --------------------------- Simple UI panels --------------------------------

def _pad_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    out = np.zeros((h, w, 3), np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out

def _resize_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    H, W = img.shape[:2]
    interp = cv2.INTER_AREA if (H > h or W > w) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)

def _label_strip(img: np.ndarray, text: str, bar_h: int = 26, bg=(0, 0, 0), fg=(0, 255, 255)) -> np.ndarray:
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), bg, -1)
    cv2.putText(vis, text, (8, int(bar_h*0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)
    return vis

def _text_panel(lines: List[str], width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    img = _label_strip(img, "Status / Legend")
    y = 40
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 1, cv2.LINE_AA)
        y += 22
        if y > height - 8: break
    return img

def _hstack(left: np.ndarray, mid: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    H = max(left.shape[0], 0 if mid is None else mid.shape[0], 0 if right is None else right.shape[0])
    Wl = left.shape[1]
    Wm = mid.shape[1] if mid is not None else Wl
    Wr = right.shape[1] if right is not None else 380
    return np.hstack([_pad_to(left, H, Wl), _pad_to(mid, H, Wm), _pad_to(right, H, Wr)])

def _show_dash(left: np.ndarray | None, mid: np.ndarray | None, right: np.ndarray | None,
               win: str = "GT Capture", tile_h: int = 480, tile_w: int = 640):
    L = _resize_to(left, tile_h, tile_w)
    M = _resize_to(mid,  tile_h, tile_w)
    R = _resize_to(right, tile_h, tile_w)
    L = _label_strip(L, "Annotation")
    M = _label_strip(M, "Reprojection / Tag debug")
    strip = np.hstack([L, M, R])
    cv2.imshow(win, strip)

def _quit_all():
    try:
        cv2.destroyAllWindows()
        try: sys.stdout.flush()
        except: pass
        try: sys.stderr.flush()
        except: pass
    finally:
        os._exit(EXIT_QUIT_ALL)

# --------------------------- Camera / Sources --------------------------------

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float
    width: int; height: int
    dist: Optional[np.ndarray] = None  # (1,5) if available

def _load_dist_from_calib(calib_path: Path) -> np.ndarray:
    try:
        y = yaml.safe_load(open(calib_path, "r"))
        dc = y.get("distortion_coefficients", {})
        return np.array([[dc.get("k1",0), dc.get("k2",0), dc.get("p1",0), dc.get("p2",0), dc.get("k3",0)]], float)
    except Exception:
        return np.zeros((1,5), float)

def load_intrinsics_from_calib(calib_path: Path) -> Intrinsics:
    y = yaml.safe_load(open(calib_path, "r"))
    cam = y["camera_matrix"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    w = int(y.get("image_width", 640))
    h = int(y.get("image_height", 480))
    dist = _load_dist_from_calib(calib_path)
    return Intrinsics(fx, fy, cx, cy, w, h, dist)

class FrameSource:
    def start(self): pass
    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:  # rgb, depth, ts_sec
        raise NotImplementedError
    def stop(self): pass
    def camera_kind(self) -> str: return "generic"

class VideoSource(FrameSource):
    def __init__(self, path: Path, fps_hint: float = 30.0):
        self.cap = cv2.VideoCapture(str(path))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or fps_hint
        self.fps_delay = 1.0 / max(1e-6, fps)
    def read(self):
        ok, bgr = self.cap.read()
        if not ok: return None, None, time.time()
        time.sleep(self.fps_delay)
        return bgr, None, time.time()
    def stop(self): self.cap.release()

class RealSenseLive(FrameSource):
    def __init__(self, width=0, height=0, fps=30):
        if rs is None: raise RuntimeError("pyrealsense2 not available.")
        self.req_w, self.req_h, self.req_fps = int(width), int(height), int(fps)
        self.w = self.h = self.fps = 0
        self.pipe = None; self.align = None; self.depth_enabled = False; self.profile = None
    def _try_start(self, w, h, fps, with_depth=True):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        depth_ok = False
        if with_depth:
            try:
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps); depth_ok = True
            except Exception:
                try:
                    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps); depth_ok = True
                except Exception:
                    depth_ok = False
        pipe = rs.pipeline(); profile = pipe.start(cfg)
        return pipe, profile, depth_ok
    def start(self):
        ctx = rs.context()
        if len(ctx.devices) == 0: raise RuntimeError("No RealSense device found.")
        trials = []
        if self.req_w > 0 and self.req_h > 0:
            trials.append((self.req_w, self.req_h, self.req_fps))
        trials += [(640,480,30), (640,360,30)]
        last_err = None
        for (w,h,fps) in trials:
            try:
                pipe, profile, depth_ok = self._try_start(w,h,fps, with_depth=True)
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
                except: pass
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
            try: self.pipe.stop()
            except: pass
    def camera_kind(self) -> str: return "intel_realsense"

class RealSenseBag(FrameSource):
    def __init__(self, bag_path: Path):
        if rs is None: raise RuntimeError("pyrealsense2 not available.")
        self.bag_path = bag_path
        self.pipe, self.align, self.profile = None, None, None
    def start(self):
        cfg = rs.config(); cfg.enable_device_from_file(str(self.bag_path), repeat_playback=False)
        self.pipe = rs.pipeline(); self.profile = self.pipe.start(cfg); self.align = rs.align(rs.stream.color)
    def read(self):
        try: frames = self.pipe.wait_for_frames()
        except Exception: return None, None, time.time()
        frames = self.align.process(frames)
        c = frames.get_color_frame(); d = frames.get_depth_frame() if frames.get_depth_frame() else None
        if not c: return None, None, time.time()
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).copy() if d else None
        ts = c.get_timestamp() / 1000.0
        return bgr, depth, ts
    def stop(self):
        if self.pipe: self.pipe.stop()
    def camera_kind(self) -> str: return "intel_realsense"

# ------------------------------ Face registry --------------------------------

@dataclass
class FaceEntry:
    object_name: str
    face_key: str
    board_yaml: Path             # absolute
    T_board_object: np.ndarray   # 4x4
    origin_id: int
    tag_size_m: float
    T_board_tag: Dict[int, np.ndarray]  # tag_id -> 4x4

def load_face_registry(project_root: Path) -> Dict[str, FaceEntry]:
    """
    Scan faces/**/<face_key>_T_board_object.yaml and build a registry: face_key -> FaceEntry
    """
    reg: Dict[str, FaceEntry] = {}
    for y in project_root.glob("faces/**/_T_board_object.yaml"):
        data = yaml.safe_load(open(y, "r"))
        obj = data["object"]
        face_key = data["face_key"]
        board_yaml = Path(data["board_yaml"])
        if not board_yaml.is_absolute():
            board_yaml = (project_root / board_yaml).resolve()
        T_bo = np.array(data["T_board_object"]["matrix"], float)
        origin_id, tag_size_m, T_board_tag = load_board(board_yaml)
        reg[face_key] = FaceEntry(
            object_name=obj,
            face_key=face_key,
            board_yaml=board_yaml,
            T_board_object=T_bo,
            origin_id=int(origin_id),
            tag_size_m=float(tag_size_m),
            T_board_tag=T_board_tag,
        )
    if not reg:
        raise SystemExit("[!] No faces registered. Run annotate_shots.py first.")
    return reg

# ------------------------------ Projections ----------------------------------

def project_points(pts3d: np.ndarray, T_cam_obj: np.ndarray, K: np.ndarray, dist: Optional[np.ndarray]) -> np.ndarray:
    R = T_cam_obj[:3,:3]; t = T_cam_obj[:3,3].reshape(3,1)
    rvec,_ = cv2.Rodrigues(R)
    uv,_ = cv2.projectPoints(pts3d.astype(np.float32), rvec, t, K, dist if dist is not None else np.zeros((1,5)))
    return uv.reshape(-1,2)

def draw_axes(img: np.ndarray, T: np.ndarray, K: np.ndarray, dist: Optional[np.ndarray], scale: float = 0.05):
    origin = np.array([[0,0,0]], np.float32)
    axes = np.array([[scale,0,0],[0,scale,0],[0,0,scale]], np.float32)
    pts = np.vstack([origin, axes])
    uv = project_points(pts, T, K, dist)
    o = tuple(np.int32(uv[0]))
    x = tuple(np.int32(uv[1])); y = tuple(np.int32(uv[2])); z = tuple(np.int32(uv[3]))
    cv2.line(img, o, x, (0,0,255), 2); cv2.line(img, o, y, (0,255,0), 2); cv2.line(img, o, z, (255,0,0), 2)

# ---------------------------- Pose estimation per frame -----------------------

def estimate_poses_multi(
    bgr: np.ndarray,
    intr: Intrinsics,
    family: str,
    faces: Dict[str, FaceEntry],
    pts3d_by_face: Dict[str, np.ndarray],
    check_tag_scale: bool,
    auto_correct_scale: bool,
    scale_tol: float,
    axes_mode: str = "both",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, dict]]:
    """
    Returns:
      anno_vis (left panel), reproj_vis (mid panel), and dict results[face_key] with:
        object, face_key, tags_used, T_cam_board, T_cam_object (4x4),
        bbox_xywh, bbox_source, diagnostics (tag_scale_*).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)
    K = np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1]], float)
    dist = intr.dist if intr.dist is not None else np.zeros((1,5), float)

    # Detect tags once per unique tag size (mixed boards supported)
    tag_sizes = sorted({f.tag_size_m for f in faces.values()})
    det_by_id: Dict[int, object] = {}
    for tag_size in tag_sizes:
        dets = detect_tags(gray, intr.fx, intr.fy, intr.cx, intr.cy, tag_size, family=family)
        for d in dets:
            det_by_id[int(d.tag_id)] = d

    anno = bgr.copy()
    reproj = bgr.copy()
    results: Dict[str, dict] = {}

    FALLBACK_SCALE = 3.0
    MIN_AREA = 6000

    for face_key, fe in faces.items():
        common = [tid for tid in fe.T_board_tag.keys() if tid in det_by_id]
        if not common and (fe.origin_id not in det_by_id):
            continue

        # Build T_cam_board from either origin tag or the best common tag
        if fe.origin_id in det_by_id and getattr(det_by_id[fe.origin_id], "pose_R", None) is not None:
            d = det_by_id[fe.origin_id]
            T_cam_tag   = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
            T_cam_board = T_cam_tag
            tag_used = fe.origin_id
        else:
            tag_used = max(common, key=lambda tid: getattr(det_by_id[tid], "decision_margin", 0.0))
            d = det_by_id[tag_used]
            T_cam_tag   = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
            T_cam_board = T_cam_tag @ inv_se3(fe.T_board_tag[tag_used])

        # Optional tag-scale check/correction
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

        # reprojection debug: a few board axes points
        try:
            pts3d = np.array([[0,0,0],[0.03,0,0],[0,0.03,0],[0,0,0.03]], np.float32)
            uv = project_points(pts3d, T_cam_board, K, dist)
            for (u,v) in uv:
                cv2.circle(reproj, (int(u),int(v)), 3, (0,0,255), -1)
        except Exception:
            pass

        # BBOX from face keypoints (if present)
        bbox_xywh = None; bbox_source = None; score_area = 0.0
        if face_key in pts3d_by_face:
            uv = project_points(pts3d_by_face[face_key], T_cam_obj, K, dist)
            for (u, v) in uv:
                cv2.circle(anno, (int(u), int(v)), 2, (255, 255, 0), -1)
            x0, y0 = uv.min(axis=0); x1, y1 = uv.max(axis=0)
            H, W = bgr.shape[:2]
            x0 = max(0.0, min(W - 1.0, float(x0))); y0 = max(0.0, min(H - 1.0, float(y0)))
            x1 = max(0.0, min(W - 1.0, float(x1))); y1 = max(0.0, min(H - 1.0, float(y1)))
            bbox_xywh = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            score_area = bbox_xywh[2] * bbox_xywh[3]
            bbox_source = "face_kps"
            cv2.rectangle(anno, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)

        # Fallback bbox around the used tag if too small or missing
        if (bbox_xywh is None) or (score_area < MIN_AREA):
            s_tag = float(fe.tag_size_m) * FALLBACK_SCALE
            hlf = 0.5 * s_tag
            square3d = np.array([[-hlf, -hlf, 0],[ hlf, -hlf, 0],[ hlf,  hlf, 0],[-hlf,  hlf, 0]], np.float32)
            uv_tag = project_points(square3d, T_cam_tag, K, dist)
            x0, y0 = uv_tag.min(axis=0); x1, y1 = uv_tag.max(axis=0)
            H, W = bgr.shape[:2]
            x0 = max(0.0, min(W - 1.0, float(x0))); y0 = max(0.0, min(H - 1.0, float(y0)))
            x1 = max(0.0, min(W - 1.0, float(x1))); y1 = max(0.0, min(H - 1.0, float(y1)))
            bbox_xywh = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            score_area = bbox_xywh[2] * bbox_xywh[3]
            bbox_source = "fallback_tag"
            cv2.rectangle(anno, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 2)

        # Axes overlay
        if axes_mode in ("both", "board"):  draw_axes(anno, T_cam_board, K, dist, scale=0.06)
        if axes_mode in ("both", "object"): draw_axes(anno, T_cam_obj,  K, dist, scale=0.04)

        # label
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
            "T_cam_object": {"matrix": T_cam_obj.tolist()},
            "diagnostics": {
                "tag_scale_ratio": float(s),
                "tag_scale_pairs": int(n_pairs),
                "tag_scale_mad": float(mad),
                "tag_scale_auto_corrected": bool(auto_correct_scale and abs(s - 1.0) > scale_tol),
            },
        }

    return anno, reproj, results

# ------------------------------- Saver ---------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_frame(
    out_dir: Path,
    idx: int,
    ts: float,
    bgr: np.ndarray,
    depth: Optional[np.ndarray],
    intr: Intrinsics,
    family: str,
    results: Dict[str, dict],
    reproj_img: Optional[np.ndarray],
    save_depth: bool,
    dataset_name: str,
):
    ensure_dir(out_dir / "frames")
    name = f"{idx:06d}_{int(ts*1000):013d}"
    img_path = out_dir / "frames" / f"{name}.png"
    meta_path = out_dir / "frames" / f"{name}.json"
    reproj_path = out_dir / "frames" / f"{name}_reproj.png"
    log_path = out_dir / f"{dataset_name}.jsonl"

    cv2.imwrite(str(img_path), bgr)
    if reproj_img is not None:
        cv2.imwrite(str(reproj_path), reproj_img)
    if save_depth and depth is not None:
        np.save(str(out_dir / "frames" / f"{name}_depth.npy"), depth)

    pretty = {}
    for fk, r in results.items():
        M = np.array(r["T_cam_object"]["matrix"], float)
        R, t = M[:3, :3], M[:3, 3]
        roll, pitch, yaw = rpy_from_R(R)
        q_xyzw = to_quat(R)
        pr = dict(r)
        pr["translation_m"] = [float(t[0]), float(t[1]), float(t[2])]
        pr["rpy_deg"] = [float(roll), float(pitch), float(yaw)]
        pr["quat_xyzw"] = [float(q) for q in q_xyzw]
        pretty[fk] = pr

    meta = {
        "timestamp_sec": float(ts),
        "camera": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "width": intr.width, "height": intr.height,
            "distortion_coefficients": None if intr.dist is None else intr.dist.reshape(-1).tolist(),
        },
        "tag_family": family,
        "faces": pretty,  # keyed by face_key
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    log_entry = {
        "idx": idx,
        "timestamp_sec": float(ts),
        "image": img_path.name,
        "reproj_image": (reproj_path.name if reproj_img is not None else None),
        "depth": (f"{name}_depth.npy" if (save_depth and depth is not None) else None),
        "tag_family": family,
        "faces": sorted(list(pretty.keys())),
        "objects": sorted({r["object"] for r in results.values()}),
    }
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(log_entry) + "\n")

# ------------------------------- Main loop -----------------------------------

def main():
    project_root = resolve_project_root(None)
    ensure_project_dirs(project_root)

    parser = argparse.ArgumentParser("Collect multi-object/multi-face GT dataset with review UI")
    parser.add_argument("--mode", choices=["live","bag","video"], required=True)
    parser.add_argument("--bag", type=str, help="Path to RealSense .bag (for mode=bag)")
    parser.add_argument("--video", type=str, help="Path to a video file (for mode=video)")
    parser.add_argument("--session", required=True, help="Dataset session name (folder under datasets/)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Root for datasets (default: <project_root>/datasets)")
    parser.add_argument("--calib", default="calib_color.yaml", help="Calibration YAML (fx,fy,cx,cy,dist)")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--continuous", action="store_true",
                        help="Do not prompt per frame; save all automatically")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0=unlimited)")
    parser.add_argument("--rs_w", type=int, default=0, help="Color width; 0=auto")
    parser.add_argument("--rs_h", type=int, default=0, help="Color height; 0=auto")
    parser.add_argument("--rs_fps", type=int, default=30, help="FPS for live mode")
    parser.add_argument("--save_depth", action="store_true",
                        help="If present, save aligned depth as .npy per frame")
    parser.add_argument("--axes", choices=["both","board","object","none"], default="both",
                        help="Which axes to draw on the left panel")
    # Tag scale options (parity with annotate_shots.py)
    parser.add_argument("--check-tag-scale", action="store_true",
                        help="Print/record scale ratio s from inter-tag distances")
    parser.add_argument("--auto-correct-scale", action="store_true",
                        help="If |s-1|>tol, divide T_cam_board translation by s")
    parser.add_argument("--scale-tol", type=float, default=0.02,
                        help="Relative tolerance (default 0.02 = 2%)")

    args = parser.parse_args()

    # Logger
    global log
    log = init_project_logger(project_root / "logs" / "collect_gt_dataset.log",
                              level="INFO", console=True)

    # Intrinsics
    calib_path = project_root / args.calib
    intr = load_intrinsics_from_calib(calib_path)
    K = np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1]], float)

    # Face registry (from <project_root>/faces)
    faces = load_face_registry(project_root)
    log.info("[i] Faces loaded: %d", len(faces))

    # Output layout
    datasets_root = Path(args.dataset_root) if args.dataset_root else (project_root / "datasets")
    out_dir = datasets_root / args.session
    ensure_dir(out_dir / "frames")

    # Choose frame source
    if args.mode == "live":
        w = args.rs_w or intr.width
        h = args.rs_h or intr.height
        src = RealSenseLive(width=w, height=h, fps=args.rs_fps)
    elif args.mode == "bag":
        if not args.bag: sys.exit("--bag is required for mode=bag")
        src = RealSenseBag(Path(args.bag))
    else:
        if not args.video: sys.exit("--video is required for mode=video")
        src = VideoSource(Path(args.video))

    # Start source
    src.start()
    cam_kind = src.camera_kind()

    cv2.namedWindow("GT Capture", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("GT Capture", 1920, 480)

    # Preload face keypoints for bbox projection
    pts3d_by_face: Dict[str, np.ndarray] = {}
    for fk, fe in faces.items():
        try:
            pts3d_dict, faces_map, _, _ = load_keypoints_fuzzy(fe.object_name, project_root)
            if fk in faces_map:
                names = faces_map[fk]
                pts3d_by_face[fk] = np.vstack([pts3d_dict[n] for n in names]).astype(np.float32)
        except Exception as e:
            log.warning("[warn] keypoints for %s not fully available: %s", fk, e)

    # Session metadata scaffold
    session_meta = {
        "version": DATASET_VERSION,
        "name": args.session,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "paths": {"root": str(datasets_root), "session_dir": str(out_dir)},
        "camera": {
            "kind": cam_kind,  # "intel_realsense" or "generic"
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "width": intr.width, "height": intr.height,
            "distortion_coefficients": None if intr.dist is None else intr.dist.reshape(-1).tolist(),
        },
        "tag_family": args.family,
        "tag_scale_controls": {
            "check_tag_scale": bool(args.check_tag_scale),
            "auto_correct_scale": bool(args.auto_correct_scale),
            "scale_tol": float(args.scale_tol),
        },
        "faces_index": {fk: {"object": fe.object_name, "board_yaml": str(fe.board_yaml)} for fk, fe in faces.items()},
        "objects_available": sorted({fe.object_name for fe in faces.values()}),
        "objects_seen_counts": {},  # filled as we save frames
        "frames_captured": 0,
    }
    (out_dir / "session.yaml").write_text(yaml.safe_dump(session_meta, sort_keys=False))

    # Main loop
    idx = 0
    paused = False
    show_help = True
    last_anno = None; last_reproj = None
    last_bgr = None; last_depth = None; last_ts = None
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
                    cv2.destroyAllWindows(); src.stop()
                    sys.exit(f"[!] Stream size {bgr.shape[1]}x{bgr.shape[0]} "
                             f"!= calibration {intr.width}x{intr.height}. "
                             f"Start with --rs_w {intr.width} --rs_h {intr.height} or fix the calibration.")

                # estimate poses
                anno, reproj, results = estimate_poses_multi(
                    bgr, intr, args.family, faces, pts3d_by_face,
                    check_tag_scale=args.check_tag_scale,
                    auto_correct_scale=args.auto_correct_scale,
                    scale_tol=args.scale_tol,
                    axes_mode=args.axes,
                )
                last_anno, last_reproj = anno, reproj
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

            # pick best per object (highest score)
            best_by_object: Dict[str, dict] = {}
            for fk, r in results.items():
                obj = r["object"]
                if (obj not in best_by_object) or (r["score"] > best_by_object[obj]["score"]):
                    best_by_object[obj] = r

            # right panel text
            lines = [
                "GT Capture",
                f"session: {args.session}",
                f"faces visible: {len(results)}   (saved: {idx})",
                "",
                "Panels:",
                "  Left  = Annotation (axes, bbox, labels)",
                "  Middle= Reprojection / tag debug",
                "  Right = Status, legend, capture feedback",
                "",
                "Keys: ENTER/y/s=accept, r/n/BACK=reject",
                "      ESC/q=abort  Q/X=quit-all",
                "      SPACE/p=cont/pause  h=help",
                "",
            ]
            if show_help:
                for obj, r in best_by_object.items():
                    M = np.array(r["T_cam_object"]["matrix"], float)
                    R, t = M[:3, :3], M[:3, 3]
                    roll, pitch, yaw = rpy_from_R(R)
                    lines.append(f"- {obj} [{r['face_key']}]")
                    lines.append(f"   t (m): [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]  | dist={np.linalg.norm(t):.3f} m")
                    lines.append(f"   rpy deg : [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]")
                    lines.append(f"   tag_used: {r['tag_used']}  bbox: {r['bbox_source']}")
                    diag = r.get("diagnostics", {})
                    lines.append(f"   tag_scale s={diag.get('tag_scale_ratio',1.0):.3f} "
                                 f"pairs={diag.get('tag_scale_pairs',0)} "
                                 f"{'AUTO' if diag.get('tag_scale_auto_corrected', False) else ''}")
                    lines.append("")

            now = time.time()
            if capture_msg and now < capture_msg_until:
                lines.append(f"*** {capture_msg} ***")

            right = _text_panel(lines, width=640, height=480)
            _show_dash(last_anno, last_reproj, right)

            # decide
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('Q'), ord('X'), ord('x')): _quit_all()
            if k == ord('h'): show_help = not show_help
            if k in (ord('p'), 32): paused = not paused  # pause/resume
            if k in (27, ord('q')):  # abort current frame (skip)
                continue

            def _capture_now():
                nonlocal idx, capture_msg, capture_msg_until, objects_seen_counts
                if last_bgr is None: return
                save_frame(out_dir, idx, last_ts, last_bgr, last_depth, intr, args.family,
                           results=best_by_object, reproj_img=last_reproj,
                           save_depth=args.save_depth, dataset_name=args.session)
                # update counters
                for r in best_by_object.values():
                    objects_seen_counts[r["object"]] = objects_seen_counts.get(r["object"], 0) + 1
                capture_msg = f"Frame #{idx:03d} captured [OK]"
                capture_msg_until = time.time() + 2.0
                idx += 1

            if k in (13, ord('y'), ord('s')):  # accept
                _capture_now()
            if args.continuous and not paused:
                _capture_now()

            if args.max_frames and idx >= args.max_frames:
                log.info("[i] Reached max_frames=%d.", args.max_frames)
                break

    except SystemExit as e:
        if e.code != EXIT_QUIT_ALL:
            log.error("Exited: %s", e)
    except BaseException as e:
        log.error("[!] Error: %s", e, exc_info=True)
    finally:
        # finalize session metadata
        session_meta["frames_captured"] = int(idx)
        session_meta["objects_seen_counts"] = {k: int(v) for k, v in sorted(objects_seen_counts.items())}
        (out_dir / "session.yaml").write_text(yaml.safe_dump(session_meta, sort_keys=False))

        try: src.stop()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
