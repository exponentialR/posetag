"""
Collect a ground-truth dataset (multi-object, multi-face) from Intel RealSense
or a recorded video. For each frame, estimate pose for every face whose board
has at least one visible AprilTag. Frame-by-frame review UI is provided.

Usage examples:
  # RealSense live (RGB + depth if available)
  python -m src.collect_gt_dataset --mode live --session run01 --calib calib_color.yaml

  # RealSense playback from a .bag
  python -m src.collect_gt_dataset --mode bag --bag path/to/recording.bag --session run02 --calib calib_color.yaml

  # Any video readable by OpenCV
  python -m src.collect_gt_dataset --mode video --video sample.mp4 --session run03 --calib calib_color.yaml

Keys:
  ENTER / y / s  -> accept & save frame
  r / n / BACKSPACE -> reject/skip frame
  ESC / q       -> abort current frame (continue stream)
  Q / X         -> quit-all immediately (stops stream + closes)
  SPACE         -> toggle continuous mode (no per-frame prompts)
  p             -> pause/resume stream
  h             -> toggle help panel
"""

from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import yaml
import math
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import pyrealsense2 as rs
except Exception:
    rs = None  # lint-friendly

# --- repo utils ---
from utils.annotation_utils import (
    detect_tags, load_board, se3, inv_se3
)

try:
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
except Exception:
    pass


from utils.annotation_utils import load_keypoints_fuzzy
from utils.gt_pose_utils import rotation_to_quat

try:
    from scipy.spatial.transform import Rotation as Rot
    def to_quat(R): return Rot.from_matrix(R).as_quat()  # [x,y,z,w]
except Exception:
    def to_quat(R): return rotation_to_quat(R)



EXIT_QUIT_ALL = 99

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

def face_quality(num_tags: int, bbox_xywh: list[float] | None) -> float:
    # prefer more tags; tie-break by bbox area (bigger is better)
    area = 0.0 if not bbox_xywh else float(bbox_xywh[2] * bbox_xywh[3])
    return (num_tags * 1_000.0) + area

def _pad_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    out = np.zeros((h, w, 3), np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out

def _text_panel(lines: List[str], width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    # header bar
    img = _label_strip(img, "Status / Legend")
    y = 40
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 1, cv2.LINE_AA)
        y += 22
        if y > height - 8: break
    return img

def _resize_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    H, W = img.shape[:2]
    interp = cv2.INTER_AREA if (H > h or W > w) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)

def _label_strip(img: np.ndarray, text: str,
                 bar_h: int = 26,
                 bg=(0, 0, 0),
                 fg=(0, 255, 255)) -> np.ndarray:
    """Add a top title bar with text to an image."""
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), bg, -1)
    cv2.putText(vis, text, (8, int(bar_h*0.75)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)
    return vis

def _hstack(left: np.ndarray, mid: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    H = max(left.shape[0], 0 if mid is None else mid.shape[0], 0 if right is None else right.shape[0])
    Wl = left.shape[1]
    Wm = mid.shape[1] if mid is not None else Wl
    Wr = right.shape[1] if right is not None else 380
    return np.hstack([_pad_to(left, H, Wl), _pad_to(mid, H, Wm), _pad_to(right, H, Wr)])

def _show_dash(left: np.ndarray | None, mid: np.ndarray | None, right: np.ndarray | None,
               win: str = "Dash", tile_h: int = 480, tile_w: int = 640):
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
    """Abstract iterator-style frame source."""
    def start(self): pass
    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:  # rgb, depth, ts_sec
        raise NotImplementedError
    def stop(self): pass

class VideoSource(FrameSource):
    def __init__(self, path: Path, fps_hint: float = 30.0):
        self.cap = cv2.VideoCapture(str(path))
        self.fps_delay = 1.0 / max(1e-6, float(self.cap.get(cv2.CAP_PROP_FPS)) or fps_hint)

    def read(self):
        ok, bgr = self.cap.read()
        if not ok: return None, None, time.time()
        time.sleep(self.fps_delay)  # gentle throttle to avoid UI storm
        return bgr, None, time.time()

    def stop(self):
        self.cap.release()

class RealSenseLive(FrameSource):
    def __init__(self, width=0, height=0, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 not available.")
        self.req_w, self.req_h, self.req_fps = int(width), int(height), int(fps)
        self.w = self.h = self.fps = 0
        self.pipe = None
        self.align = None
        self.depth_enabled = False
        self.profile = None

    def _try_start(self, w, h, fps, with_depth=True):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        depth_ok = False
        if with_depth:
            try:
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                depth_ok = True
            except Exception:
                try:
                    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
                    depth_ok = True
                except Exception:
                    depth_ok = False
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        return pipe, profile, depth_ok

    def start(self):
        ctx = rs.context()
        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device found.")

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
                if self.depth_enabled:
                    self.align = rs.align(rs.stream.color)
                    print(f"[RealSense] Started color+depth at {w}x{h}@{fps}")
                else:
                    self.align = None
                    print(f"[RealSense] Started COLOR-ONLY at {w}x{h}@{fps}")
                # warmup 10 frames
                for _ in range(10):
                    _ = self.pipe.wait_for_frames()
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
        if not c:
            return None, None, time.time()
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

class RealSenseBag(FrameSource):
    def __init__(self, bag_path: Path):
        if rs is None: raise RuntimeError("pyrealsense2 not available.")
        self.bag_path = bag_path
        self.pipe, self.align, self.profile = None, None, None

    def start(self):
        cfg = rs.config(); cfg.enable_device_from_file(str(self.bag_path), repeat_playback=False)
        self.pipe = rs.pipeline(); self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)

    def read(self):
        try:
            frames = self.pipe.wait_for_frames()
        except Exception:
            return None, None, time.time()
        frames = self.align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame() if frames.get_depth_frame() else None
        if not c: return None, None, time.time()
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).copy() if d else None
        ts = c.get_timestamp() / 1000.0
        return bgr, depth, ts

    def stop(self):
        if self.pipe: self.pipe.stop()

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

def load_face_registry(repo_root: Path) -> Dict[str, FaceEntry]:
    """
    Scan objects/faces/*/*_T_board_object.yaml and build a registry:
      face_key -> FaceEntry
    """
    reg: Dict[str, FaceEntry] = {}
    faces_root = repo_root / "objects" / "faces"
    for y in faces_root.glob("*/*_T_board_object.yaml"):
        data = yaml.safe_load(open(y, "r"))
        obj = data["object"]
        face_key = data["face_key"]
        board_yaml = Path(data["board_yaml"])
        if not board_yaml.is_absolute():
            board_yaml = (repo_root / board_yaml).resolve()
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
        raise SystemExit("[!] No faces registered. Run annotate_face_shot first.")
    return reg

# ------------------------------ PnP & overlay --------------------------------

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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, dict]]:
    """
    Returns:
      anno_vis (left panel), reproj_vis (mid panel), and a dict results keyed by face_key.
      Each result has: object, face_key, tags_used, T_cam_board, T_cam_object (4x4 lists).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)
    K = np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1]], float)
    dist = intr.dist if intr.dist is not None else np.zeros((1,5), float)

    # Detect tags once per unique tag size (so mixed boards are supported)
    tag_sizes = sorted({f.tag_size_m for f in faces.values()})
    det_by_id: Dict[int, object] = {}
    for tag_size in tag_sizes:
        dets = detect_tags(gray, intr.fx, intr.fy, intr.cx, intr.cy, tag_size, family=family)
        for d in dets:
            det_by_id[int(d.tag_id)] = d  # latest wins (they all share same 2D corners)

    anno = bgr.copy()
    reproj = bgr.copy()
    results: Dict[str, dict] = {}

    # For each face board that has any visible tag, recover T_cam_board then T_cam_object
    for face_key, fe in faces.items():
        # visible?
        common = [tid for tid in fe.T_board_tag.keys() if tid in det_by_id]
        if not common and (fe.origin_id not in det_by_id):
            continue
        if fe.origin_id in det_by_id and det_by_id[fe.origin_id].pose_R is not None:
            d = det_by_id[fe.origin_id]
            T_cam_board = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
            tag_used = fe.origin_id
        else:
            tag_used = max(common, key=lambda tid: getattr(det_by_id[tid], "decision_margin", 0.0))
            d = det_by_id[tag_used]
            T_cam_tag = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
            T_cam_board = T_cam_tag @ inv_se3(fe.T_board_tag[tag_used])
        T_cam_obj = T_cam_board @ fe.T_board_object
        try:
            pts3d = np.array([[0,0,0],[0.03,0,0],[0,0.03,0],[0,0,0.03]], np.float32)
            uv = project_points(pts3d, T_cam_board, K, dist)
            for (u,v) in uv:
                cv2.circle(reproj, (int(u),int(v)), 3, (0,0,255), -1)
        except Exception:
            pass

        bbox_xywh = None
        if face_key in pts3d_by_face:
            uv = project_points(pts3d_by_face[face_key], T_cam_obj, K, dist)  # (4,2)
            x0, y0 = uv.min(axis=0);
            x1, y1 = uv.max(axis=0)
            H, W = bgr.shape[:2]
            x0 = max(0.0, min(W - 1.0, float(x0)))
            y0 = max(0.0, min(H - 1.0, float(y0)))
            x1 = max(0.0, min(W - 1.0, float(x1)))
            y1 = max(0.0, min(H - 1.0, float(y1)))
            bbox_xywh = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            # quick overlay
            cv2.rectangle(anno, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)

        # overlay
        label = f"{fe.object_name}/{fe.face_key}"
        draw_axes(anno, T_cam_board, K, dist, scale=0.06)
        draw_axes(anno, T_cam_obj, K, dist, scale=0.04)
        cv2.putText(anno, label, (12, 24 + 18 * (hash(face_key) % 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),
                    2)

        num_tags = len(common) if common else (1 if fe.origin_id in det_by_id else 0)
        score_area = 0.0 if not bbox_xywh else float(bbox_xywh[2] * bbox_xywh[3])
        score = (num_tags * 1_000.0) + score_area

        # pack results (store 4x4 lists for JSON cleanliness)
        results[face_key] = {
            "object": fe.object_name,
            "face_key": fe.face_key,
            "board_yaml": str(fe.board_yaml),
            "tag_used": int(tag_used),
            "num_tags_visible": int(num_tags),
            "score": float(score),
            "score_tags": int(num_tags),
            "score_area_px": int(score_area),
            "T_cam_board": {"matrix": T_cam_board.tolist()},
            "T_cam_object": {"matrix": T_cam_obj.tolist()},
            "bbox_xywh": bbox_xywh,
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

    pretty = {}
    for obj, r in results.items():
        M = np.array(r["T_cam_object"]["matrix"], float)
        R, t = M[:3, :3], M[:3, 3]
        roll, pitch, yaw = rpy_from_R(R)
        pr = dict(r)
        pr["translation_m"] = [float(t[0]), float(t[1]), float(t[2])]
        pr["rpy_deg"] = [float(roll), float(pitch), float(yaw)]
        pretty[obj] = pr
        q_xyzw = to_quat(R)
        pr["quat_xyzw"] = [float(q) for q in q_xyzw]

    if reproj_img is not None:
        cv2.imwrite(str(reproj_path), reproj_img)

    if save_depth and depth is not None:
        np.save(str(out_dir / "frames" / f"{name}_depth.npy"), depth)

    meta = {
        "timestamp_sec": ts,
        "camera": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "width": intr.width, "height": intr.height,
            "distortion_coefficients": None if intr.dist is None else intr.dist.reshape(-1).tolist(),
        },
        "tag_family": family,
        "poses": pretty,
    }
    log_entry = {
        "idx": idx,
        "timestamp_sec": float(ts),
        "image": img_path.name,
        "reproj_image": (reproj_path.name if reproj_img is not None else None),
        "depth": (f"{name}_depth.npy" if (save_depth and depth is not None) else None),
        "tag_family": family,
        "objects": sorted(list(pretty.keys())),   # objects saved in this frame
    }
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(log_entry) + "\n")

    meta_path.write_text(json.dumps(meta, indent=2))

# ------------------------------- Main loop -----------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser("Collect multi-object/multi-face GT dataset with review UI")
    ap.add_argument("--mode", choices=["live","bag","video"], required=True)
    ap.add_argument("--bag", type=str, help="Path to RealSense .bag (for mode=bag)")
    ap.add_argument("--video", type=str, help="Path to a video file (for mode=video)")
    ap.add_argument("--session", required=True, help="Dataset session name (folder under datasets/)")
    ap.add_argument("--dataset_root", type=str, default=str(repo_root / "datasets"))
    ap.add_argument("--calib", default="calib_color.yaml", help="ChArUco/WY calibration (fx,fy,cx,cy,dist)")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--continuous", action="store_true", help="Do not prompt per frame; save all automatically")
    ap.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0=unlimited)")
    ap.add_argument("--rs_w", type=int, default=0, help="Color width; 0=auto")
    ap.add_argument("--rs_h", type=int, default=0, help="Color height; 0=auto")
    ap.add_argument("--rs_fps", type=int, default=30, help="FPS for live mode")
    ap.add_argument("--save_depth", action="store_true",
                    help="If present, save aligned depth as .npy per frame")

    args = ap.parse_args()

    # Intrinsics
    calib_path = Path(args.calib)
    intr = load_intrinsics_from_calib(calib_path)
    K = np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1]], float)

    # Face registry
    faces = load_face_registry(repo_root)

    # Output layout
    out_dir = Path(args.dataset_root) / args.session
    ensure_dir(out_dir / "frames")
    (out_dir / "session.yaml").write_text(yaml.safe_dump({
        "session": args.session,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "width": intr.width, "height": intr.height},
        "tag_family": args.family,
        "faces_index": {fk: {"object": fe.object_name, "board_yaml": str(fe.board_yaml)} for fk,fe in faces.items()},
    }, sort_keys=False))

    # Frame source
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

    src.start()
    cv2.namedWindow("Dash", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Dash", 1920, 480)

    idx = 0
    paused = False
    show_help = True
    last_anno = None
    last_reproj = None
    last_bgr = None
    last_depth = None
    last_ts = None
    capture_msg, capture_msg_until = "", 0.0
    pts3d_by_face: Dict[str, np.ndarray] = {}

    try:
        for fk, fe in faces.items():
            pts3d_dict, faces_map, _, _ = load_keypoints_fuzzy(fe.object_name, repo_root)
            if fk in faces_map:
                names = faces_map[fk]
                pts3d_by_face[fk] = np.vstack([pts3d_dict[n] for n in names]).astype(np.float32)
        last_results = None
        while True:
            results: Dict[str, dict] = {}
            if not paused:
                bgr, depth, ts = src.read()
                if bgr is None:
                    print("[i] End of stream.")
                    break
                if (bgr.shape[1], bgr.shape[0]) != (intr.width, intr.height):
                    cv2.destroyAllWindows()
                    src.stop()
                    sys.exit(f"[!] Stream size {bgr.shape[1]}x{bgr.shape[0]} "
                             f"does not match calibration {intr.width}x{intr.height}. "
                             f"Start with --rs_w {intr.width} --rs_h {intr.height} or fix the calibration.")

                # estimate poses
                anno, reproj, results = estimate_poses_multi(bgr, intr, args.family, faces, pts3d_by_face)
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
            best_by_object= {}
            for fk, r in results.items():
                obj = r["object"]
                if (obj not in best_by_object) or (r["score"] > best_by_object[obj]["score"]):
                    best_by_object[obj] = r

            lines = [
                "GT Capture",
                f"session: {args.session}",
                f"faces visible: {len(results)}",
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
                    lines.append(f"   tag_used: {r['tag_used']}  face: {r['face_key']}")
                    lines.append(
                        f"   score: {int(r['score'])}  (tags={r['score_tags']}, area={r['score_area_px']:,} px)")
                    lines.append("")

            now = time.time()
            if capture_msg and now < capture_msg_until:
                lines.append(f"*** {capture_msg} ***")

            right = _text_panel(lines, width=640, height=480)
            _show_dash(last_anno, last_reproj, right)

            # decide
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('Q'), ord('X'), ord('x')): _quit_all()
            if k in (27, ord('q')):  # skip this instant, keep streaming
                continue
            if k == ord('h'): show_help = not show_help
            if k in (ord('p'), 32): paused = not paused  # pause/resume


            if k in (13, ord('y'), ord('s')) and (last_bgr is not None):
                saved_idx = idx
                save_frame(out_dir, saved_idx, last_ts, last_bgr, last_depth, intr, args.family,
                           results=best_by_object,  # <--- note: best only
                           reproj_img=last_reproj,
                           save_depth=args.save_depth, dataset_name=args.session)
                capture_msg = f"Frame #{saved_idx:03d} captured [OK]"
                capture_msg_until = time.time() + 2.0
                idx += 1

            if args.continuous and not paused and (last_bgr is not None):
                saved_idx = idx
                save_frame(out_dir, idx, last_ts, last_bgr, last_depth, intr, args.family,
                           results=best_by_object,  # <-- best only, same as ENTER
                           reproj_img=last_reproj,
                           save_depth=args.save_depth, dataset_name=args.session)
                capture_msg = f"Frame #{saved_idx:03d} captured [OK]"
                capture_msg_until = time.time() + 2.0
                idx += 1

            if args.max_frames and idx >= args.max_frames:
                print("[i] Reached max_frames.")
                break
    except Exception:
        pass
    finally:
        src.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
