import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import cv2, yaml
import re, json, glob
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_DET_CACHE = {}


def _get_detector(family: str, nthreads: int = 1,
                  quad_decimate: float = 1.0, refine_edges: bool = True):
    key = (family, nthreads, quad_decimate, refine_edges)
    det = _DET_CACHE.get(key)
    if det is None:
        try:
            from pupil_apriltags import Detector
        except Exception as exc:
            raise SystemExit(
                "pupil-apriltags is not available; install PoseTag with the "
                "'apriltags' extra or run `python -m pip install pupil-apriltags`."
            ) from exc
        det = Detector(families=family,
                       nthreads=nthreads,
                       quad_decimate=quad_decimate,
                       refine_edges=refine_edges)
        _DET_CACHE[key] = det
    return det

def detect_tags(image_gray, fx, fy, cx, cy, tag_size_m, family="tag36h11"):
    # Ensure C-contiguous uint8 for the C++ side
    img = np.ascontiguousarray(image_gray, dtype=np.uint8)
    det = _get_detector(family, nthreads=1, quad_decimate=1.0, refine_edges=True)
    return det.detect(img,
                      estimate_tag_pose=True,
                      camera_params=(fx, fy, cx, cy),
                      tag_size=tag_size_m)

def _draw_prompt(img, face_key, clicked_pts, rms=None):
    vis = img.copy()
    cv2.putText(vis, f"Face: {face_key}  (click any order: 4 corners)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if rms is not None:
        cv2.putText(vis, f"fit RMS: {rms:.2f}px",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
    for i,(u,v) in enumerate(clicked_pts):
        cv2.circle(vis, (int(u),int(v)), 4, (0,255,0), -1)
        cv2.putText(vis, str(i+1), (int(u)+6, int(v)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(vis, "ENTER=save  u=undo  q=quit",
                (12, img.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return vis


def _shot_side(object_name: str) -> str | None:
    m = re.search(r'_side([A-D])$', object_name)
    return m.group(1) if m else None

def load_keypoints_fuzzy(object_name: str, repo_root: Path):
    """
    Try to locate keypoints.json for this object, allowing variant names.
    Strategy:
      1) objects/<object_name>/keypoints.json
      2) strip trailing _sideX -> objects/<base>/keypoints.json
      3) scan objects/*/keypoints.json and pick the one that has a face
         whose key endswith _sideX
    Returns: (pts3d_dict, faces_map, chosen_face_key, kp_path)
    Raises FileNotFoundError if nothing suitable is found.
    """
    obj_dir = repo_root / "objects"
    side = _shot_side(object_name)

    # 1) exact
    cand = obj_dir / object_name / "keypoints.json"
    if cand.exists():
        kp = json.loads(cand.read_text())
        pts3d = {k: np.array(v, float) for k, v in kp["points"].items()}
        faces = kp.get("faces", {})
        # prefer exact face key; else fall back by side
        face_key = object_name if object_name in faces else None
        if face_key is None and side:
            for k in faces.keys():
                if k.endswith(f"_side{side}"):
                    face_key = k; break
        return pts3d, faces, face_key, str(cand)

    # 2) strip _sideX and retry
    base = re.sub(r'_side[A-D]$', '', object_name)
    cand = obj_dir / base / "keypoints.json"
    if cand.exists():
        kp = json.loads(cand.read_text())
        pts3d = {k: np.array(v, float) for k, v in kp["points"].items()}
        faces = kp.get("faces", {})
        face_key = object_name if object_name in faces else None
        if face_key is None and side:
            for k in faces.keys():
                if k.endswith(f"_side{side}"):
                    face_key = k; break
        return pts3d, faces, face_key, str(cand)

    # 3) scan all and choose a set that contains the desired side
    if side:
        for kp_path in obj_dir.glob("*/keypoints.json"):
            kp = json.loads(kp_path.read_text())
            faces = kp.get("faces", {})
            for k in faces.keys():
                if k.endswith(f"_side{side}"):
                    pts3d = {n: np.array(v, float) for n, v in kp["points"].items()}
                    return pts3d, faces, k, str(kp_path)

    raise FileNotFoundError(f"No keypoints found for '{object_name}' (side={side}).")


def se3(R, t):
    T = np.eye(4, dtype=float); T[:3,:3] = R; T[:3,3] = t.reshape(3); return T

def inv_se3(T):
    R, t = T[:3,:3], T[:3,3]; Ti = np.eye(4, dtype=float)
    Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t; return Ti

def rz(yaw_rad):
    c,s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    return R

def load_meta(path: str | Path, strict: bool = True) -> dict:
    """
    Load metadata JSON saved alongside a shot.

    If strict=True (default), enforce legacy required keys.
    If strict=False, accept newer schemas (object_full/object_base) and
    normalize to include:
        - object_base (always)
        - object      (back-compat alias of object_base)
        - side        (if derivable)
    """
    p = Path(path)
    with p.open("r") as f:
        m = json.load(f)

    if strict:
        required = ["object", "camera", "timestamp"]
        for k in required:
            if k not in m:
                raise ValueError(f"meta missing '{k}'")
        return m

    # Prefer object_base; fall back to legacy object or object_full parsing
    obj_base = m.get("object_base")
    if not obj_base:
        obj_base = m.get("object")
    if not obj_base and m.get("object_full"):
        from utils.capture_utils import parse_base_and_side
        obj_base, side_infer = parse_base_and_side(m["object_full"])
        m.setdefault("side", side_infer)

    if obj_base:
        m.setdefault("object_base", obj_base)
        m.setdefault("object", obj_base)  # back-compat alias for older callers

    return m

def load_keypoints(obj_name: str, repo_root: Path):
    kp_path = repo_root / "objects" / obj_name / "keypoints.json"
    with kp_path.open("r") as f: kp = json.load(f)
    pts = {k: np.asarray(v, float).reshape(3) for k,v in kp["points"].items()}
    faces = kp.get("faces", {})
    return pts, faces

def load_board(board_yaml_path: Path):
    path = Path(board_yaml_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            y = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Board YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed board YAML {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read board YAML {path}: {exc}") from exc

    if not isinstance(y, dict):
        raise ValueError(f"Malformed board YAML {path}: expected a mapping.")
    if "origin_id" not in y:
        raise ValueError(f"Malformed board YAML {path}: missing origin_id.")
    try:
        origin = int(y["origin_id"])
    except Exception as exc:
        raise ValueError(f"Malformed board YAML {path}: origin_id must be an integer.") from exc
    try:
        tag_size_m = float(y.get("tag_size_m", 0.08))
    except Exception as exc:
        raise ValueError(f"Malformed board YAML {path}: tag_size_m must be numeric.") from exc
    if not np.isfinite(tag_size_m) or tag_size_m <= 0:
        raise ValueError(f"Malformed board YAML {path}: tag_size_m must be positive.")

    tags = y.get("tags", [])
    if not isinstance(tags, list) or not tags:
        raise ValueError(f"Malformed board YAML {path}: tags must be a non-empty list.")

    # Build T_board_tag for each tag id
    tb = {}
    for index, e in enumerate(tags):
        if not isinstance(e, dict):
            raise ValueError(f"Malformed board YAML {path}: tags[{index}] must be a mapping.")
        try:
            tid = int(e["id"])
            cx = float(e["cx"])
            cy = float(e["cy"])
            yaw_deg = float(e.get("yaw_deg", 0.0))
        except KeyError as exc:
            raise ValueError(f"Malformed board YAML {path}: tags[{index}] missing {exc.args[0]}.") from exc
        except Exception as exc:
            raise ValueError(f"Malformed board YAML {path}: tags[{index}] has non-numeric values.") from exc
        if not all(np.isfinite(v) for v in (cx, cy, yaw_deg)):
            raise ValueError(f"Malformed board YAML {path}: tags[{index}] values must be finite.")
        Y = math.radians(yaw_deg)
        T = np.eye(4, dtype=float)
        T[:3,:3] = rz(Y)
        T[:3,3] = np.array([cx, cy, 0.0], float)
        tb[tid] = T
    if origin not in tb:
        raise ValueError(f"Malformed board YAML {path}: origin_id {origin} is not listed in tags.")
    # Origin must be identity
    tb[origin] = np.eye(4, dtype=float)
    return origin, tag_size_m, tb



def choose_face_key(face_yaml_path: str) -> str:
    # face key = YAML basename without extension
    base = os.path.splitext(os.path.basename(face_yaml_path))[0]
    return base
