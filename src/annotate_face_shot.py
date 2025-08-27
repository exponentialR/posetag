"""
annotate_face_shot.py
Author: Samuel Adebayo

Click 4 canonical corners on a saved face shot to compute:
    T_board_object  (pose of object in the board frame for THIS FACE)

Usage
-----
  python -m src.annotate_face_shot \
    --project_root /path/to/project \
    --shot shots/<object_base>/side<Side>/<object_base>_side<Side>_<YYYYMMDD_HHMMSS>_raw.png

Inputs
------
- Assumes a sibling JSON meta file (..._meta.json) saved by capture_isc_face.py.
- Uses camera intrinsics from --calib (defaults to <project_root>/calib_color.yaml).
- Uses AprilTag detections from pupil-apriltags for board pose.

Outputs
-------
- YAML: <project_root>/faces/<object_base>/side<Side>/<face_key>_T_board_object.yaml
    • Contains T_board_object matrix, reprojection RMS, PnP parameters,
      UV corner assignments, and metadata.
- Audit images:
    • <project_root>/faces/<object_base>/side<Side>/_audit/<face_key>_<ts>_anno.png
    • <project_root>/faces/<object_base>/side<Side>/_audit/<face_key>_<ts>_reproj.png
- Log file: <project_root>/logs/annotate_face_shot.log

Requirements
------------
- pupil-apriltags
- OpenCV
- PyYAML
- numpy

Keys (during annotation)
------------------------
- Drag box then adjust corners (quad mode) or click any 4 corners (any mode).
- ENTER / y / s : accept
- r / n / BACKSPACE : redo
- u : undo last drag (quad mode only)
- q / ESC : abort this shot
- Q / X   : quit all
- SHIFT   : axis lock while dragging/clicking
"""


from __future__ import annotations
import argparse, os, sys, re, glob
import numpy as np
import cv2, yaml
from pathlib import Path
import itertools
from datetime import datetime
from utils.annotation_utils import load_meta, detect_tags, load_board, se3, inv_se3, load_keypoints_fuzzy, \
    choose_face_key
from utils.capture_utils import parse_base_and_side
from utils.project_config import resolve_project_root, ensure_project_dirs
from utils.logger import init_project_logger

EXIT_QUIT_ALL = 99


def _flush_keys(duration_ms: int = 100):
    """Drain any buffered key events (e.g., leftover ENTER) for ~duration_ms."""
    import time
    end = time.time() + (duration_ms / 1000.0)
    while time.time() < end:
        cv2.waitKey(5)


def _quit_all():
    try:
        cv2.destroyAllWindows()
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except:
            pass
    finally:
        os._exit(EXIT_QUIT_ALL)


def _pad_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    out = np.zeros((h, w, 3), np.uint8)
    hh, ww = img.shape[:2]
    out[:hh, :ww] = img
    return out


def _hstack(left: np.ndarray, middle: np.ndarray | None = None, right: np.ndarray | None = None) -> np.ndarray:
    """Always render 3 panels; blank placeholders keep layout stable."""
    H = max(left.shape[0],
            0 if middle is None else middle.shape[0],
            0 if right is None else right.shape[0])
    left_w = left.shape[1]
    mid_w = (middle.shape[1] if middle is not None else left_w)
    right_w = (right.shape[1] if right is not None else 360)
    L = _pad_to(left, H, left_w)
    M = _pad_to(middle, H, mid_w)
    R = _pad_to(right, H, right_w)
    return np.hstack([L, M, R])


def _text_panel(lines: list[str], width: int = 360, height: int = max(360, 1)) -> np.ndarray:
    # Simple right-hand debug text panel
    img = np.zeros((height, width, 3), np.uint8)
    y = 24
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
        if y > height - 8: break
    return img


def _show_dash(left: np.ndarray, mid: np.ndarray | None, right: np.ndarray | None, banner: str = ""):
    strip = _hstack(left, mid, right)
    if banner:
        cv2.rectangle(strip, (0, 0), (strip.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(strip, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Dash", strip)


def review_decide(anno_img: np.ndarray, reproj_img: np.ndarray | None, dbg_img: np.ndarray | None, rms: float) -> str:
    # Build a right-side text panel instead of overlaying across the whole window.
    lines = [
        f"Review  |  RMS={rms:.2f}px",
        "ENTER/y/s = accept",
        "r/n/BACKSPACE = redo",
        "ESC/q = abort",
        "Q/X = quit-all",
    ]
    h = anno_img.shape[0] if anno_img is not None else (reproj_img.shape[0] if reproj_img is not None else 720)
    right = _text_panel(lines, width=360, height=h)
    while True:
        _show_dash(anno_img, reproj_img, right, "")  # no banner overlay; use right panel for text
        k = cv2.waitKey(50) & 0xFF
        if k in (13, ord('y'), ord('s')):  # accept
            return "accept"
        if k in (ord('r'), ord('n'), 8):  # redo
            return "redo"
        if k in (ord('Q'), ord('X'), ord('x')):  # quit-all
            _quit_all()
        if k in (27, ord('q')):  # abort current shot
            raise SystemExit("Aborted by user during review.")


def _load_dist_from_calib(calib_path):
    try:
        y = yaml.safe_load(open(calib_path, "r"))
        dc = y["distortion_coefficients"]
        # k1,k2,p1,p2,k3 (OpenCV order)
        return np.array([[dc["k1"], dc["k2"], dc["p1"], dc["p2"], dc["k3"]]], dtype=float)
    except Exception:
        return np.zeros((1, 5), dtype=float)


def assign_corners_any_order(clicked_xy, face_names, pts3d_dict, K, dist):
    pts2d = np.array(clicked_xy, float).reshape(-1, 1, 2)
    names = list(face_names)
    best = None
    for perm in itertools.permutations(names, 4):
        X = np.array([pts3d_dict[n] for n in perm], float).reshape(-1, 1, 3)
        ok, rvec, tvec = cv2.solvePnP(X, pts2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        reproj, _ = cv2.projectPoints(X, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((reproj - pts2d) ** 2, axis=2))))
        R, _ = cv2.Rodrigues(rvec)
        Xc = (R @ X.reshape(-1, 3).T + tvec.reshape(3, 1)).T
        zmean = float(np.mean(Xc[:, 2]))
        if best is None or (err < best["err"] and zmean > 0):
            best = {"perm": perm, "rvec": rvec, "tvec": tvec, "err": err}
    if best is None:
        raise RuntimeError("PnP failed for all 24 assignments")
    mapping = {name: tuple(clicked_xy[i]) for i, name in enumerate(best["perm"])}
    ordered2d = np.array([mapping[n] for n in face_names], float)
    return mapping, ordered2d, {"rvec": best["rvec"], "tvec": best["tvec"], "err": best["err"]}


def collect_quad_editor(image_bgr, mid_img: np.ndarray | None, right_img: np.ndarray | None) -> list[
    tuple[float, float]]:
    base = image_bgr.copy()
    h, w = base.shape[:2]
    pts: list[tuple[float, float]] = []
    start_pt: tuple[float, float] | None = None
    dragging_idx: int | None = None
    history: list[list[tuple[float, float]]] = []
    cursor = (0.0, 0.0)
    last_flags = 0

    def near(a, b, r=10):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= r * r

    def clamp(u, v):
        return max(0, min(w - 1, u)), max(0, min(h - 1, v))

    def snap_axis(anchor, candidate, flags):
        if not (flags & cv2.EVENT_FLAG_SHIFTKEY): return candidate
        ax, ay = anchor;
        cx, cy = candidate
        return (cx, ay) if abs(cx - ax) >= abs(cy - ay) else (ax, cy)

    def draw_left():
        vis = base.copy()
        banner = "Annotate  |  Drag box then adjust corners. SHIFT=lock  ENTER=accept  u=undo  r=reset  q=abort  Q/X=quit-all"
        if start_pt is not None and not pts:
            x0, y0 = start_pt;
            x1, y1 = cursor
            tl = (min(x0, x1), min(y0, y1));
            br = (max(x0, x1), max(y0, y1))
            tr = (br[0], tl[1]);
            bl = (tl[0], br[1])
            for a, b in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
                cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 180, 60), 2, cv2.LINE_AA)
        if pts:
            for i in range(4):
                a = pts[i];
                b = pts[(i + 1) % 4]
                cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 200, 0), 2, cv2.LINE_AA)
            for i, (u, v) in enumerate(pts):
                color = (0, 0, 255) if i == dragging_idx else (0, 200, 0)
                cv2.circle(vis, (int(u), int(v)), 6, color, -1, cv2.LINE_AA)
                cv2.putText(vis, str(i + 1), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        _show_dash(vis, mid_img, right_img, banner)  # <- keep mid/right visible
        return vis

    left_w, left_h = w, h

    def on_mouse(event, x, y, flags, _param):
        nonlocal start_pt, pts, dragging_idx, cursor, last_flags
        last_flags = flags
        if not (0 <= x < left_w and 0 <= y < left_h):  # only left panel is interactive
            return
        cursor = clamp(float(x), float(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if not pts:
                start_pt = cursor
            else:
                for i, p in enumerate(pts):
                    if near(cursor, p, r=10):
                        dragging_idx = i
                        history.append(pts.copy())
                        break
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx is not None and pts:
                anchor = pts[dragging_idx]
                u, v = snap_axis(anchor, cursor, flags)
                pts[dragging_idx] = clamp(u, v)
            draw_left()
        elif event == cv2.EVENT_LBUTTONUP:
            if start_pt is not None and not pts:
                x0, y0 = start_pt;
                x1, y1 = cursor
                tl = clamp(min(x0, x1), min(y0, y1))
                tr = clamp(max(x0, x1), min(y0, y1))
                br = clamp(max(x0, x1), max(y0, y1))
                bl = clamp(min(x0, x1), max(y0, y1))
                pts[:] = [tl, tr, br, bl]
                start_pt = None
            dragging_idx = None
            draw_left()

    cv2.setMouseCallback("Dash", on_mouse)
    draw_left()

    while True:
        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord('q')):           raise SystemExit("Aborted.")
        if k in (ord('Q'), ord('X'), ord('x')):                  _quit_all()
        if k == ord('r'):                  pts.clear(); history.clear(); start_pt = None; draw_left()
        if k == ord('u') and history:      pts = history.pop(); draw_left()
        if k == 13 and len(pts) == 4:      return [(float(u), float(v)) for (u, v) in pts]


def collect_four_clicks_any_order(image_bgr, mid_img: np.ndarray | None, right_img: np.ndarray | None) -> list[
    tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    base = image_bgr.copy()
    h, w = base.shape[:2]
    cursor = None
    last_flags = 0

    def snap_to_axis(p0, p1, flags):
        if p0 is None or not (flags & cv2.EVENT_FLAG_SHIFTKEY):
            return (float(p1[0]), float(p1[1]))
        (x0, y0), (x1, y1) = p0, p1
        return (float(x1), float(y0)) if abs(x1 - x0) >= abs(y1 - y0) else (float(x0), float(y1))

    def redraw_left():
        vis = base.copy()
        banner = "Annotate  |  Click any 4 corners. SHIFT=lock  ENTER=accept  u=undo  r=reset  q=abort  Q/X=quit-all"
        for i, (u, v) in enumerate(pts):
            cv2.circle(vis, (int(u), int(v)), 4, (0, 200, 0), -1)
            cv2.putText(vis, str(i + 1), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            if i > 0:
                u0, v0 = pts[i - 1]
                cv2.line(vis, (int(u0), int(v0)), (int(u), int(v)), (0, 200, 0), 1, cv2.LINE_AA)
        if cursor is not None and len(pts) >= 1:
            u0, v0 = pts[-1]
            u1, v1 = snap_to_axis((u0, v0), cursor, last_flags)
            cv2.line(vis, (int(u0), int(v0)), (int(u1), int(v1)), (255, 180, 60), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, (int(u1), int(v1)), (255, 180, 60), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=1, line_type=cv2.LINE_AA)
        _show_dash(vis, mid_img, right_img, banner)  # <- keep mid/right visible
        return vis

    left_w, left_h = w, h

    def on_mouse(event, x, y, flags, _param):
        nonlocal cursor, last_flags
        last_flags = flags
        if not (0 <= x < left_w and 0 <= y < left_h): return
        if event == cv2.EVENT_MOUSEMOVE:
            cursor = (float(x), float(y));
            redraw_left()
        elif event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            p = (float(x), float(y))
            if pts: p = snap_to_axis(pts[-1], p, flags)
            pts.append(p);
            redraw_left()

    # cv2.namedWindow("Dash", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Dash", on_mouse)
    redraw_left()

    while True:
        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord('q')):           raise SystemExit("Aborted.")
        if k in (ord('Q'), ord('X'), ord('x')):                  _quit_all()
        if k == ord('u') and pts:         pts.pop(); redraw_left()
        if k == ord('r'):                 pts.clear(); redraw_left()
        if k == 13 and len(pts) == 4:     return pts


def reproj_err(pts3d, T_cam_obj, K, img, draw=True, dist=None):
    if dist is None:
        dist = np.zeros((1, 5), float)
    R = T_cam_obj[:3, :3];
    t = T_cam_obj[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts3d, rvec, t, K, dist)
    uv = uv.reshape(-1, 2)
    if draw:
        vis = img.copy()
        for (u, v) in uv:
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
        return uv, vis
    return uv, None

def _parse_shot_path(p: Path):
    # Expect: shots/<object_base>/side<Side>/<object_base>_side<Side>_<YYYYMMDD_HHMMSS>_raw.png
    # Fallback: try to extract from meta later if needed.
    m = re.search(r"shots/(.+?)/side([A-Za-z]+)/.+_side([A-Za-z]+)_(\d{8}_\d{6})_raw\.png$", str(p).replace("\\","/"))
    if m:
        obj = m.group(1)
        side = m.group(2).upper()
        ts = m.group(4)
        return obj, side, ts
    return None, None, None

def _index_shots(root: Path):
    shots_root = (root / "shots").resolve()
    found = sorted(shots_root.glob("**/*_raw.png"))
    idx: dict[str, dict[str, list[Path]]] = {}
    for p in found:
        obj, side, ts = _parse_shot_path(p)
        if obj is None:
            # still include, derive obj/side from meta if present
            meta = Path(str(p).replace("_raw.png", "_meta.json"))
            try:
                if meta.exists():
                    m = load_meta(meta, strict=False)
                    obj = m.get("object_base") or m.get("object") or "unknown"
                    side = (m.get("side") or "UNRESOLVED").upper()
            except Exception:
                obj = obj or "unknown"
                side = side or "UNRESOLVED"
        obj = obj or "unknown"; side = side or "UNRESOLVED"
        idx.setdefault(obj, {}).setdefault(side, []).append(p)
    # newest first within each side
    for obj in idx:
        for side in idx[obj]:
            idx[obj][side].sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return idx

def _render_list_panel(h: int, w: int, title: str, items: list[str], sel: int) -> np.ndarray:
    pan = np.full((h, w, 3), 240, np.uint8)
    cv2.putText(pan, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20,20,20), 2, cv2.LINE_AA)
    max_show = min(14, len(items))
    start = max(0, min(sel - max_show // 2, max(0, len(items) - max_show)))
    y = 60
    for i in range(start, start + max_show):
        if i >= len(items): break
        txt = ("> " if i == sel else "  ") + items[i]
        col = (0,0,0) if i != sel else (0,60,200)
        thk = 1 if i != sel else 2
        cv2.putText(pan, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, thk, cv2.LINE_AA)
        y += 26
    tips = "↑/↓/W/S navigate   Enter select   o switch list   Esc back   q quit"
    cv2.putText(pan, tips, (12, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1, cv2.LINE_AA)
    return pan

def _render_shot_thumb(path: Path, w: int = 420) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: return np.full((300, w, 3), 220, np.uint8)
    H, W = img.shape[:2]
    ww = min(w, max(200, w))
    hh = int(H * ww / W)
    thumb = cv2.resize(img, (ww, hh))
    can = np.full((max(hh, 300), ww, 3), 230, np.uint8)
    can[:hh, :ww] = thumb
    return can


def main():
    ap = argparse.ArgumentParser("Annotate a face shot to compute T_board_object")
    ap.add_argument("--project_root", default=None, type=Path,
                    help="Root for shots/boards/objects (default: $GTAT_PROJECT or config or ./gtat_project)")
    ap.add_argument("--shot", default=None, help="Path to *_raw.png (absolute or relative to project_root); omit to browse in-window")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--calib", default="calib_color.yaml",
                    help="Path to ChArUco calib (k1..k3); if relative, resolved under project_root.")
    ap.add_argument("--log_file", default=None,
                    help="override log path (default: <project_root>/logs/annotate_face_shot.log)")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="log verbosity")
    ap.add_argument("--log_console", action="store_true", help="also echo logs to stderr")
    ap.add_argument('--pts_type', default='quad', choices=['quad', 'any'], )
    args = ap.parse_args()

    project_root = resolve_project_root(args.project_root)
    ensure_project_dirs(project_root)
    default_log = project_root / "logs" / "annotate_face_shot.log"
    log_path = Path(args.log_file) if args.log_file else default_log
    logger = init_project_logger(log_path, level=args.log_level, console=args.log_console)
    logger.info("Using project root: %s", project_root)

    # shot_raw = Path(args.shot)
    # if not shot_raw.is_absolute():
    #     shot_raw = (project_root / shot_raw).resolve()
    # logger.info("Shot: %s", shot_raw)
    #
    # if not shot_raw.exists(): raise SystemExit(f"Shot not found: {shot_raw}")
    # if not str(shot_raw).endswith("_raw.png"):
    #     raise SystemExit("Shot path must end with _raw.png")
    shot_raw: Path | None = None
    if args.shot:
        shot_raw = Path(args.shot)
        if not shot_raw.is_absolute():
            shot_raw = (project_root / shot_raw).resolve()
    else:
        # interactive browser
        idx = _index_shots(project_root)
        objects = sorted(idx.keys())
        if not objects:
            raise SystemExit(f"No shots found under {project_root}/shots")
        obj_i = 0
        side_i = 0
        shot_i = 0
        mode = "pick_object"  # pick_object -> pick_shot
        WINDOW = "Annotate Browser"
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
        while True:
            obj = objects[obj_i]
            sides = sorted(idx[obj].keys())
            if mode == "pick_object":
                left = np.full((720, 720, 3), 200, np.uint8)
                right = _render_list_panel(720, 520, "Select object", objects, obj_i)
                _show_dash(left, None, right, "Pick object (o=switch, Enter=select)")
            else:
                side = sides[min(side_i, len(sides) - 1)]
                shots = idx[obj][side]
                if not shots:
                    mode = "pick_object";
                    continue
                shot_i = max(0, min(shot_i, len(shots) - 1))
                left = _render_shot_thumb(shots[shot_i], 760)
                info = [
                    f"Object: {obj}",
                    f"Side: {side}  | {len(shots)} shot(s)",
                    f"Shot: {shots[shot_i].name}",
                    "Enter = annotate   o = switch object   Esc = back   q = quit",
                ]
                right = _text_panel(info, width=500, height=left.shape[0])
                _show_dash(left, None, right, "Pick shot")

            k = cv2.waitKeyEx(50) & 0xFFFFFFFF
            if k in (ord('q'), 27):  # q or Esc
                if mode == "pick_shot":
                    mode = "pick_object"
                    continue
                else:
                    raise SystemExit("Aborted in browser.")
            if k in (ord('o'), ord('O')):
                mode = "pick_object"
                continue
            if mode == "pick_object":
                if k in (2490368, ord('w'), ord('W')):  # up
                    obj_i = (obj_i - 1) % len(objects)
                elif k in (2621440, ord('s'), ord('S')):  # down
                    obj_i = (obj_i + 1) % len(objects)
                elif k in (13, 10):  # enter
                    side_i = 0;
                    shot_i = 0;
                    mode = "pick_shot"
            else:
                # pick_shot: up/down = shot cycle, left/right = side cycle
                if k in (2490368, ord('w'), ord('W')):  # up
                    shot_i = (shot_i - 1) % len(idx[obj][sides[side_i]])
                elif k in (2621440, ord('s'), ord('S')):  # down
                    shot_i = (shot_i + 1) % len(idx[obj][sides[side_i]])
                elif k in (2424832, ord('a'), ord('A')):  # left
                    side_i = (side_i - 1) % len(sides);
                    shot_i = 0
                elif k in (2555904, ord('d'), ord('D')):  # right
                    side_i = (side_i + 1) % len(sides);
                    shot_i = 0
                elif k in (13, 10):  # enter -> set shot
                    shot_raw = idx[obj][sides[side_i]][shot_i]
                    logger.info("Selected shot: %s", shot_raw)
                    cv2.destroyWindow(WINDOW)
                    break
    if shot_raw is None:
        raise SystemExit("No shot selected.")

    logger.info("Shot: %s", shot_raw)
    if not shot_raw.exists(): raise SystemExit(f"Shot not found: {shot_raw}")
    if not str(shot_raw).endswith("_raw.png"):
        raise SystemExit("Shot path must end with _raw.png")

    meta_path = Path(str(shot_raw).replace("_raw.png", "_meta.json"))
    if not meta_path.exists(): raise SystemExit(f"Meta json not found: {meta_path}")
    meta = load_meta(meta_path, strict=False)

    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        cand = project_root / calib_path
        if cand.exists():
            calib_path = cand
    dist = _load_dist_from_calib(calib_path)  # use calibrated distortion

    obj_full = meta.get("object_full") or meta.get("object")  # fallback for legacy
    obj_base = meta.get("object_base") or (parse_base_and_side(obj_full)[0] if obj_full else None)

    if obj_base is None:
        raise SystemExit("Meta missing object name (object_full/object_base).")
    face_yaml = meta.get("face_yaml")

    if not face_yaml:
        raise SystemExit("Meta missing face_yaml (capture in registered mode so face YAML is recorded).")

    # Load image
    img = cv2.imread(str(shot_raw), cv2.IMREAD_COLOR)
    if img is None: raise SystemExit(f"Failed to read image: {shot_raw}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Camera intrinsics
    fx, fy = meta["camera"]["fx"], meta["camera"]["fy"]
    cx, cy = meta["camera"]["cx"], meta["camera"]["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)

    pts3d_dict, faces_map, face_key_resolved, kp_path = load_keypoints_fuzzy(obj_base, project_root)
    if face_key_resolved is None:
        raise SystemExit(f"[!] Keypoints found at {kp_path}, but no face entry matches '{obj_base}'. "
                         f"Add a faces[...] key for this face or use --object_name to a matching one.")
    logger.info("Keypoints: %s | face key: %s", kp_path, face_key_resolved)
    face_key = choose_face_key(face_yaml)
    if face_key not in faces_map:
        raise SystemExit(f"Face '{face_key}' not found in keypoints.json faces. Available: {list(faces_map.keys())}")
    names_in_order = faces_map[face_key]
    logger.info("Face '%s' click order: %s", face_key, names_in_order)

    # Board (to convert camera->tag to camera->board)
    board_yaml_path = Path(face_yaml)
    if not board_yaml_path.is_absolute():
        board_yaml_path = (project_root / board_yaml_path).resolve()
    origin_id, tag_size_m, T_board_tag = load_board(board_yaml_path)

    # Detect tags in this image (get camera->tag)
    dets = detect_tags(gray, fx, fy, cx, cy, tag_size_m, family=args.family)
    if not dets:
        raise SystemExit("No AprilTags detected in shot; cannot get board pose.")
    det_by_id = {int(d.tag_id): d for d in dets}

    # Compute T_cam_board using origin if present, else any seen tag with board mapping
    if origin_id in det_by_id and det_by_id[origin_id].pose_R is not None:
        d = det_by_id[origin_id]
        T_cam_board = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
    else:
        # pick any detected tag that exists in the board
        common = [tid for tid in det_by_id.keys() if tid in T_board_tag]
        if not common:
            raise SystemExit("Detected tags do not belong to this face’s board.")
        tid = common[0]
        d = det_by_id[tid]
        T_cam_tag = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
        T_board_tag_i = T_board_tag[tid]  # board->tag
        T_tag_board = inv_se3(T_board_tag_i)  # tag->board
        T_cam_board = T_cam_tag @ T_tag_board

    cv2.namedWindow("Dash", cv2.WINDOW_AUTOSIZE)

    mid_panel = None
    right_panel = None

    while True:
        # --- Annotate (left), keep mid/right visible while doing it ---
        if args.pts_type.lower() == 'quad':
            clicked = collect_quad_editor(img, mid_panel, right_panel)
        else:
            clicked = collect_four_clicks_any_order(img, mid_panel, right_panel)

        # Solve pose, build overlays
        mapping, ordered2d, fit = assign_corners_any_order(clicked, names_in_order, pts3d_dict, K, dist)
        R, _ = cv2.Rodrigues(fit["rvec"])
        T_cam_obj = se3(R, fit["tvec"].reshape(3))

        # anno (left)
        anno_vis = img.copy()

        def project_named(names, color, radius=3, canvas=None):
            pts = np.vstack([pts3d_dict[n] for n in names]).astype(np.float32)
            uv, _ = reproj_err(pts, T_cam_obj, K, img, draw=False, dist=dist)
            for (u, v), name in zip(uv, names):
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(canvas, (int(u), int(v)), radius, color, -1)
                    cv2.putText(canvas, name, (int(u) + 4, int(v) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        project_named(names_in_order, (0, 220, 0), radius=4, canvas=anno_vis)
        others = [n for n in pts3d_dict.keys() if n not in names_in_order]
        project_named(others, (0, 0, 255), radius=3, canvas=anno_vis)

        # reprojection (middle)
        all_pts3d = np.vstack([v for _, v in sorted(pts3d_dict.items())]).astype(np.float32)
        _, mid_panel = reproj_err(all_pts3d, T_cam_obj, K, img, draw=True, dist=dist)

        # RMS of the 4 canonical
        pts3d_canon = np.vstack([pts3d_dict[n] for n in names_in_order]).astype(np.float32)
        uv4, _ = reproj_err(pts3d_canon, T_cam_obj, K, img, draw=False, dist=dist)
        err = np.linalg.norm(uv4 - ordered2d, axis=1)
        rms_4 = float(np.sqrt((err ** 2).mean()))

        # debug (right)
        right_panel = None

        _flush_keys(120)
        # show stacked review and decide
        decision = review_decide(anno_vis, mid_panel, right_panel, rms_4)
        if decision == "redo":
            # keep mid_panel/right_panel so they persist in the next annotation pass
            continue

        # ---- Accepted: save outputs and break ----
        T_board_cam = inv_se3(T_cam_board)
        T_board_obj = T_board_cam @ T_cam_obj

        side_code = (meta.get("side") or parse_base_and_side(obj_full)[1] or "UNRESOLVED")
        side_code = str(side_code).upper()

        out_dir = (project_root / "faces" / obj_base / f"side{side_code}").resolve()
        audit_dir = out_dir / "_audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)

        ts = str(meta.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))

        out_yaml = out_dir / f"{face_key}_T_board_object.yaml"

        anno_path = audit_dir / f"{face_key}_{ts}_anno.png"
        cv2.imwrite(str(anno_path), anno_vis)

        reproj_path = audit_dir / f"{face_key}_{ts}_reproj.png"

        _, vis_all = reproj_err(
            np.vstack([v for _, v in sorted(pts3d_dict.items())]).astype(np.float32),
            T_cam_obj, K, img, draw=True
        )
        cv2.imwrite(str(reproj_path), vis_all)

        # Save YAML
        out = {
            "object": obj_base,
            "face_key": face_key,
            "board_yaml": str(board_yaml_path),
            "image": str(shot_raw),
            "rms_px": rms_4,
            "T_board_object": {"matrix": T_board_obj.tolist()},
            "corner_uv": {name: [float(mapping[name][0]), float(mapping[name][1])] for name in names_in_order},
            "pnp": {
                "rvec": [float(x) for x in fit["rvec"].ravel()],
                "tvec": [float(x) for x in fit["tvec"].ravel()],
            },
            "clicked_uv_raw": [(float(u), float(v)) for (u, v) in clicked],  # raw clicks as placed
            "face_corner_names": names_in_order,  # expected canonical order
            "assignment": {n: [float(mapping[n][0]), float(mapping[n][1])] for n in names_in_order},
            "side": side_code,
            "notes": "Any-order 4-point annotation + tag-based board pose on this image. Accepted by reviewer.",
        }

        exists = out_yaml.exists()
        if exists:
            prompt = _text_panel([
                f"File exists:",
                f"{out_yaml.name}",
                "",
                "y / ENTER = overwrite",
                "b          = keep both (make _v2, _v3, ...)",
                "ESC / q    = cancel save"
            ], width=520, height=anno_vis.shape[0])
            _show_dash(anno_vis, mid_panel, prompt, "YAML exists")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in (13, ord('y')):  # overwrite
                    break
                elif k == ord('b'):  # keep both -> bump suffix
                    i = 2
                    stem = out_yaml.stem
                    parent = out_yaml.parent
                    cand = parent / f"{stem}_v{i}.yaml"
                    while cand.exists():
                        i += 1
                        cand = parent / f"{stem}_v{i}.yaml"
                    out_yaml = cand
                    break
                elif k in (27, ord('q')):  # cancel
                    logger.info("Save cancelled by user; not writing YAML.")
                    cv2.destroyAllWindows()
                    return

        with out_yaml.open("w") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        logger.info("Wrote YAML: %s", out_yaml)
        logger.info("Audit: anno=%s  reproj=%s", anno_path, reproj_path)
        logger.info("Press any key to close.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break


if __name__ == "__main__":
    main()
