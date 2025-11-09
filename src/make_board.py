#!/usr/bin/env python3
"""
make_board.py — Build a per-face AprilTag "board" YAML and update a global registry.

What this does
--------------
- Live preview from a camera (OpenCV webcam / RealSense) or a video file.
- Detect AprilTags; you pick which IDs to include and choose an origin tag.
- Compute each selected tag's 2D offset (cx, cy) in the origin-tag frame + in-plane yaw.
- Enforce (or warn about) planarity via |z| threshold.
- Write `<project_root>/boards/<object_face>.yaml` and update
  `<project_root>/boards/tag_registry.yaml` atomically.
- Optional audit shot(s) written to `<project_root>/boards/shots/`.

Defaults are *project-aware*: if `--project_root` is omitted, we resolve one via
utils.project_config.resolve_project_root (env/config/home logic).
"""

import argparse, os, sys, yaml, numpy as np, cv2, datetime, tempfile, shutil
from math import atan2, degrees
from pathlib import Path
from typing import Optional, Tuple

# ---- Tag detector (pupil-apriltags) ----
try:
    from pupil_apriltags import Detector
except Exception as e:
    raise SystemExit("Install pupil-apriltags: pip install pupil-apriltags") from e

# ---- Optional RealSense ----
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# ---- Project root helpers ----
try:
    from utils.project_config import resolve_project_root, ensure_project_dirs
except Exception:
    resolve_project_root = None
    ensure_project_dirs = None


# --------- IO helpers ---------
def _now_iso_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _atomic_write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._tmp_yaml_", dir=str(path.parent))
    os.close(fd)
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    shutil.move(tmp, str(path))


def load_calib(path: str) -> Tuple[Tuple[float, float, float, float], np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        print(f"[!] {path} not found.")
        print("    Run ChArUco calibration to generate calib_color.yaml first.")
        sys.exit(1)
    y = yaml.safe_load(open(path)) or {}
    cm = y["camera_matrix"]; dc = y.get("distortion_coefficients", {})
    fx, fy, cx, cy = float(cm["fx"]), float(cm["fy"]), float(cm["cx"]), float(cm["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    # Distortion is not required by pupil-apriltags pose solver (assumes pinhole).
    D = np.array([[dc.get("k1", 0.0), dc.get("k2", 0.0), dc.get("p1", 0.0),
                   dc.get("p2", 0.0), dc.get("k3", 0.0)]], float)
    return (fx, fy, cx, cy), K, D


def se3(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def inv_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "updated": None, "tags": {}}
    reg = yaml.safe_load(open(path)) or {}
    reg.setdefault("version", 1)
    reg.setdefault("tags", {})
    return reg


def save_registry(path: Path, reg: dict) -> None:
    reg["updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_yaml(path, reg)


# --------- CLI ---------
class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Show defaults and preserve newlines in help text."""
    pass


def parse_args():
    ap = argparse.ArgumentParser(
        "Interactive AprilTag board builder (project-aware)",
        formatter_class=_HelpFmt,
        epilog=(
            "Workflow:\n"
            "  1) Show live preview; hit ENTER to capture.\n"
            "  2) Pick the tag IDs to include, then choose the ORIGIN ID.\n"
            "  3) We compute cx, cy (m) in the origin frame and yaw (deg) per tag.\n\n"
            "Keys:\n"
            "  ENTER capture  |  ESC quit\n"
        ),
    )
    ap.add_argument("--project_root", type=Path, default=None,
                    help="Root for boards/shots/objects/datasets (default: resolver/env/config).")
    ap.add_argument("--calib", type=str, default="calib_color.yaml",
                    help="Camera intrinsics YAML (fx, fy, cx, cy, dist).")
    ap.add_argument("--family", default="tag36h11", help="AprilTag family.")
    ap.add_argument("--tag_size_mm", type=float, default=80.0, help="Black square edge (mm).")

    # Source selection
    ap.add_argument("--source", choices=["opencv", "realsense", "video"], default="opencv",
                    help="Capture source: OpenCV webcam (default), Intel RealSense, or a video file.")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index when --source=opencv.")
    ap.add_argument("--video", type=str, default=None, help="Video path when --source=video.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    # Outputs (default under <project_root>/boards/…)
    ap.add_argument("--object_name", type=str, required=True,
                    help="Face name, e.g., connection_plate_white_sideA.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Board YAML directory (default: <project_root>/boards).")
    ap.add_argument("--shots_dir", type=str, default=None,
                    help="Audit shots directory (default: <project_root>/boards/shots).")
    ap.add_argument("--registry", type=str, default=None,
                    help="Tag registry path (default: <project_root>/boards/tag_registry.yaml).")
    ap.add_argument("--save_shot", action="store_true", help="Save captured frame (raw + annotated).")

    # Planarity controls
    ap.add_argument("--z_thresh", type=float, default=0.01, help="Max |z| (m) from plane; guards planarity.")
    ap.add_argument("--allow_nonplanar", action="store_true",
                    help="Warn but still write YAML even if |z| > z_thresh.")

    return ap.parse_args()


def _resolve_paths(args) -> Tuple[Path, Path, Path, Path]:
    """Resolve project root and default output paths."""
    # Project root
    if resolve_project_root is not None:
        pr = Path(resolve_project_root(args.project_root))
        if ensure_project_dirs is not None:
            ensure_project_dirs(pr)
    else:
        pr = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()

    # Defaults under <project_root>/boards
    boards_dir = Path(args.out_dir) if args.out_dir else pr / "boards"
    shots_dir = Path(args.shots_dir) if args.shots_dir else boards_dir / "shots"
    registry = Path(args.registry) if args.registry else boards_dir / "tag_registry.yaml"

    boards_dir.mkdir(parents=True, exist_ok=True)
    if args.save_shot:
        shots_dir.mkdir(parents=True, exist_ok=True)

    return pr, boards_dir, shots_dir, registry


def _open_source(args):
    """Return (read_frame_fn, stop_fn)."""
    if args.source == "realsense":
        if rs is None:
            sys.exit("pyrealsense2 not available; use --source opencv|video")
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pipe.start(cfg)

        def _read():
            frames = pipe.wait_for_frames()
            c = frames.get_color_frame()
            return None if not c else np.asanyarray(c.get_data())

        def _stop():
            pipe.stop()

        return _read, _stop

    if args.source == "opencv":
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            sys.exit(f"Could not open camera index {args.cam}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        def _read():
            ok, frame = cap.read()
            return frame if ok else None

        def _stop():
            cap.release()

        return _read, _stop

    # video
    if not args.video:
        sys.exit("--video path is required when --source=video")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")

    def _read():
        ok, frame = cap.read()
        return frame if ok else None

    def _stop():
        cap.release()

    return _read, _stop


# --------- Main ---------
def main():
    args = parse_args()
    project_root, boards_dir, shots_dir, registry_path = _resolve_paths(args)

    # Calib: allow relative path under project_root
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        cand = (project_root / calib_path)
        if cand.exists():
            calib_path = cand
    (fx, fy, cx, cy), K, _D = load_calib(str(calib_path))

    tag_size_m = args.tag_size_mm / 1000.0
    det = Detector(families=args.family, nthreads=4, quad_decimate=1.0, refine_edges=True)

    read_frame, stop = _open_source(args)

    # ---- Live preview & capture ----
    print(f"[i] Project root = {project_root}")
    print("[i] Showing preview. Make sure all desired tags are visible once.")
    print("[i] Press ENTER to capture this frame; ESC to quit.")

    frame = None
    dets = []
    try:
        while True:
            c = read_frame()
            if c is None:
                continue
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            # Draw IDs for situational awareness
            dd = det.detect(g, estimate_tag_pose=False)
            vis = c.copy()
            for d in dd:
                pts = d.corners.astype(int)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                cv2.putText(vis, str(int(d.tag_id)), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Select a view (ENTER to use)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return
            if k == 13:  # ENTER
                frame = c.copy()
                g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = det.detect(
                    g2, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size_m
                )
                if len(dets) == 0:
                    print("[!] No tags detected in captured frame; try again.")
                    continue

                if args.save_shot:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base = shots_dir / f"{args.object_name}_{ts}"
                    raw_path = f"{base}_raw.png"
                    ann_path = f"{base}_ann.png"
                    cv2.imwrite(raw_path, frame)
                    ann = frame.copy()
                    for d in dets:
                        pts = d.corners.astype(int)
                        cv2.polylines(ann, [pts], True, (0, 255, 0), 2)
                        cv2.putText(ann, str(int(d.tag_id)), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imwrite(ann_path, ann)
                    print(f"[i] Saved shot: {raw_path} and {ann_path}")
                break
    finally:
        stop()
        cv2.destroyAllWindows()

    # ---- Choose IDs and origin ----
    ids_visible = [int(d.tag_id) for d in dets]
    print(f"[i] Detected IDs in captured frame: {ids_visible}")
    sels = input("[?] Enter comma-separated IDs to include (e.g., 12,37): ").strip()
    try:
        sel_ids = [int(s) for s in sels.replace(" ", "").split(",") if s != ""]
    except Exception:
        print("[!] Could not parse IDs.")
        return
    sel_ids = [i for i in sel_ids if i in ids_visible]
    if len(sel_ids) < 1:
        print("[!] Need at least one valid ID.")
        return

    origin = input(f"[?] Choose ORIGIN id from {sel_ids} (default {sel_ids[0]}): ").strip()
    origin_id = int(origin) if origin else sel_ids[0]
    if origin_id not in sel_ids:
        print("[!] Origin must be one of the selected IDs.")
        return

    # ---- Build SE(3) per tag and express in origin frame ----
    det_map = {int(d.tag_id): d for d in dets}
    T_cam = {}
    for tid in sel_ids:
        d = det_map.get(tid, None)
        if d is None or d.pose_R is None or d.pose_t is None:
            print(f"[!] Missing pose for id {tid}.")
            return
        T_cam[tid] = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))

    T_org_cam = inv_se3(T_cam[origin_id])
    entries = []
    nonplanar = False
    for tid in sel_ids:
        T_org_tag = T_org_cam @ T_cam[tid]
        t = T_org_tag[:3, 3]
        R = T_org_tag[:3, :3]
        yaw = degrees(atan2(R[1, 0], R[0, 0]))  # in-plane yaw around +z
        if abs(t[2]) > args.z_thresh:
            print(f"[!] Tag {tid} z offset {t[2]:.3f} m exceeds {args.z_thresh} m.")
            nonplanar = True
        entries.append(dict(id=int(tid), cx=float(t[0]), cy=float(t[1]), yaw_deg=float(yaw)))

    if nonplanar and not args.allow_nonplanar:
        print("[!] Re-capture with a flatter view / re-mount the tags, or use --allow_nonplanar to proceed.")
        return

    entries.sort(key=lambda e: e["id"])

    # ---- Write board YAML ----
    out_path = boards_dir / f"{args.object_name}.yaml"
    board = dict(
        object=args.object_name,
        family=args.family,
        tag_size_m=tag_size_m,
        origin_id=int(origin_id),
        tags=entries,
        notes="Board frame = origin tag centre; x,y follow origin tag axes; z ≈ 0."
    )
    _atomic_write_yaml(out_path, board)
    print(f"[i] Wrote {out_path}")
    print("[i] Example entries:")
    for e in entries:
        print(f"    - {{id: {e['id']}, cx: {e['cx']:.3f}, cy: {e['cy']:.3f}, yaw_deg: {e['yaw_deg']:.1f}}}")

    # ---- Update registry ----
    reg = load_registry(registry_path)
    updated, conflicts = 0, []
    for e in entries:
        tid = str(e["id"])
        current = reg["tags"].get(tid)
        if current is None or current.get("yaml") == str(out_path):
            reg["tags"][tid] = {"object": args.object_name, "yaml": str(out_path)}
            updated += 1
        else:
            conflicts.append((tid, current["yaml"], str(out_path)))

    if conflicts:
        print("[!] Registry conflicts:")
        for tid, oldp, newp in conflicts:
            print(f"    tag {tid}: {oldp}  ->  {newp}")
        # By design we *don't* overwrite automatically here. Edit or delete the old mapping if intended.

    save_registry(registry_path, reg)
    print(f"[i] Registry updated ({updated} entries) at {registry_path}")


if __name__ == "__main__":
    main()
