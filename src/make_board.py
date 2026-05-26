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

import argparse, sys, numpy as np, cv2, datetime
from math import atan2, degrees
from pathlib import Path
from typing import Tuple

# ---- Optional RealSense ----
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

from posetag.pipelines.make_board import (
    MakeBoardError,
    build_board_yaml,
    load_calibration_yaml,
    load_registry,
    prepare_project_paths,
    preview_project_root,
    resolve_calibration_path,
    save_registry,
    update_registry_entries,
    validate_source_args,
    write_board_yaml,
)


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


def _load_detector_class():
    try:
        from pupil_apriltags import Detector
    except Exception as exc:
        raise MakeBoardError(
            "pupil-apriltags is not available; install PoseTag with the 'apriltags' "
            "extra or run `python -m pip install pupil-apriltags`."
        ) from exc
    return Detector


# --------- CLI ---------
class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Show defaults and preserve newlines in help text."""
    pass


def build_parser():
    ap = argparse.ArgumentParser(
        prog="posetag-make-board",
        description="Interactive AprilTag board builder (project-aware)",
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

    return ap


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _resolve_paths(args) -> Tuple[Path, Path, Path, Path]:
    """Resolve project root and default output paths."""
    paths = prepare_project_paths(
        project_root=args.project_root,
        object_name=args.object_name,
        out_dir=args.out_dir,
        shots_dir=args.shots_dir,
        registry=args.registry,
        save_shot=args.save_shot,
    )
    return paths.project_root, paths.boards_dir, paths.shots_dir, paths.registry_path


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
    video_path = str(Path(args.video).expanduser())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")

    def _read():
        ok, frame = cap.read()
        return frame if ok else None

    def _stop():
        cap.release()

    return _read, _stop


# --------- Main ---------
def main(argv=None):
    args = parse_args(argv)

    try:
        validate_source_args(args.source, args.video, rs)
        project_root_for_calib = preview_project_root(args.project_root)
        calib_path = resolve_calibration_path(project_root_for_calib, args.calib)
        calib = load_calibration_yaml(calib_path)
        Detector = _load_detector_class()
    except MakeBoardError as exc:
        raise SystemExit(str(exc)) from exc

    read_frame, stop = _open_source(args)
    try:
        paths = prepare_project_paths(
            project_root=args.project_root,
            object_name=args.object_name,
            out_dir=args.out_dir,
            shots_dir=args.shots_dir,
            registry=args.registry,
            save_shot=args.save_shot,
        )
    except Exception:
        stop()
        raise

    project_root = paths.project_root
    shots_dir = paths.shots_dir
    registry_path = paths.registry_path

    fx, fy, cx, cy = calib.camera_params

    tag_size_m = args.tag_size_mm / 1000.0
    det = Detector(families=args.family, nthreads=4, quad_decimate=1.0, refine_edges=True)

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
                if args.source == "video":
                    print("[i] video ended before a board frame was captured; exiting without writing board YAML.")
                    return 0
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
                print("[i] quit requested; exiting without writing board YAML.")
                return 0
            if k in (10, 13):  # ENTER
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
    out_path = paths.board_yaml_path
    board = build_board_yaml(
        object_name=args.object_name,
        family=args.family,
        tag_size_mm=args.tag_size_mm,
        origin_id=origin_id,
        entries=entries,
    )
    write_board_yaml(out_path, board)
    print(f"[i] Wrote {out_path}")
    print("[i] Example entries:")
    for e in entries:
        print(f"    - {{id: {e['id']}, cx: {e['cx']:.3f}, cy: {e['cy']:.3f}, yaw_deg: {e['yaw_deg']:.1f}}}")

    # ---- Update registry ----
    reg = load_registry(registry_path)
    registry_update = update_registry_entries(reg, board["tags"], args.object_name, out_path)

    if registry_update.conflicts:
        print("[!] Registry conflicts:")
        for conflict in registry_update.conflicts:
            print(f"    tag {conflict.tag_id}: {conflict.existing_yaml}  ->  {conflict.requested_yaml}")
        # By design we *don't* overwrite automatically here. Edit or delete the old mapping if intended.

    save_registry(registry_path, reg)
    print(f"[i] Registry updated ({registry_update.updated} entries) at {registry_path}")


if __name__ == "__main__":
    main()
