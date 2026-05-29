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
from pathlib import Path
from typing import Tuple

# ---- Optional RealSense ----
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

from posetag.pipelines.make_board import (
    MakeBoardError,
    annotate_detections,
    board_entries_from_detections,
    create_apriltag_detector,
    detect_frame_tags,
    load_detector_class,
    load_calibration_yaml,
    prepare_project_paths,
    preview_project_root,
    resolve_calibration_path,
    save_board_definition,
    validate_source_args,
)


def _load_detector_class():
    return load_detector_class()


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

    tag_size_m = args.tag_size_mm / 1000.0
    try:
        det = create_apriltag_detector(args.family, detector_class=Detector)
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
            # Draw IDs for situational awareness
            dd = detect_frame_tags(c, det, estimate_pose=False)
            vis = annotate_detections(c, dd)

            cv2.imshow("Select a view (ENTER to use)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                print("[i] quit requested; exiting without writing board YAML.")
                return 0
            if k in (10, 13):  # ENTER
                frame = c.copy()
                dets = detect_frame_tags(
                    frame,
                    det,
                    calibration=calib,
                    tag_size_m=tag_size_m,
                    estimate_pose=True,
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
                    ann = annotate_detections(frame, dets)
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

    try:
        layout = board_entries_from_detections(
            dets,
            sel_ids,
            origin_id,
            z_threshold_m=args.z_thresh,
        )
    except MakeBoardError as exc:
        print(f"[!] {exc}")
        return

    for nonplanar_tag in layout.nonplanar_tags:
        print(
            f"[!] Tag {nonplanar_tag.tag_id} z offset "
            f"{nonplanar_tag.z_offset_m:.3f} m exceeds "
            f"{nonplanar_tag.threshold_m} m."
        )
    if layout.nonplanar_tags and not args.allow_nonplanar:
        print("[!] Re-capture with a flatter view / re-mount the tags, or use --allow_nonplanar to proceed.")
        return

    entries = tuple(layout.entries)

    # ---- Write board YAML ----
    saved = save_board_definition(
        paths,
        object_name=args.object_name,
        family=args.family,
        tag_size_mm=args.tag_size_mm,
        origin_id=origin_id,
        entries=entries,
    )
    out_path = saved.board_yaml_path
    print(f"[i] Wrote {out_path}")
    print("[i] Example entries:")
    for e in entries:
        print(f"    - {{id: {e['id']}, cx: {e['cx']:.3f}, cy: {e['cy']:.3f}, yaw_deg: {e['yaw_deg']:.1f}}}")

    # ---- Update registry ----
    registry_update = saved.registry_update

    if registry_update.conflicts:
        print("[!] Registry conflicts:")
        for conflict in registry_update.conflicts:
            print(f"    tag {conflict.tag_id}: {conflict.existing_yaml}  ->  {conflict.requested_yaml}")
        # By design we *don't* overwrite automatically here. Edit or delete the old mapping if intended.

    print(f"[i] Registry updated ({registry_update.updated} entries) at {registry_path}")


if __name__ == "__main__":
    main()
