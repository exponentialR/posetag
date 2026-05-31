"""
capture_face.py — Face-shot capture for later T_board_object annotation
Author: Samuel Adebayo

Summary
-------
Capture wide shots per object face with AprilTag overlays and metadata for
subsequent T_board_object (board→object) computation. Project-aware: resolves
an active project root and writes under <project_root>/shots/… by default.

What’s new (rev)
----------------
- Full face (e.g., connection_plate_white_sideA) or base-only (e.g., connection_plate_white);
  base auto-resolves side via boards/tag_registry.yaml using live detections.
- In-window object picker ('o'); no terminal prompts in preview.
- Info + last-saved side panels ('g' toggles). Panels are UI-only.
- Safety save: press Enter twice within 3 s if expected tags aren’t seen.
- Resizable window; camera frame stays at requested resolution.

Saving layout
-------------
Default (by object and side):
  <project_root>/shots/<object_base>/side<SideLetter>/
    <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_{raw,ann}.png
    <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_meta.json

Keys
----
ENTER save (double-press to force) | Left/Right cycle faces | o object picker |
a auto-side | f next face | g panels | h help | q / ESC quit
"""

from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import cv2

from utils import capture_source
from utils.capture_utils import draw_detections, make_info_panel, make_recent_panel, text_lines
from utils.logger import init_project_logger
from posetag.pipelines.capture_face import (
    activate_capture_project_root,
    CaptureFaceError,
    build_capture_metadata,
    build_shot_paths,
    capture_timestamp,
    create_apriltag_detector,
    create_capture_output_dirs,
    faces_for_base,
    load_apriltag_detector_class,
    load_capture_calibration,
    load_capture_registry,
    load_registered_faces,
    prepare_capture_paths,
    preview_capture_project_root,
    resolve_capture_calibration_path,
    select_initial_faces,
    unique_bases,
    validate_capture_source_args,
    write_capture_outputs,
)

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 2490368, 2621440, 2424832, 2555904


UI_BG = (245, 249, 251)
UI_CARD = (255, 255, 255)
UI_BORDER = (212, 226, 234)
UI_TEXT = (36, 49, 61)
UI_MUTED = (98, 113, 126)
UI_BLUE = (160, 91, 0)
UI_GREEN = (64, 134, 29)
UI_AMBER = (24, 139, 204)
UI_RED = (44, 65, 190)
UI_DARK = (38, 45, 52)
UI_DARK_2 = (62, 72, 82)
UI_WHITE = (255, 255, 255)


def _face_label(face_entry, max_chars=28, default="unregistered"):
    """Short label for overlay: YAML basename, truncated; safe if face_entry is None."""
    if not face_entry:
        return default
    yml = face_entry.get("yaml", "")
    base = Path(yml).stem if yml else default
    if len(base) > max_chars:
        base = base[:max_chars - 1] + "..."
    return base


def _put(
    image: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    scale: float = 0.55,
    colour: Tuple[int, int, int] = UI_TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _fill_alpha(
    image: np.ndarray,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    colour: Tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, p0, p1, colour, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def object_picker_panel(
    h: int,
    w: int,
    bases: List[str],
    sel: int,
    hint: str = "Choose object",
) -> np.ndarray:
    pan = np.full((h, w, 3), UI_BG, np.uint8)
    margin = 14
    cv2.rectangle(pan, (margin, margin), (w - margin, h - margin), UI_CARD, -1)
    cv2.rectangle(pan, (margin, margin), (w - margin, h - margin), UI_BORDER, 1)
    _put(pan, hint, (margin + 16, margin + 30), scale=0.78, thickness=2)

    total = len(bases)
    status = f"{min(sel + 1, total) if total else 0}/{total} registered objects"
    _put(pan, status, (margin + 16, margin + 58), colour=UI_MUTED)
    if not bases:
        _put(
            pan,
            "No registered object bases found in the tag registry.",
            (margin + 16, margin + 96),
            colour=UI_RED,
        )
        return pan

    max_show = min(12, len(bases))
    start = max(0, min(sel - max_show // 2, len(bases) - max_show))
    end = min(len(bases), start + max_show)

    y = margin + 92
    for i in range(start, end):
        selected = i == sel
        row_top = y - 21
        row_bottom = y + 9
        if selected:
            cv2.rectangle(
                pan,
                (margin + 10, row_top),
                (w - margin - 10, row_bottom),
                (236, 246, 251),
                -1,
            )
            cv2.rectangle(
                pan,
                (margin + 10, row_top),
                (margin + 14, row_bottom),
                UI_BLUE,
                -1,
            )
        prefix = ">" if selected else " "
        colour = UI_TEXT if selected else UI_MUTED
        _put(
            pan,
            f"{prefix} {i + 1:02d}  {_truncate_middle(bases[i], 34)}",
            (margin + 20, y),
            scale=0.58,
            colour=colour,
            thickness=2 if selected else 1,
        )
        y += 34

    tips = "Up/Down or W/S navigate   Enter select   Esc cancel   1-9 jump"
    _put(pan, tips, (margin + 16, h - margin - 18), scale=0.45, colour=UI_MUTED)
    return pan


def cycle_face_selection(state: Dict, delta: int = 1) -> bool:
    """Cycle the active face explicitly and disable auto-side selection."""

    faces = state.get("faces") or []
    if not faces:
        return False
    current = int(state.get("face_idx", 0))
    state["face_idx"] = (current + int(delta)) % len(faces)
    state["auto_side"] = False
    state["face"] = faces[state["face_idx"]]
    state["save_warn"] = False
    return True


def draw_capture_overlay(
    frame: np.ndarray,
    *,
    face: Optional[Dict],
    state: Dict,
    ok: bool,
    min_expected: int,
) -> np.ndarray:
    """Draw a compact capture HUD over the camera frame."""

    vis = frame.copy()
    h, w = vis.shape[:2]
    det_ids = sorted(list(state.get("detected_ids", [])))
    exp_ids = sorted(list(set(face["tag_ids"])) if face else [])
    overlap = sorted(list(set(det_ids).intersection(exp_ids))) if face else []
    face_label = _face_label(face, max_chars=44, default="unresolved")
    status = "READY" if ok else "TAGS MISSING"
    status_colour = UI_GREEN if ok else UI_AMBER
    auto_text = "auto" if state.get("auto_side") else "manual"
    faces = state.get("faces") or []
    face_idx = int(state.get("face_idx", 0)) + 1 if faces else 0
    face_count = len(faces)

    card_w = min(w - 20, 640)
    card_h = 92
    _fill_alpha(vis, (10, 10), (10 + card_w, 10 + card_h), UI_DARK, 0.76)
    cv2.rectangle(vis, (10, 10), (10 + card_w, 10 + card_h), UI_DARK_2, 1)
    cv2.rectangle(vis, (22, 23), (34, 35), status_colour, -1)
    _put(
        vis,
        f"{status}  {face_label}",
        (42, 36),
        scale=0.62,
        colour=UI_WHITE,
        thickness=2,
    )
    _put(
        vis,
        f"mode: {auto_text}    face: {face_idx}/{face_count}    min tags: {min_expected}",
        (24, 61),
        scale=0.48,
        colour=(223, 232, 238),
    )
    _put(
        vis,
        f"seen: {_format_ids(det_ids)}    expected seen: {_format_ids(overlap)}",
        (24, 83),
        scale=0.48,
        colour=(223, 232, 238) if ok else (108, 219, 255),
        thickness=1 if ok else 2,
    )

    footer = "Enter save   Left/Right face   o object   a auto   h help   q/Esc quit"
    footer_h = 36
    y0 = max(0, h - footer_h - 10)
    _fill_alpha(vis, (10, y0), (min(w - 10, 780), y0 + footer_h), UI_DARK, 0.70)
    _put(vis, footer, (24, y0 + 24), scale=0.48, colour=UI_WHITE)

    if state.get("save_warn"):
        warn = "Tags are missing. Press Enter again within 3 seconds to force save."
        (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x0 = max(10, (w - tw) // 2 - 12)
        y1 = y0 - 12
        _fill_alpha(vis, (x0, y1 - 30), (min(w - 10, x0 + tw + 24), y1 + 4), UI_AMBER, 0.90)
        _put(vis, warn, (x0 + 12, y1 - 8), scale=0.55, colour=UI_DARK, thickness=2)

    return vis


def _format_ids(ids: List[int]) -> str:
    if not ids:
        return "-"
    if len(ids) <= 8:
        return ",".join(str(tag_id) for tag_id in ids)
    return ",".join(str(tag_id) for tag_id in ids[:8]) + f" +{len(ids) - 8}"


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(4, (max_chars - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


# Select the best face based on overlap of expected vs detected tag IDs
def best_face_by_overlap(faces: List[Dict], det_ids: Set[int], prev_idx: int | None = None) -> Tuple[Optional[int], int]:
    """Return (best_index, best_overlap_count). Tie-broken by prev_idx and tag count."""
    if not faces:
        return None, 0
    best_i, best_k, best_total = None, -1, -1
    for i, f in enumerate(faces):
        exp = set(f["tag_ids"])
        k = len(exp.intersection(det_ids))
        tot = len(exp)
        key = (k, tot, 1 if (prev_idx is not None and i == prev_idx) else 0)
        if key > (best_k, best_total, -1):
            best_i, best_k, best_total = i, k, tot
    return best_i, best_k


def main(argv=None):
    # Parse command-line arguments for capture settings
    ap = argparse.ArgumentParser(
        prog="posetag-capture-face",
        description="Capture wide shots per face for later annotation",
    )
    ap.add_argument("--project_root", type=Path, default=None,
                    help="Root for boards/shots/objects/datasets (default: resolver/env/config).")
    ap.add_argument("--calib", default="calib_color.yaml",
                    help="Camera intrinsics YAML (fx, fy, cx, cy, dist).")
    ap.add_argument("--registry", default=None,
                    help="Path to boards/tag_registry.yaml (default: <project_root>/boards/tag_registry.yaml).")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save shots (default: <project_root>/shots).")
    ap.add_argument("--object_name", default=None,
                    help="Either full (e.g. connection_plate_white_sideA) or base (e.g. connection_plate_white).")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min_expected", type=int, default=1, help="min expected tags to see from the chosen face")
    ap.add_argument("--gallery", action="store_true", default=True, help="show thumbnails in panel")
    ap.add_argument("--panel_w", type=int, default=420, help="info panel width (right side)")
    ap.add_argument("--recent_w", type=int, default=320, help="right panel width for last-saved preview")
    ap.add_argument("--layout", choices=["flat", "by_object", "by_object_side", "split_type"],
                    default="by_object_side", help="folder layout for saved captures")
    ap.add_argument("--raw_dir", default=None, help="when layout=split_type: raw images dir")
    ap.add_argument("--ann_dir", default=None, help="when layout=split_type: annotated images dir")
    ap.add_argument("--meta_dir", default=None, help="when layout=split_type: JSON metadata dir")
    ap.add_argument("--manifest", default=None, help="optional CSV file to append per-capture rows")
    ap.add_argument("--log_file", default=None,
                    help="override log path (default: <project_root>/logs/capture_face.log)")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="log verbosity")
    ap.add_argument("--log_console", action="store_true", help="also echo logs to stderr")

    # unified source options
    ap.add_argument("--source", choices=["realsense", "opencv", "video"], default="realsense",
                    help="Capture source: Intel RealSense (default), OpenCV webcam, or a video file")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index when --source=opencv")
    ap.add_argument("--video", type=str, default=None, help="Video path when --source=video")
    args = ap.parse_args(argv)

    # ---------- validate project inputs before opening hardware or writing outputs ----------
    try:
        validate_capture_source_args(args.source, args.video, capture_source.rs)
        pr = preview_capture_project_root(args.project_root)
        calibration = load_capture_calibration(resolve_capture_calibration_path(pr, args.calib))
        paths = prepare_capture_paths(
            project_root=pr,
            calib_path=calibration.path,
            registry=args.registry,
            out_dir=args.out_dir,
            manifest=args.manifest,
            log_file=args.log_file,
            layout=args.layout,
            raw_dir=args.raw_dir,
            ann_dir=args.ann_dir,
            meta_dir=args.meta_dir,
        )
        registry = load_capture_registry(paths.registry_path)
        registered_faces = load_registered_faces(
            registry,
            project_root=paths.project_root,
            registry_path=paths.registry_path,
        )
        initial_selection = select_initial_faces(args.object_name, registered_faces)
        Detector = load_apriltag_detector_class()
        det = create_apriltag_detector(args.family, detector_class=Detector)
    except CaptureFaceError as exc:
        raise SystemExit(str(exc)) from exc

    # unified source; keep this before output-directory creation so unreadable
    # videos fail without leaving capture artifacts.
    read_frame, stop, _ = capture_source.open_source(
        args.source, cam=args.cam, video=args.video, width=args.width, height=args.height, fps=args.fps, warmup=10
    )

    try:
        activate_capture_project_root(paths.project_root)
        create_capture_output_dirs(paths, layout=args.layout)
        logger = init_project_logger(paths.log_path, level=args.log_level, console=args.log_console)
    except Exception:
        stop()
        raise

    # ---------- logging & announce ----------
    logger.info("Using project root: %s", paths.project_root)
    logger.info("Using calib: %s", str(paths.calib_path))
    logger.info("Using registry: %s", str(paths.registry_path))
    logger.info("Shots will be saved to: %s", str(paths.out_dir))

    # ---------- UI state ----------
    state = {
        "object_base": None, "faces": [], "face_idx": 0, "auto_side": False,
        "detected_ids": set(), "validation_ok": True, "thumbs": [],
        "save_warn": False, "save_warn_t0": 0.0,
    }

    bases = unique_bases(registered_faces)

    # Initial object from CLI
    if initial_selection:
        state["object_base"] = initial_selection.object_base
        state["faces"] = list(initial_selection.faces)
        state["auto_side"] = initial_selection.auto_side
        state["face_idx"] = 0

    mode, pick_sel, gallery_on, show_help = ('preview' if state["object_base"] else 'pick_object', 0, bool(args.gallery), False)
    WINDOW_NAME = "Capture Face"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width + args.panel_w + args.recent_w, args.height)

    try:
        first = True
        # Main capture loop: read frames, detect tags, update UI, handle key presses
        while True:
            c = read_frame()
            if c is None:
                if args.source == "video":
                    logger.info("Video ended before another face shot was saved; exiting cleanly.")
                    return 0
                continue

            # Sync actual width/height for non-hardware-timestamped sources
            if first and args.source != "realsense":
                args.height, args.width = int(c.shape[0]), int(c.shape[1])
                first = False

            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            dd = det.detect(g, estimate_tag_pose=False)

            det_ids = {int(d.tag_id) for d in dd}
            state["detected_ids"] = det_ids

            detection_vis = draw_detections(c, dd)
            face = None

            # Resolve face from detections + selection
            if state["faces"]:
                if state["auto_side"]:
                    best_i, _ = best_face_by_overlap(state["faces"], det_ids, state.get("face_idx"))
                    if best_i is not None:
                        state["face_idx"] = best_i
                face = state["faces"][state["face_idx"]]
            state["face"] = face

            # Validate expected tags
            ok = True
            if face:
                expected = set(face["tag_ids"])
                if expected:
                    ok = len(expected.intersection(det_ids)) >= max(1, args.min_expected)
            state["validation_ok"] = ok

            vis = draw_capture_overlay(
                detection_vis,
                face=face,
                state=state,
                ok=ok,
                min_expected=args.min_expected,
            )

            # Panels
            if mode == 'pick_object':
                panel_info = object_picker_panel(args.height, args.panel_w, bases, pick_sel)
            else:
                panel_info = make_info_panel(args.height, args.panel_w, state, gallery_on=gallery_on)
                if show_help:
                    overlay = np.full((args.height, args.panel_w, 3), 230, np.uint8)
                    text_lines(overlay, [
                        "Help:",
                        "ENTER: Save (twice within 3s to force if not OK)",
                        "a: Toggle auto face/side (when base given)",
                        "Left/Right or f: Cycle faces (manual mode)",
                        "o: Pick object (on-screen picker)",
                        "g: Toggle gallery thumbnails",
                        "h: Toggle this help",
                        "q/ESC: Quit",
                    ], y0=40)
                    panel_info = cv2.addWeighted(panel_info, 0.1, overlay, 0.9, 0)

            panel_recent = make_recent_panel(args.height, args.recent_w, state.get("last_saved_ann"), gallery_on)
            combo = cv2.hconcat([vis, panel_info, panel_recent])
            cv2.imshow(WINDOW_NAME, combo)

            k = cv2.waitKeyEx(1) or 0
            ENTER_KEYS, CANCEL_KEYS = {13, 10}, {27}
            UP_KEYS, DOWN_KEYS = {KEY_UP, ord('w'), ord('k')}, {KEY_DOWN, ord('s'), ord('j')}

            # Object picker
            if mode == 'pick_object':
                if k in UP_KEYS:   pick_sel = (pick_sel - 1) % max(1, len(bases))
                elif k in DOWN_KEYS: pick_sel = (pick_sel + 1) % max(1, len(bases))
                elif ord('1') <= k <= ord('9'):
                    idx = (k - ord('1'));  pick_sel = idx if idx < len(bases) else pick_sel
                elif k in ENTER_KEYS:
                    base = bases[pick_sel] if bases else None
                    if base:
                        state["object_base"] = base
                        state["faces"] = faces_for_base(base, registered_faces)
                        state["face_idx"] = 0
                        state["auto_side"] = True
                    mode = 'preview'
                elif k in CANCEL_KEYS:
                    if state["object_base"] is None:
                        break
                    mode = 'preview'
                elif k == ord('q'):     break
                continue

            # Preview
            if k in (ord('q'),) or k in CANCEL_KEYS: break
            if k == ord('h'): show_help = not show_help
            elif k == ord('g'): gallery_on = not gallery_on
            elif k == ord('o'):
                pick_sel = bases.index(state["object_base"]) if state["object_base"] in bases else 0
                mode = 'pick_object'
            elif k == ord('a'):
                state["auto_side"] = not state["auto_side"] if state["faces"] else False
            elif k in {KEY_RIGHT, ord('f'), ord('n'), ord(']')}:
                cycle_face_selection(state, 1)
            elif k in {KEY_LEFT, ord('p'), ord('[')}:
                cycle_face_selection(state, -1)
            elif k in ENTER_KEYS:
                if face is None:
                    continue
                if not ok and set(face["tag_ids"]):
                    now = time.time()
                    if not state["save_warn"] or (now - state["save_warn_t0"] > 3.0):
                        state["save_warn"], state["save_warn_t0"] = True, now
                        continue
                state["save_warn"] = False
              
                # Generate timestamp and determine save paths based on layout.
                ts = capture_timestamp()
                obj_full = face["object"]
                shot_paths = build_shot_paths(
                    layout=args.layout,
                    out_dir=paths.out_dir,
                    object_full=obj_full,
                    timestamp=ts,
                    raw_dir=paths.raw_dir,
                    ann_dir=paths.ann_dir,
                    meta_dir=paths.meta_dir,
                )
                meta = build_capture_metadata(
                    face=face,
                    shot_paths=shot_paths,
                    detected_tag_ids=det_ids,
                    validation_ok=ok,
                    auto_face=state["auto_side"],
                    frame_shape=c.shape,
                    camera_params=calibration.camera_params,
                    timestamp=ts,
                )
                try:
                    write_capture_outputs(
                        raw_frame=c,
                        annotated_frame=vis,
                        metadata=meta,
                        shot_paths=shot_paths,
                        manifest_path=paths.manifest_path,
                        image_writer=cv2.imwrite,
                    )
                except CaptureFaceError as exc:
                    raise SystemExit(str(exc)) from exc
                state["last_saved_ann"] = vis.copy()

                logger.info(f"[+] Saved {shot_paths.raw_path}")
                logger.info(f"[+] Saved {shot_paths.ann_path}")
                logger.info(f"[+] Saved {shot_paths.meta_path}")

                # Update thumbnails for gallery panel
                try:
                    tw = min(320, vis.shape[1]); th = int(vis.shape[0] * tw / vis.shape[1])
                    state["thumbs"].append(cv2.resize(vis, (tw, th)))
                    if len(state["thumbs"]) > 12: state["thumbs"] = state["thumbs"][-12:]
                except Exception:
                    pass

    finally:
        stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
