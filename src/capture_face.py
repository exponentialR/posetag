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
ENTER save (double-press to force) | o object picker | a auto-side | f cycle faces |
g panels | h help | q / ESC quit
"""

from __future__ import annotations
import argparse, os, json, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import cv2

from utils.capture_source import open_source
from utils.intel_realsense_utils import load_calib, _face_label
from utils.capture_utils import (
    draw_detections, make_info_panel, make_recent_panel, text_lines,
    load_registry, unique_bases, faces_for_base, faces_for_object,
    parse_base_and_side, ensure_dir, timestamp, _append_manifest
)
from utils.project_config import resolve_project_root, ensure_project_dirs
from utils.logger import init_project_logger

try:
    from pupil_apriltags import Detector
except Exception as e:
    raise SystemExit("Please install pupil-apriltags: pip install pupil-apriltags") from e

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 2490368, 2621440, 2424832, 2555904


def object_picker_panel(h: int, w: int, bases: List[str], sel: int,
                        hint: str = "Select object (Enter):") -> np.ndarray:
    pan = np.full((h, w, 3), 240, np.uint8)
    cv2.putText(pan, hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    max_show = min(12, len(bases))
    start = max(0, min(sel - max_show // 2, len(bases) - max_show))
    end = min(len(bases), start + max_show)

    y = 60
    for i in range(start, end):
        s = bases[i]
        col = (0, 0, 0) if i != sel else (0, 50, 200)
        thick = 1 if i != sel else 2
        prefix = "  " if i != sel else "> "
        cv2.putText(pan, prefix + s, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, thick, cv2.LINE_AA)
        y += 26

    tips = "Move: ↑/↓ or W/S (K/J)   Select: Enter   Cancel: Esc   Jump: 1–9"
    cv2.putText(pan, tips, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return pan


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


def main():
    ap = argparse.ArgumentParser("Capture wide shots per face for later annotation")
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
    ap.add_argument("--panel_w", type=int, default=560, help="info panel width (right side)")
    ap.add_argument("--recent_w", type=int, default=480, help="right panel width for last-saved preview")
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
    args = ap.parse_args()

    # ---------- project root + path resolution ----------
    pr = Path(resolve_project_root(args.project_root))
    ensure_project_dirs(pr)

    def _under_pr(p: Optional[str | Path], default_rel: Optional[str]) -> Path:
        if p is None:
            return (pr / default_rel) if default_rel else pr
        pth = Path(p)
        return pth if pth.is_absolute() else (pr / pth)

    # calib: prefer given path; else <pr>/calib/calib_color.yaml or <pr>/calib_color.yaml
    calib_path = Path(args.calib)
    if not calib_path.exists():
        for cand in (pr / "calib" / "calib_color.yaml", pr / "calib_color.yaml"):
            if cand.exists():
                calib_path = cand
                break

    registry_path = _under_pr(args.registry, "boards/tag_registry.yaml")
    out_dir = _under_pr(args.out_dir, "shots")

    # logs
    logs_dir = pr / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = (Path(args.log_file) if args.log_file and Path(args.log_file).is_absolute()
                else logs_dir / (Path(args.log_file).name if args.log_file else "capture_face.log"))

    # split-type dirs
    if args.layout == "split_type":
        raw_dir  = _under_pr(args.raw_dir,  str(out_dir / "images"))
        ann_dir  = _under_pr(args.ann_dir,  str(out_dir / "ann"))
        meta_dir = _under_pr(args.meta_dir, str(out_dir / "meta"))
        for d in (raw_dir, ann_dir, meta_dir): d.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir = ann_dir = meta_dir = None
        out_dir.mkdir(parents=True, exist_ok=True)

    # manifest
    manifest_path = (pr / "shots" / "manifest.csv") if args.manifest is None else (
        Path(args.manifest) if Path(args.manifest).is_absolute() else (pr / args.manifest)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- logging & announce ----------
    logger = init_project_logger(log_path, level=args.log_level, console=args.log_console)
    logger.info("Using project root: %s", pr)
    logger.info("Using calib: %s", str(calib_path))
    logger.info("Using registry: %s", str(registry_path))
    logger.info("Shots will be saved to: %s", str(out_dir))

    # ---------- data & detector ----------
    (fx, fy, cx, cy), K = load_calib(str(calib_path))
    reg = load_registry(str(registry_path))
    det = Detector(families=args.family, nthreads=4, quad_decimate=1.0, refine_edges=True)

    # unified source
    read_frame, stop, _ = open_source(
        args.source, cam=args.cam, video=args.video, width=args.width, height=args.height, fps=args.fps, warmup=10
    )

    # ---------- UI state ----------
    state = {
        "object_base": None, "faces": [], "face_idx": 0, "auto_side": False,
        "detected_ids": set(), "validation_ok": True, "thumbs": [],
        "save_warn": False, "save_warn_t0": 0.0,
    }

    bases = unique_bases(reg)

    # Initial object from CLI
    cli_obj = args.object_name.strip() if args.object_name else None
    if cli_obj:
        base, side = parse_base_and_side(cli_obj)
        state["object_base"] = base
        state["faces"] = faces_for_base(base, reg, pr) if side is None else faces_for_object(cli_obj, reg, pr)
        state["auto_side"] = (side is None)
        state["face_idx"] = 0

    mode, pick_sel, gallery_on, show_help = ('preview' if state["object_base"] else 'pick_object', 0, bool(args.gallery), False)
    WINDOW_NAME = "Capture Face"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width + args.panel_w + args.recent_w, args.height)

    try:
        first = True
        while True:
            c = read_frame()
            if c is None:
                continue

            # Sync actual width/height for non-hardware-timestamped sources
            if first and args.source != "realsense":
                args.height, args.width = int(c.shape[0]), int(c.shape[1])
                first = False

            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            dd = det.detect(g, estimate_tag_pose=False)

            det_ids = {int(d.tag_id) for d in dd}
            state["detected_ids"] = det_ids

            vis = draw_detections(c, dd)
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

            # Left overlay
            label = _face_label(face) if face else "(unresolved)"
            cv2.putText(vis, f"Face: {label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if face:
                expected = set(face["tag_ids"])
                inter = sorted(list(expected.intersection(det_ids)))
                msg = f"seen={sorted(list(det_ids))}  exp&seen={inter}"
                col = (0, 200, 0) if ok else (0, 0, 255)
            else:
                msg = f"seen={sorted(list(det_ids))}"
                col = (0, 200, 200)
            cv2.putText(vis, msg, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

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
                        "f: Cycle faces (auto OFF)",
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
                        state["faces"] = faces_for_base(base, reg, pr)
                        state["face_idx"] = 0
                        state["auto_side"] = True
                    mode = 'preview'
                elif k in CANCEL_KEYS: mode = 'preview'
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
            elif k == ord('f'):
                if state["faces"] and not state["auto_side"]:
                    state["face_idx"] = (state["face_idx"] + 1) % len(state["faces"])
            elif k in ENTER_KEYS:
                if face is None:
                    continue
                if not ok and set(face["tag_ids"]):
                    now = time.time()
                    if not state["save_warn"] or (now - state["save_warn_t0"] > 3.0):
                        state["save_warn"], state["save_warn_t0"] = True, now
                        continue
                state["save_warn"] = False

                ts = timestamp()
                obj_full = face["object"] if face else (state["object_base"] or "unknown")
                base_name, side = parse_base_and_side(obj_full)
                side_code = (side or "unresolved").upper()

                if args.layout == "flat":
                    save_root, file_stub = out_dir, f"{obj_full}_{ts}"
                elif args.layout == "by_object":
                    save_root, file_stub = out_dir / base_name, f"{base_name}_side{side_code}_{ts}"
                elif args.layout == "by_object_side":
                    save_root, file_stub = out_dir / base_name / f"side{side_code}", f"{base_name}_side{side_code}_{ts}"
                else:  # split_type
                    save_root, file_stub = None, f"{base_name}_side{side_code}_{ts}"

                raw_path = str((save_root / f"{file_stub}_raw.png") if save_root else (raw_dir / f"{file_stub}_raw.png"))
                ann_path = str((save_root / f"{file_stub}_ann.png") if save_root else (ann_dir / f"{file_stub}_ann.png"))
                meta_path = str((save_root / f"{file_stub}_meta.json") if save_root else (meta_dir / f"{file_stub}_meta.json"))
                if save_root: ensure_dir(save_root)

                cv2.imwrite(raw_path, c)
                cv2.imwrite(ann_path, vis)
                state["last_saved_ann"] = vis.copy()

                meta = {
                    "object_base": base_name,
                    "object_full": obj_full,
                    "side": side_code,
                    "face_yaml": face["yaml"] if face else None,
                    "expected_tag_ids": sorted(list(set(face["tag_ids"]))) if face else [],
                    "detected_tag_ids": sorted(list(det_ids)),
                    "validation_ok": bool(ok),
                    "auto_face": bool(state["auto_side"]),
                    "image": {
                        "path_raw": raw_path, "path_ann": ann_path, "path_meta": meta_path,
                        "width": int(c.shape[1]), "height": int(c.shape[0]),
                    },
                    "camera": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
                    "timestamp": ts,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info(f"[+] Saved {raw_path}")
                logger.info(f"[+] Saved {ann_path}")
                logger.info(f"[+] Saved {meta_path}")
                _append_manifest(str(manifest_path), meta)

                # thumbnails
                try:
                    tw = min(320, vis.shape[1]); th = int(vis.shape[0] * tw / vis.shape[1])
                    state["thumbs"].append(cv2.resize(vis, (tw, th)))
                    if len(state["thumbs"]) > 12: state["thumbs"] = state["thumbs"][-12:]
                except Exception:
                    pass

    finally:
        stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
