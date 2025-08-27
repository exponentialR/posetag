"""
capture_isc_face.py — Face-shot capture for later T_board_object annotation
Author: Samuel Adebayo

Summary
-------
Capture wide shots per object face with AprilTag overlays and metadata for
subsequent T_board_object (board→object) computation.

What’s new (rev)
----------------
- Enter either a full face (e.g. connection_plate_white_sideA) or just a base
  (e.g. connection_plate_white). If a base is given, the face/side is
  auto-deduced from live detections using boards/tag_registry.yaml.
- No terminal prompts while preview is running. Press 'o' to open the in-window
  object picker; use ↑/↓/W/S/K/J to navigate; Enter selects; Esc cancels.
- Two right-hand panels:
    • Info panel: live status (seen vs expected IDs, validation), recent thumbnails,
      and concise instructions.
    • Last-saved panel: shows the most recently saved annotated frame.
  Toggle panels with 'g'. Panels are UI-only and are NOT written into the
  annotated image.
- Safety save: if expected tags are not seen, press Enter twice within 3 s to force save.
- Resizable top-level window; the camera frame remains at the requested resolution.

Saving layout
-------------
Default: by object and side
    boards/shots/<object_base>/side<SideLetter>/
      <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_{raw,ann}.png
      <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_meta.json

Filenames are de-duplicated (no repeated “…_sideA_sideA_…”). Optional CSV
manifest appends one row per save with paths, tags, OK flag, and intrinsics.

Keys (main preview)
-------------------
  ENTER  -> save (double-press within 3 s to force if validation fails)
  o      -> object picker (in-window)
  a      -> toggle auto face/side (when a base object is chosen)
  f      -> cycle faces (only when auto is OFF)
  g      -> toggle gallery / last-saved panels
  h      -> help overlay
  q or ESC -> quit

CLI highlights
--------------
  --calib PATH            Camera intrinsics YAML (fx, fy, cx, cy, dist)
  --object_name NAME      Base or full face name
  --width/--height/--fps  Stream settings (e.g., 640×480 @ 30 fps)
  --panel_w INT           Centre info-panel width (default ~560)
  --recent_w INT          Right last-saved panel width (default ~480)
  --layout {flat,by_object,by_object_side,split_type}
                         Saving layout (default: by_object_side)
  --manifest PATH         Append a CSV manifest per save
"""


from __future__ import annotations
import argparse, os, sys, json, time, datetime, re
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

from utils.intel_realsense_utils import start_rs, get_frames_with_retry, load_calib, _face_label
from utils.capture_utils import (
    draw_detections, make_info_panel, make_recent_panel, text_lines,
    load_registry, unique_bases, faces_for_base, faces_for_object,
    parse_base_and_side, ensure_dir, timestamp, _append_manifest
)

try:
    from pupil_apriltags import Detector
except Exception as e:
    raise SystemExit("Please install pupil-apriltags: pip install pupil-apriltags") from e

repo_root = Path(__file__).resolve().parents[1]


KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 2490368, 2621440, 2424832, 2555904

# ---------- path helpers ----------


def compose(vis: np.ndarray, panel: np.ndarray) -> np.ndarray:
    if vis.shape[0] != panel.shape[0]:
        # match heights
        panel = cv2.resize(panel, (panel.shape[1], vis.shape[0]))
    return cv2.hconcat([vis, panel])

# ---------- selection UI ----------


def object_picker_panel(h: int, w: int, bases: List[str], sel: int,
                        hint: str="Select object (Enter):") -> np.ndarray:
    pan = np.full((h, w, 3), 240, np.uint8)
    cv2.putText(pan, hint, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    max_show = min(12, len(bases))
    start = max(0, min(sel - max_show // 2, len(bases) - max_show))
    end = min(len(bases), start + max_show)

    y = 60
    for i in range(start, end):
        s = bases[i]
        col = (0, 0, 0) if i != sel else (0, 50, 200)
        thick = 1 if i != sel else 2
        prefix = "  " if i != sel else "> "
        cv2.putText(pan, prefix + s, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, thick, cv2.LINE_AA)
        y += 26

    tips = "Move: ↑/↓ or W/S (K/J)   Select: Enter   Cancel: Esc   Jump: 1–9"
    cv2.putText(pan, tips, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return pan


# ---------- face selection logic ----------
def best_face_by_overlap(faces: List[Dict], det_ids: Set[int], prev_idx: int|None=None) -> Tuple[Optional[int], int]:
    """Return (best_index, best_overlap_count). Tie broken by prev_idx and tag count."""
    if not faces:
        return None, 0
    best_i, best_k, best_total = None, -1, -1
    for i, f in enumerate(faces):
        exp = set(f["tag_ids"])
        k = len(exp.intersection(det_ids))
        tot = len(exp)
        key = (k, tot, 1 if (prev_idx is not None and i == prev_idx) else 0)
        # maximise k, then tot, then prefer staying on same
        if key > (best_k, best_total, -1):
            best_i, best_k, best_total = i, k, tot
    return best_i, best_k

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Capture wide shots per face for later annotation")
    ap.add_argument("--calib", default="calib_color.yaml")
    ap.add_argument("--registry", default=None,
                    help="Path to boards/tag_registry.yaml (defaults to repo_root/boards/tag_registry.yaml)")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save shots; defaults to repo_root/boards/shots")
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
                                         default = "by_object_side", help = "folder layout for saved captures")
    ap.add_argument("--raw_dir", default=None, help="when layout=split_type: raw images dir")
    ap.add_argument("--ann_dir", default=None, help="when layout=split_type: annotated images dir")
    ap.add_argument("--meta_dir", default=None, help="when layout=split_type: JSON metadata dir")
    ap.add_argument("--manifest", default="boards/shots/manifest.csv", help="optional CSV file to append per-capture rows")

    args = ap.parse_args()

    calib_path = Path(args.calib)
    if not calib_path.exists():
        alt = repo_root / "calib_color.yaml"
        if alt.exists():
            args.calib = str(alt)

    if args.registry is None:
        cand = repo_root / "boards" / "tag_registry.yaml"
        args.registry = str(cand)
    else:
        rp = Path(args.registry)
        if not rp.exists():
            cand = repo_root / args.registry
            if cand.exists():
                args.registry = str(cand)

    if args.out_dir is None:
        args.out_dir = str(repo_root / "boards" / "shots")

    # -- layout initialisation ---
    print(f"[i] Layout: {args.layout}")

    if args.layout == "split_type":
        # Fill defaults safely
        args.raw_dir = args.raw_dir or str(Path(args.out_dir) / "images")
        args.ann_dir = args.ann_dir or str(Path(args.out_dir) / "images")
        args.meta_dir = args.meta_dir or str(Path(args.out_dir) / "meta")

        for d in (args.raw_dir, args.ann_dir, args.meta_dir):
            if d:
                os.makedirs(d, exist_ok=True)
            print(f"[i] Split dirs -> raw={args.raw_dir}, ann={args.ann_dir}, meta={args.meta_dir}")
    else:
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"[i] Using calib: {args.calib}")
    print(f"[i] Using registry: {args.registry}")
    print(f"[i] Shots will be saved to: {args.out_dir}")



    (fx, fy, cx, cy), K = load_calib(args.calib)
    reg = load_registry(args.registry)
    ensure_dir(args.out_dir)

    det = Detector(families=args.family, nthreads=4, quad_decimate=1.0, refine_edges=True)

    # RealSense colour stream
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    start_rs(pipe, cfg, warmup=15)

    # ---------- UI state ----------
    state = {
        "object_base": None,   # str
        "faces": [],           # list of face dicts (maybe multi-sides)
        "face_idx": 0,         # current index in faces
        "auto_side": False,    # auto resolve face from detections
        "detected_ids": set(), # live set of tag IDs
        "validation_ok": True,
        "thumbs": [],          # recent thumbnails
        "save_warn": False,
        "save_warn_t0": 0.0,
    }

    # Prepare object list
    bases = unique_bases(reg)

    # Initial object from CLI
    cli_obj = args.object_name.strip() if args.object_name else None
    if cli_obj:
        base, side = parse_base_and_side(cli_obj)
        state["object_base"] = base
        state["faces"] = faces_for_base(base, reg, repo_root) if side is None else faces_for_object(cli_obj, reg, repo_root)
        state["auto_side"] = (side is None)
        state["face_idx"] = 0

    # Modes: 'preview' or 'pick_object'
    mode = 'preview' if state["object_base"] else 'pick_object'
    pick_sel = 0
    gallery_on = bool(args.gallery)
    show_help = False

    WINDOW_NAME = "Capture Objects' Face"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # user can resize if they want
    cv2.resizeWindow(WINDOW_NAME, args.width + args.panel_w + args.recent_w, args.height)


    try:
        while True:
            frames = get_frames_with_retry(pipe, cfg, max_retries=2, timeout_ms=2000, do_hw_reset=True)
            c = np.asanyarray(frames.get_color_frame().get_data())
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            dd = det.detect(g, estimate_tag_pose=False)

            det_ids = {int(d.tag_id) for d in dd}
            state["detected_ids"] = det_ids

            vis = draw_detections(c, dd)
            face = None

            # Resolve face
            if state["faces"]:
                if state["auto_side"]:
                    best_i, overlap_k = best_face_by_overlap(state["faces"], det_ids, state.get("face_idx"))
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

            # Overlay status lines on the left image
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
                    alpha = 0.9
                    panel_info = cv2.addWeighted(panel_info, 1 - alpha, overlay, alpha, 0)
            panel_recent = make_recent_panel(
                args.height, args.recent_w, state.get("last_saved_ann"), gallery_on
            )
            combo = cv2.hconcat([vis, panel_info, panel_recent])
            cv2.imshow(WINDOW_NAME, combo)

            k = cv2.waitKeyEx(1)
            if k == -1:
                k = 0

            ENTER_KEYS = {13, 10}  # CR/LF
            CANCEL_KEYS = {27}  # Esc (we keep 'q' for global quit in preview)
            UP_KEYS = {KEY_UP, ord('w'), ord('k')}
            DOWN_KEYS = {KEY_DOWN, ord('s'), ord('j')}

            if mode == 'pick_object':
                if k in UP_KEYS:
                    pick_sel = (pick_sel - 1) % max(1, len(bases))
                elif k in DOWN_KEYS:
                    pick_sel = (pick_sel + 1) % max(1, len(bases))
                elif ord('1') <= k <= ord('9'):  # quick jump
                    idx = (k - ord('1'))
                    if idx < len(bases):
                        pick_sel = idx
                elif k in ENTER_KEYS:
                    base = bases[pick_sel] if bases else None
                    if base:
                        state["object_base"] = base
                        state["faces"] = faces_for_base(base, reg, repo_root)
                        state["face_idx"] = 0
                        state["auto_side"] = True
                    mode = 'preview'
                elif k in CANCEL_KEYS:
                    mode = 'preview'
                elif k == ord('q'):  # allow quit from picker
                    break
                continue

            # -------- preview mode (global) --------
            if k in (ord('q'),):  # q quits from preview
                break
            if k in CANCEL_KEYS:  # Esc quits from preview
                break

            if k == ord('h'):
                show_help = not show_help
            elif k == ord('g'):
                gallery_on = not gallery_on
            elif k == ord('o'):
                pick_sel = bases.index(state["object_base"]) if state["object_base"] in bases else 0
                mode = 'pick_object'
            elif k == ord('a'):
                state["auto_side"] = not state["auto_side"] if state["faces"] else False
            elif k == ord('f'):
                if state["faces"] and not state["auto_side"]:
                    state["face_idx"] = (state["face_idx"] + 1) % len(state["faces"])
            elif k in ENTER_KEYS:
                # ENTER -> save (with 2-press safeguard if not ok)
                if face is None:
                    pass
                if not ok and face and set(face["tag_ids"]):
                    now = time.time()
                    if not state["save_warn"] or (now - state["save_warn_t0"] > 3.0):
                        state["save_warn"] = True
                        state["save_warn_t0"] = now
                        # first press just warns; need second press
                        continue
                # proceed save
                state["save_warn"] = False

                ts = timestamp()
                obj_full = (face["object"] if face else (state["object_base"] or "unknown"))
                base_name, side = parse_base_and_side(obj_full)
                side_code = (side or "unresolved").upper()

                if args.layout == "flat":
                    save_root = Path(args.out_dir)
                    file_stub = f"{obj_full}_{ts}"
                    raw_path = str(save_root / f"{file_stub}_raw.png")
                    ann_path = str(save_root / f"{file_stub}_ann.png")
                    meta_path = str(save_root / f"{file_stub}_meta.json")
                    ensure_dir(save_root)
                elif args.layout == "by_object":
                    save_root = Path(args.out_dir) / base_name
                    file_stub = f"{base_name}_side{side_code}_{ts}"
                    raw_path = str(save_root / f"{file_stub}_raw.png")
                    ann_path = str(save_root / f"{file_stub}_ann.png")
                    meta_path = str(save_root / f"{file_stub}_meta.json")
                    ensure_dir(save_root)
                elif args.layout == "by_object_side":
                    save_root = Path(args.out_dir) / base_name / f"side{side_code}"
                    file_stub = f"{base_name}_side{side_code}_{ts}"
                    raw_path = str(save_root / f"{file_stub}_raw.png")
                    ann_path = str(save_root / f"{file_stub}_ann.png")
                    meta_path = str(save_root / f"{file_stub}_meta.json")
                    ensure_dir(save_root)
                else:  # split_type
                    file_stub = f"{base_name}_side{side_code}_{ts}"
                    raw_path = str(Path(args.raw_dir) / f"{file_stub}_raw.png")
                    ann_path = str(Path(args.ann_dir) / f"{file_stub}_ann.png")
                    meta_path = str(Path(args.meta_dir) / f"{file_stub}_meta.json")

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
                        "path_raw": raw_path,
                        "path_ann": ann_path,
                        "path_meta": meta_path,
                        "width": int(args.width),
                        "height": int(args.height),
                    },
                    "camera": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
                    "timestamp": ts,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[+] Saved {raw_path}")
                print(f"[+] Saved {ann_path}")
                print(f"[+] Saved {meta_path}")
                if args.manifest:
                    _append_manifest(args.manifest, meta)

                # thumbnail
                try:
                    tw = min(320, vis.shape[1])
                    th = int(vis.shape[0] * tw / vis.shape[1])
                    thumb = cv2.resize(vis, (tw, th))
                    state["thumbs"].append(thumb)
                    if len(state["thumbs"]) > 12:
                        state["thumbs"] = state["thumbs"][-12:]
                except Exception:
                    pass


    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
