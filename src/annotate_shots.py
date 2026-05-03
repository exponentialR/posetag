"""
annotate_shots.py
=================
Interactive face annotator (single & batch) for the local project layout.

What it does
------------
Given a raw image and an AprilTag board definition, this tool:
1) Loads camera intrinsics (from the shot's meta JSON or shots/manifest.csv).
2) Detects AprilTags to estimate the board pose T_cam_board.
3) Lets you select 4 face corners (drag-quad or any-4 mode).
4) Solves PnP against the object's 3D keypoints to recover T_cam_object and
   computes T_board_object = inv(T_cam_board) @ T_cam_object.
5) Saves review images and a YAML per face with transforms & diagnostics.
6) Builds/updates faces/face_manifest.csv by scanning faces/*/*_T_board_object.yaml.
   - Adds new/updated faces (including tag_size_m).
   - Removes entries for YAMLs that were deleted offline.

Outputs
-------
- YAML: faces/<object>/sideA|B|C|D/<face_key>_T_board_object.yaml
- Images (next to the raw):
    *_ann.png     (projection of face & other keypoints)
    *_reproj.png  (all keypoints reprojection view)
- Faces manifest (auto-maintained):
    faces/face_manifest.csv with columns:
      timestamp, object, side, face_key, yaml_path, board_yaml, image, rms_px, tag_size_m

Project layout
--------------
<project_root>/
  meshes/...
  objects/<object>/keypoints.json
  faces/<object>/sideA|B|C|D/<face_key>_T_board_object.yaml     # OUTPUT HERE
  faces/face_manifest.csv                                       # auto-created/updated
  shots/
    manifest.csv                                                 # source of truth for shots
    <object>/<side>/..._raw.png, ..._ann.png, ..._meta.json
  boards/<object>_sideX.yaml                                    # AprilTag board files

Manifests
---------
Shots manifest (CSV) – minimally used fields:
  timestamp, object_base, object_full, side, face_yaml,
  path_raw, path_ann, path_meta, width, height, fx, fy, cx, cy
Required: face_yaml, path_raw (and/or path_meta), and camera intrinsics.

Faces manifest (CSV) – auto-maintained by this tool:
  timestamp (from YAML mtime), object, side, face_key, yaml_path,
  board_yaml, image, rms_px, tag_size_m

Modes
-----
1) Single shot
   python -m src.annotate_shots --shot /abs/path/to/..._raw.png

2) Batch: latest per face from shots manifest
   python -m src.annotate_shots --batch latest
   (optional) --object-filter connection_plate --side A --force --pts-type {quad|any}

3) Browse picker (interactive list of faces)
   python -m src.annotate_shots --browse
   (same toggles as batch: --object-filter, --side, --force, --pts-type ...)

4) Dry-run to see what would be annotated
   python -m src.annotate_shots --batch latest --dry-run

Keys
----
Browse view:
  ↑/W, ↓/S  select • ENTER annotate • A toggle pts-type • T check-scale
  C auto-correct-scale • F force overwrite • R reload • Q quit

Annotation view:
  - Drag a box then fine-tune 4 corners  (or click 4 corners if --pts-type any)
  - SHIFT = axis lock while dragging
  - ENTER = accept, r = reset, u = undo last corner
  - q/ESC = abort this shot, Q/X = quit-all (propagates to batch)

Notes
-----
- --check-tag-scale prints inter-tag scale ratio s; --auto-correct-scale divides
  T_cam_board translation by s when |s-1| > --scale-tol (default 0.02).
- UI auto-sizes and wraps/ellipsizes long paths/tips/log lines for readability.
- RMS reprojection error is shown during review; written into the YAML.
- Logs: <project_root>/logs/annotate_faces.log

Dependencies
------------
OpenCV, numpy, PyYAML, pupil-apriltags (for detect_tags), and project utils:
  utils.project_config: resolve_project_root, ensure_project_dirs
  utils.annotation_utils: detect_tags, load_board, se3, inv_se3, load_keypoints_fuzzy
"""


from __future__ import annotations
import argparse, csv, json, os, sys, itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2, yaml
from datetime import datetime

from utils.logger import init_project_logger
from utils.project_config import resolve_project_root, ensure_project_dirs
from utils.annotation_utils import (
    detect_tags, load_board, se3, inv_se3,
    load_keypoints_fuzzy
)

EXIT_QUIT_ALL = 99
import logging

UI_LOG: list[str] = []
log: logging.Logger  # global

_PILL = {
    "ok": ((224, 245, 228), (35, 120, 55)),
    "warn": ((241, 225, 201), (110, 85, 45)),
    "info": ((229, 238, 249), (70, 95, 160)),
}

UI_H = 720
UI_W_LEFT = 820  # was 860
UI_W_MID = 520  # was 380  ← Faces list wider
UI_W_RIGHT = 480
UI_WIN_H = 820
UI_WIN_W = UI_W_LEFT + UI_W_MID + UI_W_RIGHT


class UIBufferHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        UI_LOG.append(msg)
        if len(UI_LOG) > 200:
            del UI_LOG[:-200]


# ------- text wrapping / ellipsis helpers (ASCII only) -------
def _measure(text: str, scale=0.5, thk=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thk)
    return tw


def _ellipsize_end(text: str, max_w: int, scale=0.5, thk=1):
    if _measure(text, scale, thk) <= max_w:
        return text
    base = text
    while base and _measure(base + "...", scale, thk) > max_w:
        base = base[:-1]
    return (base + "...") if base else "..."


def _ellipsize_middle(text: str, max_w: int, scale=0.5, thk=1):
    if _measure(text, scale, thk) <= max_w:
        return text
    left, right = 0, 0
    while True:
        cand = text[:left] + "..." + text[len(text) - right:]
        if _measure(cand, scale, thk) <= max_w or (left + right) >= len(text):
            return cand if cand else "..."
        if (left <= right) and (left < len(text)):
            left += 1
        elif right < len(text):
            right += 1


def _wrap_to_width(text: str, max_w: int, scale=0.5, thk=1):
    """
    Word-wrap a single line into multiple lines that fit max_w.
    If a single token is too long (e.g., a path), ellipsize it in the middle.
    Returns a list of wrapped lines.
    """
    # Fast path
    if _measure(text, scale, thk) <= max_w:
        return [text]

    words = text.split(" ")
    if len(words) == 1:
        return [_ellipsize_middle(text, max_w, scale, thk)]

    out, cur = [], ""
    for w in words:
        token = w
        if _measure(token, scale, thk) > max_w:
            token = _ellipsize_middle(token, max_w, scale, thk)
        trial = (cur + " " + token).strip()
        if _measure(trial, scale, thk) <= max_w:
            cur = trial
        else:
            if cur:
                out.append(cur)
            # token itself may still be long; ensure it fits
            if _measure(token, scale, thk) <= max_w:
                cur = token
            else:
                out.append(_ellipsize_middle(token, max_w, scale, thk))
                cur = ""
    if cur:
        out.append(cur)
    return out


def _pill(text: str, kind: str = "info"):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad = 6
    img = np.full((th + pad * 2, tw + pad * 2, 3), _PILL[kind][0], np.uint8)
    cv2.putText(img, text, (pad, th + pad // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _PILL[kind][1], 1, cv2.LINE_AA)
    return img


def _right_column(project_root, it, args, height, w=480):
    info = [
        f"Project: {project_root}",
        "",
        f"Object: {it['object']}   side: {it['side']}",
        f"Face key: {it['face_key']}",
        f"Shot: {it['raw'].name if it['raw'] else '-'}",
        f"Board YAML: {Path(it['board']).name if it['board'] else '-'}",
        "",
        ("Status"),
        f"  ANN : {'OK' if it['ann_ok'] else 'MISSING'}" + (
            f"  (RMS {it['rms']:.2f}px)" if it.get('rms') is not None else ""),
        f"  KP  : {'OK' if it['kp_ok'] else 'missing'}",
        f"  BOARD: {'OK' if it['board_ok'] else 'missing'}",
        "",
        f"pts-type           : {args.pts_type}",
        f"check-tag-scale    : {'ON' if args.check_tag_scale else 'OFF'}",
        f"auto-correct-scale : {'ON' if args.auto_correct_scale else 'OFF'}",
        f"force overwrite    : {'ON' if args.force else 'OFF'}",
        "",
        "ENTER: annotate   A: pts   T: check-scale   C: auto-scale   F: force   R: reload   Q: quit",
        "", "Logs:",
    ]
    top_h = min(360, height)
    top = _text_panel(info, w=w, h=top_h)

    # rough estimate of how many log lines fit
    line_h = 20
    max_lines = max(1, (height - top_h) // line_h)
    tail = UI_LOG[-max_lines:] if UI_LOG else ["(no messages yet)"]
    bottom = _text_panel(tail, w=w, h=max(1, height - top_h))
    return np.vstack([top, bottom])


def _infer_object_and_side(shot_raw_path: Path, meta_json: dict | None, row) -> tuple[str, str]:
    obj = None
    side = None
    if meta_json:
        obj = meta_json.get("object_base") or meta_json.get("object")
        side = meta_json.get("side")
        full = meta_json.get("object_full")
        if (not side) and full and "_side" in full:
            side = full.split("_side")[-1][:1].upper()
    if not obj and row and getattr(row, "object_base", None):
        obj = row.object_base
    if not side and row and getattr(row, "side", None):
        side = row.side

    # Fallback: parse from shots/<object>/<side>/... path
    parts = list(shot_raw_path.parts)
    if "shots" in parts:
        i = parts.index("shots")
        if not obj and i + 1 < len(parts):
            obj = parts[i + 1]
        if not side and i + 2 < len(parts):
            side_token = parts[i + 2]  # e.g. "sideA"
            if side_token.lower().startswith("side") and len(side_token) >= 5:
                side = side_token[-1].upper()

    if not obj:
        raise SystemExit("[!] Could not infer object name (wanted meta.object_base or manifest.object_base).")
    if not side:
        side = "X"
    return obj, side.upper()


def _index_faces(project_root: Path, manifest: Path) -> List[dict]:
    rows = _read_manifest(manifest)
    latest = _latest_per_face(rows, object_filter=None, side=None)
    items = []
    for r in latest:
        obj = r.object_base
        side = r.side.upper()
        face_key = _face_key(obj, side)
        out_yaml = _out_yaml_path(project_root, obj, side)
        kp_ok = (project_root / "objects" / obj / "keypoints.json").exists()
        board_ok = Path(r.face_yaml).exists() if r.face_yaml else False

        rms = _load_rms_if_any(out_yaml) if out_yaml.exists() else None
        items.append({
            "object": obj, "side": side, "face_key": face_key, "ts": r.timestamp,
            "raw": Path(r.path_raw), "meta": Path(r.path_meta) if r.path_meta else None,
            "board": r.face_yaml, "out_yaml": out_yaml,
            "ann_ok": out_yaml.exists(), "kp_ok": kp_ok, "board_ok": board_ok,
            "rms": rms,
        })
    # stable order: objects then side
    items.sort(key=lambda d: (d["object"], d["side"]))
    return items


def _browse_and_annotate(project_root: Path, manifest: Path, args):
    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)

    items = _index_faces(project_root, manifest)
    if not items:
        log.debug("[i] Nothing in manifest to annotate.");
        return

    i = 0
    while True:
        it = items[i]
        left = _render_raw_thumb(it["raw"], target_h=UI_H, target_w=UI_W_LEFT)
        faces_w = UI_W_MID
        mid = _render_list_faces(left.shape[0], faces_w, "Faces (latest per manifest)", items, i)
        right = _right_column(project_root, it, args, height=left.shape[0], w=UI_W_RIGHT)

        strip = _hstack(left, mid, right)
        try:
            cv2.resizeWindow("Annotate", strip.shape[1], strip.shape[0])
        except Exception:
            pass

        cv2.imshow("Annotate", strip)

        k = cv2.waitKeyEx(60) & 0xFFFFFFFF
        if k in (ord('q'), 27):
            break
        elif k in (2490368, ord('w'), ord('W')):  # up
            i = (i - 1) % len(items)
        elif k in (2621440, ord('s'), ord('S')):  # down
            i = (i + 1) % len(items)
        elif k in (ord('a'), ord('A')):
            args.pts_type = "any" if args.pts_type == "quad" else "quad"
        elif k in (ord('t'), ord('T')):
            args.check_tag_scale = not args.check_tag_scale
        elif k in (ord('c'), ord('C')):
            args.auto_correct_scale = not args.auto_correct_scale
        elif k in (ord('f'), ord('F')):
            args.force = not args.force
        elif k in (ord('r'), ord('R')):
            items = _index_faces(project_root, manifest)
            i = min(i, len(items) - 1)
        elif k in (13, 10):  # ENTER → annotate selection
            if it["ann_ok"] and not args.force:
                log.info(f"[i] Skip (already annotated): {it['out_yaml']}")
                continue
            try:
                annotate_single_shot(
                    project_root, it["raw"],
                    pts_type=args.pts_type,
                    calib=args.calib,
                    check_tag_scale=args.check_tag_scale,
                    auto_correct_scale=args.auto_correct_scale,
                    scale_tol=args.scale_tol,
                )
                # refresh flags
                it["ann_ok"] = it["out_yaml"].exists()
                it["rms"] = _load_rms_if_any(it["out_yaml"]) if it["ann_ok"] else None
            except SystemExit as e:
                if str(e) == "Aborted.":
                    pass
                else:
                    log.error(e)
            except BaseException as e:
                log.error(f"[!] Error: {e}")
    cv2.destroyAllWindows()


def _estimate_tag_scale(det_by_id, T_board_tag):
    ratios = []
    ids = [k for k in det_by_id.keys() if k in T_board_tag]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pa = det_by_id[a].pose_t.reshape(3);
            pb = det_by_id[b].pose_t.reshape(3)
            d_cam = float(np.linalg.norm(pa - pb))
            ba = T_board_tag[a][:3, 3];
            bb = T_board_tag[b][:3, 3]
            d_board = float(np.linalg.norm(ba - bb))
            if d_board > 1e-9:
                ratios.append(d_cam / d_board)
    if not ratios:
        return 1.0, 0, 0
    ratios = np.asarray(ratios, float)
    return float(np.median(ratios)), int(len(ratios)), float(np.median(np.abs(ratios - np.median(ratios))))


# -------------------------- IO models --------------------------
@dataclass
class ShotRow:
    timestamp: str
    object_base: str
    object_full: str
    side: str  # "A" | "B" | "C" | "D"
    face_yaml: str
    path_raw: str
    path_ann: Optional[str]
    path_meta: Optional[str]
    width: Optional[int]
    height: Optional[int]
    fx: Optional[float]
    fy: Optional[float]
    cx: Optional[float]
    cy: Optional[float]


# -------------------------- Faces manifest (CSV) --------------------------

@dataclass
class FaceManifestRow:
    timestamp: str  # ISO time from YAML file mtime
    object: str
    side: str  # A|B|C|D
    face_key: str
    yaml_path: str  # path to *_T_board_object.yaml (prefer relative to project_root)
    board_yaml: str
    image: str
    rms_px: Optional[float]
    tag_size_m: Optional[float] = None  # <-- NEW (default keeps older call sites safe)


def _face_manifest_path(project_root: Path) -> Path:
    return project_root / "faces" / "face_manifest.csv"


def _side_from_face_key(face_key: str) -> str:
    m = re.search(r"_side([ABCD])", face_key, re.IGNORECASE)
    return m.group(1).upper() if m else "X"


def _scan_face_yamls(project_root: Path) -> List[FaceManifestRow]:
    rows: List[FaceManifestRow] = []
    root = project_root / "faces"
    if not root.exists():
        return rows

    for y in root.glob("*/*/*_T_board_object.yaml"):
        try:
            data = yaml.safe_load(open(y, "r"))
            obj = str(data.get("object", "")).strip()
            face_key = str(data.get("face_key", "")).strip() or (y.stem.replace("_T_board_object", ""))
            side = _side_from_face_key(face_key)
            board_yaml = str(data.get("board_yaml", "")).strip()
            image = str(data.get("image", "")).strip()
            rms = data.get("rms_px", None)

            # --- tag_size_m: prefer diagnostics, else load board
            tag_size_m = None
            diag = data.get("diagnostics", {}) or {}
            ts_mm = diag.get("board_tag_size_mm", None)
            if ts_mm is not None:
                try:
                    tag_size_m = float(ts_mm) / 1000.0
                except Exception:
                    tag_size_m = None
            if (tag_size_m is None) and board_yaml:
                by = Path(board_yaml)
                if not by.is_absolute():
                    by = (project_root / by).resolve()
                try:
                    _origin_id, ts_m, _T_board_tag = load_board(by)
                    tag_size_m = float(ts_m)
                except Exception:
                    pass

            ts = datetime.fromtimestamp(y.stat().st_mtime).isoformat(timespec="seconds")
            try:
                yaml_rel = str(y.relative_to(project_root))
            except Exception:
                yaml_rel = str(y)

            rows.append(FaceManifestRow(
                timestamp=ts, object=obj, side=side, face_key=face_key,
                yaml_path=yaml_rel, board_yaml=board_yaml, image=image,
                rms_px=(float(rms) if rms is not None else None),
                tag_size_m=tag_size_m
            ))
        except Exception as e:
            if 'log' in globals():
                log.warning(f"[faces-manifest] Skip bad YAML {y}: {e}")

    rows.sort(key=lambda r: (r.object, r.side, r.face_key))
    return rows


def _write_face_manifest(csv_path: Path, rows: List[FaceManifestRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "object", "side", "face_key", "yaml_path",
            "board_yaml", "image", "rms_px", "tag_size_m"  # <-- include
        ])
        for r in rows:
            w.writerow([
                r.timestamp, r.object, r.side, r.face_key, r.yaml_path,
                r.board_yaml, r.image,
                ("" if r.rms_px is None else f"{r.rms_px:.6f}"),
                ("" if r.tag_size_m is None else f"{r.tag_size_m:.6f}")  # <-- include
            ])


def _sync_face_manifest(project_root: Path) -> None:
    """Rebuild faces/face_manifest.csv from disk state (authoritative = YAML files)."""
    rows = _scan_face_yamls(project_root)
    path = _face_manifest_path(project_root)
    _write_face_manifest(path, rows)
    if 'log' in globals():
        log.info(f"[faces-manifest] Wrote {path} ({len(rows)} faces)")


def _read_manifest(path: Path) -> List[ShotRow]:
    rows: List[ShotRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(ShotRow(
                timestamp=d.get("timestamp", ""),
                object_base=d.get("object_base", ""),
                object_full=d.get("object_full", ""),
                side=str(d.get("side", "")).strip() or "",
                face_yaml=d.get("face_yaml", ""),
                path_raw=d.get("path_raw", ""),
                path_ann=d.get("path_ann") or None,
                path_meta=d.get("path_meta") or None,
                width=int(float(d["width"])) if d.get("width") else None,
                height=int(float(d["height"])) if d.get("height") else None,
                fx=float(d["fx"]) if d.get("fx") else None,
                fy=float(d["fy"]) if d.get("fy") else None,
                cx=float(d["cx"]) if d.get("cx") else None,
                cy=float(d["cy"]) if d.get("cy") else None,
            ))
    return rows


def _face_key(object_base: str, side_letter: str) -> str:
    # matches gen_keypoints faces naming
    return f"{object_base}_side{side_letter.upper()}"


def _out_yaml_path(project_root: Path, object_base: str, side_letter: str) -> Path:
    face_key = _face_key(object_base, side_letter)
    return project_root / "faces" / object_base / f"side{side_letter.upper()}" / f"{face_key}_T_board_object.yaml"


# -------------------------- Small UI helpers --------------------------

def _flush_keys(ms=120):
    import time
    deadline = time.time() + ms / 1000.0
    while time.time() < deadline:
        cv2.waitKey(5)


def _quit_all():
    try:
        cv2.destroyAllWindows()
        try:
            sys.stdout.flush()
        except:
            pass
        try:
            sys.stderr.flush()
        except:
            pass
    finally:
        os._exit(EXIT_QUIT_ALL)


def _pad(img, h, w):
    out = np.zeros((h, w, 3), np.uint8)
    if img is None: return out
    hh, ww = img.shape[:2]
    out[:hh, :ww] = img;
    return out


def _hstack(L, M=None, R=None):
    H = max(L.shape[0], 0 if M is None else M.shape[0], 0 if R is None else R.shape[0])
    l = _pad(L, H, L.shape[1]);
    m = _pad(M, H, M.shape[1] if M is not None else L.shape[1]);
    r = _pad(R, H, R.shape[1] if R is not None else 380)
    return np.hstack([l, m, r])


def _text_panel(lines: List[str], w=380, h=720, scale=0.5, thk=1, color=(240, 240, 240)):
    """Draws a list of strings with wrapping and end/middle ellipses as needed."""
    img = np.full((h, w, 3), 18, np.uint8)
    y = 24
    line_h = int(round(20 * max(0.8, scale / 0.5)))  # keep spacing nice if scale changes
    for ln in lines:
        wrapped = _wrap_to_width(ln, w - 18, scale, thk)
        for piece in wrapped:
            if y > h - 8:
                return img
            cv2.putText(img, piece, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thk, cv2.LINE_AA)
            y += line_h
    return img


def _help_panel(mode: str, h: int):
    if mode == "quad":
        lines = [
            "Annotate (quad)",
            "Drag box, then adjust 4 corners.",
            "SHIFT: axis lock",
            "ENTER: accept",
            "u: undo    r: reset",
            "q/ESC: abort   Q/X: quit-all",
        ]
    else:
        lines = [
            "Annotate (any-4)",
            "Click any 4 corners.",
            "SHIFT: axis lock",
            "ENTER: accept",
            "u: undo    r: reset",
            "q/ESC: abort   Q/X: quit-all",
        ]
    return _text_panel(lines, w=380, h=h)


def _show_dash(L, M, R, banner=""):
    strip = _hstack(L, M, R)
    try:
        cv2.resizeWindow("Annotate", strip.shape[1], strip.shape[0])
    except Exception:
        pass
    cv2.imshow("Annotate", strip)


# -------------------------- Annotation widgets --------------------------

def _load_dist_from_calib(calib_path: Optional[str]):
    if not calib_path: return np.zeros((1, 5), dtype=float)
    try:
        y = yaml.safe_load(open(calib_path, "r"))
        dc = y["distortion_coefficients"]
        return np.array([[dc["k1"], dc["k2"], dc["p1"], dc["p2"], dc.get("k3", 0.0)]], dtype=float)
    except Exception:
        return np.zeros((1, 5), dtype=float)


def _reproj(pts3d, T_cam_obj, K, img, draw=True, dist=None):
    if dist is None: dist = np.zeros((1, 5), float)
    R = T_cam_obj[:3, :3];
    t = T_cam_obj[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts3d, rvec, t, K, dist);
    uv = uv.reshape(-1, 2)
    if draw:
        vis = img.copy()
        for (u, v) in uv:
            cv2.circle(vis, (int(round(u)), int(round(v))), 2, (0, 0, 255), -1)
        return uv, vis
    return uv, None


def _collect_quad(image_bgr, mid_img=None, right_img=None):
    base = image_bgr.copy();
    h, w = base.shape[:2]
    right_help = _help_panel("quad", h)
    pts: List[Tuple[float, float]] = [];
    start = None;
    drag = None;
    hist = []
    cur = (0.0, 0.0);
    last_flags = 0

    def clamp(x, y):
        return max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))

    def near(a, b, r=10):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= r * r

    def snap(anchor, cand, flags):
        if not (flags & cv2.EVENT_FLAG_SHIFTKEY): return cand
        ax, ay = anchor;
        cx, cy = cand
        return (cx, ay) if abs(cx - ax) >= abs(cy - ay) else (ax, cy)

    def draw():
        vis = base.copy()
        if start is not None and not pts:
            x0, y0 = start;
            x1, y1 = cur
            tl = (min(x0, x1), min(y0, y1));
            tr = (max(x0, x1), min(y0, y1));
            br = (max(x0, x1), max(y0, y1));
            bl = (min(x0, x1), max(y0, y1))
            for a, b in [(tl, tr), (tr, br), (br, bl), (bl, tl)]: cv2.line(vis, a, b, (255, 180, 60), 2, cv2.LINE_AA)
        if pts:
            for i in range(4):
                a = pts[i];
                b = pts[(i + 1) % 4];
                cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 200, 0), 2, cv2.LINE_AA)
            for i, (u, v) in enumerate(pts):
                cv2.circle(vis, (int(u), int(v)), 6, (0, 200, 0), -1);
                cv2.putText(vis, str(i + 1), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        _show_dash(vis, mid_img, right_help)

    def on_mouse(event, x, y, flags, _):
        nonlocal start, drag, cur, last_flags
        last_flags = flags;
        cur = clamp(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if not pts:
                start = cur
            else:
                for i, p in enumerate(pts):
                    if near(cur, p, 10): drag = i; hist.append(pts.copy()); break
        elif event == cv2.EVENT_MOUSEMOVE:
            if drag is not None and pts:
                u, v = snap(pts[drag], cur, flags);
                pts[drag] = (u, v)
            draw()
        elif event == cv2.EVENT_LBUTTONUP:
            if start is not None and not pts:
                x0, y0 = start;
                x1, y1 = cur
                tl = (min(x0, x1), min(y0, y1));
                tr = (max(x0, x1), min(y0, y1));
                br = (max(x0, x1), max(y0, y1));
                bl = (min(x0, x1), max(y0, y1))
                pts[:] = [tl, tr, br, bl];
                start = None
            drag = None;
            draw()

    cv2.setMouseCallback("Annotate", on_mouse);
    draw()
    while True:
        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord('q')): raise SystemExit("Aborted.")
        if k in (ord('Q'), ord('X'), ord('x')): _quit_all()
        if k == ord('r'): pts.clear(); hist.clear(); start = None; draw()
        if k == ord('u') and hist: pts = hist.pop(); draw()
        if k == 13 and len(pts) == 4: return [(float(u), float(v)) for (u, v) in pts]


def _collect_any4(image_bgr, mid_img=None, right_img=None):
    pts = [];
    base = image_bgr.copy();
    h, w = base.shape[:2]
    right_help = _help_panel("any", h)
    cur = None;
    last_flags = 0

    def snap(p0, p1, flags):
        if p0 is None or not (flags & cv2.EVENT_FLAG_SHIFTKEY): return (float(p1[0]), float(p1[1]))
        (x0, y0) = p0;
        (x1, y1) = p1
        return (float(x1), float(y0)) if abs(x1 - x0) >= abs(y1 - y0) else (float(x0), float(y1))

    def draw():
        vis = base.copy()
        for i, (u, v) in enumerate(pts):
            cv2.circle(vis, (int(u), int(v)), 4, (0, 200, 0), -1);
            cv2.putText(vis, str(i + 1), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            if i > 0: cv2.line(vis, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(u), int(v)), (0, 200, 0), 1,
                               cv2.LINE_AA)
        if cur is not None and pts:
            u0, v0 = pts[-1];
            u1, v1 = snap((u0, v0), cur, last_flags)
            cv2.line(vis, (int(u0), int(v0)), (int(u1), int(v1)), (255, 180, 60), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, (int(u1), int(v1)), (255, 180, 60), cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
        _show_dash(vis, mid_img, right_help)

    def on_mouse(event, x, y, flags, _):
        nonlocal cur, last_flags;
        last_flags = flags
        if event == cv2.EVENT_MOUSEMOVE:
            cur = (float(x), float(y));
            draw()
        elif event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            p = (float(x), float(y))
            if pts: p = snap(pts[-1], p, flags)
            pts.append(p);
            draw()

    cv2.setMouseCallback("Annotate", on_mouse);
    draw()
    while True:
        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord('q')): raise SystemExit("Aborted.")
        if k in (ord('Q'), ord('X'), ord('x')): _quit_all()
        if k == ord('u') and pts: pts.pop(); draw()
        if k == ord('r'): pts.clear(); draw()
        if k == 13 and len(pts) == 4: return pts


def _assign_any_order(clicked_xy, face_names, pts3d_dict, K, dist):
    pts2d = np.array(clicked_xy, float).reshape(-1, 1, 2)
    names = list(face_names);
    best = None
    for perm in itertools.permutations(names, 4):
        X = np.array([pts3d_dict[n] for n in perm], float).reshape(-1, 1, 3)
        ok, rvec, tvec = cv2.solvePnP(X, pts2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        reproj, _ = cv2.projectPoints(X, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((reproj - pts2d) ** 2, axis=2))))
        R, _ = cv2.Rodrigues(rvec)
        Z = (R @ X.reshape(-1, 3).T + tvec.reshape(3, 1)).T[:, 2].mean()
        if best is None or (err < best["err"] and Z > 0):
            best = {"perm": perm, "rvec": rvec, "tvec": tvec, "err": err}
    if best is None: raise RuntimeError("PnP failed for all assignments")
    mapping = {name: tuple(clicked_xy[i]) for i, name in enumerate(best["perm"])}
    ordered2d = np.array([mapping[n] for n in face_names], float)
    return mapping, ordered2d, {"rvec": best["rvec"], "tvec": best["tvec"], "err": best["err"]}


def _review(anno_img, reproj_img, rms):
    lines = [
        f"Review | RMS={rms:.2f}px",
        "ENTER/y/s: accept",
        "r/n/BACKSPACE: redo",
        "ESC/q: abort this shot",
        "Q/X: quit-all (batch stops)",
    ]
    right = _text_panel(lines, 360, anno_img.shape[0])
    while True:
        _show_dash(anno_img, reproj_img, right, "")
        k = cv2.waitKey(50) & 0xFF
        if k in (13, ord('y'), ord('s')): return "accept"
        if k in (ord('r'), ord('n'), 8):  return "redo"
        if k in (ord('Q'), ord('X'), ord('x')): _quit_all()
        if k in (27, ord('q')): raise SystemExit("Aborted by user during review.")


def _render_list_faces(h, w, title, items, sel):
    pan = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(pan, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    max_show = min(22, len(items));
    start = max(0, min(sel - max_show // 2, max(0, len(items) - max_show)))
    y = 56
    for i in range(start, start + max_show):
        if i >= len(items): break
        it = items[i]

        # build right-aligned chips
        chips = []
        if it["ann_ok"]:
            chips.append(
                _pill(f"RMS {it['rms']:.2f}px", "info") if it.get("rms") is not None else _pill("ANN OK", "ok"))
        else:
            chips.append(_pill("no ann", "warn"))
        chips.append(_pill("BOARD", "ok" if it["board_ok"] else "warn"))
        chips.append(_pill("KP OK", "ok" if it["kp_ok"] else "warn"))

        x = w - 12
        for c in reversed(chips):
            ch, cw = c.shape[:2];
            x -= (cw + 6)
            pan[y - 16:y - 16 + ch, x:x + cw] = c

        avail = x - 14
        base_name = f"{it['object']}  side{it['side']}"
        name = base_name
        (tw, th), _ = cv2.getTextSize("> " + name, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        while tw > avail and len(name) > 3:
            name = name[:-4] + "..."
            (tw, th), _ = cv2.getTextSize("> " + name, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        col = (20, 70, 180) if i == sel else (30, 30, 30);
        thk = 2 if i == sel else 1
        cv2.putText(pan, ("> " if i == sel else "  ") + name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, thk,
                    cv2.LINE_AA)
        y += 30

    tips = "UP/DOWN or W/S select   ENTER annotate   A pts   C auto-scale   T check   F force   R reload   Q quit"
    tips_lines = _wrap_to_width(tips, w - 24, scale=0.46, thk=1)
    line_h = 18
    y0 = h - 10 - line_h * (len(tips_lines) - 1)
    for i, t in enumerate(tips_lines):
        cv2.putText(pan, t, (12, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1, cv2.LINE_AA)
    return pan


def _render_raw_thumb(path: Path, target_h=UI_H, target_w=UI_W_LEFT):
    if not path or not Path(path).exists():
        return np.full((target_h, target_w, 3), 32, np.uint8)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((target_h, target_w, 3), 32, np.uint8)
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    vis = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((target_h, target_w, 3), 22, np.uint8)
    out[:nh, :nw] = vis
    cv2.putText(out, Path(path).name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# -------------------------- Core per-shot flow --------------------------
def _load_K_and_face(project_root: Path, meta_path: Path, row) -> tuple[np.ndarray, str, dict]:
    """Return (K, face_yaml_abs, meta_json). Works with old and new meta."""
    meta_json = {}
    if meta_path.exists():
        try:
            meta_json = json.loads(meta_path.read_text())
        except Exception:
            pass

    # Camera intrinsics
    if meta_json.get("camera"):
        cam = meta_json["camera"]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    elif row and all(getattr(row, k) is not None for k in ("fx", "fy", "cx", "cy")):
        fx, fy, cx, cy = row.fx, row.fy, row.cx, row.cy
    else:
        raise SystemExit("[!] Camera intrinsics not found (meta.camera or manifest columns).")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)

    # face_yaml (absolute)
    face_yaml = meta_json.get("face_yaml") or (row.face_yaml if row else None)
    if not face_yaml:
        raise SystemExit("[!] face_yaml missing (meta or manifest).")
    fy_path = Path(face_yaml)
    if not fy_path.is_absolute():
        fy_path = (project_root / fy_path).resolve()

    return K, str(fy_path), meta_json

# Main function for annotating a single shot: detects tags, collects user points, solves PnP, saves results
def annotate_single_shot(project_root: Path, shot_raw_path: Path, *,
                         pts_type: str = "quad", calib: Optional[str] = None,
                         check_tag_scale: bool = False, auto_correct_scale: bool = False,
                         scale_tol: float = 0.02) -> Path:
    if not shot_raw_path.exists():
        raise SystemExit(f"[!] Shot not found: {shot_raw_path}")
    if not shot_raw_path.name.endswith("_raw.png"):
        raise SystemExit("Shot must end with _raw.png")

    # Optional manifest row
    manifest_path = project_root / "shots" / "manifest.csv"
    row = None
    if manifest_path.exists():
        for r in _read_manifest(manifest_path):
            if Path(r.path_raw) == shot_raw_path:
                row = r;
                break

    # Meta path: prefer manifest's, else sibling
    meta_path = Path(row.path_meta) if (row and row.path_meta) else Path(
        str(shot_raw_path).replace("_raw.png", "_meta.json"))
    K, face_yaml_path, meta_json = _load_K_and_face(project_root, meta_path, row)
    dist = _load_dist_from_calib(calib)

    # Infer object + side robustly
    obj, side_letter = _infer_object_and_side(shot_raw_path, meta_json, row)
    face_key = Path(face_yaml_path).stem

    img_ann = (meta_json.get("image") or {}).get("path_ann") or (
        row.path_ann if row and row.path_ann else str(shot_raw_path).replace("_raw.png", "_ann.png"))
    img_ann = Path(img_ann)
    # reprojection image path
    img_reproj = img_ann.with_name(img_ann.name.replace("_ann", "_reproj"))

    # Load image/gray
    img = cv2.imread(str(shot_raw_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[!] Failed to read image: {shot_raw_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Keypoints / faces map
    pts3d_dict, faces_map, face_key_resolved, kp_path = load_keypoints_fuzzy(obj, project_root)
    if face_key not in faces_map:
        raise SystemExit(f"[!] Face '{face_key}' not in {kp_path}. Available: {list(faces_map.keys())}")
    names_in_order = faces_map[face_key]

    # Board pose from tags: detect AprilTags and estimate camera-to-board transform
    origin_id, tag_size_m, T_board_tag = load_board(Path(face_yaml_path))
    log.info(f"[i] Board: {Path(face_yaml_path).name}  tag_size={tag_size_m * 1000:.1f} mm  origin={origin_id}")

    dets = detect_tags(gray, K[0, 0], K[1, 1], K[0, 2], K[1, 2], tag_size_m)
    if not dets:
        raise SystemExit("[!] No AprilTags detected in shot.")
    det_by_id = {int(d.tag_id): d for d in dets}
    if origin_id in det_by_id and det_by_id[origin_id].pose_R is not None:
        d = det_by_id[origin_id]
        T_cam_board = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
    else:
        common = [tid for tid in det_by_id.keys() if tid in T_board_tag]
        if not common:
            raise SystemExit("[!] Detected tags do not belong to this board.")
        tid = common[0];
        d = det_by_id[tid]
        T_cam_tag = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
        T_tag_board = inv_se3(T_board_tag[tid])
        T_cam_board = T_cam_tag @ T_tag_board
    s = 1.0;
    n_pairs = 0;
    mad = 0.0

    if check_tag_scale or auto_correct_scale:
        s, n_pairs, mad = _estimate_tag_scale(det_by_id, T_board_tag)
        log.info(f"[i] tag-scale check: s={s:.4f}  pairs={n_pairs}  MAD={mad:.4f}")

        if abs(s - 1.0) > scale_tol:
            msg = f"[!] tag_size mismatch ~{(s - 1.0) * 100:.1f}%"

            if auto_correct_scale:
                T_cam_board[:3, 3] /= s
                log.warning("tag_size mismatch ~%.1f%% -> applied 1/s to T_cam_board translation.",
                            (s - 1.0) * 100.0)
            else:
                log.warning("tag_size mismatch ~%.1f%% (consider --auto-correct-scale).",
                            (s - 1.0) * 100.0)

    # UI: collect 4 points
    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    mid = None
    clicked = (_collect_quad if pts_type == "quad" else _collect_any4)(img, mid, None)

    # Solve PnP to find best assignment of clicked points to face corners and get camera-to-object pose
    mapping, ordered2d, fit = _assign_any_order(clicked, names_in_order, pts3d_dict, K, dist)
    R, _ = cv2.Rodrigues(fit["rvec"]);
    T_cam_obj = se3(R, fit["tvec"].reshape(3))

    # Visuals for review
    def _project_named(names, col, rad=4, canvas=None):
        X = np.vstack([pts3d_dict[n] for n in names]).astype(np.float32)
        uv, _ = _reproj(X, T_cam_obj, K, img, draw=False, dist=dist)
        for (u, v), name in zip(uv, names):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(canvas, (int(u), int(v)), rad, col, -1)
                cv2.putText(canvas, name, (int(u) + 4, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

    anno = img.copy()
    _project_named(names_in_order, (0, 220, 0), 4, anno)
    others = [n for n in pts3d_dict.keys() if n not in names_in_order]
    if others: _project_named(others, (0, 0, 255), 3, anno)

    all3d = np.vstack([v for _, v in sorted(pts3d_dict.items())]).astype(np.float32)
    _, mid = _reproj(all3d, T_cam_obj, K, img, draw=True, dist=dist)

    uv4, _ = _reproj(np.vstack([pts3d_dict[n] for n in names_in_order]).astype(np.float32),
                     T_cam_obj, K, img, draw=False, dist=dist)
    err = np.linalg.norm(uv4 - ordered2d, axis=1);
    rms = float(np.sqrt((err ** 2).mean()))

    _flush_keys(120)
    while True:
        decision = _review(anno, mid, rms)
        if decision == "redo":
            clicked = (_collect_quad if pts_type == "quad" else _collect_any4)(img, mid, None)
            mapping, ordered2d, fit = _assign_any_order(clicked, names_in_order, pts3d_dict, K, dist)
            R, _ = cv2.Rodrigues(fit["rvec"]);
            T_cam_obj = se3(R, fit["tvec"].reshape(3))
            anno = img.copy()
            _project_named(names_in_order, (0, 220, 0), 4, anno)
            if others: _project_named(others, (0, 0, 255), 3, anno)
            _, mid = _reproj(all3d, T_cam_obj, K, img, draw=True, dist=dist)
            uv4, _ = _reproj(np.vstack([pts3d_dict[n] for n in names_in_order]).astype(np.float32),
                             T_cam_obj, K, img, draw=False, dist=dist)
            err = np.linalg.norm(uv4 - ordered2d, axis=1);
            rms = float(np.sqrt((err ** 2).mean()))
            continue
        break

    # Compose outputs: compute board-to-object transform and save YAML, images
    T_board_obj = inv_se3(T_cam_board) @ T_cam_obj

    out_yaml = _out_yaml_path(project_root, obj, side_letter)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    Path(img_ann).parent.mkdir(parents=True, exist_ok=True)

    # Save images
    cv2.imwrite(str(img_ann), anno)
    cv2.imwrite(str(img_reproj), mid)

    # Save YAML with all annotation data
    out = {
        "object": obj,
        "face_key": face_key,
        "board_yaml": face_yaml_path,
        "image": str(shot_raw_path),
        "rms_px": rms,
        "T_board_object": {"matrix": T_board_obj.tolist()},
        "corner_uv": {n: [float(mapping[n][0]), float(mapping[n][1])] for n in names_in_order},
        "pnp": {"rvec": [float(x) for x in fit["rvec"].ravel()],
                "tvec": [float(x) for x in fit["tvec"].ravel()]},
        "clicked_uv_raw": [(float(u), float(v)) for (u, v) in clicked],
        "face_corner_names": names_in_order,
        "assignment": {n: [float(mapping[n][0]), float(mapping[n][1])] for n in names_in_order},
        "notes": "Accepted.",
        "diagnostics": {
            "board_tag_size_mm": float(tag_size_m * 1000.0),
            "tag_scale_ratio": float(s),
            "tag_scale_pairs": int(n_pairs),
            "tag_scale_auto_corrected": bool(auto_correct_scale and abs(s - 1.0) > scale_tol),
        },
    }
    log.info("RMS=%.2f px  (%s)", rms, face_key)
    with out_yaml.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    log.info(f"[i] Wrote {out_yaml}")
    cv2.destroyAllWindows()
    log.info(f"[i] Wrote {out_yaml}")

    # Keep faces manifest in sync with disk (adds new/updated, prunes deleted)
    try:
        _sync_face_manifest(project_root)
    except Exception as e:
        log.warning(f"[faces-manifest] Sync after save failed: {e}")

    cv2.destroyAllWindows()
    return out_yaml

    return out_yaml


# -------------------------- Batch helper --------------------------
def _latest_per_face(rows: List[ShotRow], object_filter: Optional[str], side: Optional[str]) -> List[ShotRow]:
    # key = (object_base, side, face_yaml)
    latest: Dict[Tuple[str, str, str], ShotRow] = {}
    for r in rows:
        if not r.path_raw: continue
        if object_filter and object_filter.lower() not in r.object_base.lower(): continue
        if side and r.side.upper() != side.upper(): continue
        key = (r.object_base, r.side.upper(), Path(r.face_yaml).stem)
        prev = latest.get(key)
        if prev is None or r.timestamp > prev.timestamp:
            latest[key] = r
    return list(latest.values())


def _load_rms_if_any(yaml_path: Path) -> Optional[float]:
    try:
        if yaml_path.exists():
            y = yaml.safe_load(yaml_path.read_text())
            v = y.get("rms_px", None)
            return float(v) if v is not None else None
    except Exception:
        pass
    return None


# -------------------------- CLI --------------------------

def main():
   # Parse command-line arguments for single-shot, batch, or browse modes
    parser = argparse.ArgumentParser("Annotate ISC faces (single or batch via manifest).")
    parser.add_argument("--shot", type=str, help="Absolute path to *_raw.png (single-shot mode).")
    parser.add_argument("--batch", choices=["latest"], help="Batch mode from manifest.csv.")
    parser.add_argument("--manifest", type=str, help="Path to manifest.csv (default: <proj>/shots/manifest.csv).")
    parser.add_argument("--object-filter", type=str, default=None, help="Substring filter for object_base.")
    parser.add_argument("--side", type=str, default=None, help="Restrict to side letter A|B|C|D.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing YAML if present.")
    parser.add_argument("--pts-type", choices=["quad", "any"], default="quad", help="Corner input mode.")
    parser.add_argument("--calib", type=str, default=None, help="Optional ChArUco calib yaml for distortion.")
    parser.add_argument("--dry-run", action="store_true", help="List actions then exit.")
    parser.add_argument("--check-tag-scale", action="store_true", help="Print scale ratio s from inter-tag distances.")
    parser.add_argument("--auto-correct-scale", action="store_true",
                        help="If |s-1|>tol, divide T_cam_board translation by s.")
    parser.add_argument("--scale-tol", type=float, default=0.02, help="Relative tolerance (default 0.02 = 2%).")
    parser.add_argument("--browse", action="store_true",
                        help="Interactive browser (arrow keys + ENTER) to pick which face to annotate.")

    args = parser.parse_args()

    project_root = resolve_project_root(None)
    ensure_project_dirs(project_root)

    global log
    log = init_project_logger(project_root / "logs" / "annotate_shots.log",
                              level="INFO", console=True)
    uih = UIBufferHandler()
    uih.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    log.addHandler(uih)
    # Always rebuild the faces manifest from current on-disk YAMLs.
    # - Creates if missing and faces exist
    # - Updates timestamps/RMS on changed files
    # - Removes rows for YAMLs deleted offline
    try:
        _sync_face_manifest(project_root)
    except Exception as e:
        log.warning(f"[faces-manifest] Initial sync failed: {e}")

    if args.browse:
        manifest = Path(args.manifest) if args.manifest else (project_root / "shots" / "manifest.csv")

        if not manifest.exists():
            sys.exit(f"[!] manifest.csv not found: {manifest}")
        _browse_and_annotate(project_root, manifest, args)
        return
    elif args.shot:
        try:
            annotate_single_shot(project_root, Path(args.shot), pts_type=args.pts_type, calib=args.calib,
                                 check_tag_scale=args.check_tag_scale, auto_correct_scale=args.auto_correct_scale,
                                 scale_tol=args.scale_tol)

        except SystemExit as e:
            # pass through normal abort; keep exit code 1 for errors
            if str(e) != "Aborted.":
                log.error(e)
            sys.exit(1)
        except BaseException as e:
            log.error(f"[!] Error: {e}")
            sys.exit(1)
        return

    # batch mode
    manifest = Path(args.manifest) if args.manifest else (project_root / "shots" / "manifest.csv")
    if not manifest.exists():
        sys.exit(f"[!] manifest.csv not found: {manifest}")

    rows = _read_manifest(manifest)
    if args.batch == "latest":
        pick = _latest_per_face(rows, args.object_filter, args.side)
    else:
        sys.exit("[!] Choose a batch mode (e.g., --batch latest).")

    if not pick:
        log.debug("[i] Nothing to annotate.")
        return

    log.info("[i] Batch plan:")
    for r in pick:
        out_yaml = _out_yaml_path(project_root, r.object_base, r.side)
        status = "(exists)" if (out_yaml.exists() and not args.force) else ""
        log.info(f"  - {r.object_base} side{r.side}  ts={r.timestamp}  -> {out_yaml} {status}")

    if args.dry_run:
        log.info("[i] Dry-run only. Exiting.")
        return

    for r in pick:
        out_yaml = _out_yaml_path(project_root, r.object_base, r.side)
        if out_yaml.exists() and not args.force:
            log.info(f"[i] Skip (already done): {out_yaml}")
            continue
        try:
            annotate_single_shot(project_root, Path(r.path_raw), pts_type=args.pts_type, calib=args.calib,
                                 check_tag_scale=args.check_tag_scale, auto_correct_scale=args.auto_correct_scale,
                                 scale_tol=args.scale_tol)


        except SystemExit as e:
            if e.code == EXIT_QUIT_ALL or str(e) == "Aborted.":
                log.info("[i] Batch interrupted.")
                sys.exit(EXIT_QUIT_ALL if e.code == EXIT_QUIT_ALL else 1)
            log.error("%s", e)
            continue
        except BaseException as e:
            log.error("Error on %s: %s", r.path_raw, e)


if __name__ == "__main__":
    main()
