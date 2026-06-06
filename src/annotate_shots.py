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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import cv2, yaml
from datetime import datetime

from utils.logger import init_project_logger
from utils.project_config import resolve_project_root, ensure_project_dirs
from utils.annotation_utils import (
    detect_tags, load_board, se3, inv_se3,
    load_keypoints_fuzzy
)
from posetag.workflows.mesh_keypoints import split_object_face, validate_keypoint_payload

EXIT_QUIT_ALL = 99
import logging

UI_LOG: list[str] = []
log: logging.Logger  # global

_PILL = {
    "ok": ((228, 245, 224), (55, 120, 35)),
    "warn": ((213, 235, 247), (37, 102, 145)),
    "bad": ((222, 222, 241), (59, 64, 153)),
    "info": ((249, 238, 229), (160, 95, 70)),
}

UI_H = 640
UI_W_LEFT = 640
UI_W_MID = 390
UI_W_RIGHT = 320
UI_WIN_H = 820
UI_WIN_W = UI_W_LEFT + UI_W_MID + UI_W_RIGHT

KEY_UP = {2490368, 65362, 63232, ord("w"), ord("W"), ord("k"), ord("K")}
KEY_DOWN = {2621440, 65364, 63233, ord("s"), ord("S"), ord("j"), ord("J")}
KEY_PAGE_UP = {2162688, 65365, 63276}
KEY_PAGE_DOWN = {2228224, 65366, 63277}


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
    if _measure("...", scale, thk) > max_w:
        return "."
    best = "..."
    for kept in range(1, len(text) + 1):
        left = (kept + 1) // 2
        right = kept // 2
        cand = text[:left] + "..." + (text[-right:] if right else "")
        if _measure(cand, scale, thk) > max_w:
            break
        best = cand
    return best


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


def _rms_kind(rms: float | None) -> str:
    if rms is None:
        return "info"
    if rms <= 3.0:
        return "ok"
    if rms <= 8.0:
        return "warn"
    return "bad"


def _right_column(project_root, it, args, height, w=480):
    ann = "OK" if it["ann_ok"] else "MISSING"
    kp = "OK" if it["kp_ok"] else "missing"
    board = "OK" if it["board_ok"] else "missing"
    pan = np.full((height, w, 3), (249, 251, 252), np.uint8)

    def title(text: str, y: int) -> int:
        cv2.rectangle(pan, (10, y), (w - 10, y + 26), (232, 243, 248), -1)
        cv2.putText(
            pan,
            text,
            (18, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (22, 34, 48),
            1,
            cv2.LINE_AA,
        )
        return y + 38

    def row(label: str, value: str, y: int, *, kind: str | None = None) -> int:
        cv2.putText(
            pan,
            label,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (88, 104, 120),
            1,
            cv2.LINE_AA,
        )
        text = _ellipsize_middle(str(value), w - 120, 0.45, 1)
        if kind is None:
            cv2.putText(
                pan,
                text,
                (116, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (34, 48, 64),
                1,
                cv2.LINE_AA,
            )
        else:
            chip = _pill(text, kind)
            ch, cw = chip.shape[:2]
            pan[y - ch + 6:y + 6, 116:116 + min(cw, w - 126)] = chip[:, :min(cw, w - 126)]
        return y + 24

    y = 18
    y = title("Selection", y)
    y = row("Project", _ellipsize_middle(str(project_root), w - 120, 0.45, 1), y)
    y = row("Object", it["object"], y)
    y = row("Face", it["side"], y)
    y = row("Face key", it["face_key"], y)
    y = row("Shot", it["raw"].name if it["raw"] else "-", y)
    y = row("Board", Path(it["board"]).name if it["board"] else "-", y)

    y += 8
    y = title("Status", y)
    ann_value = ann
    if it.get("rms") is not None:
        ann_value = f"{ann}  RMS {it['rms']:.2f}px"
    y = row("Annot.", ann_value, y, kind=_rms_kind(it.get("rms")) if it["ann_ok"] else "warn")
    y = row("Keypoints", kp, y, kind="ok" if it["kp_ok"] else "warn")
    y = row("Board", board, y, kind="ok" if it["board_ok"] else "warn")

    y += 8
    y = title("Options", y)
    y = row("Corners", args.pts_type, y)
    y = row("Scale", "check ON" if args.check_tag_scale else "check OFF", y)
    y = row("Auto", "correct ON" if args.auto_correct_scale else "correct OFF", y)
    y = row("Force", "overwrite ON" if args.force else "overwrite OFF", y)

    y += 8
    y = title("Keys", y)
    enter_action = "ENTER redo selected" if it["ann_ok"] else "ENTER annotate"
    for line in (
        enter_action,
        "J/K, arrows, or W/S select",
        "A corner mode, T scale check",
        "C auto-scale, F force",
        "R reload, Q quit",
    ):
        if y > height - 42:
            break
        cv2.putText(
            pan,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (89, 104, 118),
            1,
            cv2.LINE_AA,
        )
        y += 22

    if y >= height - 72:
        return pan

    # rough estimate of how many log lines fit
    line_h = 20
    top_h = y + 8
    max_lines = max(1, (height - top_h) // line_h)
    log_lines = max(0, max_lines - 1)
    tail = UI_LOG[-log_lines:] if (UI_LOG and log_lines) else ["(no messages yet)"]
    bottom = _text_panel(
        ["Recent log", *tail],
        w=w,
        h=max(1, height - top_h),
        color=(231, 238, 245),
        bg=(22, 30, 38),
    )
    pan[top_h:height] = bottom[:height - top_h]
    return pan


def _normalise_side_letter(side: str | None) -> str:
    token = str(side or "").strip()
    if not token:
        return ""
    if token.lower().startswith("side") and len(token) >= 5:
        return token[-1].upper()
    return token[0].upper()


def _normalise_face_label(face: str | None) -> str:
    token = str(face or "").strip()
    if not token:
        return ""
    if token.lower() in {"unresolved", "unknown", "none", "n/a", "na"}:
        return ""
    if token.lower().startswith("side") and len(token) >= 5:
        suffix = token[4:]
        return f"side{suffix[:1].upper()}{suffix[1:]}"
    if len(token) == 1 and token.upper() in {"A", "B", "C", "D"}:
        return f"side{token.upper()}"
    return token


def _face_key(object_base: str, face_label: str) -> str:
    # matches gen_keypoints faces naming
    label = _normalise_face_label(face_label) or str(face_label).strip()
    if label.startswith(f"{object_base}_"):
        return label
    return f"{object_base}_{label}"


def _face_label_from_face_key(object_base: str, face_key: str) -> str:
    base, face = split_object_face(face_key)
    if face and base == object_base:
        return face
    prefix = f"{object_base}_"
    if face_key.startswith(prefix):
        return face_key[len(prefix):]
    return face_key


def _annotation_target(
    shot_raw_path: Path,
    meta_json: dict | None,
    row,
    *,
    board_yaml_path: Optional[Path] = None,
) -> tuple[str, str, str]:
    candidates: list[str] = []
    if board_yaml_path is not None and board_yaml_path.stem:
        candidates.append(board_yaml_path.stem)
    if meta_json:
        candidates.extend(
            str(meta_json.get(key, "")).strip()
            for key in ("object_full", "object_base", "object")
        )
    if row:
        candidates.extend(
            str(getattr(row, key, "")).strip()
            for key in ("object_full", "object_base")
        )

    # Fallback: parse from shots/<object>/<side>/... path
    path_side = ""
    parts = list(shot_raw_path.parts)
    if "shots" in parts:
        i = parts.index("shots")
        if i + 1 < len(parts):
            candidates.append(parts[i + 1])
        if i + 2 < len(parts):
            path_side = parts[i + 2]

    object_base = ""
    face_label = ""
    face_key = board_yaml_path.stem if board_yaml_path is not None else ""

    for candidate in candidates:
        if not candidate:
            continue
        base, face = split_object_face(candidate)
        if face:
            object_base = object_base or base
            face_label = face_label or face
            face_key = face_key or _face_key(base, face)
            break
        object_base = object_base or base

    side_candidates: list[str] = []
    if meta_json:
        side_candidates.append(str(meta_json.get("side", "")).strip())
    if row:
        side_candidates.append(str(getattr(row, "side", "")).strip())
    side_candidates.append(path_side)
    for side in side_candidates:
        face_label = face_label or _normalise_face_label(side)

    if face_key and object_base and not face_label:
        face_label = _face_label_from_face_key(object_base, face_key)
    if object_base and face_label and not face_key:
        face_key = _face_key(object_base, face_label)

    if not object_base:
        raise SystemExit("[!] Could not infer object name (wanted meta.object_base or manifest.object_base).")
    if not face_label:
        raise SystemExit("[!] Could not infer face label (wanted board YAML stem, meta.object_full, or manifest side).")
    if not face_key:
        face_key = _face_key(object_base, face_label)
    return object_base, face_label, face_key


def _infer_object_and_side(shot_raw_path: Path, meta_json: dict | None, row) -> tuple[str, str]:
    object_base, face_label, _face_key_value = _annotation_target(
        shot_raw_path,
        meta_json,
        row,
    )
    return object_base, face_label


def _index_faces(project_root: Path, manifest: Path) -> List[dict]:
    rows = _read_manifest(manifest)
    latest = _latest_per_face(rows, object_filter=None, side=None)
    items = []
    for r in latest:
        board_path = _resolve_project_path(project_root, r.face_yaml)
        raw_path = _resolve_project_path(project_root, r.path_raw) or Path(r.path_raw)
        obj, face_label, face_key = _annotation_target(
            raw_path,
            None,
            r,
            board_yaml_path=board_path if board_path else None,
        )
        out_yaml = _out_yaml_path(project_root, obj, face_label, face_key=face_key)
        kp_ok = (project_root / "objects" / obj / "keypoints.json").exists()
        meta_path = _resolve_project_path(project_root, r.path_meta) if r.path_meta else None
        board_ok = board_path.exists() if board_path else False

        rms = _load_rms_if_any(out_yaml) if out_yaml.exists() else None
        items.append({
            "object": obj, "side": face_label, "face_key": face_key, "ts": r.timestamp,
            "raw": raw_path, "meta": meta_path,
            "board": str(board_path) if board_path else r.face_yaml, "out_yaml": out_yaml,
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

    def on_browser_mouse(event, _x, _y, flags, _param):
        nonlocal i
        if event != cv2.EVENT_MOUSEWHEEL or not items:
            return
        try:
            delta = cv2.getMouseWheelDelta(flags)
        except AttributeError:
            delta = flags
        if delta > 0:
            i = (i - 1) % len(items)
        elif delta < 0:
            i = (i + 1) % len(items)

    while True:
        it = items[i]
        thumb_label = f"{it['object']} / {it['side']}"
        left = _render_raw_thumb(
            it["raw"],
            target_h=UI_H,
            target_w=UI_W_LEFT,
            label=thumb_label,
        )
        faces_w = UI_W_MID
        mid = _render_list_faces(left.shape[0], faces_w, "Faces (latest per manifest)", items, i)
        right = _right_column(project_root, it, args, height=left.shape[0], w=UI_W_RIGHT)

        strip = _hstack(left, mid, right)
        try:
            cv2.resizeWindow("Annotate", strip.shape[1], strip.shape[0])
        except Exception:
            pass

        cv2.setMouseCallback("Annotate", on_browser_mouse)
        cv2.imshow("Annotate", strip)

        k = cv2.waitKeyEx(60) & 0xFFFFFFFF
        if k in (ord('q'), 27):
            break
        elif k in KEY_UP:
            i = (i - 1) % len(items)
        elif k in KEY_DOWN:
            i = (i + 1) % len(items)
        elif k in KEY_PAGE_UP:
            i = (i - 6) % len(items)
        elif k in KEY_PAGE_DOWN:
            i = (i + 6) % len(items)
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
            if it["ann_ok"]:
                log.info(f"[i] Redo annotation: {it['out_yaml']}")
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


@dataclass
class AnnotationPreflight:
    shot_raw_path: Path
    annotated_path: Path
    reprojection_path: Path
    meta_path: Path
    board_yaml_path: Path
    keypoints_path: Path
    object_base: str
    side: str  # Face label, e.g. "sideA", "front", or "back".
    face_key: str
    K: np.ndarray
    meta_json: dict
    row: Optional[ShotRow]
    calib_path: Optional[Path] = None


# -------------------------- Faces manifest (CSV) --------------------------

@dataclass
class FaceManifestRow:
    timestamp: str  # ISO time from YAML file mtime
    object: str
    side: str  # A|B|C|D for side labels, or a named face such as front/back.
    face_key: str
    yaml_path: str  # path to *_T_board_object.yaml (prefer relative to project_root)
    board_yaml: str
    image: str
    rms_px: Optional[float]
    tag_size_m: Optional[float] = None  # <-- NEW (default keeps older call sites safe)


def _face_manifest_path(project_root: Path) -> Path:
    return project_root / "faces" / "face_manifest.csv"


def _side_from_face_key(face_key: str) -> str:
    base, face = split_object_face(face_key)
    if face:
        if face.lower().startswith("side") and len(face) >= 5:
            return face[-1].upper()
        return face
    m = re.search(r"_side([ABCD])", face_key, re.IGNORECASE)
    return m.group(1).upper() if m else "X"


def _scan_face_yamls(project_root: Path) -> List[FaceManifestRow]:
    rows: List[FaceManifestRow] = []
    root = project_root / "faces"
    if not root.exists():
        return rows

    for y in root.glob("*/*/*_T_board_object.yaml"):
        try:
            with y.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
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
                side=_normalise_side_letter(str(d.get("side", "")).strip()) or "",
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


def _out_yaml_path(
    project_root: Path,
    object_base: str,
    face_label: str,
    *,
    face_key: Optional[str] = None,
) -> Path:
    label = _normalise_face_label(face_label) or str(face_label).strip()
    key = face_key or _face_key(object_base, label)
    return project_root / "faces" / object_base / label / f"{key}_T_board_object.yaml"


def _resolve_project_path(project_root: Path, value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except Exception:
        return left == right


def _find_manifest_row_for_shot(
    project_root: Path,
    shot_raw_path: Path,
    *,
    manifest_path: Optional[Path] = None,
) -> Optional[ShotRow]:
    manifest = manifest_path or (project_root / "shots" / "manifest.csv")
    if not manifest.exists():
        return None
    target = shot_raw_path if shot_raw_path.is_absolute() else (project_root / shot_raw_path).resolve()
    for row in _read_manifest(manifest):
        row_raw = _resolve_project_path(project_root, row.path_raw)
        if row_raw is not None and _same_path(row_raw, target):
            return row
    return None


def _load_json_mapping(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(f"[!] {label} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[!] {label} is malformed: {path}: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"[!] Could not read {label}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"[!] {label} must contain a JSON object: {path}")
    return data


def _finite_float(value: Any, field: str, source: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise SystemExit(f"[!] {source} field '{field}' must be numeric.") from exc
    if not np.isfinite(number):
        raise SystemExit(f"[!] {source} field '{field}' must be finite.")
    return number


def _camera_matrix_from_values(values: Mapping[str, Any], source: str) -> np.ndarray:
    missing = [key for key in ("fx", "fy", "cx", "cy") if key not in values]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"[!] Camera intrinsics missing from {source}: {joined}.")
    fx = _finite_float(values["fx"], "fx", source)
    fy = _finite_float(values["fy"], "fy", source)
    cx = _finite_float(values["cx"], "cx", source)
    cy = _finite_float(values["cy"], "cy", source)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], float)


def _resolve_calib_path(project_root: Path, calib: Optional[str]) -> Optional[Path]:
    if not calib:
        return None
    raw = Path(str(calib).strip()).expanduser()
    if not str(raw):
        return None
    candidates = [raw if raw.is_absolute() else (project_root / raw).resolve()]
    if not raw.is_absolute() and len(raw.parts) == 1:
        candidates.append((project_root / "calib" / raw).resolve())
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"[!] Calibration YAML not found: {candidates[0]}")


def _validate_board_yaml(board_yaml_path: Path) -> None:
    try:
        load_board(board_yaml_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"[!] Board YAML not found: {board_yaml_path}") from exc
    except Exception as exc:
        raise SystemExit(f"[!] Malformed board YAML: {board_yaml_path}: {exc}") from exc


def _validate_keypoints(project_root: Path, object_base: str, face_key: str) -> Path:
    keypoints_path = project_root / "objects" / object_base / "keypoints.json"
    payload = _load_json_mapping(keypoints_path, "Keypoints JSON")
    errors = validate_keypoint_payload(payload, expected_faces=(face_key,))
    if errors:
        details = "; ".join(errors)
        raise SystemExit(f"[!] Malformed keypoints JSON: {keypoints_path}: {details}")
    return keypoints_path


def _as_se3_matrix(matrix: Any, label: str) -> np.ndarray:
    try:
        arr = np.asarray(matrix, dtype=float)
    except Exception as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if arr.shape != (4, 4):
        raise ValueError(f"{label} must be a 4x4 matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{label} must be a homogeneous SE(3) matrix.")
    return arr


def compose_board_object_transform(T_cam_board: Any, T_cam_object: Any) -> np.ndarray:
    """Return T_board_object using PoseTag's annotation convention."""

    T_cb = _as_se3_matrix(T_cam_board, "T_cam_board")
    T_co = _as_se3_matrix(T_cam_object, "T_cam_object")
    return inv_se3(T_cb) @ T_co


def build_annotation_record(
    *,
    object_name: str,
    face_key: str,
    board_yaml_path: str | Path,
    shot_raw_path: str | Path,
    rms_px: float,
    T_cam_board: Any,
    T_cam_object: Any,
    corner_mapping: Mapping[str, Sequence[float]],
    clicked_uv: Sequence[Sequence[float]],
    face_corner_names: Sequence[str],
    pnp_fit: Mapping[str, Any],
    tag_size_m: float,
    tag_scale_ratio: float,
    tag_scale_pairs: int,
    tag_scale_auto_corrected: bool,
) -> dict:
    T_board_obj = compose_board_object_transform(T_cam_board, T_cam_object)
    names = list(face_corner_names)
    return {
        "object": object_name,
        "face_key": face_key,
        "board_yaml": str(board_yaml_path),
        "image": str(shot_raw_path),
        "rms_px": float(rms_px),
        "T_board_object": {"matrix": T_board_obj.tolist()},
        "corner_uv": {n: [float(corner_mapping[n][0]), float(corner_mapping[n][1])] for n in names},
        "pnp": {"rvec": [float(x) for x in np.asarray(pnp_fit["rvec"]).ravel()],
                "tvec": [float(x) for x in np.asarray(pnp_fit["tvec"]).ravel()]},
        "clicked_uv_raw": [(float(u), float(v)) for (u, v) in clicked_uv],
        "face_corner_names": names,
        "assignment": {n: [float(corner_mapping[n][0]), float(corner_mapping[n][1])] for n in names},
        "notes": "Accepted.",
        "diagnostics": {
            "board_tag_size_mm": float(tag_size_m * 1000.0),
            "tag_scale_ratio": float(tag_scale_ratio),
            "tag_scale_pairs": int(tag_scale_pairs),
            "tag_scale_auto_corrected": bool(tag_scale_auto_corrected),
        },
    }


def preflight_annotation_shot(
    project_root: Path,
    shot_raw_path: Path,
    *,
    row: Optional[ShotRow] = None,
    manifest_path: Optional[Path] = None,
    calib: Optional[str] = None,
) -> AnnotationPreflight:
    root = Path(project_root).expanduser().resolve()
    shot = shot_raw_path.expanduser()
    if not shot.is_absolute():
        shot = (root / shot).resolve()
    if not shot.exists():
        raise SystemExit(f"[!] Shot image not found: {shot}")
    if not shot.name.endswith("_raw.png"):
        raise SystemExit("[!] Shot image must end with _raw.png")

    row = row or _find_manifest_row_for_shot(root, shot, manifest_path=manifest_path)
    meta_path = (
        _resolve_project_path(root, row.path_meta)
        if row and row.path_meta
        else shot.with_name(shot.name.replace("_raw.png", "_meta.json"))
    )
    if meta_path is None:
        raise SystemExit(f"[!] Shot metadata JSON not found for: {shot}")
    meta_json = _load_json_mapping(meta_path, "Shot metadata JSON")
    K, board_yaml_path_text, meta_json = _load_K_and_face(root, meta_path, row, meta_json=meta_json)
    board_yaml_path = Path(board_yaml_path_text)
    _validate_board_yaml(board_yaml_path)

    object_base, face_label, face_key = _annotation_target(
        shot,
        meta_json,
        row,
        board_yaml_path=board_yaml_path,
    )
    keypoints_path = _validate_keypoints(root, object_base, face_key)

    image = meta_json.get("image", {})
    image = image if isinstance(image, Mapping) else {}
    ann_value = image.get("path_ann") or (row.path_ann if row and row.path_ann else None)
    annotated_path = _resolve_project_path(root, ann_value) if ann_value else shot.with_name(shot.name.replace("_raw.png", "_ann.png"))
    assert annotated_path is not None
    if not annotated_path.exists():
        raise SystemExit(f"[!] Captured annotated shot image not found: {annotated_path}")
    reprojection_path = annotated_path.with_name(annotated_path.name.replace("_ann", "_reproj"))

    calib_path = _resolve_calib_path(root, calib)
    if calib_path is not None:
        _load_dist_from_calib(str(calib_path))

    return AnnotationPreflight(
        shot_raw_path=shot,
        annotated_path=annotated_path,
        reprojection_path=reprojection_path,
        meta_path=meta_path,
        board_yaml_path=board_yaml_path,
        keypoints_path=keypoints_path,
        object_base=object_base,
        side=face_label,
        face_key=face_key,
        K=K,
        meta_json=meta_json,
        row=row,
        calib_path=calib_path,
    )


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


def _text_panel(
    lines: List[str],
    w=380,
    h=720,
    scale=0.5,
    thk=1,
    color=(240, 240, 240),
    bg=(18, 18, 18),
):
    """Draws a list of strings with wrapping and end/middle ellipses as needed."""
    img = np.full((h, w, 3), bg, np.uint8)
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
            "Drag a rectangle around the face.",
            "Release to create 4 corners.",
            "Drag any corner to refine.",
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
    return _text_panel(
        lines,
        w=UI_W_RIGHT,
        h=h,
        color=(36, 52, 68),
        bg=(247, 250, 252),
    )


def _show_dash(L, M, R, banner=""):
    strip = _hstack(L, M, R)
    try:
        cv2.resizeWindow("Annotate", strip.shape[1], strip.shape[0])
    except Exception:
        pass
    cv2.imshow("Annotate", strip)


# -------------------------- Annotation widgets --------------------------

def _load_dist_from_calib(calib_path: Optional[str]):
    if not calib_path:
        return np.zeros((1, 5), dtype=float)
    path = Path(calib_path).expanduser()
    if not path.exists():
        raise SystemExit(f"[!] Calibration YAML not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            y = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SystemExit(f"[!] Calibration YAML is malformed: {path}: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"[!] Could not read calibration YAML: {path}: {exc}") from exc
    if not isinstance(y, Mapping):
        raise SystemExit(f"[!] Calibration YAML must contain a mapping: {path}")
    dc = y.get("distortion_coefficients")
    if not isinstance(dc, Mapping):
        raise SystemExit(f"[!] Calibration YAML missing distortion_coefficients: {path}")
    coeffs = [
        _finite_float(dc.get("k1"), "k1", "calibration distortion_coefficients"),
        _finite_float(dc.get("k2"), "k2", "calibration distortion_coefficients"),
        _finite_float(dc.get("p1"), "p1", "calibration distortion_coefficients"),
        _finite_float(dc.get("p2"), "p2", "calibration distortion_coefficients"),
        _finite_float(dc.get("k3", 0.0), "k3", "calibration distortion_coefficients"),
    ]
    return np.array([coeffs], dtype=float)


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


def _annotation_display_scale(img) -> float:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 1.0
    return min(UI_W_LEFT / float(w), UI_H / float(h), 1.0)


def _resize_for_annotation_display(img, scale: float):
    if img is None:
        return None
    if abs(scale - 1.0) < 1e-9:
        return img.copy()
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _display_pts_to_image_pts(
    pts: Sequence[Tuple[float, float]],
    scale: float,
) -> list[Tuple[float, float]]:
    if abs(scale - 1.0) < 1e-9:
        return [(float(u), float(v)) for u, v in pts]
    return [(float(u) / scale, float(v) / scale) for u, v in pts]


def _draw_annotation_banner(vis, lines, *, accent=(0, 160, 210)):
    if not lines:
        return
    pad_x = 14
    pad_y = 11
    line_h = 22
    max_w = 0
    for line in lines:
        (tw, _th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
        max_w = max(max_w, tw)
    x0, y0 = 16, 16
    x1 = min(vis.shape[1] - 16, x0 + max_w + pad_x * 2)
    y1 = min(vis.shape[0] - 16, y0 + line_h * len(lines) + pad_y)
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 28, 38), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + 5, y1), accent, -1)
    cv2.addWeighted(overlay, 0.82, vis, 0.18, 0, dst=vis)
    for idx, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (x0 + pad_x, y0 + 24 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (245, 248, 250),
            1,
            cv2.LINE_AA,
        )


def _draw_corner_handles(vis, pts, *, color=(0, 210, 110)):
    if not pts:
        return
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        if len(pts) == 4 or i + 1 < len(pts):
            cv2.line(
                vis,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                color,
                2,
                cv2.LINE_AA,
            )
    for i, (u, v) in enumerate(pts):
        center = (int(u), int(v))
        cv2.circle(vis, center, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, center, 8, color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(i + 1),
            (center[0] + 10, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_review_corners(
    canvas,
    pts: Sequence[Sequence[float]],
    *,
    color: tuple[int, int, int],
    names: Optional[Sequence[str]] = None,
    radius: int = 5,
    closed: bool = True,
) -> None:
    points = [(float(u), float(v)) for u, v in pts]
    if not points:
        return
    if len(points) > 1:
        pair_count = len(points) if closed and len(points) > 2 else len(points) - 1
        for i in range(pair_count):
            a = points[i]
            b = points[(i + 1) % len(points)]
            cv2.line(
                canvas,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                color,
                2,
                cv2.LINE_AA,
            )
    for i, (u, v) in enumerate(points):
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            continue
        cv2.circle(canvas, (x, y), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        if names and i < len(names):
            cv2.putText(
                canvas,
                str(names[i]),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )


def _collect_quad(image_bgr, mid_img=None, right_img=None):
    display_scale = _annotation_display_scale(image_bgr)
    base = _resize_for_annotation_display(image_bgr, display_scale)
    mid_display = _resize_for_annotation_display(mid_img, display_scale)
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
            overlay = vis.copy()
            cv2.rectangle(overlay, tl, br, (0, 170, 230), -1)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, dst=vis)
            cv2.rectangle(vis, tl, br, (0, 190, 255), 3, cv2.LINE_AA)
            cx, cy = cur
            cv2.line(vis, (cx, 0), (cx, h - 1), (0, 190, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (0, cy), (w - 1, cy), (0, 190, 255), 1, cv2.LINE_AA)
            for p in (tl, tr, br, bl):
                cv2.circle(vis, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, p, 5, (0, 190, 255), 2, cv2.LINE_AA)
        if pts:
            _draw_corner_handles(vis, pts, color=(0, 210, 110))
            _draw_annotation_banner(
                vis,
                [
                    "Refine the four corners, then press ENTER.",
                    "Drag handles. Hold SHIFT for axis lock. U undo, R reset.",
                ],
                accent=(0, 150, 92),
            )
        else:
            _draw_annotation_banner(
                vis,
                [
                    "Drag a rectangle around this face.",
                    "Release to create four adjustable corners.",
                ],
                accent=(0, 140, 190),
            )
        _show_dash(vis, mid_display, right_help)

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
        if k == 13 and len(pts) == 4: return _display_pts_to_image_pts(pts, display_scale)


def _collect_any4(image_bgr, mid_img=None, right_img=None):
    pts = [];
    display_scale = _annotation_display_scale(image_bgr)
    base = _resize_for_annotation_display(image_bgr, display_scale)
    mid_display = _resize_for_annotation_display(mid_img, display_scale)
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
        if pts:
            _draw_corner_handles(vis, pts, color=(0, 210, 110))
        if cur is not None and pts:
            u0, v0 = pts[-1];
            u1, v1 = snap((u0, v0), cur, last_flags)
            cv2.line(vis, (int(u0), int(v0)), (int(u1), int(v1)), (255, 180, 60), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, (int(u1), int(v1)), (255, 180, 60), cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
        if len(pts) == 4:
            _draw_annotation_banner(
                vis,
                ["Four corners selected. Press ENTER to solve pose."],
                accent=(0, 150, 92),
            )
        else:
            _draw_annotation_banner(
                vis,
                [
                    f"Click visible face corners: {len(pts)}/4 selected.",
                    "Hold SHIFT for axis lock. U undo, R reset.",
                ],
                accent=(0, 140, 190),
            )
        _show_dash(vis, mid_display, right_help)

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
        if k == 13 and len(pts) == 4: return _display_pts_to_image_pts(pts, display_scale)


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
    rating = "check fit" if rms > 5.0 else "good fit"
    lines = [
        "Review",
        f"RMS error  {rms:.2f}px",
        f"Status     {rating}",
        "",
        "Overlay",
        "cyan  clicked corners",
        "green solved face corners",
        "",
        "Actions",
        "ENTER / Y accept",
        "R / BACKSPACE redo",
        "ESC abort shot",
        "Q quit batch",
    ]
    display_scale = _annotation_display_scale(anno_img)
    left = _resize_for_annotation_display(anno_img, display_scale)
    mid = _resize_for_annotation_display(reproj_img, display_scale)
    right = _text_panel(
        lines,
        300,
        left.shape[0],
        scale=0.48,
        color=(36, 52, 68),
        bg=(248, 251, 253),
    )
    while True:
        _show_dash(left, mid, right, "")
        k = cv2.waitKey(50) & 0xFF
        if k in (13, ord('y'), ord('s')): return "accept"
        if k in (ord('r'), ord('n'), 8):  return "redo"
        if k in (ord('Q'), ord('X'), ord('x')): _quit_all()
        if k in (27, ord('q')): raise SystemExit("Aborted by user during review.")


def _render_list_faces(h, w, title, items, sel):
    pan = np.full((h, w, 3), (250, 252, 253), np.uint8)
    header_h = 76
    footer_h = 44
    row_h = 42
    cv2.rectangle(pan, (0, 0), (w, header_h), (238, 246, 250), -1)
    cv2.putText(
        pan,
        "Faces",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (22, 34, 48),
        2,
        cv2.LINE_AA,
    )
    subtitle = _ellipsize_middle(title.replace("Faces ", ""), w - 110, 0.42, 1)
    cv2.putText(
        pan,
        subtitle,
        (14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (89, 104, 118),
        1,
        cv2.LINE_AA,
    )
    progress = f"{sel + 1}/{len(items)}" if items else "0/0"
    (pw, ph), _ = cv2.getTextSize(progress, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(pan, (w - pw - 34, 16), (w - 14, 40), (222, 238, 247), -1)
    cv2.putText(
        pan,
        progress,
        (w - pw - 24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (49, 79, 98),
        1,
        cv2.LINE_AA,
    )
    face_x = 14
    rms_x = max(156, w - 212)
    board_x = max(rms_x + 78, w - 126)
    kp_x = max(board_x + 56, w - 62)
    for label, x in (("Face", face_x), ("RMS", rms_x), ("Board", board_x), ("KP", kp_x)):
        cv2.putText(
            pan,
            label,
            (x, header_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (86, 105, 122),
            1,
            cv2.LINE_AA,
        )

    max_show = max(1, min(len(items), (h - header_h - footer_h) // row_h))
    start = max(0, min(sel - max_show // 2, max(0, len(items) - max_show)))
    y = header_h + 4
    for i in range(start, start + max_show):
        if i >= len(items): break
        it = items[i]
        selected = i == sel

        y0 = y
        y1 = min(h - footer_h - 4, y0 + row_h - 4)
        row_bg = (227, 243, 253) if selected else (255, 255, 255)
        row_border = (11, 111, 143) if selected else (216, 229, 238)
        cv2.rectangle(pan, (8, y0), (w - 10, y1), row_bg, -1)
        cv2.rectangle(pan, (8, y0), (w - 10, y1), row_border, 2 if selected else 1)
        cv2.rectangle(
            pan,
            (8, y0),
            (12, y1),
            (11, 111, 143) if selected else (183, 151, 86),
            -1,
        )

        avail = max(60, rms_x - 22)
        base_name = f"{it['object']}  {it['side']}"
        name = _ellipsize_end(base_name, avail, scale=0.45, thk=1)
        col = (22, 34, 48) if selected else (48, 63, 79)
        cv2.putText(
            pan,
            name,
            (20, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )
        sub = "annotated" if it["ann_ok"] else "missing transform"
        cv2.putText(
            pan,
            sub,
            (20, y0 + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (95, 106, 118),
            1,
            cv2.LINE_AA,
        )

        if it["ann_ok"]:
            if it.get("rms") is None:
                rms_text = "OK"
                rms_kind = "ok"
            else:
                rms_text = f"{it['rms']:.1f}px"
                rms_kind = _rms_kind(it.get("rms"))
        else:
            rms_text = "missing"
            rms_kind = "warn"
        chips = (
            (rms_text, rms_kind, rms_x),
            ("OK" if it["board_ok"] else "miss", "ok" if it["board_ok"] else "warn", board_x),
            ("OK" if it["kp_ok"] else "miss", "ok" if it["kp_ok"] else "warn", kp_x),
        )
        for text, kind, x in chips:
            chip = _pill(text, kind)
            ch, cw = chip.shape[:2]
            max_w = max(1, min(cw, w - x - 10))
            chip_y = y0 + max(6, (row_h - ch) // 2 - 1)
            pan[chip_y:chip_y + ch, x:x + max_w] = chip[:, :max_w]
        y += row_h

    if len(items) > max_show:
        track_top = header_h + 10
        track_bottom = h - footer_h - 12
        track_h = max(1, track_bottom - track_top)
        thumb_h = max(24, int(track_h * max_show / max(1, len(items))))
        max_start = max(1, len(items) - max_show)
        thumb_y = track_top + int((track_h - thumb_h) * start / max_start)
        cv2.rectangle(pan, (w - 8, track_top), (w - 4, track_bottom), (219, 231, 239), -1)
        cv2.rectangle(pan, (w - 9, thumb_y), (w - 3, thumb_y + thumb_h), (85, 112, 138), -1)

    cv2.rectangle(pan, (0, h - footer_h), (w, h), (250, 252, 253), -1)
    tips = "J/K, arrows, W/S select   ENTER annotate/redo   A mode   R reload   Q quit"
    tips_lines = _wrap_to_width(tips, w - 24, scale=0.46, thk=1)
    line_h = 18
    y0 = h - 10 - line_h * (len(tips_lines) - 1)
    for i, t in enumerate(tips_lines):
        cv2.putText(
            pan,
            t,
            (12, y0 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (89, 104, 118),
            1,
            cv2.LINE_AA,
        )
    return pan


def _render_raw_thumb(path: Path, target_h=UI_H, target_w=UI_W_LEFT, label: str | None = None):
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
    title = label or Path(path).name
    title = _ellipsize_end(str(title), max(1, target_w - 24), scale=0.7, thk=2)
    cv2.putText(out, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# -------------------------- Core per-shot flow --------------------------
def _load_K_and_face(project_root: Path, meta_path: Path, row, *, meta_json: Optional[dict] = None) -> tuple[np.ndarray, str, dict]:
    """Return (K, face_yaml_abs, meta_json). Works with old and new meta."""
    if meta_json is None:
        meta_json = _load_json_mapping(meta_path, "Shot metadata JSON")

    # Camera intrinsics
    if isinstance(meta_json.get("camera"), Mapping):
        cam = meta_json["camera"]
        K = _camera_matrix_from_values(cam, f"metadata camera ({meta_path})")
    elif row and all(getattr(row, k) is not None for k in ("fx", "fy", "cx", "cy")):
        K = _camera_matrix_from_values(
            {"fx": row.fx, "fy": row.fy, "cx": row.cx, "cy": row.cy},
            "shots manifest row",
        )
    else:
        raise SystemExit("[!] Camera intrinsics not found (meta.camera or manifest columns).")

    # face_yaml (absolute)
    face_yaml = meta_json.get("face_yaml") or (row.face_yaml if row else None)
    if not face_yaml:
        raise SystemExit("[!] face_yaml missing (meta or manifest).")
    fy_path = _resolve_project_path(project_root, face_yaml)
    if fy_path is None:
        raise SystemExit("[!] face_yaml missing (meta or manifest).")

    return K, str(fy_path), meta_json

# Main function for annotating a single shot: detects tags, collects user points, solves PnP, saves results
def annotate_single_shot(project_root: Path, shot_raw_path: Path, *,
                         pts_type: str = "quad", calib: Optional[str] = None,
                         check_tag_scale: bool = False, auto_correct_scale: bool = False,
                         scale_tol: float = 0.02) -> Path:
    project_root = Path(project_root).expanduser().resolve()
    preflight = preflight_annotation_shot(project_root, shot_raw_path, calib=calib)
    shot_raw_path = preflight.shot_raw_path
    K = preflight.K
    face_yaml_path = str(preflight.board_yaml_path)
    dist = _load_dist_from_calib(str(preflight.calib_path) if preflight.calib_path else None)

    obj = preflight.object_base
    face_label = preflight.side
    face_key = preflight.face_key
    img_ann = preflight.annotated_path
    img_reproj = preflight.reprojection_path

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

    def _review_images(current_T_cam_obj, current_ordered2d):
        face_X = np.vstack([pts3d_dict[n] for n in names_in_order]).astype(np.float32)
        uv4, _ = _reproj(face_X, current_T_cam_obj, K, img, draw=False, dist=dist)
        err = np.linalg.norm(uv4 - current_ordered2d, axis=1)
        current_rms = float(np.sqrt((err ** 2).mean()))

        anno = img.copy()
        _draw_review_corners(
            anno,
            current_ordered2d,
            color=(255, 210, 0),
            names=[f"C{i + 1}" for i in range(len(current_ordered2d))],
            radius=4,
        )
        _draw_review_corners(
            anno,
            uv4,
            color=(0, 210, 95),
            names=names_in_order,
            radius=5,
        )

        mid = img.copy()
        _draw_review_corners(
            mid,
            current_ordered2d,
            color=(255, 210, 0),
            names=[f"C{i + 1}" for i in range(len(current_ordered2d))],
            radius=4,
        )
        _draw_review_corners(
            mid,
            uv4,
            color=(0, 210, 95),
            names=names_in_order,
            radius=4,
        )
        _draw_annotation_banner(
            mid,
            [
                f"RMS {current_rms:.2f}px",
                "Cyan = clicked corners. Green = solved face projection.",
            ],
            accent=(0, 150, 92) if current_rms <= 5.0 else (0, 145, 220),
        )
        return anno, mid, uv4, current_rms

    anno, mid, uv4, rms = _review_images(T_cam_obj, ordered2d)

    _flush_keys(120)
    while True:
        decision = _review(anno, mid, rms)
        if decision == "redo":
            clicked = (_collect_quad if pts_type == "quad" else _collect_any4)(img, mid, None)
            mapping, ordered2d, fit = _assign_any_order(clicked, names_in_order, pts3d_dict, K, dist)
            R, _ = cv2.Rodrigues(fit["rvec"]);
            T_cam_obj = se3(R, fit["tvec"].reshape(3))
            anno, mid, uv4, rms = _review_images(T_cam_obj, ordered2d)
            continue
        break

    # Compose outputs: compute board-to-object transform and save YAML, images.
    # PoseTag runtime later composes T_cam_object = T_cam_board @ T_board_object.

    out_yaml = _out_yaml_path(project_root, obj, face_label, face_key=face_key)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    Path(img_ann).parent.mkdir(parents=True, exist_ok=True)

    # Save images
    cv2.imwrite(str(img_ann), anno)
    cv2.imwrite(str(img_reproj), mid)

    # Save YAML with all annotation data
    out = build_annotation_record(
        object_name=obj,
        face_key=face_key,
        board_yaml_path=face_yaml_path,
        shot_raw_path=shot_raw_path,
        rms_px=rms,
        T_cam_board=T_cam_board,
        T_cam_object=T_cam_obj,
        corner_mapping=mapping,
        clicked_uv=clicked,
        face_corner_names=names_in_order,
        pnp_fit=fit,
        tag_size_m=tag_size_m,
        tag_scale_ratio=s,
        tag_scale_pairs=n_pairs,
        tag_scale_auto_corrected=bool(auto_correct_scale and abs(s - 1.0) > scale_tol),
    )
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
    side_filter = _normalise_side_letter(side)
    for r in rows:
        if not r.path_raw: continue
        if object_filter and object_filter.lower() not in r.object_base.lower(): continue
        if side_filter and _normalise_side_letter(r.side) != side_filter: continue
        key = (r.object_base, _normalise_side_letter(r.side), Path(r.face_yaml).stem)
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

def main(argv=None):
   # Parse command-line arguments for single-shot, batch, or browse modes
    parser = argparse.ArgumentParser("Annotate PoseTag faces (single or batch via manifest).")
    parser.add_argument("--shot", type=str, help="Absolute path to *_raw.png (single-shot mode).")
    parser.add_argument("--batch", choices=["latest"], help="Batch mode from manifest.csv.")
    parser.add_argument("--manifest", type=str, help="Path to manifest.csv (default: <proj>/shots/manifest.csv).")
    parser.add_argument("--object-filter", type=str, default=None, help="Substring filter for object_base.")
    parser.add_argument("--side", type=str, default=None, help="Restrict to side letter A|B|C|D.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing YAML if present.")
    parser.add_argument("--pts-type", choices=["quad", "any"], default="quad", help="Corner input mode.")
    parser.add_argument("--calib", type=str, default=None, help="Optional ChArUco calib yaml for distortion.")
    parser.add_argument("--dry-run", action="store_true", help="List and validate actions then exit.")
    parser.add_argument("--check-tag-scale", action="store_true", help="Print scale ratio s from inter-tag distances.")
    parser.add_argument("--auto-correct-scale", action="store_true",
                        help="If |s-1|>tol, divide T_cam_board translation by s.")
    parser.add_argument("--scale-tol", type=float, default=0.02, help="Relative tolerance (default 0.02 = 2%%).")
    parser.add_argument("--browse", action="store_true",
                        help="Interactive browser (arrow keys + ENTER) to pick which face to annotate.")

    args = parser.parse_args(argv)

    mode_count = int(bool(args.shot)) + int(bool(args.batch)) + int(bool(args.browse))
    if mode_count == 0:
        parser.error("choose one mode: --shot, --batch latest, or --browse")
    if mode_count > 1:
        parser.error("choose only one mode: --shot, --batch, or --browse")

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
            if args.dry_run:
                preflight = preflight_annotation_shot(project_root, Path(args.shot), calib=args.calib)
                log.info("[i] Dry-run single-shot annotation:")
                log.info(f"  raw image    : {preflight.shot_raw_path}")
                log.info(f"  metadata     : {preflight.meta_path}")
                log.info(f"  board YAML   : {preflight.board_yaml_path}")
                log.info(f"  keypoints    : {preflight.keypoints_path}")
                log.info(
                    "  output YAML  : "
                    f"{_out_yaml_path(project_root, preflight.object_base, preflight.side, face_key=preflight.face_key)}"
                )
                log.info(f"  review image : {preflight.annotated_path}")
                log.info(f"  reprojection : {preflight.reprojection_path}")
                log.info("[i] Dry-run only. Exiting.")
                return
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
    preflight_errors: list[str] = []
    for r in pick:
        try:
            preflight = preflight_annotation_shot(
                project_root,
                Path(r.path_raw),
                row=r,
                manifest_path=manifest,
                calib=args.calib,
            )
            out_yaml = _out_yaml_path(
                project_root,
                preflight.object_base,
                preflight.side,
                face_key=preflight.face_key,
            )
            status = "(exists)" if (out_yaml.exists() and not args.force) else ""
            log.info(
                f"  - {preflight.object_base} {preflight.side}  "
                f"ts={r.timestamp}  -> {out_yaml} {status}"
            )
        except SystemExit as e:
            message = str(e) or "annotation preflight failed"
            preflight_errors.append(f"{r.object_base} {r.side}: {message}")
            log.error(f"  - {r.object_base} {r.side}  ts={r.timestamp}: {message}")

    if args.dry_run:
        if preflight_errors:
            sys.exit("[!] Dry-run found invalid annotation input(s).")
        log.info("[i] Dry-run only. Exiting.")
        return

    for r in pick:
        try:
            preflight = preflight_annotation_shot(
                project_root,
                Path(r.path_raw),
                row=r,
                manifest_path=manifest,
                calib=args.calib,
            )
            out_yaml = _out_yaml_path(
                project_root,
                preflight.object_base,
                preflight.side,
                face_key=preflight.face_key,
            )
            if out_yaml.exists() and not args.force:
                log.info(f"[i] Skip (already done): {out_yaml}")
                continue
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
