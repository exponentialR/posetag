import argparse, os, sys, json, time, datetime, re
import yaml
import numpy as np
import cv2
try:
    import pyrealsense2 as rs
except Exception:
    rs = None
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import csv
import yaml, logging

SIDE_RE = re.compile(r"^(?P<base>.+?)_side(?P<side>[A-Za-z]+)$")
log = logging.getLogger("posetag")

def _append_manifest(manifest_path: str, meta: dict):
    """
    Append one capture to a csv manifest (creates with header if needed)
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    row = {
        "timestamp": meta.get("timestamp", ""),
        "object_base": meta.get("object_base", ""),
        "object_full": meta.get("object_full", ""),
        "side": meta.get("side", ""),
        "face_yaml": meta.get("face_yaml", "") or "",
        "path_raw": meta["image"].get("path_raw", ""),
        "path_ann": meta["image"].get("path_ann", ""),
        "path_meta": meta["image"].get("path_meta", ""),
        "width": meta["image"].get("width", 0),
        "height": meta["image"].get("height", 0),
        "fx": meta["camera"].get("fx", 0.0),
        "fy": meta["camera"].get("fy", 0.0),
        "cx": meta["camera"].get("cx", 0.0),
        "cy": meta["camera"].get("cy", 0.0),
        "detected_ids": " ".join(map(str, meta.get("detected_tag_ids", []))),
        "expected_ids": " ".join(map(str, meta.get("expected_tag_ids", []))),
        "validation_ok": int(bool(meta.get("validation_ok", False))),
        "auto_face": int(bool(meta.get("auto_face", False))),
        }
    header = list(row.keys())
    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
                w.writeheader()
        w.writerow(row)


def _resolve_yaml_path(project_root, p: str) -> str | None:
    if not p:
        return None
    p = p.replace("\\", "/")  # tolerate Windows entries
    cand = Path(p)
    if not cand.is_absolute():
        cand = project_root / cand
    if not cand.exists():
        alt = project_root / "boards" / cand.name  # fallback by basename
        if alt.exists():
            cand = alt
    return str(cand)

def _wrap_lines(lines, max_w, scale=0.60, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Word-wrap each line to fit max_w pixels."""
    out = []
    for line in lines:
        words = line.split()
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
            if tw <= max_w or not cur:
                cur = test
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
    return out

def make_recent_panel(h: int, w: int, last_img: np.ndarray, show: bool) -> np.ndarray:
    bg = (245, 249, 251)
    card = (255, 255, 255)
    border = (212, 226, 234)
    text = (36, 49, 61)
    muted = (98, 113, 126)
    panel = np.full((h, w, 3), bg, np.uint8)
    margin = 14
    cv2.rectangle(panel, (margin, margin), (w - margin, h - margin), card, -1)
    cv2.rectangle(panel, (margin, margin), (w - margin, h - margin), border, 1)
    cv2.putText(panel, "SAVED PREVIEW", (margin + 14, margin + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, text, 1, cv2.LINE_AA)

    if not show:
        cv2.putText(panel, "Panel hidden. Press g to show it.",
                    (margin + 14, margin + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, muted, 1, cv2.LINE_AA)
        return panel

    if last_img is None:
        preview_top = margin + 48
        preview_bottom = min(h - margin - 54, preview_top + max(80, h // 3))
        cv2.rectangle(
            panel,
            (margin + 14, preview_top),
            (w - margin - 14, preview_bottom),
            (236, 243, 247),
            -1,
        )
        cv2.rectangle(
            panel,
            (margin + 14, preview_top),
            (w - margin - 14, preview_bottom),
            border,
            1,
        )
        lines = ["Waiting for first save", "Auto-save or ENTER writes here"]
        y = preview_bottom + 28
        for line in _wrap_lines(lines, max(1, w - 2 * margin - 28), 0.44, 1):
            cv2.putText(panel, line, (margin + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, muted, 1, cv2.LINE_AA)
            y += 20
        return panel

    # fit-preserving resize with 12px margins
    INNER = 22
    header_h = 48
    tgt_w = max(1, w - 2*margin - 2*INNER)
    tgt_h = max(1, h - 2*margin - header_h - INNER)
    scale = min(tgt_w / last_img.shape[1], tgt_h / last_img.shape[0])
    new_w = max(1, int(last_img.shape[1] * scale))
    new_h = max(1, int(last_img.shape[0] * scale))
    img = cv2.resize(last_img, (new_w, new_h))

    x0 = margin + INNER + (tgt_w - new_w) // 2
    y0 = margin + header_h + (tgt_h - new_h) // 2
    panel[y0:y0+new_h, x0:x0+new_w] = img

    return panel


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(4, (max_chars - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def make_info_panel(
    h: int,
    w: int,
    state: dict,
    gallery_on: bool=True,
    interaction: Optional[dict]=None,
) -> np.ndarray:
    # Colours
    BG = (245, 249, 251)
    CARD_BG = (255, 255, 255)
    BORDER = (212, 226, 234)
    TITLE = (36, 49, 61)
    TEXT = (44, 58, 71)
    OK_BG = (64, 134, 29)
    BAD_BG = (24, 139, 204)

    panel = np.full((h, w, 3), BG, np.uint8)
    M, S = 14, 12  # margin, spacing
    INNER = 12
    SCALE = 0.44
    THICK = 1
    TITLE_SCALE = 0.48
    if interaction is not None:
        interaction.clear()
        interaction["queue_rows"] = []

    def draw_card(y, title, body_lines, h_fixed=None, status=None):
        inner_w = w - 2*M - 2*INNER
        wrapped = _wrap_lines(body_lines, inner_w, SCALE, THICK)
        body_h = 18 * len(wrapped)
        H = h_fixed if h_fixed is not None else (38 + body_h + INNER)
        x0, y0, x1, y1 = M, y, w - M, min(h - M, y + H)
        # card
        cv2.rectangle(panel, (x0, y0), (x1, y1), CARD_BG, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), BORDER, 1)
        # title
        cv2.putText(panel, title.upper(), (x0 + INNER, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE, TITLE, 1, cv2.LINE_AA)
        # status chip
        if status is not None:
            txt, good = status
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            pad = 6
            rx1 = x1 - INNER
            rx0 = rx1 - tw - 2*pad
            ry0 = y0 + 10
            ry1 = ry0 + th + 2*pad
            cv2.rectangle(panel, (rx0, ry0), (rx1, ry1), OK_BG if good else BAD_BG, -1)
            cv2.putText(panel, txt, (rx0 + pad, ry1 - pad - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        # body
        yy = y0 + 40
        for s in wrapped:
            if yy + 18 >= y1 - INNER: break  # truncate if needed
            cv2.putText(panel, s, (x0 + INNER, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, SCALE, TEXT, THICK, cv2.LINE_AA)
            yy += 18
        return y1

    def draw_queue_card(y, queue_faces, current, captured):
        max_rows = min(6, len(queue_faces))
        start = max(0, min(current - max_rows // 2, len(queue_faces) - max_rows))
        end = min(len(queue_faces), start + max_rows)
        done = 0
        for item in queue_faces:
            name = str(item.get("object", "")).strip()
            if name in captured:
                done += 1

        row_h = 24
        H = 50 + row_h * max_rows + INNER
        x0, y0, x1, y1 = M, y, w - M, min(h - M, y + H)
        cv2.rectangle(panel, (x0, y0), (x1, y1), CARD_BG, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), BORDER, 1)
        cv2.putText(panel, "FACE QUEUE", (x0 + INNER, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE, TITLE, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{done}/{len(queue_faces)} captured",
                    (x0 + INNER, y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, SCALE, TEXT, THICK, cv2.LINE_AA)

        yy = y0 + 68
        for index in range(start, end):
            item = queue_faces[index]
            name = str(item.get("object", "")).strip() or os.path.splitext(
                os.path.basename(str(item.get("yaml", "")))
            )[0]
            marker = "[x]" if name in captured else "[ ]"
            selected = index == current
            row_top = yy - 18
            row_bottom = min(y1 - INNER, yy + 6)
            if interaction is not None:
                interaction.setdefault("queue_rows", []).append(
                    (index, x0 + INNER - 2, row_top, x1 - INNER + 2, row_bottom)
                )
            if selected:
                cv2.rectangle(
                    panel,
                    (x0 + INNER - 4, row_top),
                    (x1 - INNER + 4, row_bottom),
                    (236, 246, 251),
                    -1,
                )
                cv2.rectangle(
                    panel,
                    (x0 + INNER - 4, row_top),
                    (x0 + INNER, row_bottom),
                    OK_BG if name in captured else BAD_BG,
                    -1,
                )
            pointer = ">" if selected else " "
            colour = OK_BG if name in captured else (TEXT if selected else (98, 113, 126))
            cv2.putText(
                panel,
                f"{pointer} {marker} {_truncate_middle(name, 34)}",
                (x0 + INNER + 4, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                SCALE,
                colour,
                1 if not selected else 2,
                cv2.LINE_AA,
            )
            yy += row_h
        return y1

    # ----- gather state
    obj_base = state.get("object_base")
    face = state.get("face")
    auto = state.get("auto_side", False)
    det_ids = sorted(list(state.get("detected_ids", [])))
    exp_ids = sorted(list(set(face["tag_ids"])) if face else [])
    overlap = sorted(list(set(det_ids).intersection(exp_ids))) if face else []
    ok = state.get("validation_ok", True)
    save_warn = state.get("save_warn", False)

    # ----- INFO card
    lines = []
    if obj_base: lines.append(f"Object base: {obj_base}")
    if face:
        b, side = parse_base_and_side(face['object'])
        face_key = os.path.splitext(os.path.basename(face["yaml"]))[0]
        lines += [
            f"Resolved face: {face['object']} (side={side or '?'})",
            f"Face key: {face_key}",
        ]
    else:
        lines.append("Resolved face: (none)")
    faces = state.get("faces") or []
    face_idx = int(state.get("face_idx", 0)) + 1 if faces else 0
    lines += [
        f"Mode: {'auto face' if auto else 'manual face'}",
        f"Face navigation: {face_idx}/{len(faces)}",
        f"Seen IDs: {det_ids}",
    ]
    if face:
        lines += [f"Expected: {exp_ids}", f"Overlap: {overlap}"]

    y = M
    y = draw_card(y, "Capture status", lines, status=("OK" if ok else "NOT OK", ok)) + S

    # ----- QUEUE card
    queue_faces = state.get("faces") or []
    captured = state.get("captured_faces") or set()
    if queue_faces:
        current = int(state.get("face_idx", 0))
        y = draw_queue_card(y, queue_faces, current, captured) + S

    # ----- INSTRUCTIONS card (ASCII-only)
    instr = [
        "- ENTER: Save; double-press to force",
        "- Click or scroll: Move through face queue",
        "- Arrow keys: Move through face queue",
        "- a: Auto/manual; h: Help; q/Esc: Quit",
    ]
    y = draw_card(y, "Controls", instr) + S

    # ----- GALLERY card (single column, truncates safely)
    if gallery_on:
        thumbs = state.get("thumbs", []) or []
        if thumbs:
            x0, y0, x1 = M, y, w - M
            inner_w = x1 - x0 - 2*INNER
            y_cursor = y0 + 42
            # header
            y1 = min(h - M, y0 + 42 + 8)  # temp
            cv2.rectangle(panel, (x0, y0), (x1, h - M), CARD_BG, -1)
            cv2.rectangle(panel, (x0, y0), (x1, h - M), BORDER, 1)
            cv2.putText(panel, "RECENT SHOTS", (x0 + INNER, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE, TITLE, 1, cv2.LINE_AA)
            # thumbnails (most recent first)
            for th in thumbs[::-1]:
                if th is None: continue
                scale = min(1.0, inner_w / th.shape[1])
                th_res = cv2.resize(th, (int(th.shape[1]*scale), int(th.shape[0]*scale)))
                h_avail = (h - M) - y_cursor - INNER
                if th_res.shape[0] > h_avail:
                    if h_avail <= 0: break
                    th_res = th_res[:h_avail, :]
                # clip-safe paste
                y2 = min(y_cursor + th_res.shape[0], panel.shape[0])
                xL = x0 + INNER
                xR = min(xL + th_res.shape[1], panel.shape[1])
                if y_cursor >= y2 or xL >= xR: break
                panel[y_cursor:y2, xL:xR] = th_res[:y2 - y_cursor, :xR - xL]
                y_cursor = y2 + 8
            # advance y to bottom of card
            y = min(h - M, y_cursor + INNER) + S
        else:
            y = draw_card(y, "Recent shots", ["(none yet)"]) + S

    # footer warn strip
    if save_warn:
        warn = "Press ENTER again within 3s to confirm save"
        (tw, th), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(panel, warn, (max(M, (w - tw)//2), h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BAD_BG, 2, cv2.LINE_AA)

    return panel


# ---------- registry helpers ----------
def load_registry(path: str) -> Dict:
    if not os.path.exists(path):
        log.warning(f"[!] tag registry not found: {path}")
        return {"version": 1, "tags": {}}
    reg = yaml.safe_load(open(path, "r")) or {}
    reg.setdefault("tags", {})
    return reg

def faces_for_object(object_name: str, registry: Dict, project_root) -> List[Dict]:
    """Return list of face entries for the *full* object name.
       Each entry: dict(yaml, object, tag_ids=set(...))"""
    by_yaml = {}
    for tid, rec in registry["tags"].items():
        try:
            obj = rec.get("object")
            yml_raw = rec.get("yaml")
            if obj is None or yml_raw is None:
                continue
            if obj != object_name:
                continue
            yml = _resolve_yaml_path(project_root, yml_raw)
            if not yml:
                log.warning(f"[!] Could not resolve yaml path: {yml_raw}")
                continue
            by_yaml.setdefault(yml, {"object": obj, "tag_ids": set()})
            by_yaml[yml]["tag_ids"].add(int(tid))
        except Exception:
            continue

    faces = []
    for yml, info in by_yaml.items():
        try:
            with open(yml, "r") as f:
                yy = yaml.safe_load(f) or {}
            obj_field = yy.get("object", info["object"])
            tags = yy.get("tags", [])
            if tags and not info["tag_ids"]:
                info["tag_ids"] = set(int(t["id"]) for t in tags if "id" in t)
            faces.append({
                "yaml": yml,
                "object": obj_field,
                "tag_ids": sorted(list(info["tag_ids"]))
            })
        except Exception as e:
            log.warning(f"[!] Could not read {yml}: {e}")
    return faces

def parse_base_and_side(name: str) -> Tuple[str, Optional[str]]:
    m = SIDE_RE.match(name)
    if m:
        return m.group("base"), m.group("side")
    return name, None

def unique_bases(registry: Dict) -> List[str]:
    bases: Set[str] = set()
    for rec in registry.get("tags", {}).values():
        obj = rec.get("object")
        if not obj:
            continue
        b, _ = parse_base_and_side(obj)
        bases.add(b)
    return sorted(bases)

def objects_for_base(base: str, registry: Dict) -> List[str]:
    """All full object names for a base (e.g., ..._sideA/B/C/D)."""
    fulls = set()
    for rec in registry.get("tags", {}).values():
        obj = rec.get("object")
        if not obj:
            continue
        b, _ = parse_base_and_side(obj)
        if b == base:
            fulls.add(obj)
    return sorted(fulls)

def faces_for_base(base: str, registry: Dict, repo_root) -> List[Dict]:
    out = []
    for full in objects_for_base(base, registry):
        out.extend(faces_for_object(full, registry, repo_root))
    # stable order: by object, then yaml basename
    out.sort(key=lambda f: (f["object"], os.path.basename(f["yaml"])))
    return out

# ---------- UI helpers ----------
def draw_detections(img, dets, colour=(0, 255, 0)):
    vis = img.copy()
    for d in dets:
        pts = d.corners.astype(int)
        cv2.polylines(vis, [pts], True, colour, 2)
        cv2.putText(vis, str(int(d.tag_id)), tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def text_lines(panel, lines, x=12, y0=28, dy=22, colour=(0,0,0), scale=0.6, thick=1):
    y = y0
    for s in lines:
        cv2.putText(panel, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)
        y += dy
