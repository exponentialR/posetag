from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import json as _json

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import trimesh
except Exception:
    trimesh = None

# --------------------- Globals / UI look ---------------------
UI_H = 720
UI_W_LEFT = 860
UI_W_MID = 420
UI_W_RIGHT = 480
UI_WIN_W = UI_W_LEFT + UI_W_MID + UI_W_RIGHT
PILL = {
    "ok":   ((224,245,228), (35,120,55)),
    "warn": ((241,225,201), (110,85,45)),
    "info": ((229,238,249), (70,95,160)),
}

MESH_EXTS = {".obj", ".ply", ".stl", ".glb", ".gltf", ".off"}

def state_path(project_root: Path) -> Path:
    return project_root / "canonical_keypoints" / ".state.json"

def load_state(project_root: Path) -> dict:
    p = state_path(project_root)
    if p.exists():
        try:
            return _json.loads(p.read_text())
        except Exception:
            pass
    return {
        "last_mesh": None,
        "mode": "auto",
        "method": "fps",
        "count": 16,
        "view": "prompt",
    }

def save_state(project_root: Path, **kwargs):
    p = state_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = load_state(project_root)
    st.update({k: v for k, v in kwargs.items() if v is not None})
    p.write_text(_json.dumps(st, indent=2))



# --------------------- Mesh IO ---------------------

def find_meshes(project_root: Path, pattern: Optional[str], single_path: Optional[str]) -> List[Path]:
    if single_path:
        p = Path(single_path)
        return [p] if p.exists() else []
    if pattern:
        return sorted([p for p in Path().glob(pattern) if p.suffix.lower() in MESH_EXTS])
    meshes_dir = project_root / "meshes"
    return sorted([p for p in meshes_dir.rglob("*") if p.suffix.lower() in MESH_EXTS])

def load_mesh_any(path: Path):
    """Return (open3d_mesh or None, V [N,3], F [M,3])"""
    if o3d is not None:
        try:
            m = o3d.io.read_triangle_mesh(str(path))
            if m.has_vertices() and len(m.triangles) > 0:
                m.compute_vertex_normals()
                V = np.asarray(m.vertices, float)
                F = np.asarray(m.triangles, int)
                return m, V, F
        except Exception:
            pass
    if trimesh is None:
        raise RuntimeError(f"Failed to load mesh {path}; trimesh not available.")
    tm = trimesh.load(str(path), force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(g for g in tm.dump().geometry.values()))
    if not isinstance(tm, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type for {path}")
    V = np.asarray(tm.vertices, float)
    F = np.asarray(tm.faces, int)
    m_o3d = None
    if o3d is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
        mesh.compute_vertex_normals()
        m_o3d = mesh
    return m_o3d, V, F


# --------------------- Auto KPs ---------------------

def fps(points: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), points.shape[0]))
    c = points.mean(0)
    d0 = np.linalg.norm(points - c, axis=1)
    idx0 = int(np.argmax(d0))
    sel = np.empty(k, int)
    sel[0] = idx0
    mind = np.full(points.shape[0], np.inf)
    last = points[idx0]
    mind = np.minimum(mind, np.linalg.norm(points - last, axis=1))
    for i in range(1, k):
        j = int(np.argmax(mind))
        sel[i] = j
        last = points[j]
        mind = np.minimum(mind, np.linalg.norm(points - last, axis=1))
    return sel

def curvature_scores(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    if trimesh is not None:
        try:
            tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
            bb = V.max(0) - V.min(0)
            rad = float(np.linalg.norm(bb)) * 0.01 + 1e-9
            from trimesh.curvature import discrete_gaussian_curvature_measure
            curv = discrete_gaussian_curvature_measure(tm, tm.vertices, radius=rad)
            return np.abs(curv).astype(float)
        except Exception:
            pass
    # fallback: neighbor-distance saliency
    from scipy.spatial import cKDTree
    k = min(16, max(2, V.shape[0] // 200))
    d = cKDTree(V).query(V, k=k)[0]
    return d.mean(1)

def auto_keypoints(V: np.ndarray, F: np.ndarray, num: int, method: str) -> np.ndarray:
    K = max(4, int(num))
    m = method.lower()
    if m == "fps":
        ids = fps(V, K)
        return V[ids]
    elif m in ("curvature_fps", "curvature"):
        s = curvature_scores(V, F)
        k_cand = min(V.shape[0], max(K * 5, K))
        cand = np.argsort(-s)[:k_cand]
        ids = fps(V[cand], K)
        return V[cand][ids]
    else:
        raise ValueError(f"Unknown auto method: {method}")

# --------------------- Save ---------------------
def save_keypoints(out_dir: Path, mesh_path: Path, mode: str, method: Optional[str], pts: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "mesh_path": str(mesh_path.as_posix()),
        "mode": mode,
        "method": (method or None),
        "num_keypoints": int(pts.shape[0]),
        "keypoints": [[float(x), float(y), float(z)] for (x, y, z) in np.asarray(pts, float)],
    }
    out = out_dir / f"{mesh_path.stem}_keypoints.json"
    out.write_text(json.dumps(rec, indent=2))
    return out

def existing_kp_file(project_root: Path, mesh_path: Path) -> Optional[Path]:
    p = (project_root / "canonical_keypoints" / f"{mesh_path.stem}_keypoints.json")
    return p if p.exists() else None


# --------------------- UI helpers ---------------------

def measure(text: str, scale=0.5, thk=1):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thk)
    return tw

def ellipsize_middle(text: str, max_w: int, scale=0.5, thk=1):
    if measure(text, scale, thk) <= max_w:
        return text
    left, right = 0, 0
    while True:
        cand = text[:left] + "..." + text[len(text) - right:]
        if measure(cand, scale, thk) <= max_w or (left + right) >= len(text):
            return cand if cand else "..."
        if left <= right and left < len(text): left += 1
        elif right < len(text): right += 1

def wrap_to_width(text: str, max_w: int, scale=0.5, thk=1):
    if measure(text, scale, thk) <= max_w:
        return [text]
    words = text.split(" ")
    if len(words) == 1:
        return [ellipsize_middle(text, max_w, scale, thk)]
    out, cur = [], ""
    for w in words:
        tok = w if measure(w, scale, thk) <= max_w else ellipsize_middle(w, max_w, scale, thk)
        trial = (cur + " " + tok).strip()
        if measure(trial, scale, thk) <= max_w:
            cur = trial
        else:
            if cur: out.append(cur)
            cur = tok if measure(tok, scale, thk) <= max_w else ellipsize_middle(tok, max_w, scale, thk)
    if cur: out.append(cur)
    return out

def pill(text: str, kind: str):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad = 6
    img = np.full((th + 2*pad, tw + 2*pad, 3), PILL[kind][0], np.uint8)
    cv2.putText(img, text, (pad, th + pad//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PILL[kind][1], 1, cv2.LINE_AA)
    return img

def pad(img, h, w):
    out = np.zeros((h, w, 3), np.uint8)
    if img is None: return out
    hh, ww = img.shape[:2]
    out[:hh, :ww] = img
    return out

def hstack(L, M, R):
    H = max(L.shape[0], M.shape[0], R.shape[0])
    return np.hstack([pad(L, H, L.shape[1]), pad(M, H, M.shape[1]), pad(R, H, R.shape[1])])

def text_panel(lines: List[str], w=380, h=720, scale=0.5, thk=1, color=(240,240,240)):
    img = np.full((h, w, 3), 18, np.uint8)
    y = 24
    line_h = int(round(20 * max(0.8, scale / 0.5)))
    for ln in lines:
        for piece in wrap_to_width(ln, w - 18, scale, thk):
            if y > h - 8: return img
            cv2.putText(img, piece, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thk, cv2.LINE_AA)
            y += line_h
    return img


# --------------------- Modal overlays (GUI-only) ---------------------
def modal_int_overlay(win_name: str,
                       base_img: np.ndarray,
                       prompt: str,
                       default: int = 16,
                       min_value: int = 4,
                       presets: Tuple[int, ...] = (4, 8, 16, 32)) -> Optional[int]:
    """
    In-window numeric picker. Returns chosen int (>= min_value) or None if canceled.
    - OK/Cancel moved to the BOTTOM row (more intuitive).
    - Dialog auto-sizes and wraps long text.
    Keys: digits, Backspace, Enter=OK, ESC=Cancel, +/- (±1), PgUp/PgDn (±10).
    """
    value = max(min_value, int(default))
    editing = False

    img_h, img_w = base_img.shape[:2]
    pad = 16
    scale_prompt = 0.64
    thk_prompt = 1
    info_text = "Type digits / Backspace.  +/- = ±1,  PgUp/PgDn = ±10,  Enter=OK, ESC=Cancel"

    # component sizes
    chip_w, chip_h = 64, 32
    btn_w, btn_h = 90, 40
    big_w, big_h = 110, 44

    # compute a good width
    min_w = 620
    max_w = min(img_w - 80, 1000)
    chips_row_w = len(presets) * chip_w + (len(presets) - 1) * 8 + 2 * pad
    prompt_w = measure(prompt, scale_prompt, thk_prompt) + 2 * pad
    info_w = measure(info_text, 0.45, 1) + 2 * pad
    box_w = max(min_w, min(max_w, max(prompt_w, chips_row_w, info_w)))

    # wrap texts to width
    prompt_lines = wrap_to_width(prompt, box_w - 2 * pad, scale_prompt, thk_prompt)
    info_lines = wrap_to_width(info_text, box_w - 2 * pad, 0.45, 1)

    # compute height from dynamic content
    line_h1 = 26  # prompt lines height
    gap = 12
    # layout blocks: prompt, "Presets", chips, +/- row, info, big buttons
    prompt_h = len(prompt_lines) * line_h1
    label_h = 18
    chips_h = chip_h
    pm_row_h = btn_h      # plus/minus row
    info_h = len(info_lines) * 20
    total_blocks = prompt_h + gap + label_h + gap + chips_h + gap + pm_row_h + gap + info_h + gap + big_h
    box_h = total_blocks + 2 * pad

    # center box
    box_x = (img_w - box_w) // 2
    box_y = (img_h - box_h) // 2

    # compute rects (x1,y1,x2,y2)
    y_cursor = box_y + pad

    # prompt lines area
    prompt_region_top = y_cursor
    y_cursor += prompt_h + gap

    # "Presets"
    presets_label_y = y_cursor + label_h - 4
    y_cursor += label_h + gap

    # chips row
    chips_y = y_cursor
    y_cursor += chips_h + gap

    # +/- row
    pm_y = y_cursor
    y_cursor += pm_row_h + gap

    # info lines
    info_top = y_cursor
    y_cursor += info_h + gap

    # big buttons row (bottom)
    ok_btn = (box_x + box_w - pad - big_w, y_cursor, box_x + box_w - pad, y_cursor + big_h)
    cancel_btn = (ok_btn[0] - 8 - big_w, y_cursor, ok_btn[0] - 8, y_cursor + big_h)

    # small +/- buttons (two on left, two on right)
    minus10 = (box_x + pad, pm_y, box_x + pad + btn_w, pm_y + btn_h)
    minus1  = (minus10[2] + 8, pm_y, minus10[2] + 8 + btn_w, pm_y + btn_h)
    plus1   = (box_x + box_w - pad - 2 * btn_w - 8, pm_y, box_x + box_w - pad - btn_w - 8, pm_y + btn_h)
    plus10  = (box_x + box_w - pad - btn_w, pm_y, box_x + box_w - pad, pm_y + btn_h)

    # chips rects
    chips = []
    cx = box_x + pad
    for p in presets:
        chips.append((cx, chips_y, cx + chip_w, chips_y + chip_h, p))
        cx += chip_w + 8

    def inside(r, x, y):
        x1, y1, x2, y2 = r
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    done, canceled = False, False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal value, done, canceled, editing
        if event == cv2.EVENT_LBUTTONDOWN:
            if inside(minus10, x, y): value = max(min_value, value - 10); editing = False
            elif inside(minus1, x, y): value = max(min_value, value - 1); editing = False
            elif inside(plus1, x, y): value = max(min_value, value + 1); editing = False
            elif inside(plus10, x, y): value = max(min_value, value + 10); editing = False
            elif inside(ok_btn, x, y): done = True
            elif inside(cancel_btn, x, y): canceled = True
            else:
                for (x1,y1,x2,y2,p) in chips:
                    if inside((x1,y1,x2,y2), x, y):
                        value = max(min_value, int(p)); editing = False
                        break

    cv2.setMouseCallback(win_name, on_mouse)
    try:
        while not (done or canceled):
            frame = base_img.copy()
            overlay = frame.copy(); overlay[:] = (0, 0, 0)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

            # dialog card
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (46,46,46), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (90,90,90), 1, cv2.LINE_AA)

            # prompt
            y = prompt_region_top
            for ln in prompt_lines:
                cv2.putText(frame, ln, (box_x + pad, y + 20), cv2.FONT_HERSHEY_SIMPLEX, scale_prompt,
                            (235,235,235), thk_prompt, cv2.LINE_AA)
                y += line_h1

            # current value big
            disp = str(max(min_value, int(value)))
            (tw, th), _ = cv2.getTextSize(disp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(frame, disp, (box_x + pad, prompt_region_top + th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (240,240,240), 2, cv2.LINE_AA)

            # 'Presets' label
            cv2.putText(frame, "Presets:", (box_x + pad, presets_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)

            # chips
            for (x1,y1,x2,y2,p) in chips:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (64,64,64), -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (100,100,100), 1, cv2.LINE_AA)
                s = str(p); (stw, sth), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(frame, s, (x1 + (x2-x1-stw)//2, y1 + (y2-y1+sth)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)

            # +/- buttons
            def draw_btn(r, label):
                x1,y1,x2,y2 = r
                cv2.rectangle(frame, (x1,y1), (x2,y2), (64,64,64), -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (100,100,100), 1, cv2.LINE_AA)
                (stw, sth), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(frame, label, (x1 + (x2-x1-stw)//2, y1 + (y2-y1+sth)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
            draw_btn(minus10, "-10"); draw_btn(minus1, "-1"); draw_btn(plus1, "+1"); draw_btn(plus10, "+10")

            # info (wrapped)
            y = info_top
            for ln in info_lines:
                cv2.putText(frame, ln, (box_x + pad, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (160,160,160), 1, cv2.LINE_AA)
                y += 20

            # OK / Cancel at bottom
            def draw_big_btn(r, label, filled=False):
                x1,y1,x2,y2 = r
                cv2.rectangle(frame, (x1,y1), (x2,y2), (55,90,170) if filled else (64,64,64), -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (110,110,110), 1, cv2.LINE_AA)
                (stw, sth), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(frame, label, (x1 + (x2-x1-stw)//2, y1 + (y2-y1+sth)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
            draw_big_btn(cancel_btn, "Cancel", filled=False)
            draw_big_btn(ok_btn, "OK", filled=True)

            cv2.imshow(win_name, frame)
            k = cv2.waitKeyEx(25) & 0xFFFFFFFF
            if k in (ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7'),ord('8'),ord('9')):
                d = int(chr(k))
                if not editing:
                    value = d
                    editing = True
                else:
                    value = int(str(value) + str(d))
            elif k in (8, 127):  # Backspace/Delete
                if editing:
                    s = str(value)[:-1]
                    value = int(s) if s else min_value
                    if not s:
                        editing = False
                else:
                    value = min_value
            elif k in (ord('+'), ord('=')): value = value + 1; editing = False
            elif k in (ord('-'), ord('_')): value = max(min_value, value - 1); editing = False
            elif k == 2162688:  value = value + 10; editing = False        # PgUp
            elif k == 2228224:  value = max(min_value, value - 10); editing = False  # PgDn
            elif k in (13, 10): done = True
            elif k in (27,):    canceled = True
        return None if canceled else max(min_value, int(value))
    finally:
        cv2.setMouseCallback(win_name, lambda *a, **k: None)

def modal_yes_no_overlay(win_name: str,
                          base_img: np.ndarray,
                          question: str,
                          default: str = "no") -> bool:
    """
    Blocks in-place inside the existing OpenCV window (win_name).
    Draws a dimmed overlay with a centered Yes/No dialog.
    Returns True for Yes, False for No.
    """
    img_h, img_w = base_img.shape[:2]

    # Dialog geometry
    box_w, box_h = 520, 170
    box_x = (img_w - box_w) // 2
    box_y = (img_h - box_h) // 2
    pad = 16
    btn_w, btn_h = 160, 44
    yes_rect = (box_x + pad, box_y + box_h - pad - btn_h,
                box_x + pad + btn_w, box_y + box_h - pad)
    no_rect  = (box_x + box_w - pad - btn_w, box_y + box_h - pad - btn_h,
                box_x + box_w - pad,            box_y + box_h - pad)

    # State
    choice = None

    def inside(r, x, y):
        x1, y1, x2, y2 = r
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal choice
        if event == cv2.EVENT_LBUTTONDOWN:
            if inside(yes_rect, x, y): choice = True
            elif inside(no_rect, x, y): choice = False

    cv2.setMouseCallback(win_name, on_mouse)

    try:
        while choice is None:
            frame = base_img.copy()

            # dim background
            overlay = frame.copy()
            overlay[:] = (0, 0, 0)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

            # dialog box
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (46,46,46), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (90,90,90), 1, cv2.LINE_AA)

            # question
            cv2.putText(frame, question, (box_x + pad, box_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235,235,235), 1, cv2.LINE_AA)

            # buttons
            def draw_btn(rect, label, is_default=False):
                x1, y1, x2, y2 = rect
                bg = (55, 90, 170) if is_default else (64,64,64)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100,100,100), 1, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
                cx, cy = (x1 + x2)//2, (y1 + y2)//2 + th//2
                cv2.putText(frame, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (240,240,240), 1, cv2.LINE_AA)

            draw_btn(yes_rect, "Yes  (Y)", default.lower() == "yes")
            draw_btn(no_rect,  "No   (N)", default.lower() == "no")
            cv2.putText(frame, "Enter = default,  ESC = cancel",
                        (box_x + pad, box_y + box_h - btn_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)
            k = cv2.waitKeyEx(25) & 0xFFFFFFFF
            if k in (ord('y'), ord('Y')):
                choice = True
            elif k in (ord('n'), ord('N')):
                choice = False
            elif k in (13, 10, 27):  # Enter or ESC -> default
                choice = True if default.lower() == "yes" else False

        return bool(choice)
    finally:
        # restore to key event only (no mouse grab)
        cv2.setMouseCallback(win_name, lambda *a, **k: None)


def modal_yes_no_window(question: str, default: str = "no") -> bool:
    """
    Standalone Yes/No window that auto-sizes and wraps the question text.
    Returns True for Yes, False for No.
    """
    scale = 0.65; thk = 1
    pad = 16
    min_w, max_w = 520, 1000
    # pick width based on text, then wrap
    w_est = measure(question, scale, thk) + 2 * pad + 80
    w = max(min_w, min(max_w, w_est))
    lines = wrap_to_width(question, w - 2 * pad, scale, thk)
    line_h = 26
    text_h = line_h * len(lines)
    h = max(170, 90 + text_h)
    win = "Confirm"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    try:
        base = np.full((h, w, 3), 28, np.uint8)
        while True:
            frame = base.copy()
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (90,90,90), 1, cv2.LINE_AA)
            y = 32
            for ln in lines:
                cv2.putText(frame, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240,240,240), thk, cv2.LINE_AA)
                y += line_h
            cv2.putText(frame, "[Y]es / [N]o — Enter = default", (pad, h-24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)
            k = cv2.waitKeyEx(25) & 0xFFFFFFFF
            if k in (ord('y'), ord('Y')): return True
            if k in (ord('n'), ord('N')): return False
            if k in (13, 10, 27):
                return True if default.lower() == "yes" else False
    finally:
        try: cv2.destroyWindow(win)
        except Exception: pass


def render_list_meshes(h, w, title, items, sel, project_root: Path):
    pan = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(pan, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35,35,35), 2, cv2.LINE_AA)
    max_show = min(22, len(items))
    start = max(0, min(sel - max_show // 2, max(0, len(items) - max_show)))
    y = 56
    for i in range(start, start + max_show):
        if i >= len(items): break
        mp = items[i]
        chips = []
        ex = existing_kp_file(project_root, mp)
        chips.append(pill("KPs OK", "ok") if ex else pill("no KPs", "warn"))
        x = w - 12
        for c in reversed(chips):
            ch, cw = c.shape[:2]; x -= (cw + 6)
            pan[y-16:y-16+ch, x:x+cw] = c
        name = mp.name
        avail = x - 14
        (tw,_),_ = cv2.getTextSize("> " + name, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        while tw > avail and len(name) > 3:
            name = name[:-4] + "..."
            (tw,_),_ = cv2.getTextSize("> " + name, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        col = (20, 70, 180) if i == sel else (30,30,30); thk = 2 if i==sel else 1
        cv2.putText(pan, ("> " if i==sel else "  ") + name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, thk, cv2.LINE_AA)
        y += 30

    tips = "UP/DOWN select   ENTER run   M mode   F method   +/- count   K set count   G gen missing   V view   R reload   Q quit"
    lines = wrap_to_width(tips, w - 24, 0.46, 1)
    line_h = 18; y0 = h - 10 - line_h * (len(lines) - 1)
    for i, t in enumerate(lines):
        cv2.putText(pan, t, (12, y0 + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90,90,90), 1, cv2.LINE_AA)
    return pan


# --------------------- Viewer ---------------------
def view_points(mesh_o3d, pts: np.ndarray, title: str):
    if o3d is None:
        print("[view] open3d not installed; cannot visualize")
        return
    geoms = []
    if mesh_o3d is not None:
        geoms.append(mesh_o3d)
    if pts is not None and len(pts):
        # sphere radius ~ 2% of bbox diag
        V = np.asarray(mesh_o3d.vertices)
        diag = float(np.linalg.norm(V.max(0) - V.min(0))) + 1e-9
        r = 0.02 * diag
        for p in pts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=6)
            s.compute_vertex_normals()
            s.translate(p.tolist())
            s.paint_uniform_color([1.0, 1.0, 0.0])
            geoms.append(s)
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1280, height=900)


# --------------------- Thumbnails ---------------------
def render_mesh_thumb(path: Path, target_h=UI_H, target_w=UI_W_LEFT) -> np.ndarray:
    """Render an offscreen thumbnail of the mesh. Several fallbacks."""
    bg = np.full((target_h, target_w, 3), 22, np.uint8)
    # Try Open3D OffscreenRenderer first
    if o3d is not None:
        try:
            m = o3d.io.read_triangle_mesh(str(path))
            if m.has_vertices() and len(m.triangles) > 0:
                m.compute_vertex_normals()
                bb = m.get_axis_aligned_bounding_box()
                center = bb.get_center()
                extent = np.linalg.norm(bb.get_extent())
                R = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.05)
                vis = o3d.visualization.rendering.OffscreenRenderer(target_w, target_h)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                vis.scene.add_geometry("mesh", m, mat)
                vis.scene.add_geometry("axes", R, mat)
                # Camera
                eye = center + np.array([1.5, 1.2, 1.0]) * (0.8 * extent + 1e-6)
                vis.scene.camera.look_at(center, eye, np.array([0, 0, 1.0]))
                img = vis.render_to_image()
                vis.release()
                if img is not None:
                    arr = np.asarray(img)
                    if arr.ndim == 3:
                        # Open3D returns RGBA; convert to BGR
                        if arr.shape[2] == 4:
                            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                        else:
                            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        return arr
        except Exception:
            pass
    # Trimesh fallback: save_image offscreen (may need pyglet)
    if trimesh is not None:
        try:
            sc = trimesh.load(str(path))
            if isinstance(sc, trimesh.Trimesh):
                sc = sc.scene()
            data = sc.save_image(resolution=(target_w, target_h), visible=True)
            if data is not None:
                import PIL.Image as Image
                import io
                im = Image.open(io.BytesIO(data))
                arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGR)
                return arr
        except Exception:
            pass
    # Fallback: label only
    cv2.putText(bg, path.name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return bg

