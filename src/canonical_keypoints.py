"""
generate_canonical_keypoints.py
===============================
Interactively (or automatically) create **canonical 3D keypoints** for meshes
under the project layout. Designed to match the repo’s project-root pattern
(via `resolve_project_root(...)`) and to feel like the existing
`annotate_shots.py` UI.

What it does
------------
- Scans meshes in <project_root>/meshes/** (supports .obj/.ply/.stl via trimesh).
- Two modes:
  1) **auto**   — sample K canonical keypoints (K ≥ 4; typical presets: 4, 8, 16).
                  Methods: `fps` (Farthest-Point Sampling), `curvature_fps`
                  (curvature-weighted FPS). After generation, optionally preview
                  the points on the mesh.
  2) **manual** — point picker UI to place, move, or delete 3D canonical keypoints
                  using the mouse (on an interactive render of the mesh).
- Saves one JSON **per mesh** in <project_root>/canonical_keypoints/<mesh_stem>.json
  with positions in the **mesh’s canonical/object frame** (units = mesh units).

New in this version
-------------------
- **State persistence** in `<project_root>/canonical_keypoints/.state.json`:
  remembers the last selected mesh and UI settings
  (`mode`, `method`, `count`, `view`) so you can quit & resume later.
- **Progress indicator** in the right panel: `N / M reviewed`
  (counts meshes that already have a saved keypoint file).
- Robust to missing/corrupt state; safe defaults are applied automatically.

Project layout
--------------
<project_root>/
  meshes/                               # inputs (OBJ/PLY/STL…)
  canonical_keypoints/                  # outputs (JSON) + .state.json
    <mesh_stem>.json
    .state.json                         # UI state (last selection & settings)
  utils/
    project_config.py                   # resolve_project_root, ensure_project_dirs
  ...

Output schema (per mesh)
------------------------
{
  "mesh_file": "<relative/path/from_project_root>",
  "mesh_stem": "<stem>",
  "generated_by": "generate_canonical_keypoints.py",
  "mode": "auto" | "manual",
  "method": "fps" | "curvature_fps",            # present for auto
  "count": <int>,                                # requested K
  "points": [[x, y, z], ...],                    # canonical 3D points (object frame)
  "metadata": {
    "timestamp": "<ISO8601>",
    "num_vertices": <int>,
    "num_faces": <int>,
    "notes": "<free text>"
  }
}

CLI
---
Browse picker (recommended):
  python -m src.generate_canonical_keypoints --browse

Single mesh (direct path):
  python -m src.generate_canonical_keypoints --mesh meshes/part.obj --mode auto --count 16
  python -m src.generate_canonical_keypoints --mesh meshes/part.obj --mode manual

Options
-------
--mode {auto,manual}          Default: auto (browser lets you toggle)
--method {fps,curvature_fps}  Auto mode sampling method (default: fps)
--count K                     Number of keypoints (clamped to K ≥ 4)
--mesh PATH                   Process a single mesh file
--browse                      Open the 3-column UI browser (like annotate_shots)
--reset-state                 Ignore/clear .state.json and start fresh
--no-preview                  In auto mode, skip the "view points?" prompt

UI (browse)
-----------
- ↑/W, ↓/S        : select mesh
- A               : toggle mode (auto/manual)
- M               : cycle auto method (fps ↔ curvature_fps)
- +/- or [/]      : decrease/increase K (keypoint count; min 4)
- V               : toggle auto preview (ask/yes/no)
- ENTER           : run on selected mesh with current settings
- R               : rescan meshes
- Q               : quit (state is saved automatically)

UI (manual editor)
------------------
- Left click       : add point (on surface) or drag existing point to move
- Right click/DEL  : delete nearest point
- SHIFT (hold)     : axis/plane snap while dragging (viewport dependent)
- Z / Y            : undo / redo
- R                : reset current mesh keypoints (clear)
- ENTER            : save & exit editor
- ESC / q          : abort without saving
- Q / X            : quit-all immediately

Notes
-----
- Points are stored in the mesh’s **object frame** as loaded by trimesh
  (i.e., after any baked transforms in the file). No world transforms are applied.
- If a canonical keypoint file already exists for a mesh, loading it will seed
  the manual editor and count toward the progress in the browser.
- Auto preview respects `--no-preview` and the current **view** preference
  (`prompt` | `yes` | `no`) remembered in `.state.json`.

Dependencies
------------
- numpy, trimesh, open3d (or pyrender/pyglet for simple viewer), opencv-python,
  PyYAML, and project utils:
    utils.project_config: resolve_project_root, ensure_project_dirs

Logs
----
<project_root>/logs/generate_canonical_keypoints.log

"""


from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import json as _json


from utils.project_config import resolve_project_root, ensure_project_dirs


# ---- deps (soft) ----
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

def _state_path(project_root: Path) -> Path:
    return project_root / "canonical_keypoints" / ".state.json"

def _load_state(project_root: Path) -> dict:
    p = _state_path(project_root)
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

def _save_state(project_root: Path, **kwargs):
    p = _state_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = _load_state(project_root)
    st.update({k: v for k, v in kwargs.items() if v is not None})
    p.write_text(_json.dumps(st, indent=2))

# --------------------- Mesh IO ---------------------

def _find_meshes(project_root: Path, pattern: Optional[str], single_path: Optional[str]) -> List[Path]:
    if single_path:
        p = Path(single_path)
        return [p] if p.exists() else []
    if pattern:
        return sorted([p for p in Path().glob(pattern) if p.suffix.lower() in MESH_EXTS])
    meshes_dir = project_root / "meshes"
    return sorted([p for p in meshes_dir.rglob("*") if p.suffix.lower() in MESH_EXTS])

def _load_mesh_any(path: Path):
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

def _fps(points: np.ndarray, k: int) -> np.ndarray:
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

def _curvature_scores(V: np.ndarray, F: np.ndarray) -> np.ndarray:
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

def _auto_keypoints(V: np.ndarray, F: np.ndarray, num: int, method: str) -> np.ndarray:
    K = max(4, int(num))
    m = method.lower()
    if m == "fps":
        ids = _fps(V, K)
        return V[ids]
    elif m in ("curvature_fps", "curvature"):
        s = _curvature_scores(V, F)
        k_cand = min(V.shape[0], max(K * 5, K))
        cand = np.argsort(-s)[:k_cand]
        ids = _fps(V[cand], K)
        return V[cand][ids]
    else:
        raise ValueError(f"Unknown auto method: {method}")

# --------------------- Manual picking ---------------------

def _pick_points_o3d(mesh_o3d, window="Pick canonical keypoints"):
    if o3d is None:
        raise RuntimeError("manual mode needs open3d")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window, width=1280, height=900)
    vis.add_geometry(mesh_o3d)
    vis.run()
    ids = vis.get_picked_points()
    vis.destroy_window()
    V = np.asarray(mesh_o3d.vertices)
    return V[np.asarray(ids, int)] if len(ids) else np.empty((0,3), float)

# --------------------- Viewer ---------------------

def _view_points(mesh_o3d, pts: np.ndarray, title: str):
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

# --------------------- Save ---------------------

def _save_keypoints(out_dir: Path, mesh_path: Path, mode: str, method: Optional[str], pts: np.ndarray) -> Path:
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

def _existing_kp_file(project_root: Path, mesh_path: Path) -> Optional[Path]:
    p = (project_root / "canonical_keypoints" / f"{mesh_path.stem}_keypoints.json")
    return p if p.exists() else None

# --------------------- Thumbnails ---------------------

def _render_mesh_thumb(path: Path, target_h=UI_H, target_w=UI_W_LEFT) -> np.ndarray:
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

# --------------------- UI helpers ---------------------

def _measure(text: str, scale=0.5, thk=1):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thk)
    return tw

def _ellipsize_middle(text: str, max_w: int, scale=0.5, thk=1):
    if _measure(text, scale, thk) <= max_w:
        return text
    left, right = 0, 0
    while True:
        cand = text[:left] + "..." + text[len(text) - right:]
        if _measure(cand, scale, thk) <= max_w or (left + right) >= len(text):
            return cand if cand else "..."
        if left <= right and left < len(text): left += 1
        elif right < len(text): right += 1

def _wrap_to_width(text: str, max_w: int, scale=0.5, thk=1):
    if _measure(text, scale, thk) <= max_w:
        return [text]
    words = text.split(" ")
    if len(words) == 1:
        return [_ellipsize_middle(text, max_w, scale, thk)]
    out, cur = [], ""
    for w in words:
        tok = w if _measure(w, scale, thk) <= max_w else _ellipsize_middle(w, max_w, scale, thk)
        trial = (cur + " " + tok).strip()
        if _measure(trial, scale, thk) <= max_w:
            cur = trial
        else:
            if cur: out.append(cur)
            cur = tok if _measure(tok, scale, thk) <= max_w else _ellipsize_middle(tok, max_w, scale, thk)
    if cur: out.append(cur)
    return out

def _pill(text: str, kind: str):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad = 6
    img = np.full((th + 2*pad, tw + 2*pad, 3), PILL[kind][0], np.uint8)
    cv2.putText(img, text, (pad, th + pad//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PILL[kind][1], 1, cv2.LINE_AA)
    return img

def _pad(img, h, w):
    out = np.zeros((h, w, 3), np.uint8)
    if img is None: return out
    hh, ww = img.shape[:2]
    out[:hh, :ww] = img
    return out

def _hstack(L, M, R):
    H = max(L.shape[0], M.shape[0], R.shape[0])
    return np.hstack([_pad(L, H, L.shape[1]), _pad(M, H, M.shape[1]), _pad(R, H, R.shape[1])])

def _text_panel(lines: List[str], w=380, h=720, scale=0.5, thk=1, color=(240,240,240)):
    img = np.full((h, w, 3), 18, np.uint8)
    y = 24
    line_h = int(round(20 * max(0.8, scale / 0.5)))
    for ln in lines:
        for piece in _wrap_to_width(ln, w - 18, scale, thk):
            if y > h - 8: return img
            cv2.putText(img, piece, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thk, cv2.LINE_AA)
            y += line_h
    return img

def _render_list_meshes(h, w, title, items, sel, project_root: Path):
    pan = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(pan, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35,35,35), 2, cv2.LINE_AA)
    max_show = min(22, len(items))
    start = max(0, min(sel - max_show // 2, max(0, len(items) - max_show)))
    y = 56
    for i in range(start, start + max_show):
        if i >= len(items): break
        mp = items[i]
        chips = []
        ex = _existing_kp_file(project_root, mp)
        chips.append(_pill("KPs OK", "ok") if ex else _pill("no KPs", "warn"))
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
    tips = "UP/DOWN select   ENTER run   M mode   F method   +/- count   V view   R reload   Q quit"
    lines = _wrap_to_width(tips, w - 24, 0.46, 1)
    line_h = 18; y0 = h - 10 - line_h * (len(lines) - 1)
    for i, t in enumerate(lines):
        cv2.putText(pan, t, (12, y0 + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90,90,90), 1, cv2.LINE_AA)
    return pan

# --------------------- UI main ---------------------

def _ui_browse(project_root: Path, meshes: List[Path], args):
    cv2.namedWindow("Canonical KPs", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Canonical KPs", UI_WIN_W, UI_H)

    st = _load_state(project_root)
    mode = st.get("mode", args.mode)
    method = st.get("method", args.method)
    kcount = max(4, int(st.get("count", args.num)))
    view_after = st.get("view", args.view)

    i = 0
    if meshes:
        last = st.get("last_mesh")
        if last:
            last = Path(last)
            # try absolute match
            try:
                i = next(j for j, m in enumerate(meshes) if m.resolve() == last.resolve())
            except StopIteration:
                # fall back to stem match
                try:
                    i = next(j for j, m in enumerate(meshes) if m.stem == last.stem)
                except StopIteration:
                    i = 0
    # simple thumb cache
    cache = {}

    while True:
        mp = meshes[i]
        # left: thumbnail
        if mp not in cache:
            cache[mp] = _render_mesh_thumb(mp, target_h=UI_H, target_w=UI_W_LEFT)
        left = cache[mp].copy()
        cv2.putText(left, mp.name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # mid: list
        mid = _render_list_meshes(left.shape[0], UI_W_MID, "Meshes", meshes, i, project_root)

        # right: settings and status
        ex = _existing_kp_file(project_root, mp)
        done = sum(1 for m in meshes if _existing_kp_file(project_root, m))
        total = len(meshes)

        lines = [
            f"Progress: {done}/{total} reviewed",
            "",
            "Settings",
            f"  Mode:   {mode}",
            f"  Method: {method if mode=='auto' else '-'}",
            f"  Count:  {kcount}",
            f"  View:   {view_after}",
            "",
            "Selected mesh",
            f"  Path: {str(mp)}",
            f"  Status: {'HAS keypoints' if ex else 'no keypoints yet'}",
            "",
            "ENTER: run on selected",
            "M: toggle mode (auto/manual)",
            "F: toggle auto method (fps/curvature_fps)",
            "+/-: adjust count (min 4)",
            "V: toggle view (yes/no/prompt)",
            "R: reload list",
            "Q: quit",
        ]
        right = _text_panel(lines, w=UI_W_RIGHT, h=left.shape[0])

        cv2.imshow("Canonical KPs", _hstack(left, mid, right))
        k = cv2.waitKeyEx(60) & 0xFFFFFFFF

        if k in (ord('q'), ord('Q'), 27):  # ESC also quits
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
            break
        elif k in (2490368, ord('w'), ord('W')):  # up
            i = (i - 1) % len(meshes)
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (2621440, ord('s'), ord('S')):  # down
            i = (i + 1) % len(meshes)
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('m'), ord('M')):
            mode = "manual" if mode == "auto" else "auto"
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('f'), ord('F')):
            method = "curvature_fps" if method == "fps" else "fps"
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('+'), ord('=')):
            kcount = min(9999, kcount + 1)
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('-'), ord('_')):
            kcount = max(4, kcount - 1)
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('v'), ord('V')):
            view_after = {"prompt":"yes","yes":"no","no":"prompt"}[view_after]
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('r'), ord('R')):
            meshes[:] = _find_meshes(project_root, args.glob, args.mesh)  # refresh
            cache.clear()
            i = min(i, len(meshes) - 1)
            _save_state(project_root,
                        last_mesh=str(meshes[i]),
                        mode=mode, method=method, count=kcount, view=view_after)
        elif k in (13, 10):  # ENTER = run
            # process selected file
            try:
                mesh_o3d, V, F = _load_mesh_any(mp)
                if mode == "auto":
                    pts = _auto_keypoints(V, F, kcount, method)
                    out = _save_keypoints(project_root / "canonical_keypoints", mp, "auto", method, pts)
                    # optionally view
                    doit = view_after
                    if view_after == "prompt":
                        ans = input("View keypoints on mesh? [y/N] ").strip().lower()
                        doit = "yes" if ans in ("y", "yes") else "no"
                    if doit == "yes":
                        _view_points(mesh_o3d, pts, f"Auto KPs ({mp.name})")
                else:
                    if o3d is None:
                        print("[!] Manual mode needs open3d; install with pip.")
                    else:
                        print("Open3D window: click vertices to select; close window to save.")
                        pts = _pick_points_o3d(mesh_o3d, f"Pick Canonical KPs: {mp.name}")
                        if len(pts):
                            out = _save_keypoints(project_root / "canonical_keypoints", mp, "manual", None, pts)
                            if view_after in ("yes", "prompt"):
                                _view_points(mesh_o3d, pts, f"Manual KPs ({mp.name})")
                        else:
                            print("[i] No points picked; skipped saving.")
                _save_state(project_root,
                            last_mesh=str(meshes[i]),
                            mode=mode, method=method, count=kcount, view=view_after)
                # update right panel status immediately
                cache.pop(mp, None)  # optional; keep thumbnail as-is
            except BaseException as e:
                print(f"[!] Error on {mp}: {e}")

    cv2.destroyAllWindows()

# --------------------- CLI entry ---------------------

def main():
    project_root = resolve_project_root(None)
    ensure_project_dirs(project_root)

    ap = argparse.ArgumentParser("Canonical keypoints from meshes")
    ap.add_argument("--browse", action="store_true", help="Open browse UI")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--mesh", type=str, default=None, help="Process a single mesh path")
    ap.add_argument("--glob", type=str, default=None, help="Glob (e.g., 'meshes/**/*.obj')")
    ap.add_argument("--num", type=int, default=16, help="Number of keypoints (min 4)")
    ap.add_argument("--method", choices=["fps", "curvature_fps"], default="fps",
                    help="Auto method")
    ap.add_argument("--view", choices=["yes", "no", "prompt"], default="prompt",
                    help="Open viewer after generation")
    args = ap.parse_args()

    meshes = _find_meshes(project_root, args.glob, args.mesh)
    if not meshes:
        sys.exit(f"[!] No meshes found (looked in {project_root/'meshes'} or pattern/mesh provided)")

    if args.browse:
        _ui_browse(project_root, meshes, args)
        return

    out_dir = project_root / "canonical_keypoints"
    for mp in meshes:
        m, V, F = _load_mesh_any(mp)
        if args.mode == "auto":
            pts = _auto_keypoints(V, F, args.num, args.method)
            _save_keypoints(out_dir, mp, "auto", args.method, pts)
            if args.view == "yes" or (args.view == "prompt" and input("View? [y/N] ").strip().lower() in ("y","yes")):
                _view_points(m, pts, f"Auto KPs ({mp.name})")
        else:
            if o3d is None:
                sys.exit("[!] Manual mode needs open3d")
            pts = _pick_points_o3d(m, f"Pick Canonical KPs: {mp.name}")
            if len(pts):
                _save_keypoints(out_dir, mp, "manual", None, pts)
                if args.view in ("yes","prompt"):
                    _view_points(m, pts, f"Manual KPs ({mp.name})")
            else:
                print(f"[i] No points picked for {mp.name}; skipped.")

if __name__ == "__main__":
    main()
