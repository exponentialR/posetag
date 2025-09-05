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
from utils.generate_canonical_utils import (
    load_mesh_any, auto_keypoints, save_keypoints, existing_kp_file,
    load_state, save_state, find_meshes, view_points,
    render_mesh_thumb, render_list_meshes, text_panel,
    hstack, modal_yes_no_window, modal_yes_no_overlay, modal_int_overlay,
    pad, UI_W_LEFT, UI_W_MID, UI_W_RIGHT, UI_WIN_W, UI_H
)

# ---- deps (soft) ----
try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import trimesh
except Exception:
    trimesh = None


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
    return V[np.asarray(ids, int)] if len(ids) else np.empty((0, 3), float)


# --------------------- UI main ---------------------
def _ui_browse(project_root: Path, meshes: List[Path], args):
    cv2.namedWindow("Canonical KPs", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Canonical KPs", UI_WIN_W, UI_H)

    st = load_state(project_root)
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

    mp0 = meshes[i]
    if mp0 not in cache:
        cache[mp0] = render_mesh_thumb(mp0, target_h=UI_H, target_w=UI_W_LEFT)
    left0 = cache[mp0].copy()
    mid0 = render_list_meshes(left0.shape[0], UI_W_MID, "Meshes", meshes, i, project_root)
    ex0 = existing_kp_file(project_root, mp0)
    done0 = sum(1 for m in meshes if existing_kp_file(project_root, m))
    total0 = len(meshes)
    right0 = text_panel([
        f"Progress: {done0}/{total0} reviewed",
        "", "Settings",
        f"  Mode:   {mode}",
        f"  Method: {method if mode == 'auto' else '-'}",
        f"  Count:  {kcount}",
        f"  View:   {view_after}",
        "", "Selected mesh",
        f"  Path: {str(mp0)}",
        f"  Status: {'HAS keypoints' if ex0 else 'no keypoints yet'}",
    ], w=UI_W_RIGHT, h=left0.shape[0])
    frame0 = hstack(left0, mid0, right0)

    missing = [m for m in meshes if not existing_kp_file(project_root, m)]
    if mode == "auto" and len(missing) > 0:
        newk = modal_int_overlay("Canonical KPs", frame0, f"{len(missing)} meshes missing KPs — set K (min 4):",
                                 default=kcount, min_value=4, presets=(4, 8, 16, 32))
        if newk is not None:
            kcount = max(4, int(newk))
            save_state(project_root, count=kcount)
        if modal_yes_no_overlay("Canonical KPs", frame0, "Generate now for all missing?", default="no"):
            for mp in missing:
                try:
                    mesh_o3d, V, F = load_mesh_any(mp)
                    pts = auto_keypoints(V, F, kcount, method)
                    save_keypoints(project_root / "canonical_keypoints", mp, "auto", method, pts)
                except BaseException as e:
                    print(f"[!] Error on {mp}: {e}")
            cache.clear()
    while True:
        mp = meshes[i]
        # left: thumbnail
        if mp not in cache:
            cache[mp] = render_mesh_thumb(mp, target_h=UI_H, target_w=UI_W_LEFT)
        left = cache[mp].copy()
        cv2.putText(left, mp.name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # mid: list
        mid = render_list_meshes(left.shape[0], UI_W_MID, "Meshes", meshes, i, project_root)

        # right: settings and status
        ex = existing_kp_file(project_root, mp)
        done = sum(1 for m in meshes if existing_kp_file(project_root, m))
        total = len(meshes)

        lines = [
            f"Progress: {done}/{total} reviewed",
            "",
            "Settings",
            f"  Mode:   {mode}",
            f"  Method: {method if mode == 'auto' else '-'}",
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
        right = text_panel(lines, w=UI_W_RIGHT, h=left.shape[0])

        cv2.imshow("Canonical KPs", hstack(left, mid, right))
        k = cv2.waitKeyEx(60) & 0xFFFFFFFF

        if k in (ord('q'), ord('Q'), 27):  # ESC also quits
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
            break
        elif k in (2490368, ord('w'), ord('W')):  # up
            i = (i - 1) % len(meshes)
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (2621440, ord('s'), ord('S')):  # down
            i = (i + 1) % len(meshes)
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('m'), ord('M')):
            mode = "manual" if mode == "auto" else "auto"
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('f'), ord('F')):
            method = "curvature_fps" if method == "fps" else "fps"
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('+'), ord('=')):
            kcount = min(9999, kcount + 1)
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('-'), ord('_')):
            kcount = max(4, kcount - 1)
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('v'), ord('V')):
            view_after = {"prompt": "yes", "yes": "no", "no": "prompt"}[view_after]
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (ord('k'), ord('K')):  # set K via numeric modal
            frame = hstack(left, mid, right)
            newk = modal_int_overlay("Canonical KPs", frame, "Set keypoint count K (min 4):", default=kcount,
                                     min_value=4, presets=(4, 8, 16, 32))
            if newk is not None:
                kcount = max(4, int(newk))
                save_state(project_root, count=kcount)

        elif k in (ord('g'), ord('G')):  # batch-generate for all missing
            missing = [m for m in meshes if not existing_kp_file(project_root, m)]
            if mode == "auto" and len(missing) > 0:
                frame = hstack(left, mid, right)
                newk = modal_int_overlay("Canonical KPs", frame, f"{len(missing)} missing — set K for batch (min 4):",
                                         default=kcount, min_value=4, presets=(4, 8, 16, 32))
                if newk is not None:
                    kcount = max(4, int(newk))
                    save_state(project_root, count=kcount)
                if modal_yes_no_overlay("Canonical KPs", frame, "Proceed with batch generation?", default="yes"):
                    for mp2 in missing:
                        try:
                            mesh_o3d2, V2, F2 = load_mesh_any(mp2)
                            pts2 = auto_keypoints(V2, F2, kcount, method)
                            save_keypoints(project_root / "canonical_keypoints", mp2, "auto", method, pts2)
                        except BaseException as e:
                            print(f"[!] Error on {mp2}: {e}")
                    cache.clear()
        elif k in (ord('r'), ord('R')):
            meshes[:] = find_meshes(project_root, args.glob, args.mesh)  # refresh
            cache.clear()
            i = min(i, len(meshes) - 1)
            save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount, view=view_after)
        elif k in (13, 10):  # ENTER = run
            # process selected file
            try:
                mesh_o3d, V, F = load_mesh_any(mp)
                if mode == "auto":
                    pts = auto_keypoints(V, F, kcount, method)
                    out = save_keypoints(project_root / "canonical_keypoints", mp, "auto", method, pts)
                    # optionally view
                    frame = hstack(left, mid, right)
                    doit = view_after
                    if view_after == "prompt":
                        doit = "yes" if modal_yes_no_overlay("Canonical KPs", frame, "View keypoints on mesh now?",
                                                             default="no") else "no"

                        # ans = input("View keypoints on mesh? [y/N] ").strip().lower()
                        # doit = "yes" if ans in ("y", "yes") else "no"
                    if doit == "yes":
                        view_points(mesh_o3d, pts, f"Auto KPs ({mp.name})")
                else:
                    if o3d is None:
                        print("[!] Manual mode needs open3d; install with pip.")
                    else:
                        print("Open3D window: click vertices to select; close window to save.")
                        pts = _pick_points_o3d(mesh_o3d, f"Pick Canonical KPs: {mp.name}")
                        if len(pts):
                            out = save_keypoints(project_root / "canonical_keypoints", mp, "manual", None, pts)
                            if view_after == "yes" or (view_after == "prompt" and
                                                       modal_yes_no_overlay("Canonical KPs", hstack(left, mid, right),
                                                                            "View picked keypoints?", default="yes")):
                                view_points(mesh_o3d, pts, f"Manual KPs ({mp.name})")
                        else:
                            print("[i] No points picked; skipped saving.")
                save_state(project_root, last_mesh=str(meshes[i]), mode=mode, method=method, count=kcount,
                           view=view_after)
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
    ap.add_argument("--num", type=int, default=512, help="Number of keypoints (min 4)")
    ap.add_argument("--count", type=int, dest="num", help="Alias for --num (min 4)")
    ap.add_argument("--method", choices=["fps", "curvature_fps"], default="curvature_fps",
                    help="Auto method")
    ap.add_argument("--view", choices=["yes", "no", "prompt"], default="prompt",
                    help="Open viewer after generation")
    args = ap.parse_args()

    meshes = find_meshes(project_root, args.glob, args.mesh)
    if not meshes:
        sys.exit(f"[!] No meshes found (looked in {project_root / 'meshes'} or pattern/mesh provided)")

    if args.browse:
        _ui_browse(project_root, meshes, args)
        return

    out_dir = project_root / "canonical_keypoints"
    for mp in meshes:
        m, V, F = load_mesh_any(mp)
        if args.mode == "auto":
            pts = auto_keypoints(V, F, args.num, args.method)
            save_keypoints(out_dir, mp, "auto", args.method, pts)
            if args.view == "yes" or (
                    args.view == "prompt" and modal_yes_no_window("View keypoints now?", default="no")):
                view_points(m, pts, f"Auto KPs ({mp.name})")
        else:
            if o3d is None:
                sys.exit("[!] Manual mode needs open3d")
            pts = _pick_points_o3d(m, f"Pick Canonical KPs: {mp.name}")
            if len(pts):
                save_keypoints(out_dir, mp, "manual", None, pts)
                if args.view == "yes" or (
                        args.view == "prompt" and modal_yes_no_window("View picked keypoints?", default="yes")):
                    view_points(m, pts, f"Manual KPs ({mp.name})")
            else:
                print(f"[i] No points picked for {mp.name}; skipped.")


if __name__ == "__main__":
    main()
