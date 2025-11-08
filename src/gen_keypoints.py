"""
gen_keypoints.py
================
Interactive browser to generate canonical keypoints (axis-aligned box corners)
for OBJ meshes and to preview them in a single stacked OpenCV window.

Project layout (no CLI args needed)
-----------------------------------
<project_root>/
  meshes/<object>.obj                 # input meshes
  objects/<object>/
    keypoints.json                    # generated AABB corners + faces
    object_config.yaml                # mesh metadata (optional)
  faces/<object>/sideA|B|C|D/*.yaml   # OPTIONAL face annotations

What this tool does
-------------------
- Lists meshes under meshes/*.obj and shows per-mesh badges:
    KP OK  -> objects/<name>/keypoints.json exists
    CFG OK -> objects/<name>/object_config.yaml exists
    no ann -> no YAMLs found under faces/<name>/side*/...
- Computes an axis-aligned bounding box (AABB) in the *object* frame.
  If objects/<name>/object_config.yaml contains mesh.T_mesh_object.matrix
  (4x4 homogeneous), it is applied to mesh vertices before the AABB is built.
- Saves:
    keypoints.json:
      {
        "units_to_m": <float>,
        "points": { "<corner_name>": [x,y,z], ... },
        "faces":  { "<name>_sideA|B|C|D": [corner_name0..3] }
      }
    object_config.yaml (when enabled):
      mesh.path (relative if possible), mesh.units_to_m, mesh.T_mesh_object
- Provides in-panel previews so you never leave the main window.

Panels
------
Left   : Either a normalised XY scatter of mesh vertices or a 3-view (XY/XZ/YZ)
         overlay with the AABB cuboid; toggle with V.
Middle : Mesh list with status pills (KP/CFG/ann).
Right  : Project details, current settings, key bindings.

Keys
----
W / S or Arrow Up / Down : Navigate meshes
ENTER / G                : Generate keypoints.json (prompts if it exists)
V                        : Toggle in-panel preview (points <-> 3-view)
Z                        : Open3D interactive preview (separate window; optional)
O                        : Toggle writing objects/<name>/object_config.yaml
A                        : Toggle auto side mapping
                           - Auto: A..D mapped from the two thinnest axes
                           - Manual: +X, -X, +Y, -Y map to A..D
u / j                    : units_to_m +0.001 / -0.001
U / J                    : units_to_m +0.01  / -0.01
Q / ESC                  : Quit

Generate behaviour
------------------
- If keypoints.json already exists you can:
    Y/ENTER -> overwrite
    B       -> keep both (appends _v2, _v3, ...)
    ESC/Q   -> cancel
- When write-config is ON, an object_config.yaml is written (identity T by default).
- If no face annotations are found for the object, a friendly reminder is shown.

Tips / Troubleshooting
----------------------
- Units:
    • Unity exports: 1 Unity unit = 1 metre → set units_to_m = 1.0 (typical).
    • CAD exports: often millimetres → units_to_m = 0.001.
    • Blender/Maya: depends on scene settings; check raw extents.
- If the 3-view looks empty, ensure units_to_m is sensible and that the OBJ has real 'v' lines.
  Large meshes are subsampled in the scatter for speed.
- Open3D is optional; only the Z preview requires it.

Run
---
python -m src.gen_keypoints
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2, yaml

from gt6dof_atag.utils.project_config import resolve_project_root, ensure_project_dirs
from gt6dof_atag.utils.logger import init_project_logger

def _pill_width(text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 6   # same as _draw_pill
    gap = 12  # spacing between pills
    return tw + pad*2 + gap



# ---------------- Geometry ----------------

def load_obj_vertices(path: Path) -> np.ndarray:
    vs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *rest = line.strip().split()
                try: vs.append([float(x), float(y), float(z)])
                except ValueError: pass
    if not vs:
        raise SystemExit(f"[!] No vertices found in {path}")
    return np.asarray(vs, dtype=np.float64)

def corners_from_bounds(vmin, vmax, r=6):
    xmin, ymin, zmin = vmin; xmax, ymax, zmax = vmax
    R = lambda x: round(float(x), r)
    return {
        "xmin_ymin_zmin": [R(xmin), R(ymin), R(zmin)],
        "xmin_ymin_zmax": [R(xmin), R(ymin), R(zmax)],
        "xmin_ymax_zmin": [R(xmin), R(ymax), R(zmin)],
        "xmin_ymax_zmax": [R(xmin), R(ymax), R(zmax)],
        "xmax_ymin_zmin": [R(xmax), R(ymin), R(zmin)],
        "xmax_ymin_zmax": [R(xmax), R(ymin), R(zmax)],
        "xmax_ymax_zmin": [R(xmax), R(ymax), R(zmin)],
        "xmax_ymax_zmax": [R(xmax), R(ymax), R(zmax)],
    }

def face_corner_names(axis: str, sign: int):
    assert axis in ("x","y","z") and sign in (-1, +1)
    if axis == "x":
        side = "xmin" if sign < 0 else "xmax"
        return [f"{side}_ymax_zmin", f"{side}_ymax_zmax", f"{side}_ymin_zmax", f"{side}_ymin_zmin"]
    if axis == "y":
        side = "ymin" if sign < 0 else "ymax"
        return [f"xmax_{side}_zmin", f"xmax_{side}_zmax", f"xmin_{side}_zmax", f"xmin_{side}_zmin"]
    side = "zmin" if sign < 0 else "zmax"
    return [f"xmax_ymax_{side}", f"xmin_ymax_{side}", f"xmin_ymin_{side}", f"xmax_ymin_{side}"]


def auto_side_mapping(extents: np.ndarray) -> Tuple[str, str]:
    order = np.argsort(extents)  # thinnest first
    ab_axis = ["x","y","z"][order[0]]
    cd_axis = ["x","y","z"][order[1]]
    return ab_axis, cd_axis

def apply_T_mesh_object(V: np.ndarray, cfg_path: Path) -> np.ndarray:
    try:
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            T = cfg.get("mesh", {}).get("T_mesh_object", {}).get("matrix", None)
            if T is not None and len(T)==4 and len(T[0])==4:
                T = np.array(T, dtype=np.float64)
                V = (np.c_[V, np.ones(len(V))] @ T.T)[:, :3]
    except Exception:
        pass
    return V

# ---------------- UI constants ----------------

# Wider list + status so nothing gets truncated.
LEFT_W, MID_W, RIGHT_W = 1040, 560, 560
PANEL_H = 900

BG = (245,245,245); FG=(40,40,40); ACC=(60,90,220)
OK=(30,140,30); MUTED=(120,120,120); WARN=(0,140,255); EDGE=(0,0,0)

ICON_OK="OK"; ICON_NO="--"  # ASCII only, avoids OpenCV '???'

# ---------------- UI helpers ----------------

def _pad_to(img, h, w):
    out = np.full((h, w, 3), BG, np.uint8)
    if img is None: return out
    hh, ww = img.shape[:2]; out[:hh, :ww] = img; return out

def _hstack(a,b,c):
    H = max(a.shape[0], b.shape[0], c.shape[0])
    return np.hstack([_pad_to(a,H,a.shape[1]), _pad_to(b,H,b.shape[1]), _pad_to(c,H,c.shape[1])])

def _draw_pill(img, x, y, text, col):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad=6; cv2.rectangle(img, (x, y-th-pad), (x+tw+pad*2, y+pad//2), (230,230,230), -1, cv2.LINE_AA)
    cv2.putText(img, text, (x+pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return x+tw+pad*2+6

def _wrap_lines(text, width_chars=40):
    words=text.split(); lines=[]; cur=""
    for w in words:
        if len(cur)+len(w)+1>width_chars:
            lines.append(cur); cur=w
        else:
            cur = (cur+" "+w).strip()
    if cur: lines.append(cur)
    return lines

def _text_panel(lines, width=RIGHT_W, height=PANEL_H):
    img = np.full((height, width, 3), 252, np.uint8)
    y=34
    for ln in lines:
        if ln=="---":
            cv2.line(img,(12,y),(width-12,y),(210,210,210),1,cv2.LINE_AA); y+=10; continue
        for seg in _wrap_lines(ln, width_chars=max(28, width//12)):
            cv2.putText(img, seg, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, FG, 1, cv2.LINE_AA); y+=22
        if y>height-18: break
    return img

def _render_list_panel(h, w, title, items, sel):
    pan = np.full((h, w, 3), 250, np.uint8)
    cv2.putText(pan, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FG, 2, cv2.LINE_AA)
    max_show = min(28, len(items))
    start = max(0, min(sel - max_show//2, max(0, len(items)-max_show)))
    y=58
    for i in range(start, start+max_show):
        if i>=len(items): break
        name, has_kp, has_cfg, has_ann = items[i]
        prefix = "> " if i==sel else "  "
        col = ACC if i==sel else FG; thk=2 if i==sel else 1
        cv2.putText(pan, prefix+name, (16,y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, col, thk, cv2.LINE_AA)
        x = w-12
        # pills right-aligned
        for text, colour in reversed([
            (f"ann {ICON_OK}" if has_ann else "no ann", OK if has_ann else WARN),
            (f"CFG {ICON_OK}" if has_cfg else f"CFG {ICON_NO}", OK if has_cfg else MUTED),
            (f"KP {ICON_OK}" if has_kp else f"KP {ICON_NO}", OK if has_kp else MUTED),
        ]):
            x -= _pill_width(text)
            _draw_pill(pan, x, y, text, colour)

        y += 28
    tips = "W/S or arrows: select   ENTER/G: generate   V: in-panel preview   Z: Open3D   O: cfg   A: auto   u/U j/J: units   Q: quit"
    cv2.putText(pan, tips, (16, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
    return pan

def _grid(img, step=80, col=(235,235,235)):
    h,w=img.shape[:2]
    for x in range(step, w, step):
        cv2.line(img,(x,0),(x,h),col,1,cv2.LINE_AA)
    for y in range(step, h, step):
        cv2.line(img,(0,y),(w,y),col,1,cv2.LINE_AA)

def _points_panel(V: np.ndarray, w=LEFT_W, h=PANEL_H):
    img = np.full((h,w,3),255,np.uint8)
    _grid(img)
    Vc = V - V.mean(axis=0, keepdims=True)
    span = Vc.ptp(axis=0).max()
    if span>0: Vc /= span
    xy = Vc[:,[0,1]]*0.46 + 0.5
    pts = np.clip((xy*np.array([w-40,h-40])+20).astype(int),0,[w-1,h-1])
    step = max(1, len(pts)//8000)
    for (u,v) in pts[::step]:
        img[v,u] = (40,40,40)
    cv2.putText(img,"XY view (normalised)",(16,32),cv2.FONT_HERSHEY_SIMPLEX,0.7,(20,20,20),2,cv2.LINE_AA)
    return img

def _edges_of_cuboid():
    return [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]

def _proj_view(title, V, C, w, h):
    axes = {"XY":(0,1), "XZ":(0,2), "YZ":(1,2)}[title]
    P = V[:,axes]; K = C[:,axes]
    allp = np.vstack([P, K])
    mn = allp.min(axis=0); mx = allp.max(axis=0)
    span = np.maximum(mx-mn, 1e-9)
    def to_px(X):
        Xn = (X-mn)/span
        return np.clip((Xn*np.array([w-40,h-40])+20).astype(int),0,[w-1,h-1])
    img = np.full((h,w,3),255,np.uint8); _grid(img)
    for (u,v) in to_px(P)[::max(1,len(P)//8000)]:
        img[v,u]=(50,50,50)
    Cpx = to_px(K);
    for a,b in _edges_of_cuboid():
        cv2.line(img, tuple(Cpx[a]), tuple(Cpx[b]), (0,120,255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{title} view", (14,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 2, cv2.LINE_AA)
    return img

def _three_view(V, C, w=LEFT_W, h=PANEL_H):
    tw = w//3
    return np.hstack([_proj_view("XY",V,C,tw,h), _proj_view("XZ",V,C,tw,h), _proj_view("YZ",V,C,tw,h)])

# ---------------- Payload & preview ----------------

def _make_faces_mapping(obj_name: str, extents, auto=True):
    def parse_face(s): return (s[1].lower(), +1 if s[0]=="+" else -1)
    if auto:
        a_axis, c_axis = auto_side_mapping(extents)
        facespec = [f"+{a_axis.upper()}", f"-{a_axis.upper()}",
                    f"+{c_axis.upper()}", f"-{c_axis.upper()}"]
    else:
        facespec = ["+X","-X","+Y","-Y"]
    A = face_corner_names(*parse_face(facespec[0]))
    B = face_corner_names(*parse_face(facespec[1]))
    C = face_corner_names(*parse_face(facespec[2]))
    D = face_corner_names(*parse_face(facespec[3]))
    faces = {
        f"{obj_name}_sideA": A,
        f"{obj_name}_sideB": B,
        f"{obj_name}_sideC": C,
        f"{obj_name}_sideD": D,
    }
    return faces, facespec

def _any_annotations_exist(project_root: Path, obj_name: str) -> bool:
    return any((project_root/"faces"/obj_name).glob("side*/**/*.yaml"))

def _compute_payload(project_root: Path, obj_name: str, obj_path: Path,
                     units_to_m: float, auto_sides: bool, round_dec=6):
    V = load_obj_vertices(obj_path)
    V = apply_T_mesh_object(V, project_root/"objects"/obj_name/"object_config.yaml")
    vmin = V.min(axis=0)*units_to_m; vmax = V.max(axis=0)*units_to_m
    points = corners_from_bounds(vmin, vmax, r=round_dec)
    faces, facespec = _make_faces_mapping(obj_name, vmax-vmin, auto=auto_sides)
    payload = {"units_to_m": float(units_to_m), "points": points, "faces": faces, "facespec": facespec}
    cuboid = np.array(list(points.values()), dtype=np.float64)
    return payload, V*units_to_m, cuboid

def _open3d_preview(obj_name: str, obj_path: Path, kp_dict: dict):
    try:
        import open3d as o3d
    except Exception:
        print("[!] open3d not installed; preview disabled."); return
    s = float(kp_dict.get("units_to_m",1.0))
    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
    mesh.scale(s, center=[0,0,0]); geoms=[mesh]
    def sph(c,r=0.003,col=(1,0.5,0)):
        m=o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12); m.paint_uniform_color(col); m.translate(np.array(c,float)); return m
    for p in kp_dict["points"].values(): geoms.append(sph(p))
    o3d.visualization.draw_geometries(geoms, window_name=f"{obj_name} preview", width=1200, height=800)

# ---------------- Browser ----------------

def _index_meshes(project_root: Path):
    ms = sorted((project_root/"meshes").glob("*.obj"))
    out=[]
    for p in ms:
        name=p.stem
        kp=project_root/"objects"/name/"keypoints.json"
        cfg=project_root/"objects"/name/"object_config.yaml"
        ann=_any_annotations_exist(project_root,name)
        out.append({"name":name,"path":p,"kp_path":kp,"cfg_path":cfg,
                    "has_kp":kp.exists(),"has_cfg":cfg.exists(),"has_ann":ann})
    return out

def main():
    pr = resolve_project_root(None); ensure_project_dirs(pr)
    log = init_project_logger(pr/"logs"/"gen_keypoints.log", level="INFO", console=True)

    entries=_index_meshes(pr)
    if not entries: raise SystemExit(f"[!] No .obj files in {pr}/meshes")

    i=0; auto=True; units=1.0; write_cfg=True; mode="bbox"
    win="Gen Keypoints"; cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, LEFT_W+MID_W+RIGHT_W, PANEL_H)

    while True:
        e=entries[i]
        V = apply_T_mesh_object(load_obj_vertices(e["path"]), e["cfg_path"])
        # left = _three_view(*(lambda payload: (payload[1], payload[2]))(_compute_payload(pr, e["name"], e["path"], units, auto))) if mode=="bbox" else _points_panel(V, LEFT_W, PANEL_H)
        if mode == "bbox":
            _, V_scaled, cuboid = _compute_payload(pr, e["name"], e["path"], units, auto)
            left = _three_view(V_scaled, cuboid, LEFT_W, PANEL_H)
        else:
            left = _points_panel(V, LEFT_W, PANEL_H)
        names=[(en["name"], en["has_kp"], en["has_cfg"], en["has_ann"]) for en in entries]
        mid = _render_list_panel(PANEL_H, MID_W, "Meshes (meshes/*.obj)", names, i)

        status = [
            str(pr), "---",
            f"Mesh: {e['path'].name}",
            f"Object: {e['name']}",
            f"keypoints.json: {ICON_OK if e['has_kp'] else ICON_NO}",
            f"object_config.yaml: {ICON_OK if e['has_cfg'] else ICON_NO}",
            f"face annotations: {'OK' if e['has_ann'] else 'no ann (none found)'}",
            "---",
            f"units_to_m: {units:.3f}  (tip: mm to m is 0.001)",
            f"auto sides: {'ON' if auto else 'OFF'}",
            f"write object_config: {'ON' if write_cfg else 'OFF'}",
            f"in-panel preview: {mode}",
            "---",
            "ENTER/G : generate (confirm overwrite if exists)",
            "V       : toggle in-panel preview (points / 3-view)",
            "Z       : Open3D interactive preview",
            "O       : toggle writing object_config.yaml",
            "A       : toggle auto sides",
            "u/j     : units_to_m +/- 0.001    U/J : +/- 0.01",
            "Q/ESC   : quit",
        ]
        right = _text_panel(status, RIGHT_W, PANEL_H)

        cv2.imshow(win, _hstack(left, mid, right))
        k = cv2.waitKeyEx(50) & 0xFFFFFFFF

        if k in (ord('q'), 27): break
        elif k in (2490368, ord('w'), ord('W')): i = (i-1) % len(entries)
        elif k in (2621440, ord('s'), ord('S')): i = (i+1) % len(entries)
        elif k in (ord('o'), ord('O')): write_cfg = not write_cfg
        elif k in (ord('a'), ord('A')): auto = not auto
        elif k in (ord('v'), ord('V')): mode = "points" if mode=="bbox" else "bbox"
        elif k in (ord('z'), ord('Z')):
            payload,_,_ = _compute_payload(pr, e["name"], e["path"], units, auto)
            _open3d_preview(e["name"], e["path"], payload)
        elif k == ord('u'): units = max(1e-9, units+0.001)
        elif k == ord('j'): units = max(1e-9, units-0.001)
        elif k == ord('U'): units = max(1e-9, units+0.01)
        elif k == ord('J'): units = max(1e-9, units-0.01)
        elif k in (13, 10, ord('g'), ord('G')):
            payload,_,_ = _compute_payload(pr, e["name"], e["path"], units, auto)
            facespec = payload.pop("facespec")
            kp_path = pr/"objects"/e["name"]/ "keypoints.json"
            out_root = kp_path.parent; out_root.mkdir(parents=True, exist_ok=True)

            save_path = kp_path
            if kp_path.exists():
                prompt = _text_panel([
                    "keypoints.json exists:", str(kp_path), "---",
                    "Y/ENTER: overwrite   B: keep both (_v2, _v3, ...)   ESC/Q: cancel"
                ], RIGHT_W, PANEL_H)
                cv2.imshow(win, _hstack(left, mid, prompt))
                while True:
                    kk = cv2.waitKey(0) & 0xFF
                    if kk in (13, ord('y'), ord('Y')): break
                    if kk in (ord('b'), ord('B')):
                        idx=2; cand=kp_path.with_stem(kp_path.stem+f"_v{idx}")
                        while cand.exists():
                            idx+=1; cand=kp_path.with_stem(kp_path.stem+f"_v{idx}")
                        save_path=cand; break
                    if kk in (27, ord('q'), ord('Q')): save_path=None; break
                if save_path is None: continue

            with open(save_path,"w") as f: json.dump(payload, f, indent=2)

            if write_cfg:
                cfg = {
                    "mesh": {
                        "path": str(e["path"].relative_to(pr) if e["path"].is_relative_to(pr) else e["path"]),
                        "units_to_m": float(units),
                        "T_mesh_object": {"matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
                    },
                    "notes": f"Sides auto-mapped: A,B,C,D = {facespec}"
                }
                with open(out_root/"object_config.yaml","w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

            log.info("Wrote %s%s", save_path, " and object_config.yaml" if write_cfg else "")
            entries[i]["has_kp"]=True
            entries[i]["has_cfg"]=entries[i]["has_cfg"] or write_cfg
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
