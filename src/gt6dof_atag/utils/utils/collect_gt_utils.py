from typing import Dict, List

import math
import numpy as np
import cv2, sys, os
import time


def _dominant_color_near_polygon(bgr: np.ndarray, uv: np.ndarray) -> tuple[int,int,int]:
    """
    Pick a stable, object-like color from pixels under the projected tag quad.
    Returns a BGR tuple.
    """
    mask = np.zeros(bgr.shape[:2], np.uint8)
    pts = np.int32(uv.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, pts, 255)
    # Erode a bit so we avoid black tag border / background bleed
    mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
    sel = bgr[mask > 0]
    if sel.size == 0:
        return (0, 255, 255)  # fallback: yellow

    # Use median (robust to highlights/shadows)
    med = np.median(sel, axis=0).astype(np.uint8)  # BGR

    # Punch up saturation a little so it reads well on video
    hsv = cv2.cvtColor(med.reshape(1,1,3), cv2.COLOR_BGR2HSV).reshape(3)
    H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
    S = max(S, 140)          # ensure some saturation
    V = min(max(V, 90), 230) # keep within display-friendly range
    out = cv2.cvtColor(np.uint8([[[H, S, V]]]), cv2.COLOR_HSV2BGR).reshape(3)
    return (int(out[0]), int(out[1]), int(out[2]))  # B,G,R

def _draw_tag_quad(img: np.ndarray, uv: np.ndarray, color_bgr: tuple[int,int,int]):
    """Draw a visible quad (black under-stroke + colored over-stroke)."""
    pts = np.int32(uv.reshape(-1,1,2))
    cv2.polylines(img, [pts], True, (0,0,0), 4, cv2.LINE_AA)      # outline
    cv2.polylines(img, [pts], True, color_bgr, 2, cv2.LINE_AA)    # colored stroke



def _ts_tag_from_epoch(ts:float) -> str:
    """Convert epoch timestamp to a compact string suitable for tag IDs."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(ts))

def _map_class_id_and_name(object_name: str) -> tuple[int, str]:
    n = object_name.lower()
    if "connection_plate" in n or ("connection" in n and "plate" in n):
        return 2, "connection_plate"
    if "full_assembly" in n or ("full" in n and "assembly" in n):
        return 3, "full_assembly"
    if "column" in n:
        return 1, "column"
    return 0, object_name  # fallback: keep original name

def rpy_from_R(R: np.ndarray) -> tuple[float,float,float]:
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    return tuple(np.degrees([roll, pitch, yaw]).tolist())



def _estimate_tag_scale(det_by_id: Dict[int, object], T_board_tag: Dict[int, np.ndarray]) -> tuple[float,int,float]:
    """Median ratio of inter-tag distances (camera / board)."""
    ratios: List[float] = []
    ids = [k for k in det_by_id.keys() if k in T_board_tag]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pa = det_by_id[a].pose_t.reshape(3).astype(float)
            pb = det_by_id[b].pose_t.reshape(3).astype(float)
            d_cam = float(np.linalg.norm(pa - pb))
            ba = T_board_tag[a][:3, 3]; bb = T_board_tag[b][:3, 3]
            d_board = float(np.linalg.norm(ba - bb))
            if d_board > 1e-9:
                ratios.append(d_cam / d_board)
    if not ratios:
        return 1.0, 0, 0.0
    ratios = np.asarray(ratios, float)
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    return med, int(len(ratios)), mad


def _pad_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    out = np.zeros((h, w, 3), np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out

def _resize_to(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    H, W = img.shape[:2]
    interp = cv2.INTER_AREA if (H > h or W > w) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)

def _label_strip(img: np.ndarray, text: str, bar_h: int = 26, bg=(0, 0, 0), fg=(0, 255, 255)) -> np.ndarray:
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), bg, -1)
    cv2.putText(vis, text, (8, int(bar_h*0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)
    return vis

def _text_panel(lines: List[str], width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    img = _label_strip(img, "Status / Legend")
    y = 40
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 1, cv2.LINE_AA)
        y += 22
        if y > height - 8: break
    return img

def _hstack(left: np.ndarray, mid: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    H = max(left.shape[0], 0 if mid is None else mid.shape[0], 0 if right is None else right.shape[0])
    Wl = left.shape[1]
    Wm = mid.shape[1] if mid is not None else Wl
    Wr = right.shape[1] if right is not None else 380
    return np.hstack([_pad_to(left, H, Wl), _pad_to(mid, H, Wm), _pad_to(right, H, Wr)])

def _show_dash(left: np.ndarray | None, mid: np.ndarray | None, right: np.ndarray | None,
               win: str = "GT Capture", tile_h: int = 480, tile_w: int = 640):
    L = _resize_to(left, tile_h, tile_w)
    M = _resize_to(mid,  tile_h, tile_w)
    R = _resize_to(right, tile_h, tile_w)
    L = _label_strip(L, "Annotation")
    M = _label_strip(M, "Reprojection / Tag debug")
    strip = np.hstack([L, M, R])
    cv2.imshow(win, strip)

