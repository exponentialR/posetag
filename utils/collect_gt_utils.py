from typing import Any, Dict, List, Sequence

import math
import numpy as np
import cv2, sys, os
import time


def _rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Convert RGB literals to OpenCV's BGR order."""

    return (int(b), int(g), int(r))


def _normalize_capture_key(raw_key: int | None) -> int:
    """Normalize an OpenCV waitKey/waitKeyEx result to an ASCII key code."""

    if raw_key is None or int(raw_key) < 0:
        return -1
    return int(raw_key) & 0xFF


def _capture_key_requests_quit(raw_key: int | None) -> bool:
    """Return true when a GT Capture key should close the capture loop."""

    key = _normalize_capture_key(raw_key)
    return key in (27, ord("q"), ord("Q"), ord("x"), ord("X"))


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

def _label_strip(
    img: np.ndarray,
    text: str,
    bar_h: int = 30,
    bg=_rgb(20, 28, 36),
    fg=_rgb(245, 248, 250),
) -> np.ndarray:
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), bg, -1)
    cv2.rectangle(vis, (0, bar_h - 2), (vis.shape[1], bar_h), _rgb(22, 119, 150), -1)
    cv2.putText(
        vis,
        text,
        (10, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        fg,
        1,
        cv2.LINE_AA,
    )
    return vis

def _text_panel(lines: List[str], width: int = 640, height: int = 480) -> np.ndarray:
    img = np.full((height, width, 3), _rgb(31, 39, 49), np.uint8)
    img = _label_strip(img, "Status / Legend")
    y = 48
    for ln in lines:
        cv2.putText(
            img,
            ln,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            _rgb(224, 232, 240),
            1,
            cv2.LINE_AA,
        )
        y += 21
        if y > height - 8: break
    return img


def _capture_status_panel(
    *,
    session: str,
    faces_visible: int,
    saved_count: int,
    object_summaries: Sequence[Dict[str, Any]],
    paused: bool = False,
    continuous: bool = False,
    auto_capture_enabled: bool = False,
    auto_capture_status: str = "",
    show_help: bool = True,
    capture_message: str = "",
    width: int = 520,
    height: int = 480,
) -> np.ndarray:
    """Render the right-side GT capture status panel."""

    img = np.full((height, width, 3), _rgb(235, 241, 246), np.uint8)
    _draw_header(
        img,
        title="PoseTag GT Capture",
        subtitle=f"session {session}",
        state=("PAUSED" if paused else "LIVE"),
        state_color=_rgb(52, 103, 190) if paused else _rgb(29, 126, 70),
    )

    y = 78
    stat_w = (width - 44) // 3
    for idx, (label, value, accent) in enumerate(
        (
            ("visible faces", str(faces_visible), _rgb(22, 119, 150)),
            ("saved frames", str(saved_count), _rgb(29, 126, 70)),
            (
                "capture mode",
                "smart auto" if auto_capture_enabled else ("continuous" if continuous else "review"),
                _rgb(132, 94, 16),
            ),
        )
    ):
        _draw_metric_card(
            img,
            x=14 + idx * (stat_w + 8),
            y=y,
            w=stat_w,
            h=62,
            label=label,
            value=value,
            accent=accent,
        )

    y += 78
    if capture_message:
        _draw_message_card(img, 14, y, width - 28, 44, capture_message)
        y += 58
    if auto_capture_enabled and auto_capture_status:
        _draw_auto_capture_card(img, 14, y, width - 28, 50, auto_capture_status)
        y += 64

    footer_h = 90 if show_help else 42
    footer_y = height - footer_h - 12
    _draw_section_title(img, "Selected object poses", 14, y)
    y += 22
    if object_summaries:
        shown = 0
        for summary in object_summaries[:4]:
            if y + 82 > footer_y - 10:
                break
            y = _draw_object_summary(img, 14, y, width - 28, summary)
            shown += 1
            y += 8
        remaining = len(object_summaries) - shown
        if remaining > 0 and y < height - 96:
            _put_text(
                img,
                f"+ {remaining} more object{'s' if remaining != 1 else ''}",
                (22, y + 18),
                scale=0.45,
                color=_rgb(76, 88, 100),
            )
            y += 28
    else:
        _draw_empty_state(
            img,
            14,
            y,
            width - 28,
            82,
            "No registered board face is visible.",
            "Move a tagged object face into view before saving.",
        )
        y += 96

    if y < footer_y - 8:
        _draw_divider(img, 14, footer_y - 8, width - 28)
    if show_help:
        _draw_controls_footer(img, 14, footer_y, width - 28, footer_h)
    else:
        _draw_empty_state(
            img,
            14,
            footer_y,
            width - 28,
            footer_h,
            "Controls hidden",
            "Press h to show keys.",
        )
    return img


def _draw_header(
    img: np.ndarray,
    *,
    title: str,
    subtitle: str,
    state: str,
    state_color: tuple[int, int, int],
) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 62), _rgb(16, 28, 42), -1)
    cv2.rectangle(img, (0, 60), (w, 62), _rgb(22, 119, 150), -1)
    _put_text(img, title, (16, 25), scale=0.62, color=_rgb(248, 251, 253), thickness=2)
    _put_text(img, subtitle, (17, 49), scale=0.42, color=_rgb(190, 204, 216))
    pill_w = max(74, 18 + len(state) * 10)
    x0 = w - pill_w - 14
    _rounded_rect(img, x0, 16, pill_w, 30, state_color, radius=8)
    _put_text(
        img,
        state,
        (x0 + 12, 36),
        scale=0.43,
        color=_rgb(255, 255, 255),
        thickness=1,
    )


def _draw_metric_card(
    img: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    accent: tuple[int, int, int],
) -> None:
    _rounded_rect(img, x, y, w, h, _rgb(250, 252, 253), radius=8)
    cv2.rectangle(img, (x, y), (x + 4, y + h), accent, -1)
    _put_text(img, label, (x + 12, y + 20), scale=0.34, color=_rgb(82, 96, 110))
    value_scale = 0.62 if len(value) <= 8 else 0.45
    _put_text(img, value, (x + 12, y + 47), scale=value_scale, color=_rgb(26, 39, 52), thickness=2)


def _draw_message_card(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    message: str,
) -> None:
    _rounded_rect(img, x, y, w, h, _rgb(220, 246, 232), radius=8)
    cv2.rectangle(img, (x, y), (x + 4, y + h), _rgb(29, 126, 70), -1)
    _put_text(img, message, (x + 14, y + 28), scale=0.46, color=_rgb(20, 94, 55), thickness=1)


def _draw_auto_capture_card(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    message: str,
) -> None:
    _rounded_rect(img, x, y, w, h, _rgb(243, 248, 252), radius=8)
    cv2.rectangle(img, (x, y), (x + 4, y + h), _rgb(22, 119, 150), -1)
    _put_text(img, "Smart auto-capture", (x + 14, y + 19), scale=0.36, color=_rgb(72, 86, 100))
    _put_text(img, message, (x + 14, y + 39), scale=0.42, color=_rgb(20, 52, 70), thickness=1)


def _draw_section_title(img: np.ndarray, text: str, x: int, y: int) -> None:
    _put_text(img, text, (x, y + 16), scale=0.46, color=_rgb(30, 43, 56), thickness=2)


def _draw_object_summary(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    summary: Dict[str, Any],
) -> int:
    h = 82
    _rounded_rect(img, x, y, w, h, _rgb(250, 252, 253), radius=8)
    object_name = str(summary.get("object", "(unknown object)"))
    face_key = str(summary.get("face_key", "(unknown face)"))
    _put_text(img, object_name, (x + 12, y + 21), scale=0.47, color=_rgb(18, 31, 44), thickness=2)
    _put_text(img, face_key, (x + 12, y + 42), scale=0.38, color=_rgb(91, 105, 119))

    t = summary.get("translation_m", (0.0, 0.0, 0.0))
    rpy = summary.get("rpy_deg", (0.0, 0.0, 0.0))
    dist = float(summary.get("distance_m", 0.0))
    tag_scale = float(summary.get("tag_scale_ratio", 1.0))
    pairs = int(summary.get("tag_scale_pairs", 0))
    auto = bool(summary.get("tag_scale_auto_corrected", False))
    row1 = f"t m [{float(t[0]):+.2f}, {float(t[1]):+.2f}, {float(t[2]):+.2f}]  d={dist:.2f}"
    row2 = f"rpy [{float(rpy[0]):+.0f}, {float(rpy[1]):+.0f}, {float(rpy[2]):+.0f}] deg"
    row3 = (
        f"tag {summary.get('tag_used', '?')}  bbox {summary.get('bbox_source', '?')}  "
        f"scale {tag_scale:.3f}/{pairs}{' auto' if auto else ''}"
    )
    x2 = x + max(210, int(w * 0.44))
    _put_text(img, row1, (x2, y + 20), scale=0.36, color=_rgb(35, 49, 64))
    _put_text(img, row2, (x2, y + 41), scale=0.36, color=_rgb(35, 49, 64))
    _put_text(img, row3, (x2, y + 62), scale=0.34, color=_rgb(83, 96, 109))
    return y + h


def _draw_empty_state(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    detail: str,
) -> None:
    _rounded_rect(img, x, y, w, h, _rgb(250, 252, 253), radius=8)
    _put_text(img, title, (x + 14, y + 26), scale=0.45, color=_rgb(42, 55, 70), thickness=2)
    _put_text(img, detail, (x + 14, y + 52), scale=0.38, color=_rgb(92, 106, 120))


def _draw_controls_footer(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    _rounded_rect(img, x, y, w, h, _rgb(16, 28, 42), radius=8)
    _put_text(img, "Controls", (x + 14, y + 22), scale=0.43, color=_rgb(242, 246, 249), thickness=2)
    controls = (
        "ENTER / y / s  save / force",
        "r / n / backspace  reject",
        "space / p  pause",
        "h  hide controls",
        "ESC / q / Q  close",
        "x / X  close",
    )
    col_w = (w - 28) // 2
    for idx, text in enumerate(controls):
        col = idx % 2
        row = idx // 2
        _put_text(
            img,
            text,
            (x + 14 + col * col_w, y + 46 + row * 17),
            scale=0.36,
            color=_rgb(200, 214, 226),
        )


def _draw_divider(img: np.ndarray, x: int, y: int, w: int) -> None:
    cv2.line(img, (x, y), (x + w, y), _rgb(205, 216, 226), 1, cv2.LINE_AA)


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.45,
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        str(text),
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _rounded_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    *,
    radius: int = 8,
) -> None:
    radius = max(1, min(radius, w // 2, h // 2))
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)

def _hstack(left: np.ndarray, mid: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    H = max(left.shape[0], 0 if mid is None else mid.shape[0], 0 if right is None else right.shape[0])
    Wl = left.shape[1]
    Wm = mid.shape[1] if mid is not None else Wl
    Wr = right.shape[1] if right is not None else 380
    return np.hstack([_pad_to(left, H, Wl), _pad_to(mid, H, Wm), _pad_to(right, H, Wr)])

def _dashboard_strip(
    left: np.ndarray | None,
    mid: np.ndarray | None,
    right: np.ndarray | None,
    *,
    tile_h: int = 480,
    tile_w: int = 560,
    right_w: int = 520,
) -> np.ndarray:
    L = _resize_to(left, tile_h, tile_w)
    M = _resize_to(mid,  tile_h, tile_w)
    R = _resize_to(right, tile_h, right_w)
    L = _label_strip(L, "Annotation review")
    M = _label_strip(M, "Reprojection / tag debug")
    gutter = np.full((tile_h, 10, 3), _rgb(224, 232, 238), np.uint8)
    return np.hstack([L, gutter, M, gutter, R])


def _show_dash(left: np.ndarray | None, mid: np.ndarray | None, right: np.ndarray | None,
               win: str = "GT Capture", tile_h: int = 480, tile_w: int = 560,
               right_w: int = 520):
    strip = _dashboard_strip(
        left,
        mid,
        right,
        tile_h=tile_h,
        tile_w=tile_w,
        right_w=right_w,
    )
    cv2.imshow(win, strip)
    return strip
