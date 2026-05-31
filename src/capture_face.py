"""
capture_face.py — Face-shot capture for later T_board_object annotation
Author: Samuel Adebayo

Summary
-------
Capture wide shots per object face with AprilTag overlays and metadata for
subsequent T_board_object (board→object) computation. Project-aware: resolves
an active project root and writes under <project_root>/shots/… by default.

What is current
----------------
- Full face (e.g., connection_plate_white_sideA) or base-only (e.g., connection_plate_white);
  base auto-resolves side via boards/tag_registry.yaml using live detections.
- In-window registered-face queue ('o'); no terminal prompts in preview.
- Optional stable-tag auto-capture for queued faces.
- Info + last-saved side panels ('g' toggles). Panels are UI-only.
- Safety save: press Enter twice within 3 s if expected tags aren’t seen.
- Resizable window; camera frame stays at requested resolution.

Saving layout
-------------
Default (by object and side):
  <project_root>/shots/<object_base>/side<SideLetter>/
    <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_{raw,ann}.png
    <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_meta.json

Keys
----
ENTER save (double-press to force) | arrow keys cycle face queue |
o queue | a auto-side | g panels | h help | q / ESC quit
"""

from __future__ import annotations
import argparse, csv, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Mapping, Sequence

import numpy as np
import cv2

from utils import capture_source
from utils.capture_utils import draw_detections, make_info_panel, make_recent_panel, text_lines
from utils.logger import init_project_logger
from posetag.pipelines.capture_face import (
    activate_capture_project_root,
    CaptureFaceError,
    build_capture_metadata,
    build_shot_paths,
    capture_timestamp,
    create_apriltag_detector,
    create_capture_output_dirs,
    load_apriltag_detector_class,
    load_capture_calibration,
    load_capture_registry,
    load_registered_faces,
    parse_base_and_side,
    prepare_capture_paths,
    preview_capture_project_root,
    resolve_capture_calibration_path,
    select_initial_faces,
    validate_capture_source_args,
    write_capture_outputs,
)

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 2490368, 2621440, 2424832, 2555904
KEY_UP_CODES = {KEY_UP, 63232, 65362, 82}
KEY_DOWN_CODES = {KEY_DOWN, 63233, 65364, 84}
KEY_LEFT_CODES = {KEY_LEFT, 63234, 65361, 81}
KEY_RIGHT_CODES = {KEY_RIGHT, 63235, 65363, 83}


UI_BG = (245, 249, 251)
UI_CARD = (255, 255, 255)
UI_BORDER = (212, 226, 234)
UI_TEXT = (36, 49, 61)
UI_MUTED = (98, 113, 126)
UI_BLUE = (160, 91, 0)
UI_GREEN = (64, 134, 29)
UI_AMBER = (24, 139, 204)
UI_RED = (44, 65, 190)
UI_DARK = (38, 45, 52)
UI_DARK_2 = (62, 72, 82)
UI_WHITE = (255, 255, 255)


def _face_label(face_entry, max_chars=28, default="unregistered"):
    """Short label for overlay: YAML basename, truncated; safe if face_entry is None."""
    if not face_entry:
        return default
    yml = face_entry.get("yaml", "")
    base = Path(yml).stem if yml else default
    if len(base) > max_chars:
        base = base[:max_chars - 1] + "..."
    return base


def _put(
    image: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    scale: float = 0.55,
    colour: Tuple[int, int, int] = UI_TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _fill_alpha(
    image: np.ndarray,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    colour: Tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, p0, p1, colour, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def object_picker_panel(
    h: int,
    w: int,
    bases: List[str],
    sel: int,
    hint: str = "Choose object",
) -> np.ndarray:
    pan = np.full((h, w, 3), UI_BG, np.uint8)
    margin = 14
    cv2.rectangle(pan, (margin, margin), (w - margin, h - margin), UI_CARD, -1)
    cv2.rectangle(pan, (margin, margin), (w - margin, h - margin), UI_BORDER, 1)
    _put(pan, hint, (margin + 16, margin + 30), scale=0.78, thickness=2)

    total = len(bases)
    status = f"{min(sel + 1, total) if total else 0}/{total} registered objects"
    _put(pan, status, (margin + 16, margin + 58), colour=UI_MUTED)
    if not bases:
        _put(
            pan,
            "No registered object bases found in the tag registry.",
            (margin + 16, margin + 96),
            colour=UI_RED,
        )
        return pan

    max_show = min(12, len(bases))
    start = max(0, min(sel - max_show // 2, len(bases) - max_show))
    end = min(len(bases), start + max_show)

    y = margin + 92
    for i in range(start, end):
        selected = i == sel
        row_top = y - 21
        row_bottom = y + 9
        if selected:
            cv2.rectangle(
                pan,
                (margin + 10, row_top),
                (w - margin - 10, row_bottom),
                (236, 246, 251),
                -1,
            )
            cv2.rectangle(
                pan,
                (margin + 10, row_top),
                (margin + 14, row_bottom),
                UI_BLUE,
                -1,
            )
        prefix = ">" if selected else " "
        colour = UI_TEXT if selected else UI_MUTED
        _put(
            pan,
            f"{prefix} {i + 1:02d}  {_truncate_middle(bases[i], 34)}",
            (margin + 20, y),
            scale=0.58,
            colour=colour,
            thickness=2 if selected else 1,
        )
        y += 34

    tips = "Up/Down or W/S navigate   Enter select   Esc cancel   1-9 jump"
    _put(pan, tips, (margin + 16, h - margin - 18), scale=0.45, colour=UI_MUTED)
    return pan


def face_queue_panel(
    h: int,
    w: int,
    faces: Sequence[Mapping[str, object]],
    sel: int,
    *,
    captured_faces: Optional[Set[str]] = None,
    hint: str = "Capture queue",
    interaction: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Draw a scrollable registered-face queue for the OpenCV side panel."""

    pan = np.full((h, w, 3), UI_BG, np.uint8)
    margin = 14
    captured = captured_faces or set()
    total = len(faces)
    sel = max(0, min(sel, total - 1)) if total else 0

    cv2.rectangle(pan, (margin, margin), (w - margin, h - margin), UI_CARD, -1)
    cv2.rectangle(pan, (margin, margin), (w - margin, h - margin), UI_BORDER, 1)
    _put(pan, hint, (margin + 16, margin + 30), scale=0.78, thickness=2)
    if interaction is not None:
        interaction.clear()
        interaction["queue_rows"] = []
        interaction["queue_rect"] = (margin, margin, w - margin, h - margin)

    done = sum(1 for face in faces if face_key(face) in captured)
    status = f"{done}/{total} captured"
    _put(pan, status, (margin + 16, margin + 58), colour=UI_MUTED)
    if not faces:
        _put(
            pan,
            "No registered faces were found in the tag registry.",
            (margin + 16, margin + 96),
            colour=UI_RED,
        )
        return pan

    max_show = min(11, len(faces))
    start = max(0, min(sel - max_show // 2, len(faces) - max_show))
    end = min(len(faces), start + max_show)

    y = margin + 92
    for i in range(start, end):
        face = faces[i]
        key = face_key(face)
        selected = i == sel
        captured_here = key in captured
        row_top = y - 22
        row_bottom = y + 12
        if interaction is not None:
            interaction.setdefault("queue_rows", []).append(
                (i, margin + 10, row_top, w - margin - 10, row_bottom)
            )
        if selected:
            cv2.rectangle(
                pan,
                (margin + 10, row_top),
                (w - margin - 10, row_bottom),
                (236, 246, 251),
                -1,
            )
            cv2.rectangle(
                pan,
                (margin + 10, row_top),
                (margin + 14, row_bottom),
                UI_GREEN if captured_here else UI_BLUE,
                -1,
            )
        marker = "[x]" if captured_here else "[ ]"
        prefix = ">" if selected else " "
        colour = UI_GREEN if captured_here else (UI_TEXT if selected else UI_MUTED)
        label = _truncate_middle(face_label(face), 34)
        _put(
            pan,
            f"{prefix} {marker} {label}",
            (margin + 20, y),
            scale=0.54,
            colour=colour,
            thickness=2 if selected else 1,
        )
        y += 36

    tips = "Click row or scroll   Enter save   q/Esc quit"
    _put(pan, tips, (margin + 16, h - margin - 18), scale=0.44, colour=UI_MUTED)
    return pan


def face_key(face: Mapping[str, object]) -> str:
    """Return the queue key for a registered face."""

    return str(face.get("object", "")).strip() or str(face.get("yaml", "")).strip()


def face_label(face: Mapping[str, object]) -> str:
    """Return the user-facing queue label for a registered face."""

    return face_key(face) or Path(str(face.get("yaml", ""))).stem or "unregistered"


def load_existing_captured_faces(manifest_path: Path) -> Set[str]:
    """Return valid captured face labels already present in the manifest."""

    captured: Set[str] = set()
    if not manifest_path.exists():
        return captured
    try:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                validation = str(row.get("validation_ok", "")).strip().lower()
                if validation not in {"1", "true", "yes"}:
                    continue
                object_full = str(row.get("object_full", "")).strip()
                if object_full:
                    captured.add(object_full)
    except OSError:
        return set()
    return captured


def load_recent_saved_images(
    manifest_path: Path,
    *,
    project_root: Optional[Path] = None,
    limit: int = 12,
) -> List[np.ndarray]:
    """Return recent annotated/raw images already listed in the manifest."""

    if not manifest_path.exists() or limit <= 0:
        return []
    try:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []

    root = project_root or manifest_path.parent
    images_reversed: List[np.ndarray] = []
    for row in reversed(rows):
        for key in ("path_ann", "path_raw"):
            image_path = _resolve_manifest_image_path(root, row.get(key, ""))
            if image_path is None or not image_path.exists():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                images_reversed.append(image)
                break
        if len(images_reversed) >= limit:
            break
    return list(reversed(images_reversed))


def _resolve_manifest_image_path(root: Path, value: str) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else root / path


def first_uncaptured_index(
    faces: Sequence[Mapping[str, object]],
    captured_faces: Set[str],
    *,
    start: int = 0,
) -> Optional[int]:
    """Return the next uncaptured face index, wrapping once through the queue."""

    if not faces:
        return None
    count = len(faces)
    for offset in range(count):
        idx = (start + offset) % count
        if face_key(faces[idx]) not in captured_faces:
            return idx
    return None


def cycle_face_selection(state: Dict, delta: int = 1) -> bool:
    """Cycle the active face explicitly and disable auto-side selection."""

    faces = state.get("faces") or []
    if not faces:
        return False
    current = int(state.get("face_idx", 0))
    state["face_idx"] = (current + int(delta)) % len(faces)
    state["auto_side"] = False
    state["face"] = faces[state["face_idx"]]
    base, _side = parse_base_and_side(str(state["face"].get("object", "")))
    state["object_base"] = base or state.get("object_base")
    state["save_warn"] = False
    return True


def select_face_index(state: Dict, index: int, *, auto_side: bool = False) -> bool:
    """Select a face by index and update display state."""

    faces = state.get("faces") or []
    if not faces:
        return False
    state["face_idx"] = max(0, min(int(index), len(faces) - 1))
    state["auto_side"] = bool(auto_side)
    state["face"] = faces[state["face_idx"]]
    base, _side = parse_base_and_side(str(state["face"].get("object", "")))
    state["object_base"] = base or state.get("object_base")
    state["save_warn"] = False
    state["auto_candidate"] = None
    state["auto_streak"] = 0
    return True


def queue_row_at(
    rows: Sequence[Tuple[int, int, int, int, int]],
    x: int,
    y: int,
) -> Optional[int]:
    """Return the queue item index under local panel coordinates."""

    for index, x0, y0, x1, y1 in rows:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return int(index)
    return None


def mouse_wheel_queue_delta(flags: int) -> int:
    """Return a face-selection delta from an OpenCV mouse-wheel event."""

    try:
        raw_delta = int(cv2.getMouseWheelDelta(flags))
    except Exception:
        raw_delta = int(flags)
    if raw_delta > 0:
        return -1
    if raw_delta < 0:
        return 1
    return 0


def capture_face_mouse_action(
    event: int,
    x: int,
    y: int,
    flags: int,
    layout: Mapping[str, object],
) -> Optional[Tuple[str, int]]:
    """Translate an OpenCV mouse event into a queue selection action."""

    wheel_events = {
        getattr(cv2, "EVENT_MOUSEWHEEL", 10),
        getattr(cv2, "EVENT_MOUSEHWHEEL", 11),
    }
    if event in wheel_events:
        delta = mouse_wheel_queue_delta(flags)
        if delta:
            return ("scroll", delta)
        return None

    if event != getattr(cv2, "EVENT_LBUTTONDOWN", 1):
        return None

    frame_w = int(layout.get("frame_w", 0) or 0)
    panel_w = int(layout.get("panel_w", 0) or 0)
    if not (frame_w <= x < frame_w + panel_w):
        return None
    rows = layout.get("queue_rows") or ()
    index = queue_row_at(rows, x - frame_w, y)
    if index is None:
        return None
    return ("select", index)


def update_auto_capture_state(
    state: Dict,
    face: Optional[Mapping[str, object]],
    ok: bool,
    *,
    now: float,
    required_frames: int,
    cooldown_s: float,
) -> bool:
    """Return true when a valid face has remained stable long enough to save."""

    if face is None or not ok:
        state["auto_candidate"] = None
        state["auto_streak"] = 0
        return False

    key = face_key(face)
    if key in (state.get("captured_faces") or set()):
        state["auto_candidate"] = key
        state["auto_streak"] = 0
        return False

    if state.get("auto_candidate") == key:
        state["auto_streak"] = int(state.get("auto_streak", 0)) + 1
    else:
        state["auto_candidate"] = key
        state["auto_streak"] = 1

    last_save = float(state.get("last_auto_save_t", 0.0))
    return (
        int(state["auto_streak"]) >= max(1, int(required_frames))
        and now - last_save >= max(0.0, float(cooldown_s))
    )


def select_queued_faces(
    requested_faces: Sequence[str],
    registered_faces: Sequence[Mapping[str, object]],
) -> List[Dict]:
    """Resolve repeated full-face queue selections against registered faces."""

    requested = [str(face).strip() for face in requested_faces if str(face).strip()]
    if not requested:
        return []

    by_name = {
        str(face.get("object", "")).strip(): dict(face)
        for face in registered_faces
        if str(face.get("object", "")).strip()
    }
    selected: List[Dict] = []
    missing: List[str] = []
    for name in dict.fromkeys(requested):
        face = by_name.get(name)
        if face is None:
            missing.append(name)
        else:
            selected.append(face)
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise CaptureFaceError(
            f"No registered face found for queued selection {names}."
        )
    return selected


def draw_capture_overlay(
    frame: np.ndarray,
    *,
    face: Optional[Dict],
    state: Dict,
    ok: bool,
    min_expected: int,
) -> np.ndarray:
    """Draw a compact capture HUD over the camera frame."""

    vis = frame.copy()
    h, w = vis.shape[:2]
    det_ids = sorted(list(state.get("detected_ids", [])))
    exp_ids = sorted(list(set(face["tag_ids"])) if face else [])
    overlap = sorted(list(set(det_ids).intersection(exp_ids))) if face else []
    face_label = _face_label(face, max_chars=44, default="unresolved")
    status = "READY" if ok else "TAGS MISSING"
    status_colour = UI_GREEN if ok else UI_AMBER
    auto_text = "auto" if state.get("auto_side") else "manual"
    faces = state.get("faces") or []
    face_idx = int(state.get("face_idx", 0)) + 1 if faces else 0
    face_count = len(faces)

    card_w = min(w - 20, 640)
    card_h = 92
    _fill_alpha(vis, (10, 10), (10 + card_w, 10 + card_h), UI_DARK, 0.76)
    cv2.rectangle(vis, (10, 10), (10 + card_w, 10 + card_h), UI_DARK_2, 1)
    cv2.rectangle(vis, (22, 23), (34, 35), status_colour, -1)
    _put(
        vis,
        f"{status}  {face_label}",
        (42, 36),
        scale=0.62,
        colour=UI_WHITE,
        thickness=2,
    )
    _put(
        vis,
        f"mode: {auto_text}    face: {face_idx}/{face_count}    min tags: {min_expected}",
        (24, 61),
        scale=0.48,
        colour=(223, 232, 238),
    )
    _put(
        vis,
        f"seen: {_format_ids(det_ids)}    expected seen: {_format_ids(overlap)}",
        (24, 83),
        scale=0.48,
        colour=(223, 232, 238) if ok else (108, 219, 255),
        thickness=1 if ok else 2,
    )

    footer = "Enter save   Click/scroll queue   Arrows queue   h help   q/Esc quit"
    footer_h = 36
    y0 = max(0, h - footer_h - 10)
    _fill_alpha(vis, (10, y0), (min(w - 10, 780), y0 + footer_h), UI_DARK, 0.70)
    _put(vis, footer, (24, y0 + 24), scale=0.48, colour=UI_WHITE)

    if state.get("save_warn"):
        warn = "Tags are missing. Press Enter again within 3 seconds to force save."
        (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x0 = max(10, (w - tw) // 2 - 12)
        y1 = y0 - 12
        _fill_alpha(vis, (x0, y1 - 30), (min(w - 10, x0 + tw + 24), y1 + 4), UI_AMBER, 0.90)
        _put(vis, warn, (x0 + 12, y1 - 8), scale=0.55, colour=UI_DARK, thickness=2)

    return vis


def _format_ids(ids: List[int]) -> str:
    if not ids:
        return "-"
    if len(ids) <= 8:
        return ",".join(str(tag_id) for tag_id in ids)
    return ",".join(str(tag_id) for tag_id in ids[:8]) + f" +{len(ids) - 8}"


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(4, (max_chars - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


# Select the best face based on overlap of expected vs detected tag IDs
def best_face_by_overlap(faces: List[Dict], det_ids: Set[int], prev_idx: int | None = None) -> Tuple[Optional[int], int]:
    """Return (best_index, best_overlap_count). Tie-broken by prev_idx and tag count."""
    if not faces:
        return None, 0
    best_i, best_k, best_total = None, -1, -1
    for i, f in enumerate(faces):
        exp = set(f["tag_ids"])
        k = len(exp.intersection(det_ids))
        tot = len(exp)
        key = (k, tot, 1 if (prev_idx is not None and i == prev_idx) else 0)
        if key > (best_k, best_total, -1):
            best_i, best_k, best_total = i, k, tot
    return best_i, best_k


def main(argv=None):
    # Parse command-line arguments for capture settings
    ap = argparse.ArgumentParser(
        prog="posetag-capture-face",
        description="Capture wide shots per face for later annotation",
    )
    ap.add_argument("--project_root", type=Path, default=None,
                    help="Root for boards/shots/objects/datasets (default: resolver/env/config).")
    ap.add_argument("--calib", default="calib_color.yaml",
                    help="Camera intrinsics YAML (fx, fy, cx, cy, dist).")
    ap.add_argument("--registry", default=None,
                    help="Path to boards/tag_registry.yaml (default: <project_root>/boards/tag_registry.yaml).")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save shots (default: <project_root>/shots).")
    ap.add_argument("--object_name", default=None,
                    help="Either full (e.g. connection_plate_white_sideA) or base (e.g. connection_plate_white).")
    ap.add_argument("--queue_face", action="append", default=[],
                    help="full registered face to include in the capture queue; may be repeated")
    ap.add_argument("--capture_all", action="store_true",
                    help="queue all registered faces from the tag registry")
    ap.add_argument("--auto_capture", action="store_true",
                    help="auto-save when the selected face's expected tags are stable")
    ap.add_argument("--auto_capture_frames", type=int, default=8,
                    help="valid consecutive frames required before auto-save")
    ap.add_argument("--auto_capture_cooldown", type=float, default=1.0,
                    help="minimum seconds between auto-saves")
    ap.add_argument("--exit_when_complete", action="store_true",
                    help="exit cleanly after the queued faces have been captured")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min_expected", type=int, default=1, help="min expected tags to see from the chosen face")
    ap.add_argument("--gallery", action="store_true", default=True, help="show thumbnails in panel")
    ap.add_argument("--panel_w", type=int, default=420, help="info panel width (right side)")
    ap.add_argument("--recent_w", type=int, default=320, help="right panel width for last-saved preview")
    ap.add_argument("--layout", choices=["flat", "by_object", "by_object_side", "split_type"],
                    default="by_object_side", help="folder layout for saved captures")
    ap.add_argument("--raw_dir", default=None, help="when layout=split_type: raw images dir")
    ap.add_argument("--ann_dir", default=None, help="when layout=split_type: annotated images dir")
    ap.add_argument("--meta_dir", default=None, help="when layout=split_type: JSON metadata dir")
    ap.add_argument("--manifest", default=None, help="optional CSV file to append per-capture rows")
    ap.add_argument("--log_file", default=None,
                    help="override log path (default: <project_root>/logs/capture_face.log)")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="log verbosity")
    ap.add_argument("--log_console", action="store_true", help="also echo logs to stderr")

    # unified source options
    ap.add_argument("--source", choices=["realsense", "opencv", "video"], default="realsense",
                    help="Capture source: Intel RealSense (default), OpenCV webcam, or a video file")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index when --source=opencv")
    ap.add_argument("--video", type=str, default=None, help="Video path when --source=video")
    args = ap.parse_args(argv)
    if args.auto_capture_frames <= 0:
        raise SystemExit("--auto_capture_frames must be positive")
    if args.auto_capture_cooldown < 0:
        raise SystemExit("--auto_capture_cooldown must be zero or greater")
    if args.queue_face and (args.object_name or args.capture_all):
        raise SystemExit("--queue_face cannot be combined with --object_name or --capture_all")

    # ---------- validate project inputs before opening hardware or writing outputs ----------
    try:
        validate_capture_source_args(args.source, args.video, capture_source.rs)
        pr = preview_capture_project_root(args.project_root)
        calibration = load_capture_calibration(resolve_capture_calibration_path(pr, args.calib))
        paths = prepare_capture_paths(
            project_root=pr,
            calib_path=calibration.path,
            registry=args.registry,
            out_dir=args.out_dir,
            manifest=args.manifest,
            log_file=args.log_file,
            layout=args.layout,
            raw_dir=args.raw_dir,
            ann_dir=args.ann_dir,
            meta_dir=args.meta_dir,
        )
        registry = load_capture_registry(paths.registry_path)
        registered_faces = load_registered_faces(
            registry,
            project_root=paths.project_root,
            registry_path=paths.registry_path,
        )
        queued_faces = select_queued_faces(args.queue_face, registered_faces)
        initial_selection = (
            None
            if queued_faces
            else select_initial_faces(args.object_name, registered_faces)
        )
        Detector = load_apriltag_detector_class()
        det = create_apriltag_detector(args.family, detector_class=Detector)
    except CaptureFaceError as exc:
        raise SystemExit(str(exc)) from exc

    # unified source; keep this before output-directory creation so unreadable
    # videos fail without leaving capture artifacts.
    read_frame, stop, _ = capture_source.open_source(
        args.source, cam=args.cam, video=args.video, width=args.width, height=args.height, fps=args.fps, warmup=10
    )

    try:
        activate_capture_project_root(paths.project_root)
        create_capture_output_dirs(paths, layout=args.layout)
        logger = init_project_logger(paths.log_path, level=args.log_level, console=args.log_console)
    except Exception:
        stop()
        raise

    # ---------- logging & announce ----------
    logger.info("Using project root: %s", paths.project_root)
    logger.info("Using calib: %s", str(paths.calib_path))
    logger.info("Using registry: %s", str(paths.registry_path))
    logger.info("Shots will be saved to: %s", str(paths.out_dir))

    # ---------- UI state ----------
    captured_faces = load_existing_captured_faces(paths.manifest_path)
    existing_thumbs = load_recent_saved_images(
        paths.manifest_path,
        project_root=paths.project_root,
    )
    state = {
        "object_base": None, "faces": [], "face_idx": 0, "auto_side": False,
        "detected_ids": set(), "validation_ok": True, "thumbs": existing_thumbs,
        "save_warn": False, "save_warn_t0": 0.0,
        "captured_faces": captured_faces, "auto_candidate": None,
        "auto_streak": 0, "last_auto_save_t": 0.0,
        "last_saved_ann": existing_thumbs[-1] if existing_thumbs else None,
    }

    if queued_faces:
        state["faces"] = list(queued_faces)
        first_missing = first_uncaptured_index(state["faces"], captured_faces)
        select_face_index(state, first_missing if first_missing is not None else 0)
    elif args.capture_all or initial_selection is None:
        state["faces"] = list(registered_faces)
        first_missing = first_uncaptured_index(state["faces"], captured_faces)
        select_face_index(state, first_missing if first_missing is not None else 0)
    elif initial_selection:
        state["object_base"] = initial_selection.object_base
        state["faces"] = list(initial_selection.faces)
        state["auto_side"] = initial_selection.auto_side
        first_missing = first_uncaptured_index(state["faces"], captured_faces)
        select_face_index(
            state,
            first_missing if first_missing is not None else 0,
            auto_side=initial_selection.auto_side,
        )

    mode = "preview" if state["faces"] and (queued_faces or args.capture_all or initial_selection) else "pick_face"
    pick_sel = int(state.get("face_idx", 0))
    gallery_on, show_help = bool(args.gallery), False
    WINDOW_NAME = "Capture Face"
    mouse_state: Dict[str, object] = {"layout": {}, "action": None}

    def on_mouse(event, x, y, flags, param) -> None:
        action = capture_face_mouse_action(
            event,
            int(x),
            int(y),
            int(flags),
            mouse_state.get("layout") or {},
        )
        if action is not None:
            mouse_state["action"] = action

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width + args.panel_w + args.recent_w, args.height)
    try:
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    except Exception:
        # Headless tests patch window creation; real HighGUI windows support this.
        pass

    def save_current_face(raw_frame, annotated_frame, face_entry, validation_ok: bool) -> None:
        ts = capture_timestamp()
        obj_full = face_entry["object"]
        shot_paths = build_shot_paths(
            layout=args.layout,
            out_dir=paths.out_dir,
            object_full=obj_full,
            timestamp=ts,
            raw_dir=paths.raw_dir,
            ann_dir=paths.ann_dir,
            meta_dir=paths.meta_dir,
        )
        meta = build_capture_metadata(
            face=face_entry,
            shot_paths=shot_paths,
            detected_tag_ids=state.get("detected_ids", set()),
            validation_ok=validation_ok,
            auto_face=state["auto_side"],
            frame_shape=raw_frame.shape,
            camera_params=calibration.camera_params,
            timestamp=ts,
        )
        try:
            write_capture_outputs(
                raw_frame=raw_frame,
                annotated_frame=annotated_frame,
                metadata=meta,
                shot_paths=shot_paths,
                manifest_path=paths.manifest_path,
                image_writer=cv2.imwrite,
            )
        except CaptureFaceError as exc:
            raise SystemExit(str(exc)) from exc
        state["last_saved_ann"] = annotated_frame.copy()
        state["captured_faces"].add(face_key(face_entry))

        logger.info(f"[+] Saved {shot_paths.raw_path}")
        logger.info(f"[+] Saved {shot_paths.ann_path}")
        logger.info(f"[+] Saved {shot_paths.meta_path}")

        try:
            tw = min(320, annotated_frame.shape[1])
            th = int(annotated_frame.shape[0] * tw / annotated_frame.shape[1])
            state["thumbs"].append(cv2.resize(annotated_frame, (tw, th)))
            if len(state["thumbs"]) > 12:
                state["thumbs"] = state["thumbs"][-12:]
        except Exception:
            pass

    def advance_after_capture() -> bool:
        faces = state.get("faces") or []
        if not faces:
            return False
        next_idx = first_uncaptured_index(
            faces,
            state.get("captured_faces", set()),
            start=int(state.get("face_idx", 0)) + 1,
        )
        if next_idx is None:
            return False
        return select_face_index(state, next_idx)

    try:
        first = True
        # Main capture loop: read frames, detect tags, update UI, handle key presses
        while True:
            c = read_frame()
            if c is None:
                if args.source == "video":
                    logger.info("Video ended before another face shot was saved; exiting cleanly.")
                    return 0
                continue

            # Sync actual width/height for non-hardware-timestamped sources
            if first and args.source != "realsense":
                args.height, args.width = int(c.shape[0]), int(c.shape[1])
                first = False

            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            dd = det.detect(g, estimate_tag_pose=False)

            det_ids = {int(d.tag_id) for d in dd}
            state["detected_ids"] = det_ids

            detection_vis = draw_detections(c, dd)
            face = None

            # Resolve face from detections + selection
            if state["faces"]:
                if state["auto_side"]:
                    best_i, _ = best_face_by_overlap(state["faces"], det_ids, state.get("face_idx"))
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

            vis = draw_capture_overlay(
                detection_vis,
                face=face,
                state=state,
                ok=ok,
                min_expected=args.min_expected,
            )

            # Panels
            info_layout: Dict[str, object] = {}
            if mode == 'pick_face':
                panel_info = face_queue_panel(
                    args.height,
                    args.panel_w,
                    state.get("faces", []),
                    pick_sel,
                    captured_faces=state.get("captured_faces", set()),
                    interaction=info_layout,
                )
            else:
                panel_info = make_info_panel(
                    args.height,
                    args.panel_w,
                    state,
                    gallery_on=gallery_on,
                    interaction=info_layout,
                )
                if show_help:
                    overlay = np.full((args.height, args.panel_w, 3), 230, np.uint8)
                    text_lines(overlay, [
                        "Help:",
                        "ENTER: Save (twice within 3s to force if not OK)",
                        "Click queue rows or scroll: Select face",
                        "Up/Down or Left/Right: Move through face queue",
                        "a: Toggle auto face/side (when base given)",
                        "o: Open face queue",
                        "g: Toggle gallery thumbnails",
                        "h: Toggle this help",
                        "q/ESC: Quit",
                    ], y0=40)
                    panel_info = cv2.addWeighted(panel_info, 0.1, overlay, 0.9, 0)
                    info_layout = {}

            panel_recent = make_recent_panel(
                args.height,
                args.recent_w,
                state.get("last_saved_ann"),
                gallery_on,
                thumbs=state.get("thumbs", []),
            )
            combo = cv2.hconcat([vis, panel_info, panel_recent])
            mouse_state["layout"] = {
                "frame_w": int(vis.shape[1]),
                "panel_w": int(panel_info.shape[1]),
                "queue_rows": tuple(info_layout.get("queue_rows", ())),
            }
            cv2.imshow(WINDOW_NAME, combo)

            if args.auto_capture and mode == "preview" and update_auto_capture_state(
                state,
                face,
                ok,
                now=time.time(),
                required_frames=args.auto_capture_frames,
                cooldown_s=args.auto_capture_cooldown,
            ):
                state["save_warn"] = False
                save_current_face(c, vis, face, ok)
                state["last_auto_save_t"] = time.time()
                state["auto_streak"] = 0
                advanced = advance_after_capture()
                if not advanced and args.exit_when_complete:
                    logger.info("Queued face-shot capture is complete; exiting cleanly.")
                    return 0

            k = cv2.waitKeyEx(1) or 0
            mouse_action = mouse_state.get("action")
            mouse_state["action"] = None
            ENTER_KEYS, CANCEL_KEYS = {13, 10}, {27}
            UP_KEYS = KEY_UP_CODES | {ord('w'), ord('k')}
            DOWN_KEYS = KEY_DOWN_CODES | {ord('s'), ord('j')}
            LEFT_KEYS = KEY_LEFT_CODES | {ord('p'), ord('[')}
            RIGHT_KEYS = KEY_RIGHT_CODES | {ord('f'), ord('n'), ord(']')}

            # Face queue picker
            if mode == 'pick_face':
                faces = state.get("faces") or []
                if mouse_action:
                    action, value = mouse_action
                    if action == "scroll":
                        pick_sel = (pick_sel + int(value)) % max(1, len(faces))
                    elif action == "select" and faces:
                        pick_sel = max(0, min(int(value), len(faces) - 1))
                        select_face_index(state, pick_sel)
                        mode = 'preview'
                    continue
                if k in UP_KEYS:   pick_sel = (pick_sel - 1) % max(1, len(faces))
                elif k in DOWN_KEYS: pick_sel = (pick_sel + 1) % max(1, len(faces))
                elif ord('1') <= k <= ord('9'):
                    idx = (k - ord('1'));  pick_sel = idx if idx < len(faces) else pick_sel
                elif k in ENTER_KEYS:
                    if faces:
                        select_face_index(state, pick_sel)
                    mode = 'preview'
                elif k in CANCEL_KEYS:
                    if not state.get("faces"):
                        break
                    mode = 'preview'
                elif k == ord('q'):     break
                continue

            # Preview
            faces = state.get("faces") or []
            if mouse_action:
                action, value = mouse_action
                if action == "select" and faces:
                    select_face_index(state, int(value))
                elif action == "scroll":
                    cycle_face_selection(state, int(value))
                continue
            if k in (ord('q'),) or k in CANCEL_KEYS: break
            if k == ord('h'): show_help = not show_help
            elif k == ord('g'): gallery_on = not gallery_on
            elif k == ord('o'):
                pick_sel = int(state.get("face_idx", 0))
                mode = 'pick_face'
            elif k == ord('a'):
                state["auto_side"] = not state["auto_side"] if state["faces"] else False
            elif k in RIGHT_KEYS or k in DOWN_KEYS:
                cycle_face_selection(state, 1)
            elif k in LEFT_KEYS or k in UP_KEYS:
                cycle_face_selection(state, -1)
            elif k in ENTER_KEYS:
                if face is None:
                    continue
                if not ok and set(face["tag_ids"]):
                    now = time.time()
                    if not state["save_warn"] or (now - state["save_warn_t0"] > 3.0):
                        state["save_warn"], state["save_warn_t0"] = True, now
                        continue
                state["save_warn"] = False
                save_current_face(c, vis, face, ok)
                advanced = advance_after_capture()
                if not advanced and args.exit_when_complete:
                    logger.info("Queued face-shot capture is complete; exiting cleanly.")
                    return 0

    finally:
        stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
