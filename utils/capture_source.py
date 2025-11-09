#!/usr/bin/env python3
"""
capture_source.py — unified camera/video opener for RealSense, OpenCV webcams, and videos.

API
---
open_source(source, *, cam=0, video=None, width=640, height=480, fps=30, warmup=0)
    -> (read_frame: () -> np.ndarray|None, stop: () -> None, info: dict)

- `read_frame()` returns a BGR image or None if a frame isn’t available.
- `stop()` releases/tears down the source.
- `info` gives 'kind' and the requested capture parameters; callers may still
  want to re-sync to *actual* frame size after the first frame (OpenCV cams).
"""
from __future__ import annotations
from typing import Callable, Tuple, Optional, Dict
import cv2, numpy as np

try:
    import pyrealsense2 as rs  # optional
except Exception:
    rs = None


def open_source(
    source: str,
    *,
    cam: int = 0,
    video: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup: int = 0,
) -> Tuple[Callable[[], Optional[np.ndarray]], Callable[[], None], Dict]:
    """
    Open a capture source and return (read_frame, stop, info).

    Parameters
    ----------
    source : {"realsense","opencv","video"}
        Capture backend to use.
    cam : int
        OpenCV camera index when `source="opencv"`.
    video : str | None
        Path to a video file when `source="video"`.
    width, height, fps : int
        Requested stream parameters. (OpenCV webcams: best-effort.)
    warmup : int
        For RealSense: number of frames to discard after start.

    Returns
    -------
    read_frame : () -> np.ndarray | None
        Function returning a BGR frame or None on failure.
    stop : () -> None
        Function that releases/tears down the capture.
    info : dict
        {'kind': 'realsense'|'opencv'|'video', 'req': {'width', 'height', 'fps'}}
    """
    info = {"kind": source, "req": {"width": int(width), "height": int(height), "fps": int(fps)}}

    if source == "realsense":
        if rs is None:
            raise SystemExit("pyrealsense2 not available; use --source opencv|video")
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        pipe.start(cfg)
        for _ in range(max(0, int(warmup))):
            _ = pipe.wait_for_frames()

        def _read():
            fs = pipe.wait_for_frames()
            cf = fs.get_color_frame()
            return None if not cf else np.asanyarray(cf.get_data())

        def _stop():
            try:
                pipe.stop()
            except Exception:
                pass

        return _read, _stop, info

    if source == "opencv":
        cap = cv2.VideoCapture(int(cam))
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {cam}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))

        def _read():
            ok, frame = cap.read()
            return frame if ok else None

        def _stop():
            cap.release()

        return _read, _stop, info

    # video
    if not video:
        raise SystemExit("--video path is required when --source=video")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    def _read():
        ok, frame = cap.read()
        return frame if ok else None

    def _stop():
        cap.release()

    return _read, _stop, info
