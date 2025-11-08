"""
Colour ChArUco camera calibration (OpenCV webcam, Intel RealSense, or video file).

What it does
------------
- Detects ArUco/ChArUco corners from a live camera or video.
- Lets you collect multiple views (SPACE), then solves intrinsics/distortion (ENTER).
- Writes a `calib_color.yaml` with explicit fields (fx, fy, cx, cy, k1..k3, image size, RMS).
- If `--project_root` is provided, initializes a project tree and defaults output to
  `<root>/calib/calib_color.yaml` (unless `--out` is set).

Typical usage
-------------
# Generic webcam (default source)
python charuco_calibrate.py --source opencv --cam 0 --squares-x 5 --squares-y 3 \
  --square-length-mm 50 --marker-length-mm 37 --dict 7X7_100

# RealSense
python charuco_calibrate.py --source realsense --width 640 --height 480 --fps 30

# From a video
python charuco_calibrate.py --source video --video sample.mp4
"""

import argparse, yaml
import sys
import numpy as np
import cv2
import os, datetime
from pathlib import Path

# RealSense is optional; general webcams/video should work without it
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# Project bootstrap helpers (optional)
try:
    from utils.project_config import resolve_project_root, ensure_project_dirs
except Exception:
    resolve_project_root = None
    ensure_project_dirs = None


class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Show defaults and preserve newlines in help/epilog."""
    pass


def parse_args():
    ap = argparse.ArgumentParser(
        "Colour ChArUco calibration (RealSense, generic webcam, or video)",
        formatter_class=_HelpFmt,
        epilog=(
            "Controls:\n"
            "  SPACE  add sample    ENTER  solve    q  quit\n\n"
            "Tips:\n"
            "  - Print the board at 100% and pass correct --square-length-mm / --marker-length-mm.\n"
            "  - For OpenCV webcams, requested --width/--height are best-effort; the YAML records actual size."
        ),
    )
    ap.add_argument("--squares-x", type=int, default=3)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--square-length-mm", type=float, default=50.0,
                    help="Size of one ChArUco square (mm).")
    ap.add_argument("--marker-length-mm", type=float, default=37.0,
                    help="Size of the ArUco marker inside each square (mm).")
    ap.add_argument("--dict", type=str, default="7X7_50",
                    help="ArUco dictionary (e.g., 4X4_50, 5X5_250, 6X6_1000, 7X7_1000, APRILTAG_36H11).")
    ap.add_argument("--width", type=int, default=640,
                    help="Requested capture width (best effort for OpenCV webcams).")
    ap.add_argument("--height", type=int, default=480,
                    help="Requested capture height (best effort for OpenCV webcams).")
    ap.add_argument("--fps", type=int, default=30,
                    help="Requested frames per second (best effort for OpenCV webcams).")
    ap.add_argument("--min-corners", type=int, default=1, help="min ChArUco corners per sample")
    ap.add_argument("--min-samples", type=int, default=30, help="min accepted samples before solve")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--auto-interval", type=int, default=10)
    ap.add_argument("--source", choices=["opencv", "realsense", "video"], default="opencv",
                    help="Camera source: generic OpenCV webcam (default), Intel RealSense, or a video file")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index when --source=opencv")
    ap.add_argument("--video", type=str, default=None, help="Video path when --source=video")
    ap.add_argument("--project_root", type=Path, default=None,
                    help="If set, initialize project layout and (when --out is not given) save to <project_root>/calib/calib_color.yaml")
    ap.add_argument("--out", type=str, default=None,
                    help="Output YAML file (default: <root>/calib/calib_color.yaml if --project_root is set; otherwise calib_color.yaml in CWD)")
    return ap.parse_args()


def get_dictionary(name: str):
    """Return an OpenCV ArUco dictionary object from a human-friendly name."""
    name = name.upper().replace("-", "_")
    aruco = cv2.aruco
    MAP = {
        "4X4_50": aruco.DICT_4X4_50, "4X4_100": aruco.DICT_4X4_100,
        "4X4_250": aruco.DICT_4X4_250, "4X4_1000": getattr(aruco, "DICT_4X4_1000", None),
        "5X5_50": aruco.DICT_5X5_50, "5X5_100": aruco.DICT_5X5_100,
        "5X5_250": aruco.DICT_5X5_250, "5X5_1000": aruco.DICT_5X5_1000,
        "6X6_50": aruco.DICT_6X6_50, "6X6_100": aruco.DICT_6X6_100,
        "6X6_250": aruco.DICT_6X6_250, "6X6_1000": aruco.DICT_6X6_1000,
        "7X7_50": aruco.DICT_7X7_50, "7X7_100": aruco.DICT_7X7_100,
        "7X7_250": aruco.DICT_7X7_250, "7X7_1000": aruco.DICT_7X7_1000,
        "APRILTAG_36H11": getattr(aruco, "DICT_APRILTAG_36h11", None),
    }
    if name not in MAP or MAP[name] is None:
        raise ValueError(f"Unsupported dictionary {name}")
    return aruco.getPredefinedDictionary(MAP[name])


def make_board(aruco, sx, sy, square_m, marker_m, dictionary):
    """Create a ChArUco board using the OpenCV API available in the installed version."""
    if hasattr(aruco, "CharucoBoard") and callable(getattr(aruco, "CharucoBoard")):
        return aruco.CharucoBoard((sx, sy), square_m, marker_m, dictionary)
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(sx, sy, square_m, marker_m, dictionary)
    raise RuntimeError("Your OpenCV build lacks both CharucoBoard APIs.")


def calibrate_from_charuco(samples_corners, samples_ids, board, image_size):
    """Run calibration using Charuco corners; falls back to classic calibrateCamera if needed."""
    aruco = cv2.aruco
    if hasattr(aruco, "calibrateCameraCharuco"):
        return aruco.calibrateCameraCharuco(
            charucoCorners=samples_corners,
            charucoIds=samples_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None, distCoeffs=None
        )
    if hasattr(aruco, "calibrateCameraCharucoExtended"):
        out = aruco.calibrateCameraCharucoExtended(
            charucoCorners=samples_corners,
            charucoIds=samples_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None, distCoeffs=None
        )
        return out[:5]

    # Fallback: assemble object/image points per view then calibrate
    if hasattr(board, "getChessboardCorners"):
        all_corners3d = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        all_corners3d = board.chessboardCorners
    else:
        raise RuntimeError("ChArUco board missing expected corner accessors")

    all_corners3d = np.asarray(all_corners3d, dtype=np.float32)
    N_all = all_corners3d.shape[0]

    objpoints, imgpoints = [], []
    for ch_c, ch_id in zip(samples_corners, samples_ids):
        if ch_c is None or ch_id is None:
            continue
        img = ch_c.reshape(-1, 2).astype(np.float32)
        ids = ch_id.reshape(-1).astype(np.int32)
        mask = (ids >= 0) & (ids < N_all)
        if not np.any(mask):
            continue
        ids = ids[mask]
        img = img[mask]
        if len(ids) < 6:
            continue
        obj = all_corners3d[ids, :]
        objpoints.append(obj)
        imgpoints.append(img)

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid views for fallback calibration (got {len(objpoints)}, need ≥10).")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=image_size,
        cameraMatrix=None, distCoeffs=None
    )
    return rms, K, dist, rvecs, tvecs


def main():
    """CLI entry point: capture frames, collect ChArUco corners, calibrate, and write YAML."""
    args = parse_args()

    # Project-aware default output path
    if getattr(args, "project_root", None) is not None:
        if resolve_project_root is None or ensure_project_dirs is None:
            print("Warning: --project_root provided but project helpers unavailable; proceeding without init.",
                  file=sys.stderr)
            pr = Path(args.project_root)
        else:
            pr = Path(resolve_project_root(args.project_root))
            ensure_project_dirs(str(pr))
        if args.out is None:
            out_path = pr / "calib" / "calib_color.yaml"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            args.out = str(out_path)
    else:
        if args.out is None:
            args.out = "calib_color.yaml"

    aruco = cv2.aruco
    dictionary = get_dictionary(args.dict)

    # Prepare calibration image dump folder
    calibration_folder = "calib_images"
    os.makedirs(calibration_folder, exist_ok=True)
    calib_folders = sorted(
        [fd for fd in os.listdir(calibration_folder) if os.path.isdir(os.path.join(calibration_folder, fd))]
    )
    last_sub_calib_folder = calib_folders[-1] if len(calib_folders) > 0 else None
    if last_sub_calib_folder is None:
        new_sub_calib_folder = os.path.join(calibration_folder, "set_01")
    else:
        last_index = int(last_sub_calib_folder.split("_")[-1])
        new_index = last_index + 1
        new_sub_calib_folder = os.path.join(calibration_folder, f"set_{new_index:02d}")
    os.makedirs(new_sub_calib_folder, exist_ok=True)
    print(f"[i] Created new calibration image folder: {new_sub_calib_folder}")

    # Build ChArUco board
    square_m = args.square_length_mm / 1000.0
    marker_m = args.marker_length_mm / 1000.0
    board = make_board(aruco, args.squares_x, args.squares_y, square_m, marker_m, dictionary)

    # Detector setup
    has_charuco_detector = hasattr(aruco, "CharucoDetector")
    if has_charuco_detector:
        chdet = aruco.CharucoDetector(board)
    else:
        det_params = aruco.DetectorParameters_create()

    # --- Camera/video source setup ---
    if args.source == "realsense":
        if rs is None:
            sys.exit("pyrealsense2 not available; use --source opencv|video")
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        profile = pipe.start(cfg)

        def _read_frame():
            frames = pipe.wait_for_frames()
            cframe = frames.get_color_frame()
            return None if not cframe else np.asanyarray(cframe.get_data())

        def _stop():
            pipe.stop()

    elif args.source == "opencv":
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            sys.exit(f"Could not open camera index {args.cam}")
        # Best effort; actual size may differ
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        def _read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        def _stop():
            cap.release()

    elif args.source == "video":
        if not args.video:
            sys.exit("--video path is required when --source=video")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Could not open video: {args.video}")

        def _read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        def _stop():
            cap.release()

    samples_corners, samples_ids = [], []
    auto_tick = 0

    print("[i] Move/tilt the board; SPACE=add, ENTER=solve, q=quit")
    try:
        first = True
        while True:
            color = _read_frame()
            if color is None:
                continue

            # For non-RealSense, sync image_size to actual stream
            if first and args.source != "realsense":
                args.height, args.width = int(color.shape[0]), int(color.shape[1])
                first = False

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            if has_charuco_detector:
                out = chdet.detectBoard(gray)
                if isinstance(out, tuple) and len(out) >= 2:
                    ch_corners, ch_ids = out[0], out[1]
                else:
                    ch_corners, ch_ids = None, None
                corners, ids = None, None
            else:
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=det_params)
                if ids is not None and len(ids) > 0:
                    aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None, parameters=det_params)
                _, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners, markerIds=ids, image=gray, board=board
                )

            vis = color.copy()
            if not has_charuco_detector and ids is not None and len(ids) > 0:
                vis = aruco.drawDetectedMarkers(vis, corners, ids)
            if ch_corners is not None and ch_ids is not None and len(ch_corners) > 0:
                aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

            good = ch_corners is not None and ch_ids is not None and len(ch_corners) >= args.min_corners
            cv2.putText(vis, f"samples={len(samples_corners)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(vis, "BOARD: OK" if good else "BOARD: not ready", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0) if good else (0, 0, 255), 2)
            cv2.imshow("ChArUco Calibration", vis)
            k = cv2.waitKey(1) & 0xFF

            if args.auto and good:
                auto_tick += 1
                if auto_tick % args.auto_interval == 0:
                    samples_corners.append(ch_corners.copy())
                    samples_ids.append(ch_ids.copy())
                    print(f"[+] auto sample {len(samples_corners)} ({len(ch_corners)} corners)")
            elif k == ord(' '):
                if good:
                    samples_corners.append(ch_corners.copy())
                    samples_ids.append(ch_ids.copy())
                    print(f"[+] sample {len(samples_corners)} ({len(ch_corners)} corners)")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = os.path.join(new_sub_calib_folder, f"calib_{ts}.png")
                    cv2.imwrite(img_filename, color)
                    print(f"[i] Saved calibration image: {img_filename}")
                else:
                    print("[!] not enough corners; get closer / reduce glare / tilt more.")
            elif k == 13:
                break
            elif k == ord('q'):
                raise KeyboardInterrupt
    finally:
        _stop()
        cv2.destroyAllWindows()

    need = max(10, args.min_samples)
    if len(samples_corners) < need:
        raise SystemExit(f"Need at least {need} good samples; got {len(samples_corners)}.")

    img_size = (args.width, args.height)
    ret, K, dist, rvecs, tvecs = calibrate_from_charuco(
        samples_corners, samples_ids, board, img_size
    )

    # ---- write YAML with explicit keys ----
    dist = np.asarray(dist, dtype=float).reshape(1, -1)  # ensure shape (1, N)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    k1 = float(dist[0, 0]) if dist.shape[1] > 0 else 0.0
    k2 = float(dist[0, 1]) if dist.shape[1] > 1 else 0.0
    p1 = float(dist[0, 2]) if dist.shape[1] > 2 else 0.0
    p2 = float(dist[0, 3]) if dist.shape[1] > 3 else 0.0
    k3 = float(dist[0, 4]) if dist.shape[1] > 4 else 0.0

    print(f"[i] Reprojection RMS = {ret:.3f} px")
    print("[i] K =\n", K)
    print("[i] dist =", dist.ravel())

    data = dict(
        image_width=args.width,
        image_height=args.height,
        camera_matrix=dict(
            fx=fx, fy=fy, cx=cx, cy=cy,
            data=K.tolist()
        ),
        distortion_coefficients=dict(
            k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
            data=dist.tolist()
        ),
        reproj_rms=float(ret),
        model="plumb_bob",
        notes=f"ChArUco {args.squares_x}x{args.squares_y}, "
              f"square={args.square_length_mm}mm, marker={args.marker_length_mm}mm, dict={args.dict}"
    )
    with open(args.out, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[i] wrote {args.out}")


if __name__ == "__main__":
    main()
