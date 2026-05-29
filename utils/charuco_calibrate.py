#!/usr/bin/env python3
"""
Colour ChArUco camera calibration (OpenCV webcam, Intel RealSense, or video file).

What it does
------------
- Detects ArUco/ChArUco corners from a live camera or video.
- Lets you collect multiple views (SPACE), then solves intrinsics/distortion (ENTER).
- Saves ALL artefacts under the active project:
    <project_root>/calib/
      ├─ calib_color.yaml                # latest calibration
      ├─ images/set_XX/*.png             # captured raw samples (auto-incremented set)
      └─ runs/<UTC-ISO>/                 # per-run snapshot
           ├─ config.yaml                # args, board spec, image size, source info, samples
           └─ calib_color.yaml           # calibration for this run
- Project resolution priority: --project_root → $POSETAG_PROJECT
  (legacy $GTAT_PROJECT still accepted) → config current → last/most recent
  under ~/posetag (legacy ~/gt-6dof still discovered) → new timestamped project.

Typical usage
-------------
# Generic webcam (default source)
posetag-calib-charuco --source opencv --cam 0 --squares-x 5 --squares-y 3 \
  --square-length-mm 50 --marker-length-mm 37 --dict 7X7_100

# RealSense
posetag-calib-charuco --source realsense --width 640 --height 480 --fps 30

# From a video
posetag-calib-charuco --source video --video sample.mp4
"""

from __future__ import annotations
import argparse, yaml
import sys, os, datetime, shutil
from pathlib import Path
import numpy as np
import cv2

# RealSense is optional; general webcams/video should work without it
try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def _ensure_checkout_import_paths() -> None:
    """Make direct ``python utils/charuco_calibrate.py`` work from a checkout."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_src = repo_root / "src"
    for path in (repo_root, repo_src):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Canonical PoseTag helpers.  The capture/calibration loop remains in this
# legacy module temporarily; pure validation, IO, and YAML helpers live in
# posetag.pipelines so they can be tested without camera hardware.
try:
    from posetag.pipelines.charuco_calibration import (  # type: ignore
        CharucoCalibrationError,
        build_calibration_yaml,
        get_dictionary,
        prepare_project_io,
        validate_capture_args,
        write_calibration_yaml,
    )
    from posetag.workflows.calibration_guidance import (  # type: ignore
        analyze_calibration_guidance,
        auto_capture_decision,
        guidance_overlay_lines,
        observation_from_charuco_corners,
        parse_grid_shape,
    )
except ModuleNotFoundError:
    _ensure_checkout_import_paths()
    from posetag.pipelines.charuco_calibration import (  # type: ignore
        CharucoCalibrationError,
        build_calibration_yaml,
        get_dictionary,
        prepare_project_io,
        validate_capture_args,
        write_calibration_yaml,
    )
    from posetag.workflows.calibration_guidance import (  # type: ignore
        analyze_calibration_guidance,
        auto_capture_decision,
        guidance_overlay_lines,
        observation_from_charuco_corners,
        parse_grid_shape,
    )


class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Show defaults and preserve newlines in help/epilog."""
    pass


def build_parser():
    ap = argparse.ArgumentParser(
        "Colour ChArUco calibration (RealSense, generic webcam, or video)",
        formatter_class=_HelpFmt,
        epilog=(
            "Controls:\n"
            "  SPACE  add manual sample    ENTER  solve    q  quit\n\n"
            "Tips:\n"
            "  - Print the board at 100% and pass correct --square-length-mm / --marker-length-mm.\n"
            "  - Guided auto-capture saves samples as you cover frame grid cells.\n"
            "  - Follow the preview guidance to cover frame edges/corners and vary distance.\n"
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
    ap.add_argument("--min-corners", type=int, default=4,
                    help="min ChArUco corners per sample; values below 4 are promoted to 4")
    ap.add_argument("--min-samples", type=int, default=30, help="min accepted samples before solve")
    ap.add_argument("--guided-auto", action=argparse.BooleanOptionalAction, default=True,
                    help="Automatically save good frames that improve guided coverage.")
    ap.add_argument("--coverage-grid", type=str, default="3x3",
                    help="Frame coverage grid for guided capture, e.g. 3x3 or 4x4.")
    ap.add_argument("--samples-per-cell", type=int, default=1,
                    help="Minimum accepted samples required in each coverage grid cell.")
    ap.add_argument("--guided-auto-cooldown", type=int, default=8,
                    help="Frames to wait after a guided automatic sample.")
    ap.add_argument("--auto", action="store_true",
                    help="Legacy timed auto-sampling fallback; guided auto is preferred.")
    ap.add_argument("--auto-interval", type=int, default=10)
    ap.add_argument("--source", choices=["opencv", "realsense", "video"], default="opencv",
                    help="Camera source: generic OpenCV webcam (default), Intel RealSense, or a video file")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index when --source=opencv")
    ap.add_argument("--video", type=str, default=None, help="Video path when --source=video")
    ap.add_argument("--project_root", type=Path, default=None,
                    help="Explicit project root; otherwise resolved via config/env/home logic.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output YAML file (default: <project_root>/calib/calib_color.yaml)")
    return ap


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def make_board(aruco, sx, sy, square_m, marker_m, dictionary):
    """Create a ChArUco board using the OpenCV API available in the installed version."""
    if hasattr(aruco, "CharucoBoard") and callable(getattr(aruco, "CharucoBoard")):
        return aruco.CharucoBoard((sx, sy), square_m, marker_m, dictionary)
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(sx, sy, square_m, marker_m, dictionary)
    raise RuntimeError("Your OpenCV build lacks both CharucoBoard APIs.")


def _prepare_project_io(args) -> tuple[Path, Path, Path, Path, Path]:
    """
    Resolve project root and prepare:
      calib_dir, images_root, run_dir (unique), out_yaml.
    Returns (project_root, calib_dir, images_root, run_dir, out_yaml)
    """
    project_io = prepare_project_io(args.project_root, args.out)
    return (
        project_io.project_root,
        project_io.calib_dir,
        project_io.images_dir,
        project_io.run_dir,
        project_io.out_yaml,
    )


def _open_capture_source(args):
    """Open the requested capture source without creating project artifacts."""

    if args.source == "realsense":
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pipe.start(cfg)

        def _read_frame():
            frames = pipe.wait_for_frames()
            cframe = frames.get_color_frame()
            return None if not cframe else np.asanyarray(cframe.get_data())

        def _stop():
            pipe.stop()

        return _read_frame, _stop

    if args.source == "opencv":
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.cam}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        def _read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        def _stop():
            cap.release()

        return _read_frame, _stop

    if args.source == "video":
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")

        def _read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        def _stop():
            cap.release()

        return _read_frame, _stop

    raise SystemExit(f"Unsupported source: {args.source}")


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


def _draw_guidance_overlay(vis, guidance_state):
    """Render state-driven calibration guidance in the OpenCV preview."""

    y = 120
    for line in guidance_overlay_lines(guidance_state):
        cv2.putText(
            vis,
            line[:92],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            vis,
            line[:92],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 60, 20),
            1,
        )
        y += 26

    _draw_guidance_grid(vis, guidance_state)


def _draw_guidance_grid(vis, guidance_state):
    height, width = vis.shape[:2]
    rows, cols = guidance_state.grid_shape
    longest_side = max(rows, cols)
    cell_size = max(16, min(28, int(min(width, height) * 0.2 / longest_side)))
    grid_width = cell_size * cols
    x0 = max(20, width - grid_width - 20)
    y0 = 20
    covered = set(guidance_state.covered_cells)
    target = guidance_state.recommendation.target_cell
    current = (
        guidance_state.current_observation.cell
        if guidance_state.current_observation is not None
        else None
    )

    for cell in guidance_state.all_cells:
        row, col = _guidance_cell_row_col(cell, guidance_state.grid_shape)
        x1 = x0 + col * cell_size
        y1 = y0 + row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        if cell in covered:
            cv2.rectangle(vis, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 80, 35), -1)
        if cell == target:
            color = (0, 220, 255)
            thickness = 2
        elif cell == current:
            color = (255, 180, 0)
            thickness = 2
        elif cell in covered:
            color = (0, 170, 70)
            thickness = 1
        else:
            color = (150, 150, 150)
            thickness = 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)


def _guidance_cell_row_col(cell: str, grid_shape: tuple[int, int]) -> tuple[int, int]:
    if grid_shape == (3, 3) and cell == "center":
        return 1, 1
    if grid_shape == (3, 3) and "-" in cell:
        row_name, col_name = cell.split("-", 1)
        row = {"top": 0, "middle": 1, "bottom": 2}[row_name]
        col = {"left": 0, "center": 1, "right": 2}[col_name]
        return row, col
    if cell.startswith("r") and "c" in cell:
        row_text, col_text = cell[1:].split("c", 1)
        return int(row_text) - 1, int(col_text) - 1
    raise ValueError(f"Unsupported guidance cell: {cell}")


def _guidance_observation(ch_corners, args):
    return observation_from_charuco_corners(
        ch_corners,
        (int(args.width), int(args.height)),
        min_corners=int(args.min_corners),
        grid_shape=args.coverage_grid_shape,
    )


def _validate_guided_capture_args(args) -> None:
    try:
        args.coverage_grid_shape = parse_grid_shape(args.coverage_grid)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.samples_per_cell < 1:
        raise SystemExit("--samples-per-cell must be at least 1.")
    if args.guided_auto_cooldown < 0:
        raise SystemExit("--guided-auto-cooldown must be zero or greater.")


def _minimum_required_samples(args) -> int:
    rows, cols = args.coverage_grid_shape
    return max(10, int(args.min_samples), rows * cols * int(args.samples_per_cell))


def _save_calibration_sample(
    *,
    ch_corners,
    ch_ids,
    color,
    images_dir: Path,
    samples_corners: list,
    samples_ids: list,
    saved_images: list[str],
    guidance_observation,
    guidance_observations: list,
    accepted_guidance_samples: list[dict],
    reason: str,
    label: str,
) -> Path:
    samples_corners.append(ch_corners.copy())
    samples_ids.append(ch_ids.copy())
    if guidance_observation is not None:
        guidance_observations.append(guidance_observation)

    sample_index = len(samples_corners)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_filename = images_dir / f"calib_{ts}_{sample_index:03d}.png"
    cv2.imwrite(str(img_filename), color)
    saved_images.append(str(img_filename))

    record = {
        "index": sample_index,
        "file": str(img_filename),
        "reason": reason,
        "auto": reason != "manual",
        "corner_count": int(len(ch_corners)),
    }
    if guidance_observation is not None:
        record.update(
            cell=guidance_observation.cell,
            row=int(guidance_observation.row) + 1,
            col=int(guidance_observation.col) + 1,
            grid=list(guidance_observation.grid_shape),
            scale_bucket=guidance_observation.scale_bucket,
        )
    accepted_guidance_samples.append(record)
    print(f"[+] {label} {sample_index} ({len(ch_corners)} corners) -> {img_filename}")
    return img_filename


def main(argv=None):
    """CLI entry point: capture frames, collect ChArUco corners, calibrate, and write YAML."""
    args = parse_args(argv)
    _validate_guided_capture_args(args)

    try:
        validate_capture_args(args.source, args.video, rs)
        dictionary = get_dictionary(args.dict)
    except CharucoCalibrationError as exc:
        raise SystemExit(str(exc)) from exc
    args.min_corners = max(4, args.min_corners)

    # ----- ArUco/board setup -----
    aruco = cv2.aruco

    square_m = args.square_length_mm / 1000.0
    marker_m = args.marker_length_mm / 1000.0
    board = make_board(aruco, args.squares_x, args.squares_y, square_m, marker_m, dictionary)

    has_charuco_detector = hasattr(aruco, "CharucoDetector")
    if has_charuco_detector:
        chdet = aruco.CharucoDetector(board)
    else:
        det_params = aruco.DetectorParameters_create()

    # ----- Source setup -----
    _read_frame, _stop = _open_capture_source(args)

    try:
        # ----- Project-aware IO layout -----
        pr, calib_dir, images_dir, run_dir, out_yaml = _prepare_project_io(args)
        print(f"[i] Project root = {pr}")
        print(f"[i] Samples folder = {images_dir}")
        print(f"[i] Run folder = {run_dir}")
        args.out = str(out_yaml)  # ensure consistent path downstream
    except Exception:
        _stop()
        raise

    # ----- Capture loop -----
    samples_corners, samples_ids = [], []
    guidance_observations = []
    accepted_guidance_samples: list[dict] = []
    saved_images: list[str] = []
    auto_tick = 0
    guided_auto_cooldown = 0

    if args.guided_auto:
        print(
            "[i] Guided auto-capture is on; move the board through highlighted "
            "grid cells. SPACE=manual add, ENTER=solve, q=quit"
        )
    else:
        print("[i] Move the board around the frame; SPACE=add, ENTER=solve, q=quit")
    try:
        first = True
        while True:
            color = _read_frame()
            if color is None:
                if args.source == "video":
                    print("[i] video ended; solving with collected samples.")
                    break
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
            guidance_observation = _guidance_observation(ch_corners, args)
            guidance_state = analyze_calibration_guidance(
                guidance_observation,
                guidance_observations,
                min_samples=args.min_samples,
                grid_shape=args.coverage_grid_shape,
                samples_per_cell=args.samples_per_cell,
                auto_capture=args.guided_auto,
            )
            guided_decision = (
                auto_capture_decision(
                    guidance_state,
                    cooldown_frames_remaining=guided_auto_cooldown,
                )
                if args.guided_auto
                else None
            )
            cv2.putText(vis, f"samples={len(samples_corners)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(vis, "BOARD: OK" if good else "BOARD: not ready", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0) if good else (0, 0, 255), 2)
            _draw_guidance_overlay(vis, guidance_state)
            cv2.imshow("ChArUco Calibration", vis)
            k = cv2.waitKey(1) & 0xFF

            if k == 13:
                break
            if k == ord('q'):
                print("[i] quit requested; exiting without calibration.")
                return 0

            saved_sample = False
            if k == ord(' '):
                if good:
                    _save_calibration_sample(
                        ch_corners=ch_corners,
                        ch_ids=ch_ids,
                        color=color,
                        images_dir=images_dir,
                        samples_corners=samples_corners,
                        samples_ids=samples_ids,
                        saved_images=saved_images,
                        guidance_observation=guidance_observation,
                        guidance_observations=guidance_observations,
                        accepted_guidance_samples=accepted_guidance_samples,
                        reason="manual",
                        label="sample",
                    )
                    saved_sample = True
                else:
                    print("[!] not enough corners; get closer / reduce glare / tilt more.")
            elif good and guided_decision is not None and guided_decision.should_capture:
                _save_calibration_sample(
                    ch_corners=ch_corners,
                    ch_ids=ch_ids,
                    color=color,
                    images_dir=images_dir,
                    samples_corners=samples_corners,
                    samples_ids=samples_ids,
                    saved_images=saved_images,
                    guidance_observation=guidance_observation,
                    guidance_observations=guidance_observations,
                    accepted_guidance_samples=accepted_guidance_samples,
                    reason=guided_decision.reason,
                    label="guided sample",
                )
                guided_auto_cooldown = args.guided_auto_cooldown
                saved_sample = True
            elif args.auto and good:
                auto_tick += 1
                if auto_tick % args.auto_interval == 0:
                    _save_calibration_sample(
                        ch_corners=ch_corners,
                        ch_ids=ch_ids,
                        color=color,
                        images_dir=images_dir,
                        samples_corners=samples_corners,
                        samples_ids=samples_ids,
                        saved_images=saved_images,
                        guidance_observation=guidance_observation,
                        guidance_observations=guidance_observations,
                        accepted_guidance_samples=accepted_guidance_samples,
                        reason="legacy_timed_auto",
                        label="auto sample",
                    )
                    saved_sample = True

            if saved_sample:
                if args.guided_auto:
                    guided_auto_cooldown = args.guided_auto_cooldown
                next_guidance = analyze_calibration_guidance(
                    guidance_observation,
                    guidance_observations,
                    min_samples=args.min_samples,
                    grid_shape=args.coverage_grid_shape,
                    samples_per_cell=args.samples_per_cell,
                    auto_capture=args.guided_auto,
                )
                print(f"[guide] {next_guidance.recommendation.message}")
            elif guided_auto_cooldown > 0:
                guided_auto_cooldown -= 1
    finally:
        _stop()
        cv2.destroyAllWindows()

    need = _minimum_required_samples(args)
    if len(samples_corners) < need:
        raise SystemExit(f"Need at least {need} good samples; got {len(samples_corners)}.")

    # ----- Solve & write results -----
    img_size = (args.width, args.height)
    # ---- filter weak samples before solving ----
    REQUIRED = args.min_corners  # charuco solver needs at least 4 per view
    filtered_corners, filtered_ids = [], []
    for ch_c, ch_id in zip(samples_corners, samples_ids):
        if ch_c is not None and ch_id is not None and len(ch_c) >= REQUIRED:
            filtered_corners.append(ch_c)
            filtered_ids.append(ch_id)

    print(f"[i] using {len(filtered_corners)}/{len(samples_corners)} samples "
          f"(dropped {len(samples_corners) - len(filtered_corners)} < {REQUIRED} corners)")

    if len(filtered_corners) < need:
        raise SystemExit(f"Not enough valid samples after filtering: "
                         f"{len(filtered_corners)} < {need}")

    img_size = (args.width, args.height)
    ret, K, dist, rvecs, tvecs = calibrate_from_charuco(
        filtered_corners, filtered_ids, board, img_size
    )

    # ret, K, dist, rvecs, tvecs = calibrate_from_charuco(samples_corners, samples_ids, board, img_size)

    # write main YAML (latest)
    dist = np.asarray(dist, dtype=float).reshape(1, -1)  # ensure shape (1, N)

    print(f"[i] Reprojection RMS = {ret:.3f} px")
    print("[i] K =\n", K)
    print("[i] dist =", dist.ravel())

    calib_data = build_calibration_yaml(
        image_width=args.width,
        image_height=args.height,
        camera_matrix=K,
        distortion_coefficients=dist,
        reproj_rms=float(ret),
        notes=f"ChArUco {args.squares_x}x{args.squares_y}, square={args.square_length_mm}mm, "
              f"marker={args.marker_length_mm}mm, dict={args.dict}",
    )
    write_calibration_yaml(out_yaml, calib_data)
    print(f"[i] wrote {out_yaml}")

    # Run snapshot: config + calibration copy
    run_cfg = dict(
        timestamp=Path(run_dir).name,
        project_root=str(pr),
        source=dict(kind=args.source, cam=args.cam if args.source == "opencv" else None,
                    video=args.video if args.source == "video" else None,
                    width=args.width, height=args.height, fps=args.fps),
        board=dict(squares_x=args.squares_x, squares_y=args.squares_y,
                   square_length_mm=args.square_length_mm, marker_length_mm=args.marker_length_mm,
                   dict=args.dict),
        thresholds=dict(min_corners=args.min_corners, min_samples=args.min_samples,
                        required_samples=need, auto=args.auto,
                        auto_interval=args.auto_interval),
        guidance=dict(coverage_grid=list(args.coverage_grid_shape),
                      samples_per_cell=args.samples_per_cell,
                      guided_auto=args.guided_auto,
                      guided_auto_cooldown=args.guided_auto_cooldown,
                      accepted_samples=accepted_guidance_samples),
        image_size=dict(width=args.width, height=args.height),
        samples=dict(count=len(saved_images), files=saved_images),
        reproj_rms=float(ret),
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f)
    with open(run_dir / "calib_color.yaml", "w") as f:
        yaml.safe_dump(calib_data, f)
    print(f"[i] run snapshot saved in {run_dir}")


if __name__ == "__main__":
    main()
