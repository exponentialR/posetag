# Step 6: Collect Dataset / Pose Outputs

Step 6 is the runtime pose-labelling workflow. At this point PoseTag should
already know the camera calibration, AprilTag board layouts, face-shot
provenance, object keypoints where available, and the annotated
`T_board_object` transform for each object face.

The runtime pose composition is:

```text
T_cam_object = T_cam_board @ T_board_object
```

Do not invert or reorder this composition. `T_cam_board` maps board-frame
points into the camera frame, `T_board_object` maps object-frame points into the
board frame, and `T_cam_object` maps object-frame points into the camera frame.
Matrices are 4x4 homogeneous transforms stored as row-major nested lists and
are intended for column-vector multiplication.

## Command

```bash
posetag-collect --project_root my_project --mode opencv --session run01 --dry-run
```

`--dry-run` validates required inputs and source arguments without opening a
camera, creating a dataset session, or writing frame outputs. Run this first on
a prepared project.

## Optional GUI Launch

The optional `posetag-gui` dashboard exposes dataset collection as Stage 8.
When calibration, board definitions, face shots, mesh keypoints, and
annotations are complete, Stage 8 shows readiness, expected outputs, and two
actions:

- **Dry Run** launches `posetag-collect --dry-run` with `POSETAG_PROJECT` set
  for the selected project. It validates inputs and exits without opening a
  camera or writing dataset outputs.
- **Start Collection** launches the existing `posetag-collect` workflow with
  the selected session/source values. The dashboard defaults to smart
  auto-capture; the OpenCV/RealSense capture window still owns frame review
  and saving.

The GUI is process orchestration only. It does not reimplement AprilTag
detection, pose estimation, best-face selection, transform composition, or
dataset writing.

## Required Inputs

PoseTag validates these before opening the capture source:

- `calib/calib_color.yaml`
- `boards/tag_registry.yaml`
- board YAMLs referenced by the face annotations and registry
- `faces/face_manifest.csv`
- annotation YAMLs listed by the face manifest
- each annotation YAML must contain `T_board_object.matrix`

`objects/<object>/keypoints.json` is optional for collection. When it exists,
PoseTag uses object or face keypoints for bounding boxes and review metadata.
When it is missing, collection can still run and falls back to tag-derived
boxes.

Relative calibration names such as `--calib calib_color.yaml` resolve first to
`<project_root>/calib/calib_color.yaml`.

## Source Modes

OpenCV webcam:

```bash
posetag-collect --project_root my_project --mode opencv --cam 0 --session run01 \
  --auto-capture
```

Live Intel RealSense:

```bash
posetag-collect --project_root my_project --mode live --session run02 \
  --rs_w 640 --rs_h 480 --rs_fps 30
```

Add `--save_depth` to save aligned depth arrays when depth is available.

RealSense `.bag` playback:

```bash
posetag-collect --project_root my_project --mode bag --bag path/to/rec.bag \
  --session run03
```

OpenCV-readable video:

```bash
posetag-collect --project_root my_project --mode video --video sample.mp4 \
  --session run04
```

`--mode live` and `--mode bag` require `pyrealsense2`. Install the RealSense
extra or use `--mode opencv` / `--mode video` when RealSense support is not
available.

## Runtime Behavior

For each frame, PoseTag:

1. Checks the stream resolution against `calib/calib_color.yaml`
   `image_width` / `image_height`.
2. Detects AprilTags once per tag size.
3. Estimates `T_cam_board` for visible faces.
4. Composes `T_cam_object = T_cam_board @ T_board_object`.
5. Scores visible faces and keeps the best face per object.
6. Shows annotation review, reprojection/debug, and structured status panels.
7. Saves the accepted frame and pose label when the user accepts it, when
   smart auto-capture accepts it, or every frame when `--continuous` is
   enabled.

The best-face score prioritizes visible tag count, then image-space area.
When the origin tag is visible, it is used for `T_cam_board`; otherwise PoseTag
uses the visible board tag with the highest AprilTag decision margin and the
stored `T_board_tag` transform from the board YAML.

The status panel reports visible faces, saved frames, capture mode, selected
object pose summaries, tag-scale diagnostics, and keyboard controls. This panel
is display-only; it does not change pose estimation, frame acceptance, or output
schemas.

## Smart Auto-Capture

`--auto-capture` is the preferred low-effort collection mode. It waits for:

- detection quality: minimum visible tags and bbox area
- optional tag-scale quality: reject candidates whose measured tag spacing is
  too far from the board definition when scale diagnostics are available
- pose stability: consecutive frames within translation and rotation deltas
- cooldown: minimum time between saved frames
- usefulness: first sample, a new image coverage grid cell, a new distance bin,
  or a sufficiently different pose

Common tuning flags:

```bash
posetag-collect --mode opencv --session run01 --auto-capture \
  --auto-stable-frames 5 \
  --auto-cooldown-sec 1.0 \
  --auto-grid 3x3 \
  --auto-min-translation-delta-m 0.04 \
  --auto-min-rotation-delta-deg 8.0
```

`--continuous` remains available as the old crude save-every-frame mode, but it
is mutually exclusive with `--auto-capture`.

## Controls

- `ENTER` / `y` / `s`: accept and save current view; force-save in smart mode
- `SPACE` / `p`: pause or resume
- `r` / `n` / `BACKSPACE`: reject or skip frame
- `h`: toggle help
- `ESC` / `q` / `Q`: close capture cleanly
- `x` / `X`: close capture cleanly

## Outputs

Accepted frames are written under:

```text
datasets/<session>/
  images/
    Run_<YYYYMMDD-HHMMSS>_rgb_frame_<000000>.png
  annotated/
    Run_<YYYYMMDD-HHMMSS>_annotated_frame_<000000>.png
  reproj/
    Run_<YYYYMMDD-HHMMSS>_reproj_frame_<000000>.png
  depth/
    Run_<YYYYMMDD-HHMMSS>_depth_frame_<000000>.npy
  annotations/
    Run_<YYYYMMDD-HHMMSS>_frame_<000000>.json
  session.yaml
  <session>.jsonl
```

`images/` contains the saved RGB/BGR frame with detected tags covered to reduce
visual tag leakage. `annotated/` contains overlays, bounding boxes, labels, and
axes. `reproj/` contains the reprojection/debug view. `depth/` is written only
when `--save_depth` is set and the source supplies aligned depth.

`session.yaml` stores session-level provenance: source mode, camera kind,
calibration path, intrinsics, tag family, pose convention, face index, optional
missing keypoints, object counts, and frame count. `<session>.jsonl` is an
append-only rolling capture log with frame filenames, timestamps, tag family,
and saved object names.

## Per-Frame Pose Schema

Each `annotations/*.json` file includes:

```json
{
  "version": "1.0",
  "metadata": {
    "dataset_name": "run01",
    "frame_index": 0,
    "timestamp": 1724631234.123,
    "tag_family": "tag36h11"
  },
  "pose_convention": {
    "composition": "T_cam_object = T_cam_board @ T_board_object",
    "translation_units": "meters",
    "quaternion_order": "x, y, z, w",
    "rpy_order": "roll, pitch, yaw",
    "angle_units": "degrees"
  },
  "image_filename": "Run_<tag>_rgb_frame_<idx>.png",
  "annotated_image": "Run_<tag>_annotated_frame_<idx>.png",
  "reproj_image": "Run_<tag>_reproj_frame_<idx>.png",
  "depth": null,
  "capture": {
    "mode": "smart_auto",
    "status": "ready",
    "reason": "coverage_cell",
    "trigger_object": "connection_plate_white",
    "trigger_face": "connection_plate_white_sideA",
    "stable_count": 5
  },
  "camera_intrinsics": {
    "fx": 600.0,
    "fy": 610.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480
  },
  "camera_extrinsics": {
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 0.0, 1.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "rpy_deg": [0.0, 0.0, 0.0],
    "reference_face": "connection_plate_white_sideA",
    "reference_board": "boards/connection_plate_white_sideA.yaml"
  },
  "objects": [
    {
      "object_name": "connection_plate_white",
      "selected_face": "connection_plate_white_sideA",
      "selected_board": "boards/connection_plate_white_sideA.yaml",
      "class_id": 2,
      "class_name": "connection_plate",
      "2D_center": [0.5, 0.5],
      "width": 0.25,
      "height": 0.18,
      "bbox_xywh_px": [160.0, 120.0, 160.0, 86.0],
      "bbox_source": "face_kps",
      "transforms": {
        "T_cam_board": {"matrix": [[...]]},
        "T_board_object": {"matrix": [[...]]},
        "T_cam_object": {"matrix": [[...]]}
      },
      "quality": {
        "tag_used": 52,
        "num_tags_visible": 2,
        "score": 2500.0,
        "score_tags": 2,
        "score_area_px": 500,
        "bbox_source": "face_kps",
        "tag_scale_ratio": 1.0,
        "tag_scale_pairs": 1,
        "tag_scale_mad": 0.0,
        "tag_scale_auto_corrected": false
      },
      "6DOF_pose": {
        "position": [0.27, -0.07, 1.54],
        "orientation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation_units": "meters",
        "orientation_order": "roll, pitch, yaw",
        "orientation_units": "degrees",
        "rotation_quaternion_order": "x, y, z, w",
        "dimensions": [0.1, 0.1, 0.0]
      }
    }
  ]
}
```

Compatibility notes:

- `camera_extrinsics.orientation` is retained as the quaternion field used by
  earlier docs; `camera_extrinsics.rotation` is an explicit alias in the same
  `[x, y, z, w]` order.
- `objects[].6DOF_pose.orientation` is roll, pitch, yaw in degrees.
- `objects[].6DOF_pose.rotation` is the object quaternion in `[x, y, z, w]`
  order.
- `objects[].transforms.T_cam_object` is checked in tests against
  `T_cam_board @ T_board_object`.
- `capture` is optional provenance for frame acceptance. It records whether
  the frame was saved by manual review, smart auto-capture, or continuous mode;
  it does not change pose semantics.

## Common Failure Modes

- `--mode video` without `--video` fails before capture.
- An unreadable `--video` path fails before capture.
- `--mode bag` without `--bag` fails before capture.
- `--mode live` or `--mode bag` fails clearly when `pyrealsense2` is missing.
- `--continuous` together with `--auto-capture` fails before capture because
  those policies conflict.
- Missing or malformed `calib/calib_color.yaml` fails before capture.
- Calibration YAML must include positive `image_width` and `image_height`.
- Missing or malformed `boards/tag_registry.yaml` fails before capture.
- Board YAMLs missing from the registry or annotation records fail before
  capture.
- Missing `faces/face_manifest.csv` fails before capture unless
  `--allow-face-scan` is supplied for legacy projects.
- Missing or malformed annotation YAMLs fail before capture.
- Annotation YAMLs missing `T_board_object.matrix` fail before capture.
- Runtime stream resolution mismatch with calibration exits with a clear
  message. Use matching `--width` / `--height` or `--rs_w` / `--rs_h`, or
  recalibrate at the capture resolution.

## Verify Before Review / Export

After collection:

1. Confirm `datasets/<session>/session.yaml` records the expected calibration,
   source mode, and pose convention.
2. Inspect a few `annotations/*.json` records and verify every object includes
   `selected_face`, `selected_board`, `transforms`, `quality`, and `6DOF_pose`.
3. Confirm `T_cam_object = T_cam_board @ T_board_object` for representative
   saved objects.
4. Review `annotated/` and `reproj/` images for plausible selected faces,
   bounding boxes, and axes.
5. Confirm `<session>.jsonl` has one row per accepted frame.
