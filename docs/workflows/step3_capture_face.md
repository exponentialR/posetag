# Step 3: Capture Face Shots For Annotation

This workflow validates the PoseTag Step 3 command:

```bash
posetag-capture-face --project_root <path> --source opencv --cam 0 --object_name <object_or_face>
```

Step 3 captures wide reference images of object faces after Step 2 has created
board YAML files and `boards/tag_registry.yaml`. These images are used by the
later annotation workflow to establish board-to-object transforms.

Step 3 does not implement annotation, board-to-object transforms, dataset
collection, or pose estimation. It records images, AprilTag detections, the
registered face identity, camera intrinsics, and reproducible file paths.

## Scientific Assumptions

- Step 1 has produced a valid colour-camera calibration YAML.
- Step 2 has produced one board YAML per physical face and a tag registry that
  maps AprilTag IDs to those board YAML files.
- The `--family` value matches the printed AprilTag family.
- Face identity is resolved from expected tag IDs in `boards/tag_registry.yaml`
  and the referenced board YAML files.
- `expected_tag_ids` are the IDs registered for the selected face.
- `detected_tag_ids` are the AprilTag IDs detected in the saved image.
- The command does not compute `T_board_object`, `T_cam_board`, or
  `T_cam_object`.
- Camera intrinsics in metadata are copied from Step 1 calibration for
  provenance; no calibration solve is performed in Step 3.

## Required Inputs From Earlier Steps

With:

```bash
--project_root my_project --calib calib_color.yaml
```

PoseTag first looks for:

```text
my_project/calib/calib_color.yaml
```

If a relative calibration path is supplied and that project path does not
exist, PoseTag also checks the literal relative path and `<project_root>/<path>`
for legacy compatibility. Absolute paths are used exactly as supplied.

The default tag registry path is:

```text
my_project/boards/tag_registry.yaml
```

Every registry entry used by Step 3 must reference an existing board YAML, and
the registry object/tag IDs must match the referenced board YAML. The registry
membership for a referenced board must cover all tag IDs listed in that board
YAML, so Step 3 does not under-report expected tags for stale registries.

## Command Examples

Generic OpenCV webcam:

```bash
posetag-capture-face --project_root my_project \
  --source opencv --cam 0 \
  --object_name connection_plate_white
```

Intel RealSense colour stream:

```bash
posetag-capture-face --project_root my_project \
  --source realsense --width 640 --height 480 --fps 30 \
  --object_name connection_plate_white
```

Video file:

```bash
posetag-capture-face --project_root my_project \
  --source video --video sample.mp4 \
  --object_name connection_plate_white
```

Specific face:

```bash
posetag-capture-face --project_root my_project \
  --source opencv --cam 0 \
  --object_name connection_plate_white_sideA
```

Queue all registered faces with stable-tag auto-capture:

```bash
posetag-capture-face --project_root my_project \
  --source opencv --cam 0 \
  --capture_all --auto_capture --exit_when_complete
```

Queue a selected subset of registered faces:

```bash
posetag-capture-face --project_root my_project \
  --source opencv --cam 0 \
  --queue_face connection_plate_white_sideA \
  --queue_face connection_plate_white_sideB \
  --auto_capture --exit_when_complete
```

The default source remains `realsense` for compatibility with the legacy script.
For a normal USB webcam, pass `--source opencv --cam 0` explicitly.

## Guided GUI Stage 5 Flow

The optional `posetag-gui` dashboard presents face-shot capture as Stage 5 in
the calibration-first workflow. The preferred GUI path opens a native
batch-guided capture window that uses shared PoseTag workflow/pipeline helpers
for camera or video frames, AprilTag detection, overlays, metadata writing, and
manifest updates. The `posetag-capture-face` OpenCV CLI remains the
reproducible terminal fallback.

The Stage 5 panel:

- reads registered object bases and full face names from
  `boards/tag_registry.yaml`
- shows the registered face capture queue with captured/missing status
- opens a Stage-4-style batch window for all missing faces, a selected subset,
  or the current face
- supports mouse/trackpad scrolling and clicks on the queue; selecting an
  already captured face enables Retake Now
- shows a saved-shot gallery from `shots/manifest.csv`, collapsed into a
  compact thumbnail stack; clicking or double-clicking the stack opens its
  individual shots, previews the selected saved shot, and shows its
  raw/annotated/metadata paths
- offers batch capture, selected-face capture, and current-face capture actions
- validates the calibration YAML, registry, selected object/face or queue, source,
  video path, and output paths before launch
- supports webcam/OpenCV, RealSense, and video sources
- writes the same raw image, annotated image, metadata JSON, and manifest row
  as the CLI workflow
- previews the exact `posetag-capture-face` command and keeps a copy-command
  fallback
- starts native queue mode with stable-tag auto-capture enabled by default
- refreshes project status when `shots/manifest.csv` appears or changes

The CLI/OpenCV capture window also loads recent manifest images at startup, so
an already captured project shows saved-shot previews instead of an empty
"waiting for first save" panel when you use the terminal fallback.

Stage 5 status is coverage-based. It is complete only when every registered
board face has at least one valid saved shot with existing raw image,
annotated image, metadata JSON, and manifest row. Partial coverage, missing
files, malformed metadata, or shots saved with `validation_ok: false` keep the
stage in a missing or needs-attention state.

## Object And Face Selection

`--object_name` accepts either:

- a base object name such as `connection_plate_white`
- a full face name such as `connection_plate_white_sideA`

When a base object is supplied, PoseTag keeps auto-side mode enabled and selects
the best registered face by overlap between expected and detected tag IDs. When
a full face is supplied, PoseTag uses that face directly.

If `--object_name` is omitted, the OpenCV viewer starts from the registered
face queue. Use `--capture_all` to queue every registered face explicitly.
Use repeated `--queue_face` arguments to capture a selected subset.

The optional auto-capture controls are:

- `--auto_capture`: save automatically when the selected face is valid and
  stable.
- `--auto_capture_frames`: number of consecutive valid frames required before
  auto-save.
- `--auto_capture_cooldown`: minimum seconds between auto-saves.
- `--exit_when_complete`: exit cleanly after every queued face has been saved.

## Viewer Controls

- `ENTER`: save the current frame.
- `ENTER` twice within 3 seconds: force-save when expected tags are missing.
- click a visible queue row: select that face.
- mouse wheel or trackpad scroll: move through the face queue.
- arrow keys: move through the face queue, including macOS/OpenCV arrow codes.
- `o`: open the face queue picker.
- Up/down or `W` / `S` / `K` / `J`: move in the queue picker.
- `a`: toggle auto-side mode.
- `Left` / `Right`, `f` / `n`, or `[` / `]`: cycle faces and switch to
  manual face selection.
- `g`: toggle gallery panels.
- `h`: toggle help overlay.
- `q` or `ESC`: quit cleanly without saving another shot.

If a video reaches EOF before another save, the command exits cleanly.

## Project Directory Side Effects

Default outputs:

```text
my_project/
  shots/
    manifest.csv
    <object_base>/
      side<SideLetter>/
        <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_raw.png
        <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_ann.png
        <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_meta.json
  logs/
    capture_face.log
```

The default layout is `by_object_side`. Alternative layouts:

- `--layout flat` writes raw, annotated, and metadata files directly in
  `shots/`.
- `--layout by_object` writes files under `shots/<object_base>/`.
- `--layout split_type` writes separate raw, annotated, and metadata folders.

Example split layout:

```bash
posetag-capture-face --project_root my_project \
  --source opencv --cam 0 \
  --object_name connection_plate_white \
  --layout split_type \
  --raw_dir shots/images \
  --ann_dir shots/ann \
  --meta_dir shots/meta
```

Use `--out_dir`, `--manifest`, and `--log_file` to override default output
locations. Relative output paths are resolved under the project root.

Invalid source arguments, missing calibration, malformed calibration, missing
registry files, malformed registries, missing board YAML references, and unknown
object selections fail before camera/video preview and before Step 3 capture
artifacts are written.

## Metadata JSON Schema

Each saved shot writes a `*_meta.json` file. The current schema is:

```json
{
  "object_base": "connection_plate_white",
  "object_full": "connection_plate_white_sideA",
  "side": "A",
  "face_yaml": "my_project/boards/connection_plate_white_sideA.yaml",
  "expected_tag_ids": [52, 53],
  "detected_tag_ids": [52],
  "validation_ok": true,
  "auto_face": true,
  "image": {
    "path_raw": "my_project/shots/connection_plate_white/sideA/..._raw.png",
    "path_ann": "my_project/shots/connection_plate_white/sideA/..._ann.png",
    "path_meta": "my_project/shots/connection_plate_white/sideA/..._meta.json",
    "width": 640,
    "height": 480
  },
  "camera": {
    "fx": 600.0,
    "fy": 610.0,
    "cx": 320.0,
    "cy": 240.0
  },
  "timestamp": "20260529_120000"
}
```

The timestamp format is `YYYYMMDD_HHMMSS`. Paths are recorded exactly as written
by the current command.

## Manifest CSV Schema

By default, Step 3 appends one row per saved shot to:

```text
my_project/shots/manifest.csv
```

Fields:

- `timestamp`
- `object_base`
- `object_full`
- `side`
- `face_yaml`
- `path_raw`
- `path_ann`
- `path_meta`
- `width`
- `height`
- `fx`
- `fy`
- `cx`
- `cy`
- `detected_ids`
- `expected_ids`
- `validation_ok`
- `auto_face`

`detected_ids` and `expected_ids` are space-separated integer tag IDs in the
CSV row.

## Common Failure Modes

- `--source video` without `--video` fails with a clear message.
- Missing or unreadable video paths fail before Step 3 capture artifacts are
  created.
- `--source realsense` fails clearly when `pyrealsense2` is not installed.
- Missing `calib/calib_color.yaml` fails before preview.
- Malformed calibration YAML fails with a schema-oriented error.
- Missing `boards/tag_registry.yaml` fails before preview.
- Malformed or empty tag registries fail before preview.
- Registry entries that reference missing board YAML files fail before preview.
- Registry object/tag mismatches with referenced board YAML files fail before
  preview.
- Registry membership that does not cover all tag IDs in the referenced board
  YAML fails before preview.
- Unknown `--object_name` selections fail before preview.
- `ESC` / `q` exits cleanly without writing another shot.
- Video EOF exits cleanly without hanging.

## Verify Before Step 4

Before moving to face annotation:

1. Confirm `shots/manifest.csv` exists and has one row per saved reference
   image.
2. Confirm each row's raw, annotated, and metadata paths exist.
3. Load each `*_meta.json` file and confirm `object_full`, `face_yaml`,
   `expected_tag_ids`, `detected_tag_ids`, image size, and camera intrinsics are
   plausible.
4. Confirm the saved raw image shows the full physical face needed for later
   annotation.
5. Confirm the annotated image overlays the detected tag IDs expected for that
   face.

## Current Implementation Note

The public `posetag-capture-face` entry point still delegates to the legacy
top-level `src/capture_face.py` interactive OpenCV viewer. Pure validation,
path, registered-face, metadata, and manifest helpers now live in
`posetag.pipelines.capture_face` so the Step 3 contract can be tested without
camera hardware.

Migrating the full interactive viewer into a package-first module remains
future technical debt and is intentionally outside issue #37.
