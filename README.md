# PoseTag
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-green)
![NumPy](https://img.shields.io/badge/numpy-supported-yellow)
![PyPI](https://img.shields.io/pypi/v/opencv-contrib-python)
![PyPI](https://img.shields.io/pypi/v/pyrealsense2)
![PyPI](https://img.shields.io/pypi/v/apriltag)
![System Overview](media/Screenshot%20from%202025-08-28%2011-06-45.png)

PoseTag is scientific Python software for generating reproducible ground-truth
6-DoF object poses from RGB using AprilTags, camera calibration, per-face board
definitions, and Intel RealSense / webcam / video capture.
It is designed for training RGB-only pose networks in pick-and-place and
assembly-style scenarios.

The repository is being migrated from legacy names such as `GT-6DoF-ATag`,
`gt6dof_atag`, and `gtat*`. The canonical identity is now:

- Project/display name: `PoseTag`
- Repository name: `posetag`
- Python import namespace: `posetag`
- Primary CLI: `posetag`

Legacy names are still available as temporary compatibility aliases. New docs
and examples below prefer the canonical PoseTag naming.

## What this repo is for

A reproducible pipeline to convert AprilTags plus a few clicks into
high-quality 6-DoF ground truth for rigid objects, then capture multi-object
scenes with readable poses for training and validation of RGB-only pose
estimators.

## Contents

- **Calibration**: ChArUco colour camera calibration to `calib_color.yaml`
  with `fx`, `fy`, `cx`, `cy`, distortion, image size, and RMS.
- **Board building**: per-face AprilTag layout from a single frame to
  `<object_face>.yaml` plus a global `boards/tag_registry.yaml`.
- **Shot capture**: face shots with overlays and metadata to
  `shots/*_{raw,ann,meta}`.
- **Mesh keypoints**: canonical 3D keypoints per object in
  `objects/<object>/keypoints.json`.
- **Annotation**: click 4 face corners on a shot to compute `T_board_object`
  for that face.
- **Batch annotation**: drive the annotator across faces and shots
  automatically.
- **Ground-truth capture**: live / bag / video multi-object capture with
  per-frame review, best-face selection, and optional depth.
- **At runtime**: `T_cam_object = T_cam_board @ T_board_object`.

## Installation

Install in editable mode from a fresh clone:

```bash
python3 -m pip install -e .
```

Optional extras:

```bash
python3 -m pip install -e ".[apriltags]"
python3 -m pip install -e ".[gui]"
python3 -m pip install -e ".[realsense]"
python3 -m pip install -e ".[scipy]"
```

## Developer checks

Validate GitHub Actions workflows locally before commit or push:

```bash
./scripts/check-workflows.sh
```

The script uses a local `actionlint` binary when available, or falls back to
the `rhysd/actionlint` Docker image if Docker is installed.

Workflow changes are also linted automatically in GitHub Actions on pull
requests and pushes to `develop`.

## Project Management

Create or select an active PoseTag project:

```bash
posetag project new --name my_project
posetag project ls
posetag project current
posetag project home
```

By default PoseTag uses `~/posetag` as the projects home. During the migration
it still discovers legacy project homes under `~/gt-6dof` and still accepts the
legacy `GTAT_PROJECT` / `GTAT_PROJECTS_DIR` environment variables.

## Workflow Dashboard

PoseTag includes an optional PySide6 dashboard for inspecting workflow status
from the same package helpers used by tests and command-line tools:

```bash
python3 -m pip install -e ".[gui]"
posetag-gui --project_root my_project
```

The dashboard shows the selected project root, stages 0-8, a project-health
summary, checked paths, warnings, errors, next recommended action, and copyable
command previews. The GUI order is project setup, ChArUco board generation,
camera calibration, object AprilTag generation, board definitions, face-shot
capture, annotation, dataset collection, and review/export. In Stage 1 it can
generate the ChArUco board PNG, PDF, and metadata YAML using the same package
helper as `posetag-gen-charuco`. In Stage 2 it reads the generated ChArUco
metadata, autofills the calibration board parameters, lets you choose webcam,
RealSense, or video source settings, and can launch the existing
`posetag-calib-charuco` workflow with a `Run Calibration` button when
readiness checks pass. The panel keeps a copyable command fallback, streams
process output into a compact log, and watches for the expected
`<project_root>/calib/calib_color.yaml` output path. In Stage 3 it can
generate object AprilTag PNG/PDF sheets with `posetag-gen-tags`, including tag
size, ID list or start/count range, paper, DPI, layout, prefix, and output
folder settings. It previews the first generated PNG sheet and can open
selected folders in the system file manager. In Stage 4 it guides object board
definition setup for `posetag-make-board`: object/face name, tag size, family,
calibration YAML, webcam/RealSense/video source settings, optional output paths,
and expected `boards/<object_name>.yaml` plus `boards/tag_registry.yaml`
status. The GUI splits the board identity into an object label and side/face
label, then passes the composed board definition name, for example
`connection_plate_white_sideA`, to the same board-building writer used by the
CLI. Stage 4 offers a native Batch Guided Capture flow for repeated object
types and instances such as `connection_plate | 01-08 | sideA, sideB`. The
dashboard keeps the object queue, tag size, source, and calibration path
visible, with single-board, bulk-paste, advanced output, and CLI fallback
details in expandable sections. It keeps the camera open, walks through the
generated board queue, auto-captures when the visible tag IDs are stable, lets
you confirm the selected tag IDs and origin tag, and writes each board YAML
plus the tag registry through the shared board-building helpers. It can also
persist the pre-capture queue to
`<project_root>/boards/board_building_queue.yaml` as a reloadable draft. That
draft records planned object rows and capture settings only; PoseTag does not
create board YAML or tag-registry entries until a real capture is saved. It can
also launch the existing `posetag-make-board` workflow as a fallback, stream
process output, and provide a prompt-response box for the selected tag IDs and
origin tag while keeping the OpenCV `ENTER`/`ESC` controls unchanged. In Stage
5 it can open a native guided face-shot capture window with a registered face
queue, webcam/RealSense/video source settings, calibration and registry paths,
output layout, expected manifest path, stable-tag auto-capture for missing
faces, and retake controls for already captured faces. Users can start the full
missing-face batch, capture a selected subset, or capture one current face.
The Stage 5 dashboard and capture window keep manifest-backed saved shots
collapsed into a compact thumbnail stack; clicking the stack opens its
individual shots, previews the selected shot, and shows raw/annotated/metadata
paths. Copy Command keeps the
`posetag-capture-face` OpenCV CLI fallback. Stage
5 status is coverage-based: it is complete only when
every registered board face has at least one valid saved raw image, annotated
image, metadata JSON, and `shots/manifest.csv` row. The
calibration capture window also shows
state-driven guidance based on detected ChArUco corners and accepted sample
coverage, such as moving the board toward missing frame edges/corners or
changing distance. Guided auto-capture saves good frames as the board covers
the grid, while `SPACE` remains available as a manual override. The GUI does
not reimplement AprilTag rendering, calibration solving, board-frame math,
board YAML schemas, registry writing, annotation, or dataset workflows.

## Scientific Contract

The core pose composition must remain:

```text
T_cam_object = T_cam_board @ T_board_object
```

Current conventions used by the repository include:

- translations in meters
- board YAML files storing `tag_size_m` in meters
- quaternion outputs in `xyzw` order where emitted
- roll / pitch / yaw outputs reported in degrees where emitted

If any of those conventions change later, code, tests, docs, and output schema
notes should change together.

---

## Quick start

### 0) Generate AprilTags for printing

Create printable **AprilTag 36h11** sheets to stick on object faces.
The normal project workflow is:

```bash
posetag-gen-tags --project_root my_project --tag-size-mm 40 --ids 1-4 --dpi 150
```

This initializes or reuses `my_project/` and writes sheets to:

```text
my_project/boards/patterns/
```

Preview of generated sheet content:

![AprilTag sheet preview](media/step0_apriltag_sheet_preview.png)

Print at **100%** / **Actual size** so the black square edge matches
`--tag-size-mm`. Do not use “Fit to page”.

**Without a project root**

```bash
posetag-gen-tags --tag-size-mm 40 --ids 1-4 --out_dir apriltags_out
```

**Range of IDs**

```bash
posetag-gen-tags --project_root my_project --tag-size-mm 80 --ids 46-49
```

**Non-contiguous IDs**

```bash
posetag-gen-tags --project_root my_project --tag-size-mm 40 --ids 19,20,21,22,27,28,29,30
```

If the requested IDs do not fit on one sheet, PoseTag writes deterministic
page-numbered PNGs, for example `page01of03`, `page02of03`, and `page03of03`.
This tiny-paper command is mainly useful for testing pagination:

```bash
posetag-gen-tags --project_root my_project --tag-size-mm 40 --paper-mm 80x80 --ids 1-3 --dpi 100
```

Options you might tweak later: `--paper {A4,LETTER,LEGAL}`, `--paper-mm WxH`,
`--orientation`, `--dpi`, and `--out_dir`.

Standard PoseTag installs write both PNG and PDF sheets.

In the calibration-first GUI workflow, generate these object tags from
`posetag-gui` Stage 3 after `calib/calib_color.yaml` exists. The Stage 3 panel
uses the same generator, shows the exact command fallback, refreshes project
status after generation, previews the first generated sheet, and reminds you to
print at **100%** / **Actual Size** without “Fit to page”.

---

### 1) Calibrate the colour camera (ChArUco)

![calibration-grab](media/calibration.gif)

First generate the exact ChArUco calibration board you will print:

```bash
posetag-gen-charuco --project_root my_project \
  --squares-x 3 --squares-y 5 \
  --square-length-mm 50 --marker-length-mm 37 --dict 7X7_50 \
  --paper A4 --dpi 300
```

This writes a printable PNG, PDF, and metadata YAML under:

```text
my_project/calib/boards/
```

Print at **100%** / **Actual Size** and do not use “Fit to page”. Verify the
printed square edge with a ruler before calibration.

The optional `posetag-gui` dashboard provides the same ChArUco board setup from
the Stage 1 ChArUco panel and refreshes project status after writing the files.
The Stage 2 calibration panel then reads the generated metadata, autofills the
matching board arguments, lets you choose webcam/OpenCV, RealSense, or video
source settings, shows the guided auto-capture / `SPACE` / `ENTER` / `q`
controls, and can run the existing calibration command as a child process. The
OpenCV calibration window still owns capture and solving: guided auto-capture
saves good frames as the board covers missing grid cells, `SPACE` manually adds
a sample, `ENTER` solves, and `q` quits. During capture, PoseTag overlays
deterministic guidance derived from the current ChArUco observation and
accepted sample coverage, rather than randomly rotating prompts. The Stage 2
panel exposes coverage rows, coverage columns, and samples-per-cell controls,
and passes those settings to the same command shown in the copy-command
preview. It refreshes status after the process exits or after
`calib/calib_color.yaml` appears, and it only marks the launch successful when
that YAML passes the current status/schema checks. When the calibration YAML is
present, the right panel shows a parsed calibration result summary with image
size, RMS, intrinsics, distortion, output path, and latest run snapshot, while
keeping raw YAML available through an explicit view/open action.

Then calibrate with the same board parameters. Move the board through the
coverage grid; guided auto-capture saves useful samples and `SPACE` can still
force a manual sample. Press `ENTER` to solve once the guidance says coverage
looks good. The
calibration command outputs
`calib_color.yaml` containing `fx`, `fy`, `cx`, `cy`, distortion, image size,
and reprojection RMS. With `--project_root`, the latest calibration defaults to
`<root>/calib/calib_color.yaml`. Each run also stores raw calibration samples
under `<root>/calib/images/set_XX/` and a timestamped snapshot under
`<root>/calib/runs/<UTC-timestamp>/`.

**Generic webcam**

```bash
posetag-calib-charuco --project_root my_project --source opencv --cam 0 \
  --squares-x 3 --squares-y 5 --square-length-mm 50 --marker-length-mm 37 --dict 7X7_50
```

**Intel RealSense**

```bash
posetag-calib-charuco --project_root my_project --source realsense \
  --width 640 --height 480 --fps 30 \
  --squares-x 3 --squares-y 5 --square-length-mm 50 --marker-length-mm 37 --dict 7X7_50
```

**Video file**

```bash
posetag-calib-charuco --project_root my_project --source video --video sample.mp4 \
  --squares-x 3 --squares-y 5 --square-length-mm 50 --marker-length-mm 37 --dict 7X7_50
```

Ensure `--squares-x`, `--squares-y`, `--square-length-mm`,
`--marker-length-mm`, and `--dict` match the generated board metadata exactly.
OpenCV webcams treat `--width` / `--height` as best-effort; the YAML records
the actual stream size used during calibration. Use the recorded `image_width`
/ `image_height` for downstream capture.

For the full Step 1 contract, failure modes, and verification checklist, see
[`docs/workflows/step1_calibrate_charuco.md`](docs/workflows/step1_calibrate_charuco.md).

Sample `calib_color.yaml`:

```yaml
camera_matrix:
  cx: 327.64591864758034
  cy: 191.92344695734758
  data:
  - - 1130.1711396910728
    - 0.0
    - 327.64591864758034
  - - 0.0
    - 1137.6243734485495
    - 191.92344695734758
  - - 0.0
    - 0.0
    - 1.0
  fx: 1130.1711396910728
  fy: 1137.6243734485495
distortion_coefficients:
  data:
  - - -0.4270123490995857
    - 6.863413191402844
    - -0.010742366737405607
    - -0.0025807724510832734
    - -63.07160252228832
  k1: -0.4270123490995857
  k2: 6.863413191402844
  k3: -63.07160252228832
  p1: -0.010742366737405607
  p2: -0.0025807724510832734
image_height: 480
image_width: 640
model: plumb_bob
notes: ChArUco 3x5, square=50.0mm, marker=37.0mm, dict=7X7_50
reproj_rms: 0.13782644794293097
```

---

### 2) Build per-face boards and tag registry

![make-board](media/make_board.gif)

Capture one view with all tags on a face visible, pick an **origin** ID, and
write a face YAML plus update the project tag registry. The board frame is the
origin tag centre; board x/y axes follow the origin tag axes; planar boards
should have z offsets approximately equal to `0`.

**OpenCV webcam**

```bash
posetag-make-board --project_root my_project \
  --source opencv --cam 0 \
  --object_name connection_plate_white_sideA \
  --calib calib_color.yaml \
  --tag_size_mm 80
```

**Intel RealSense**

```bash
posetag-make-board --project_root my_project \
  --source realsense --width 640 --height 480 --fps 30 \
  --object_name connection_plate_white_sideA \
  --calib calib_color.yaml \
  --tag_size_mm 80
```

**From a video**

```bash
posetag-make-board --project_root my_project \
  --source video --video sample.mp4 \
  --object_name connection_plate_white_sideA \
  --calib calib_color.yaml \
  --tag_size_mm 80
```

`--tag_size_mm` is the physical black-square edge size of the AprilTags in
millimetres. The board YAML stores this as `tag_size_m` in metres.

With `--project_root my_project --calib calib_color.yaml`, PoseTag first looks
for the Step 1 calibration at:

```text
my_project/calib/calib_color.yaml
```

Absolute calibration paths are used exactly as supplied. Relative paths retain
legacy compatibility after the project calibration lookup.

Controls and prompts:

- `ENTER`: capture the current frame and estimate tag poses.
- `ESC`: exit cleanly without writing board YAML.
- After capture, enter the comma-separated tag IDs to include, then choose the
  origin ID from the selected IDs.

**Outputs**

- `my_project/boards/<object_name>.yaml`
  with `origin_id`, `tag_size_m`, and per-tag `cx`, `cy`, and `yaw_deg`.
- `my_project/boards/tag_registry.yaml`
  with tag-ID to face-YAML mappings.
- Optional `my_project/boards/shots/*` audit images when `--save_shot` is
  enabled.

Tips:

- add `--project_root <path>` to create or select a project explicitly
- otherwise the resolver uses your current or last project
- use `--out_dir`, `--registry`, and `--shots_dir` to override board, registry,
  and audit-shot paths
- use `--z_thresh` (default `0.01` m) to enforce planarity
- use `--allow_nonplanar` to proceed with a warning
- source and calibration failures are checked before board YAML or registry
  output artifacts are created

The optional `posetag-gui` dashboard exposes this as guided Stage 4. It
prepares board-building state from object label, side/face label, tag size,
family, calibration YAML path, webcam/RealSense/video source settings, and
optional output paths. The composed board definition name is the single
`--object_name` value used by the existing CLI and by
`boards/<object_name>.yaml`; sideA-sideF presets help keep repeated object
faces labelled consistently.

The preferred GUI path is **Start Batch**. Stage 4 carries over the Stage 3
object-tag size when it can, either from the current GUI value or from the
latest generated pattern filename. Add object rows from the compact
object/instances/sides/tag-size controls, or expand **Bulk Paste** for
multi-object rows in this format:

```text
connection_plate | 01-08 | sideA, sideB | 80
column | 01-04 | sideA-sideB | 40
column | 01-04 | sideC-sideD | 80
```

PoseTag generates board names such as `connection_plate_01_sideA`, keeps one
native live preview open, detects pose-estimated tags, gives stability
guidance, auto-captures once the visible IDs hold steady, then lets you confirm
the selected tag IDs and origin tag before saving and advancing. The save step
uses the per-board tag size and the shared board-frame, planarity, board YAML,
and tag-registry helpers. The existing board YAML schema stores one
`tag_size_m` per board definition, so different sides can use different sizes
as separate queued boards. Pressing **Add Object** immediately saves the
planned queue and current capture settings to
`<project_root>/boards/board_building_queue.yaml`; reopening the dashboard
loads that draft so rows can be selected, inspected, or captured later. This
draft is not a board definition and does not mark Stage 4 complete. The board
name is the idempotent identity: recapturing `connection_plate_01_sideA`
replaces `boards/connection_plate_01_sideA.yaml` and removes stale registry
entries that used to point at that same board YAML.
The dashboard still keeps single **Guided Capture**, copyable
`posetag-make-board`, and CLI launch paths as fallbacks; the CLI paths explain
that `ENTER` captures and `ESC` quits, then accept selected tag IDs plus
origin tag through the GUI prompt-response box.

**Example board YAML** (`connection_plate_white_sideA.yaml`)

```yaml
object: connection_plate_white_sideA
family: tag36h11
tag_size_m: 0.08
origin_id: 52
tags:
- id: 52
  cx: 0.0
  cy: 0.0
  yaw_deg: 0.0
- id: 53
  cx: 0.1
  cy: -0.2
  yaw_deg: 180.0
notes: "Board frame = origin tag centre; x,y follow origin tag axes; z ≈ 0."
```

**Example tag registry** (`tag_registry.yaml`)

```yaml
version: 1
updated: "2025-11-09T18:50:42Z"
tags:
  "52":
    object: connection_plate_white_sideA
    yaml: my_project/boards/connection_plate_white_sideA.yaml
  "53":
    object: connection_plate_white_sideA
    yaml: my_project/boards/connection_plate_white_sideA.yaml
```

The registry is updated each time you run `posetag-make-board`. Recapturing the
same board YAML removes stale tag IDs for that board before writing the new
membership. If the same tag ID is already mapped to another board YAML, PoseTag
warns and preserves the existing mapping so conflicts are not silently
overwritten.

Faces may use different tag sizes; each face YAML stores its own `tag_size_m`.

---

### 3) Capture face shots for annotation

![capture-face](media/capture_face.gif)

Once per-face boards and the tag registry exist, the next step is to capture
wide shots of each face with detections overlaid. These images and metadata are
later used to solve the board-to-object transform. This step records reference
images and provenance only; annotation, board-to-object transforms, dataset
collection, and pose estimation happen later.

Detailed workflow notes: [`docs/workflows/step3_capture_face.md`](docs/workflows/step3_capture_face.md)

**Quick start**

In the GUI, open Stage 5 and use **Start Batch** for the guided face-shot queue;
use **Capture Selected** or **Capture Current** for a smaller run. The command
below is the reproducible CLI fallback.

```bash
posetag-capture-face --project_root my_project \
  --source opencv --cam 0 \
  --object_name connection_plate_white
# Optional UI sizing
#   --panel_w 420
#   --recent_w 320
# Optional saving layout / manifest
#   --layout {flat,by_object,by_object_side,split_type}
#   --manifest shots/manifest.csv
```

**Other sources**

```bash
posetag-capture-face --project_root my_project --source realsense --object_name connection_plate_white
posetag-capture-face --source opencv --cam 0 --object_name connection_plate_white
posetag-capture-face --source video --video sample.mp4 --object_name connection_plate_white
```

The command's legacy default source is `realsense`; use `--source opencv` for a
regular USB webcam or laptop camera.

**Notes**

- Outputs live under your active project:
  `<root>/shots/<object_base>/side<Side>/..._{raw,ann}.png` and `..._meta.json`
- The GUI and CLI both write raw images, annotated images, metadata JSON, and
  `shots/manifest.csv` rows.
- `--manifest` appends rows to `<root>/shots/manifest.csv`
- `--calib` and `--registry` default to
  `<root>/calib/calib_color.yaml` and `<root>/boards/tag_registry.yaml`
- The registry must reference readable board YAML files created by
  `posetag-make-board`.
- Missing calibration, missing or malformed registries, incomplete registry
  membership, missing board YAML references, unknown object selections, missing
  `--video`, unreadable videos, and unavailable RealSense support fail clearly
  before capture artifacts are written.

**Workflow in the viewer**

- `o` picker with arrow keys or `W/S/K/J`, `Enter` to select
- `a` auto-side on or off
- `Left` / `Right`, `f` / `n`, or `[` / `]` cycle faces and switch to manual
  face selection
- `ENTER` save, with double-press within 3 seconds to force a save if expected
  tags are missing
- `g` panels, `h` help, `q` or `ESC` quit

You can pass either a base object name such as
`connection_plate_white` or a full face such as
`connection_plate_white_sideA`.

**Optional layout and UI sizing**

```bash
posetag-capture-face --layout split_type --raw_dir shots/images --ann_dir shots/ann --meta_dir shots/meta
posetag-capture-face --panel_w 420 --recent_w 320
```

```text
shots/
  <object_base>/
    side<SideLetter>/
      <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_raw.png
      <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_ann.png
      <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_meta.json
```

`*_meta.json` includes object identity, side, face YAML, expected and detected
tag IDs, validation status, camera intrinsics, image size, timestamp, and save
paths. `shots/manifest.csv` appends one row per saved shot with the same core
provenance fields.

**Guided GUI Stage 5**

The optional `posetag-gui` dashboard exposes this workflow as guided Stage 5
after Stage 4 board definitions are valid. It shows the registered face queue
from `boards/tag_registry.yaml`, validates calibration, registry, source,
video, and output settings, and can launch `posetag-capture-face` in the
current Python environment. Like Stage 4, it has a batch action for all
missing faces, a selected-subset action from the queue, and a current-face
action for one-off captures. The OpenCV capture window remains the capture UI,
but GUI launches default to stable-tag auto-capture; `ENTER` still saves
manually, and queue rows can be selected with mouse clicks, scroll/trackpad
gestures, or arrow keys. The dashboard streams process output, keeps a
copyable command fallback, shows saved face-shot thumbnails from the manifest,
and refreshes coverage when `shots/manifest.csv` is written or updated.
When reopening capture on an existing project, the OpenCV saved-preview panel
loads recent manifest images instead of starting empty.

**Alternative layouts**

- `--layout flat` writes everything in `shots/`
- `--layout by_object` writes to `shots/<object_base>/...`
- `--layout split_type` writes separate raw, annotated, and metadata folders

**Manifest (optional)**

When `--manifest` is enabled, PoseTag appends one row per save to
`shots/manifest.csv`. The current manifest includes fields such as:

- `timestamp`
- `object_base`, `object_full`, `side`
- `face_yaml`
- `path_raw`, `path_ann`, `path_meta`
- `width`, `height`, `fx`, `fy`, `cx`, `cy`
- `detected_ids`, `expected_ids`
- `validation_ok`, `auto_face`

---

### 4) Derive annotation-ready 3D keypoints from meshes (once per object)

The canonical command for the annotation-ready mesh-keypoint stage is:

```bash
posetag-gen-keypoints --project_root my_project \
  --mesh meshes/connection_plate_white.obj
```

It computes the 8 AABB corners in the object frame, writes per-face corner
orderings, and can also write `object_config.yaml`. The output consumed by
annotation is:

```text
objects/<object>/keypoints.json
```

PoseTag infers known objects and faces from `boards/*.yaml`,
`boards/tag_registry.yaml`, and `shots/manifest.csv`. It groups board names
such as `<object>_sideA`, `<object>_front`, and `<object>_back` under the base
object. If the mesh filename matches a known object, the object name is filled
from the filename:

```bash
posetag-gen-keypoints --project_root my_project \
  --mesh meshes/connection_plate_white.obj
```

If the mesh filename differs from the PoseTag object identity, pass the mapping
explicitly:

```bash
posetag-gen-keypoints --project_root my_project \
  --mesh meshes/unity_export_plate_v12.obj \
  --object_name connection_plate_white
```

Expected layout:

```text
<project_root>/
  meshes/<object>.obj                 # supported annotation-ready mesh input
  objects/<object>/keypoints.json
  objects/<object>/object_config.yaml
  boards/<object>_sideA.yaml
  boards/tag_registry.yaml
  shots/manifest.csv
```

What it does:

- lists known objects and supported meshes with `--list`
- computes an axis-aligned bounding box in the object frame
- writes `keypoints.json` containing corner points plus face mappings
- includes face mappings for every inferred board/registry/manifest face key
- optionally writes `object_config.yaml` with mesh metadata and
  `T_mesh_object`
- refuses to overwrite an existing `keypoints.json` unless `--force` or
  `--keep-both` is supplied
- keeps the legacy OpenCV browser available through `--browse`

The annotation-ready generator currently supports `.obj` meshes. Coordinates
are saved in metres after applying `--units_to_m`. The default
`T_mesh_object` is identity, meaning mesh frame and object frame are the same.
Stage 6 requires finite mesh/keypoint coordinates and a `faces` entry for every
inferred or captured object face before annotation can proceed.

Legacy browser:

```bash
python3 -m gen_keypoints
```

Browser controls:

- `W/S` or arrow keys to navigate meshes
- `ENTER` or `G` to generate `keypoints.json`
- `V` to toggle preview style
- `Z` for Open3D preview
- `O` to toggle writing `object_config.yaml`
- `A` to toggle auto side mapping

Typical outputs:

- `objects/<object_name>/keypoints.json`
- `objects/<object_name>/object_config.yaml`

`canonical_keypoints/` outputs from `python3 -m generate_canonical_keypoints`
are separate sampled-point files and are not consumed by annotation. Use
`objects/<object>/keypoints.json` for the PoseTag annotation workflow.

The optional `posetag-gui` dashboard shows this as Stage 6 between face-shot
capture and annotation. It lists inferred objects, shows each expected
`meshes/<object>.obj` input and `objects/<object>/keypoints.json` output,
can import a selected `.obj` into the project mesh folder, and generates
keypoints for that object when `keypoints.json` is absent using the selected
units-to-metre scale. It can also remove the selected object's Stage 6 geometry
artifacts so the mesh/keypoints can be reuploaded.
Clicking an object shows an interactive OBJ viewport when a mesh is staged;
drag to rotate, secondary-drag to pan, and use the mouse wheel to zoom. The
stage remains missing until
`objects/<object>/keypoints.json` exists and validates.

Optional mesh visualisation:

```bash
python3 -m view_keypoints --project_root my_project \
  --object_name connection_plate_white --show_axes
```

Detailed Step 4 notes: `docs/workflows/step4_mesh_keypoints.md`.

---

### 5) Annotate each face shot to recover `T_board_object`

Click 4 corners; the tool detects tags to get `T_cam_board`, solves
`T_cam_object`, and then derives `T_board_object = inv(T_cam_board) @ T_cam_object`.
This step follows mesh-keypoint generation and expects
`objects/<object>/keypoints.json`. Face labels may be generic sides such as
`sideA` or named faces such as `front` and `back`; face-suffixed captures such
as `column_white_front` use `objects/column_white/keypoints.json`.

**Single shot**

```bash
posetag-annotate --shot /abs/path/to/shot_raw.png
```

**Batch across faces using the latest shot per face**

```bash
posetag-annotate --batch latest
posetag-annotate --batch latest --object-filter connection_plate_white
posetag-annotate --batch latest --force
```

**Browse picker**

```bash
posetag-annotate --browse
```

In the GUI, Stage 7 shows the expected face-transform queue, the current
annotation output count, dry-run/batch launch controls, and process logs. The
**Open Annotator** button launches the existing `posetag-annotate --browse`
workflow with `POSETAG_PROJECT` set for the selected project; corner clicking
still happens in the OpenCV annotator window.

**Dry run**

```bash
posetag-annotate --shot /abs/path/to/shot_raw.png --dry-run
posetag-annotate --batch latest --dry-run
```

Required inputs:

- `shots/manifest.csv` for batch and browse mode
- raw shot image, captured annotated/reference image, and metadata JSON from
  `posetag-capture-face`
- board YAML referenced by the shot metadata or manifest row
- camera intrinsics from metadata `camera` or manifest `fx`, `fy`, `cx`, `cy`
- `objects/<object>/keypoints.json` from `posetag-gen-keypoints`

**Outputs**

- YAML: `faces/<object>/<face-label>/<face_key>_T_board_object.yaml`, for
  example `faces/column_white/front/column_white_front_T_board_object.yaml`
- Review images next to the raw shot:
  `*_ann.png` and `*_reproj.png`
- Auto-maintained face manifest:
  `faces/face_manifest.csv`

Batch annotation uses `shots/manifest.csv` as its source of truth for captures
and automatically rebuilds `faces/face_manifest.csv` from the YAML outputs.
`--batch latest` selects the latest saved shot per object/side/board group;
`--dry-run` validates the selected images, metadata, board YAML, intrinsics,
optional calibration distortion YAML, and keypoints JSON before opening any UI.

Useful notes:

- `--check-tag-scale` reports inter-tag scale ratio `s`
- `--auto-correct-scale` divides `T_cam_board` translation by `s` when
  `|s-1| > --scale-tol`
- RMS reprojection error is shown during review and written into the YAML

Detailed Step 5 notes: `docs/workflows/step5_annotate_faces.md`.

---

### 6) Collect a ground-truth dataset (multi-object / multi-face)

Live RealSense, RealSense `.bag`, or any OpenCV-readable video. For each frame:

- detect tags once, even when faces use mixed tag sizes
- choose the best visible face per object
- compose `T_cam_object = T_cam_board @ T_board_object`
- review annotated, reprojection, and status panels before saving

**Live (RGB, optional aligned depth)**

```bash
posetag-collect --mode live --session run01 --calib calib_color.yaml --rs_w 640 --rs_h 480 --rs_fps 30
# add --save_depth to save aligned depth .npy per accepted frame
```

**Playback from `.bag`**

```bash
posetag-collect --mode bag --bag path/to/rec.bag --session run02 --calib calib_color.yaml
```

**Any video**

```bash
posetag-collect --mode video --video sample.mp4 --session run03 --calib calib_color.yaml
```

**Keys**

- `ENTER` / `y` / `s`: accept and save current view
- `SPACE` / `p`: pause or resume
- `r` / `n` / `BACKSPACE`: reject or skip frame
- `h`: toggle help
- `ESC` / `q`: abort current frame
- `Q` / `X`: quit all

Important: the stream resolution must match `calib_color.yaml`
(`image_width` / `image_height`). Use `--rs_w` / `--rs_h` or recalibrate.

**Outputs (per accepted frame)**

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

`session.yaml` stores session-wide metadata for the run, and
`<session>.jsonl` is a rolling capture log listing frame names, timestamps, and
saved objects.

**Per-frame annotation schema (abridged)**

```json
{
  "version": "1.0",
  "metadata": {
    "dataset_name": "<session>",
    "frame_index": 0,
    "timestamp": 1724631234.123,
    "tag_family": "tag36h11"
  },
  "image_filename": "Run_<tag>_rgb_frame_<idx>.png",
  "annotated_image": "Run_<tag>_annotated_frame_<idx>.png",
  "reproj_image": "Run_<tag>_reproj_frame_<idx>.png",
  "depth": "Run_<tag>_depth_frame_<idx>.npy",
  "camera_intrinsics": {
    "fx": 594.97,
    "fy": 601.72,
    "cx": 307.15,
    "cy": 267.01,
    "width": 640,
    "height": 480
  },
  "camera_extrinsics": {
    "position": [tx, ty, tz],
    "orientation": [x, y, z, w],
    "rpy_deg": [roll, pitch, yaw]
  },
  "objects": [
    {
      "class_id": 2,
      "class_name": "connection_plate",
      "2D_center": [cx, cy],
      "width": 0.25,
      "height": 0.18,
      "6DOF_pose": {
        "position": [tx, ty, tz],
        "orientation": [roll, pitch, yaw],
        "rotation": [x, y, z, w],
        "dimensions": [dx, dy, dz]
      }
    }
  ]
}
```

Notes on interpretation:

- `position` fields are in meters
- `camera_extrinsics.orientation` is a quaternion in `[x, y, z, w]` order
- `objects[].6DOF_pose.orientation` is roll, pitch, yaw in degrees
- `objects[].6DOF_pose.rotation` is a quaternion in `[x, y, z, w]` order
- 2D centers and box sizes are normalized by image width and height
- the raw image saved in `images/` is tag-covered to suppress AprilTags visually

Common options include:

- `--continuous`
- `--max_frames N`
- `--save_depth`
- `--axes {both,board,object,none}`
- `--bbox-mode {auto,face,object}`
- `--check-tag-scale`
- `--auto-correct-scale`

Selection notes:

- per object, the dataset stage keeps the best visible face using tag count
  first and area-based score as a tiebreaker
- when the origin tag is visible, it is preferred for `T_cam_board`
- otherwise PoseTag uses the visible common tag with the highest
  `decision_margin`

---

## Ground-truth composition

At capture time:

1. Detect AprilTags in the RGB frame.
2. Pick a visible tag `i` on the face.
3. Use the tag solver to compute `T_cam_tag_i` from camera intrinsics and the
   face’s `tag_size_m`.
4. Use the board YAML to get `T_board_tag_i`.
5. Compute `T_cam_board = T_cam_tag_i @ inv(T_board_tag_i)`.
6. Use annotation output to load `T_board_object` for that face.
7. Compose the final pose:

   ```text
   T_cam_object = T_cam_board @ T_board_object
   ```

Pose outputs can then include translation in meters, roll/pitch/yaw in degrees,
and quaternions in `xyzw` order depending on the stage.

---

## Block diagram

```mermaid
flowchart LR
  A["Calibrate camera (ChArUco)<br/>posetag-calib-charuco<br/>-> calib_color.yaml"]
  B["Build per-face boards<br/>posetag-make-board<br/>-> boards/*.yaml + boards/tag_registry.yaml"]
  C["Capture face shots<br/>posetag-capture-face<br/>-> raw/ann PNG + meta JSON"]
  D["Annotate faces<br/>posetag-annotate<br/>-> T_board_object per face"]
  E["Meshes (.obj)"]
  F["Generate mesh keypoints<br/>posetag-gen-keypoints<br/>-> objects/&lt;object&gt;/keypoints.json"]
  G["GT capture<br/>posetag-collect<br/>Live/Bag/Video; best face per object"]
  H["Detect tags + compute T_cam_board"]
  I["Compose T_cam_object = T_cam_board @ T_board_object<br/>-> save RGB / pose"]
  J["Train RGB-only pose net"]

  A --> B --> C --> F --> D --> G
  E --> F
  G --> H --> I --> J
```

---

## File layout

```text
<project_root>/
  boards/
    patterns/
    shots/
    tag_registry.yaml
    <object_face>.yaml
  calib/
    calib_color.yaml
  faces/
    <object>/sideA|B|C|D/<face_key>_T_board_object.yaml
    face_manifest.csv
  meshes/
    *.obj
  objects/
    <object_name>/
      keypoints.json
      object_config.yaml
  shots/
    <object_base>/side<SideLetter>/*_{raw,ann,meta}.png|json
    manifest.csv
  datasets/
    <session>/
      images/
      annotated/
      reproj/
      depth/
      annotations/
      session.yaml
      <session>.jsonl
  logs/
```

## Repository entry points

During Phase 0, the public CLI is PoseTag-first, but some utilities are still
module-invoked while packaging catches up. The current practical entry points
are:

- `posetag-gen-tags` -> AprilTag sheet generation
- `posetag-gen-charuco` -> ChArUco calibration board generation
- `posetag-calib-charuco` -> ChArUco colour calibration
- `posetag-make-board` -> per-face board YAML and tag registry creation
- `posetag-capture-face` -> face-shot capture
- `posetag-gen-keypoints` -> annotation-ready mesh keypoint generation
- `posetag-annotate` -> single, batch, and browse annotation flows
- `posetag-collect` -> dataset capture
- `posetag-gui` -> optional workflow dashboard for status, ChArUco board
  setup, guided calibration launch, object AprilTag generation, and
  board-building guidance
- `python3 -m gen_keypoints` -> legacy mesh-keypoint browser
- `python3 -m generate_canonical_keypoints` -> experimental sampled keypoint
  files under `canonical_keypoints/` (not annotation-ready)
- `python3 -m view_keypoints` -> mesh/keypoint visualisation

---

## Notes and tips

- **Resolution must match calibration**:
  if `calib_color.yaml` is 640×480, start live capture at 640×480.
- **Best face selection**:
  the capture pipeline prefers faces with more visible tags and then uses other
  ranking signals such as projected geometry.
- **Depth saving**:
  `--save_depth` writes aligned depth `.npy` arrays per accepted frame.
- **Pose fields**:
  translation is in meters; roll/pitch/yaw are in degrees; quaternions are
  emitted in documented order for that stage.
- **RealSense stability**:
  the repo limits thread counts and warms up capture in several paths to reduce
  driver hiccups.

## Migration

See [MIGRATION.md](MIGRATION.md) for the old-to-new naming map and the
compatibility aliases intentionally retained during Phase 0.
