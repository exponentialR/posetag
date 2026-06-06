# Step 5: Annotate Faces / Board-To-Object Transforms

This workflow validates the PoseTag annotation command:

```bash
posetag-annotate
```

Annotation connects one captured tagged face to the full object coordinate
frame. It uses a saved face shot, the face board YAML, camera intrinsics, and
annotation-ready mesh keypoints to compute:

```text
T_board_object = inv(T_cam_board) @ T_cam_object
```

The runtime pose composition remains:

```text
T_cam_object = T_cam_board @ T_board_object
```

Step 5 follows mesh-keypoint generation. It expects
`objects/<object>/keypoints.json` from `posetag-gen-keypoints`; the sampled
files under `canonical_keypoints/` are not annotation inputs.

## Inputs

The default batch and browse modes read:

```text
<project_root>/shots/manifest.csv
```

Each selected manifest row must identify one saved face shot and normally
comes from `posetag-capture-face`.

Required per-shot inputs:

- raw shot image: `path_raw`, normally `*_raw.png`
- captured annotated/reference image: `path_ann`, normally `*_ann.png`
- metadata JSON: `path_meta`, normally `*_meta.json`
- board YAML: `face_yaml`
- camera intrinsics: `fx`, `fy`, `cx`, `cy` in metadata `camera`, or in the
  manifest row
- object keypoints: `objects/<object>/keypoints.json`

The metadata JSON should include:

```json
{
  "object_base": "connection_plate_white",
  "object_full": "connection_plate_white_sideA",
  "side": "A",
  "face_yaml": "boards/connection_plate_white_sideA.yaml",
  "image": {
    "path_raw": "shots/connection_plate_white/sideA/..._raw.png",
    "path_ann": "shots/connection_plate_white/sideA/..._ann.png",
    "path_meta": "shots/connection_plate_white/sideA/..._meta.json"
  },
  "camera": {
    "fx": 600.0,
    "fy": 610.0,
    "cx": 320.0,
    "cy": 240.0
  }
}
```

`--calib` is optional and supplies distortion coefficients for PnP/reprojection.
When provided, relative paths are resolved under the project root; a bare
`calib_color.yaml` also resolves to `calib/calib_color.yaml`.

The board YAML must contain `origin_id`, positive `tag_size_m`, and a non-empty
`tags` list with finite `id`, `cx`, `cy`, and optional `yaw_deg` values. The
origin tag must be listed in `tags`.

The keypoints JSON must contain a positive `units_to_m`, a non-empty `points`
mapping, and a `faces` mapping with the face key matching the board YAML stem,
for example:

```json
{
  "units_to_m": 1.0,
  "points": {
    "p0": [0.0, 0.0, 0.0],
    "p1": [0.1, 0.0, 0.0],
    "p2": [0.1, 0.1, 0.0],
    "p3": [0.0, 0.1, 0.0]
  },
  "faces": {
    "connection_plate_white_sideA": ["p0", "p1", "p2", "p3"]
  }
}
```

Face labels may be generic side labels such as `sideA` or named labels such as
`front` and `back`. If a captured shot stores the face in `object_base` or
`object_full`, for example `column_white_front`, annotation resolves the
keypoints under the base object directory:

```text
objects/column_white/keypoints.json
faces/column_white/front/column_white_front_T_board_object.yaml
```

The board YAML stem and keypoint face key remain `column_white_front`.

## Commands

Annotate one shot:

```bash
posetag-annotate --shot /abs/path/to/connection_plate_white_sideA_20260606_120000_raw.png
```

Validate one shot without opening the annotation UI:

```bash
posetag-annotate --shot /abs/path/to/connection_plate_white_sideA_20260606_120000_raw.png --dry-run
```

Batch annotate the latest saved shot for each object/side in
`shots/manifest.csv`:

```bash
posetag-annotate --batch latest
```

Useful batch filters:

```bash
posetag-annotate --batch latest --object-filter connection_plate_white
posetag-annotate --batch latest --side A
posetag-annotate --batch latest --force
```

Validate the batch plan without opening annotation windows:

```bash
posetag-annotate --batch latest --dry-run
```

Open the interactive browser:

```bash
posetag-annotate --browse
```

## Guided GUI Stage 7

The optional `posetag-gui` dashboard shows annotation as Stage 7. It displays
the expected face-transform queue, current `faces/face_manifest.csv` status,
annotation YAML counts, and process logs. The **Open Annotator** action launches
the existing `posetag-annotate --browse` workflow with `POSETAG_PROJECT` set to
the selected project. **Dry Run** executes `posetag-annotate --batch latest
--dry-run`, and **Run Batch** executes `posetag-annotate --batch latest`.

Stage 7 also exposes the annotator options that are otherwise keyboard toggles
inside the OpenCV window:

- corner mode: `quad` rectangle refinement or `any` four-corner clicking
- check tag scale
- auto-correct scale
- force overwrite

The **Remove Transform** action deletes only the selected
`*_T_board_object.yaml` outputs and refreshes `faces/face_manifest.csv`. Use
normal multi-select gestures, such as Shift-click or Command-click on macOS, to
remove several transforms at once. It does not delete board YAMLs, face shots,
object meshes, or keypoints.

The GUI does not reimplement corner selection or pose solving. Manual clicking,
PnP, and YAML writing remain in the existing OpenCV annotator so the annotation
math and output schema stay unchanged.

## Interaction

In annotation mode, PoseTag detects the AprilTags on the face to estimate
`T_cam_board`, then asks the user to click four face corners.

Controls:

- `ENTER`: accept the current corner selection or review result
- `J` / `K`, arrow keys, or `W` / `S`: move through the face browser queue
- `r`: reset / redo
- `u`: undo the last corner in click mode
- `q` or `ESC`: abort the current shot
- `Q` or `X`: quit all pending batch work

`--pts-type quad` lets the user drag a quadrilateral and adjust its corners.
`--pts-type any` lets the user click four corners directly.
To redo an already annotated face in the OpenCV browser, select the row and
press `ENTER`. Batch mode still skips existing annotation YAMLs unless
`--force` is supplied.

The face browser reports reprojection RMS in pixels. Low RMS indicates that
the clicked corners, object keypoints, board pose, and calibration agree well.
High RMS values should be treated as a review warning: redo the clicks, verify
the selected face/keypoints, and check calibration/tag-size consistency before
using the transform downstream.

## Outputs

For an object `connection_plate_white` and side `A`, PoseTag writes:

```text
faces/connection_plate_white/sideA/connection_plate_white_sideA_T_board_object.yaml
```

For a named face such as `column_white_front`, PoseTag writes:

```text
faces/column_white/front/column_white_front_T_board_object.yaml
```

The YAML schema is:

```yaml
object: connection_plate_white
face_key: connection_plate_white_sideA
board_yaml: /abs/path/to/boards/connection_plate_white_sideA.yaml
image: /abs/path/to/shots/connection_plate_white/sideA/..._raw.png
rms_px: 0.42
T_board_object:
  matrix:
    - [1.0, 0.0, 0.0, 0.02]
    - [0.0, 1.0, 0.0, 0.03]
    - [0.0, 0.0, 1.0, 0.04]
    - [0.0, 0.0, 0.0, 1.0]
corner_uv:
  p0: [10.0, 20.0]
pnp:
  rvec: [0.0, 0.0, 0.0]
  tvec: [0.0, 0.0, 1.0]
clicked_uv_raw:
  - [10.0, 20.0]
face_corner_names: [p0, p1, p2, p3]
assignment:
  p0: [10.0, 20.0]
notes: Accepted.
diagnostics:
  board_tag_size_mm: 80.0
  tag_scale_ratio: 1.0
  tag_scale_pairs: 1
  tag_scale_auto_corrected: false
```

PoseTag also writes review images next to the selected shot path:

```text
*_ann.png
*_reproj.png
```

Finally, it rebuilds:

```text
faces/face_manifest.csv
```

`faces/face_manifest.csv` is derived from
`faces/*/*/*_T_board_object.yaml`. If a YAML is deleted offline, the next
annotation sync removes that manifest row.

Manifest columns:

```text
timestamp, object, side, face_key, yaml_path, board_yaml, image, rms_px, tag_size_m
```

## Transform Convention

Annotation solves a camera-to-object pose from clicked 2D corners and the
object keypoints:

```text
T_cam_object
```

It estimates a camera-to-board pose from AprilTag detections and the board YAML:

```text
T_cam_board
```

The saved annotation transform is:

```text
T_board_object = inv(T_cam_board) @ T_cam_object
```

Downstream dataset collection must use:

```text
T_cam_object = T_cam_board @ T_board_object
```

Do not invert or reorder this composition unless the code, tests, docs,
examples, and output schema are updated together.

## Batch Behavior

`--batch latest` groups `shots/manifest.csv` rows by manifest object, side,
and board YAML stem, then selects the latest timestamp per group. Preflight
resolves face-suffixed labels such as `column_white_front` to the base object
`column_white` before checking `objects/<object>/keypoints.json`. Existing
annotation YAMLs are skipped unless `--force` is supplied.

`--dry-run` validates the selected raw image, metadata JSON, board YAML,
intrinsics, optional calibration distortion YAML, and keypoints JSON before
opening any UI. Dry-run exits non-zero if any selected row fails preflight.

Batch annotation is still interactive for each selected shot. Hardware-free
tests validate the batch planning and preflight path, but not the manual click
sequence.

## Common Failure Modes

- Missing `shots/manifest.csv` in batch/browse mode fails before annotation.
- Missing `path_raw` image fails as a missing shot image.
- Missing or malformed `path_meta` fails as a metadata JSON error.
- Missing `face_yaml` fails before tag detection.
- Missing or malformed board YAML fails before tag detection.
- Missing camera intrinsics fails before tag detection.
- Missing or malformed `objects/<object>/keypoints.json` fails before UI.
- Missing or malformed `--calib` fails before UI when the option is supplied.
- Unsupported batch mode values are rejected by CLI argument parsing.
- Running without `--shot`, `--batch latest`, or `--browse` fails with a mode
  selection error.

## Verify Before Dataset Collection

Before running `posetag-collect`:

1. Confirm every object/side needed for collection has a
   `*_T_board_object.yaml` file under `faces/`.
2. Confirm `faces/face_manifest.csv` lists the same YAML files.
3. Inspect `*_ann.png` and `*_reproj.png` for plausible corner assignment and
   low reprojection error.
4. Confirm the board YAML and keypoints JSON object/face names match.
5. Keep using `T_cam_object = T_cam_board @ T_board_object` downstream.
