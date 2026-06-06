# Step 5 Annotation Validation Report

Date: 2026-06-06

Branch: `feature/38-annotation-validation`

Issue: #38 `[workflow] Validate Step 4: face annotation and board-to-object transforms`

## Scope

This pass validates `posetag-annotate` after the mesh-keypoint stage. In the
current PoseTag workflow order, annotation follows:

1. ChArUco calibration
2. board YAML / tag registry creation
3. face-shot capture
4. annotation-ready mesh keypoints at `objects/<object>/keypoints.json`
5. face annotation / `T_board_object`

No dataset collection behavior was implemented or changed.

## Scientific Contract

The annotation transform convention is unchanged:

```text
T_board_object = inv(T_cam_board) @ T_cam_object
```

Runtime composition remains:

```text
T_cam_object = T_cam_board @ T_board_object
```

Hardware-free tests verify that a synthetic `T_cam_object` composed from
`T_cam_board @ T_board_object` recovers the original `T_board_object`.

## Inputs Validated

`posetag-annotate` now has a preflight path for single-shot and batch dry-run
use. It validates:

- selected raw shot path
- captured annotated/reference shot path
- shot metadata JSON
- face board YAML
- metadata/manifest camera intrinsics
- optional calibration distortion YAML supplied with `--calib`
- annotation-ready keypoints at `objects/<object>/keypoints.json`
- selected batch rows from `shots/manifest.csv`

The preflight runs before AprilTag detection or OpenCV annotation windows.

## Outputs Validated

Tests cover the annotation YAML record shape, including:

- `object`
- `face_key`
- `board_yaml`
- `image`
- `rms_px`
- `T_board_object.matrix`
- `corner_uv`
- `pnp`
- `clicked_uv_raw`
- `face_corner_names`
- `assignment`
- `diagnostics.board_tag_size_mm`
- `diagnostics.tag_scale_ratio`
- `diagnostics.tag_scale_pairs`
- `diagnostics.tag_scale_auto_corrected`

Tests also cover `faces/face_manifest.csv` rebuild behavior. The manifest is
derived from `faces/*/*/*_T_board_object.yaml`, includes `tag_size_m`, and
prunes rows when YAML files are deleted.

## Implementation Notes

- `posetag-annotate --help` works through the editable-install console script
  and `python -m posetag.cli.annotate --help`.
- The packaged CLI wrapper now accepts `main(argv)` like the other PoseTag
  command wrappers.
- `pupil-apriltags` is loaded lazily, so help and dry-run do not require the
  optional AprilTag detector dependency.
- Board YAML loading now reports clear missing/malformed schema errors.
- `--dry-run` validates selected single-shot or batch inputs and exits before
  UI/detection.
- Batch support remains the existing interactive `--batch latest` flow:
  latest shot per object/side/board group, optional `--object-filter`,
  `--side`, and `--force`.
- Face-suffixed captures such as `column_white_front` or `column_white_back`
  now resolve to base-object keypoints at `objects/column_white/keypoints.json`
  while preserving the board/keypoint face key and writing outputs such as
  `faces/column_white/front/column_white_front_T_board_object.yaml`.
- The GUI-independent workflow status helpers now inspect Stage 7 annotation
  YAMLs and `faces/face_manifest.csv` instead of returning a placeholder.
- Stage 6 dashboard support now generates mesh keypoints after OBJ import when
  the selected object has no canonical keypoint file. The guarded
  "Generate Mesh Keypoints" fallback writes absent
  `objects/<object>/keypoints.json` files with an explicit units-to-metre scale
  and refuses to overwrite existing or invalid keypoint JSON. Stage 6 can also
  remove the selected object's OBJ/keypoint/config artifacts without deleting
  upstream boards, shots, or registry entries.
- Stage 7 dashboard support now displays the annotation queue, expected
  transform YAML counts, readiness messages, and annotator process logs. It can
  launch `posetag-annotate --browse`, `--batch latest`, and `--batch latest
  --dry-run` with `POSETAG_PROJECT` set for the selected project. The OpenCV
  annotator remains the annotation engine.
- Stage 7 now exposes the annotator's corner mode, tag-scale check,
  auto-correct scale, and force-overwrite options in the dashboard. These
  controls are propagated into launched and copied commands.
- Stage 7 can remove one or more selected board-to-object transform YAMLs and
  refresh `faces/face_manifest.csv` without deleting board YAMLs, face shots,
  meshes, or keypoints. The removal helper accepts absolute paths, project-root
  relative paths, and dashboard display paths such as `my_project/faces/...`.
- The OpenCV annotation browser now uses a compact face table with RMS severity
  chips, a lighter selection/options panel, and reliable `J/K`, arrow, and
  `W/S` navigation. High RMS values are surfaced as a fit-quality warning, not
  as a neutral status. Annotated rows can be redone directly by pressing
  `ENTER` on the selected row; batch mode still requires `--force` before
  replacing existing YAMLs.

## Documentation Updated

- README annotation workflow summary
- `docs/workflows/step5_annotate_faces.md`

The docs now place annotation after mesh keypoints and describe required
inputs, outputs, batch behavior, failure modes, and transform composition.

## Validation Commands

Validation run during this pass:

```bash
python3 -m unittest tests.test_step5_annotate_faces -v
python3 -m unittest tests.test_step4_mesh_keypoints -v
python3 -m unittest tests.test_workflow_gui -v
python3 -m unittest tests.test_workflow_status -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q src utils tests
git diff --check
python3 -m posetag.cli.annotate --help
.venv/bin/posetag-annotate --help
.venv/bin/python -m posetag.gui.app --help
.venv/bin/posetag-gen-keypoints --help
env POSETAG_PROJECT=my_project .venv/bin/posetag-annotate --batch latest --dry-run
env POSETAG_PROJECT=my_project .venv/bin/posetag-annotate --batch latest --dry-run --pts-type any --check-tag-scale
```

All commands passed on 2026-06-06. The full suite currently runs 280 tests.

An initial targeted compile check also passed while implementing the CLI and
preflight changes:

```bash
python3 -m compileall -q src/annotate_shots.py utils/annotation_utils.py src/posetag/cli/annotate.py
```

## Remaining Technical Debt

- The interactive annotation UI and solver still live in the legacy
  top-level `annotate_shots.py` module.
- Hardware-free tests do not simulate AprilTag pose detection or manual corner
  clicking.
- `*_ann.png` is used as the annotation review output path, which may overwrite
  the captured reference annotation image from Step 3.
- Batch mode is validated for planning/preflight, but the per-shot click loop
  remains manual.
- The Stage 7 GUI now launches and tracks the existing annotator, but a native
  Qt annotation canvas remains future work.
- Dataset and export status remain placeholders until their workflow validation
  issues are implemented.

## Manual Interactive Checks Still Needed

Before merging a release-quality annotation validation PR, run at least one
real or recorded project through:

```bash
posetag-annotate --batch latest --dry-run
posetag-annotate --shot /abs/path/to/<face>_raw.png
posetag-annotate --batch latest
```

Inspect the saved YAML, `*_ann.png`, `*_reproj.png`, and
`faces/face_manifest.csv`. Confirm the clicked corner order, reprojection RMS,
and downstream composition remain physically plausible.
