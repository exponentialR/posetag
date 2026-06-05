# Step 4 Mesh-Keypoint Validation Report

Date: 2026-06-04

Issue: #70, `[workflow] Validate mesh keypoint generation before annotation`

Branch: `feature/70-mesh-keypoint-validation`

## Current Implementation Summary

`posetag-gen-keypoints` is the canonical public command for generating
annotation-ready mesh keypoints. It wraps the existing `gen_keypoints` behavior
and writes:

```text
objects/<object>/keypoints.json
```

The legacy interactive OpenCV browser remains available through:

```text
python3 -m gen_keypoints
posetag-gen-keypoints --browse
```

The annotation-ready generator currently supports `.obj` meshes through the
existing lightweight OBJ vertex loader. The newer
`generate_canonical_keypoints` utility remains separate: it writes sampled
point files under `canonical_keypoints/`, which are not consumed by annotation.

## What Was Improved

- Added the `posetag-gen-keypoints` entry point.
- Added `--project_root`, `--mesh`, `--object_name`, `--units_to_m`,
  `--force`, `--keep-both`, `--list`, and `--browse` behavior to
  `gen_keypoints`.
- Added non-interactive generation for one mesh, so hardware-free tests can
  validate `objects/<object>/keypoints.json` without opening the OpenCV UI.
- Added object identity helpers that infer known objects/faces from:
  - `boards/*.yaml`
  - `boards/tag_registry.yaml`
  - `shots/manifest.csv`
- Grouped common object-face board suffixes such as `_front`, `_back`,
  `_left`, `_right`, `_top`, and `_bottom` under the base PoseTag object while
  preserving the existing `_sideA`, `_sideB`, ... convention.
- Validated that annotation-ready `faces` mappings cover every inferred or
  captured object face before Stage 6 can complete.
- Rejected non-finite `units_to_m`, OBJ vertices, transformed bounds, and
  keypoint coordinates so `NaN`/`Infinity` cannot enter annotation geometry.
- Added safe mesh filename autofill when the mesh stem matches a known object
  or when no prior object identities exist.
- Added explicit object mapping for mismatched mesh filenames via
  `--object_name`.
- Restricted annotation-ready mesh validation to `.obj`.
- Restricted the separate canonical sampler scan list to `.obj`, `.ply`, and
  `.stl`, matching its documented scope.
- Improved `view_keypoints --help` so it does not require Open3D until an
  actual visualization is requested.
- Improved `generate_canonical_keypoints --help` / project-root parsing so
  help does not resolve or create a project first.
- Added `posetag-gui` / project-status visibility for mesh keypoints as Stage
  6, before annotation.
- Improved the Stage 6 dashboard command preview so known projects suggest a
  concrete first missing object, for example
  `meshes/column_white.obj --object_name column_white`, instead of only
  placeholder values.
- Added a Stage 6 object-geometry panel that lists inferred objects, expected
  `meshes/<object>.obj` inputs, expected
  `objects/<object>/keypoints.json` outputs, an `Import OBJ` action for the
  selected object, and per-object / batch copy-command actions.
- Added an interactive dashboard OBJ preview for the selected Stage 6 object,
  with mouse rotate, pan, zoom, and reset behavior when the mesh is staged.
- Clarified the dashboard status split so a staged OBJ mesh does not mark the
  stage complete until `objects/<object>/keypoints.json` exists and validates.
- Documented Step 4 in `docs/workflows/step4_mesh_keypoints.md`.
- Updated README and Step 3 docs so mesh keypoints appear before annotation.
- Updated local `AGENTS.md` durable notes and `TASKS.md` checklist for issue
  #70.

## Commands Tested

```bash
python3 -m unittest tests.test_step4_mesh_keypoints -v
```

Result:

```text
12 tests passed
```

```bash
python3 -m unittest discover -s tests -v
```

Result:

```text
258 tests passed
```

```bash
python3 -m unittest tests.test_workflow_status tests.test_workflow_gui tests.test_step4_mesh_keypoints -v
```

Result:

```text
67 tests passed
```

```bash
python3 -m compileall -q src utils tests
```

Result: passed with no output.

```bash
git diff --check
```

Result: passed with no output.

```bash
python3 -m pip install -e .
```

Result: editable install completed successfully for `posetag==0.1.0`, and pip
reported the new `posetag-gen-keypoints` script.

```bash
$(python3 - <<'PY'
import site
from pathlib import Path
print(Path(site.getuserbase()) / 'bin' / 'posetag-gen-keypoints')
PY
) --help
```

Result: installed `posetag-gen-keypoints --help` resolved successfully.

Additional help checks:

```bash
python3 -m gen_keypoints --help
python3 -m view_keypoints --help
python3 -m generate_canonical_keypoints --help
python3 -m posetag.cli.gen_keypoints --help
```

Result: all resolved successfully without opening viewers.

Additional dashboard-model check:

```bash
python3 - <<'PY'
from pathlib import Path
from posetag.gui.models import inspect_project_view
root = Path("my_project")
stage6 = {model.stage_id: model for model in inspect_project_view(root)}[6]
print(stage6.message)
print(stage6.checked_paths)
print(stage6.command_preview)
PY
```

Result: the local project inferred two objects, `column_white` and
`connection_plate_white`, checked their annotation-ready
`objects/<object>/keypoints.json` paths, and previewed:

```text
posetag-gen-keypoints --project_root my_project --mesh my_project/meshes/column_white.obj --object_name column_white
```

## Outputs And Schema Verified

Hardware-free tests verify:

- `meshes/<object>.obj` writes
  `objects/<object>/keypoints.json`.
- explicit mesh/object mapping writes
  `objects/<object>/keypoints.json` even when the mesh filename differs.
- `object_config.yaml` records:
  - mesh path
  - `units_to_m`
  - identity `T_mesh_object` by default
- `keypoints.json` contains:
  - finite positive `units_to_m`
  - eight AABB corner points
  - `faces` mappings for `<object>_sideA` through `<object>_sideD`
  - matching `<object>_<face>` mappings for inferred/captured aliases such as
    `front` and `back`
  - exactly four point names per face mapping
- Stage 6 refuses existing keypoint JSON that omits any inferred/captured face
  key.
- OBJ and JSON geometry containing `NaN` or `Infinity` is rejected.
- existing `keypoints.json` requires `--force` or `--keep-both`.
- unsupported mesh extensions fail clearly.
- missing mesh paths fail clearly.
- known object identities are inferred from boards, registry, and face-shot
  manifest rows.
- common physical face suffixes such as `front` and `back` are grouped under
  the base object identity.
- dashboard object rows expose the expected OBJ import path and matching
  `posetag-gen-keypoints` command.
- dashboard object selection can load OBJ vertices for a display-only mesh
  preview without requiring Open3D.

## Remaining Technical Debt

- The full interactive OpenCV browser still lives in top-level
  `src/gen_keypoints.py`. This branch adds testable non-interactive behavior
  but does not migrate the whole UI into `src/posetag/pipelines/`.
- The annotation-ready generator still uses an AABB corner model. Non-box-like
  objects may require a later, deliberately designed keypoint schema or mesh
  alignment workflow.
- Non-identity `T_mesh_object` behavior is preserved and documented, but this
  branch does not add a new editor or calibration workflow for estimating that
  transform.
- The GUI now shows mesh keypoints as a guided OBJ staging, preview, and
  command-preview stage, but a native 3D mesh editor or in-GUI keypoint
  generation remains future work.
- `canonical_keypoints/` sampled outputs remain separate legacy/experimental
  artifacts and are not migrated into annotation.

## Manual / Interactive Checks Still Needed

- Run `posetag-gen-keypoints --browse` on a real project with meshes and
  inspect the OpenCV browser.
- Run `posetag-gen-keypoints --mesh ...` on the maintainer's real object mesh
  and verify the saved AABB faces match the intended physical sides.
- Run `python3 -m view_keypoints --project_root <project> --object_name <object>`
  with Open3D installed to inspect the mesh, keypoints, and face quads.
- Confirm downstream annotation selects the expected face key from
  `objects/<object>/keypoints.json` on real face shots.
