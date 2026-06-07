# Step 6 Dataset Collection Validation Report

Issue: #39 `[workflow] Validate Step 5: dataset collection and pose outputs`

Branch: `feature/39-dataset-collection-validation`

## Purpose

This validation pass hardens `posetag-collect`, the workflow that records
ground-truth pose-labelled dataset frames after calibration, board building,
face-shot capture, mesh keypoint generation, and face annotation are complete.

The scientific pose invariant is preserved:

```text
T_cam_object = T_cam_board @ T_board_object
```

## What Changed

- Added `posetag.pipelines.collect_dataset` with hardware-free helpers for:
  - source argument validation
  - calibration, registry, face manifest, board YAML, and annotation preflight
  - dataset path resolution
  - session metadata construction
  - frame pose-record serialization
  - transform composition checks
  - quaternion `[x, y, z, w]` serialization
- Updated `posetag-collect` / `src/collect_gt_dataset.py` to:
  - support `main(argv)` through the canonical CLI wrapper
  - add `--project_root`, `--registry`, `--face_manifest`, and `--dry-run`
  - add `--mode opencv` for generic OpenCV webcam capture
  - keep `--mode live` as RealSense live, `--mode bag` as RealSense playback,
    and `--mode video` as OpenCV-readable video playback
  - validate required project inputs before opening capture sources
  - require calibration `image_width` and `image_height` for collection
  - reject missing video/bag arguments, unreadable videos, missing RealSense
    support, missing registries, missing manifests, missing board YAMLs, and
    annotation YAMLs missing `T_board_object.matrix`
  - include explicit object name, selected face, selected board, transforms,
    quality fields, units, and rotation ordering in per-frame JSON labels
  - add `--auto-capture` as a quality-gated smart collection policy over the
    existing pose results
  - keep manual review as a force-save path and keep `--continuous` as the
    legacy save-every-pose-labelled-frame mode
  - reject accepted saves when no object pose was estimated, so frame-only
    captures do not masquerade as pose-labelled dataset annotations
  - keep the existing interactive capture loop and best-face selection logic
- Added smart auto-capture helpers and tests for stability gates, cooldown,
  image-grid coverage, distance/pose diversity, candidate quality rejection,
  and pose-delta calculations.
- Updated README collection notes and added
  `docs/workflows/step6_collect_dataset.md`.
- Added hardware-free tests in `tests/test_step6_collect_dataset.py`.
- Wired Stage 8 project-status inspection so `posetag-gui` no longer reports
  collection as `not_applicable` after annotation is complete. A collection-
  ready project with no dataset sessions now reports Stage 8 as missing/ready
  to run, and malformed dataset pose records surface as needs-attention.
- Added `posetag.workflows.collect_dataset` with GUI-independent launch helpers
  for Stage 8 command construction, source/session validation, dry-run launch,
  collection launch, and process-state summaries.
- Added a Stage 8 dashboard action panel with session/source controls,
  smart auto-capture controls, **Dry Run**, **Start Collection**, process state,
  and stdout/stderr logging. The GUI launches the existing `posetag-collect`
  process and does not duplicate detection, pose estimation, transform
  composition, or dataset writing.
- Refined the runtime GT Capture review window while preserving capture and
  pose logic: annotation and reprojection panes now use clearer labels, and the
  right panel shows structured session metrics, selected object pose summaries,
  capture feedback, and keyboard controls instead of the older yellow text
  dump.
- Simplified GT Capture quit behavior: `ESC`, `q`, `Q`, `x`, and `X` now close
  the capture loop through normal cleanup, and the window close button is
  detected when OpenCV reports it.
- Updated the GUI command preview for collection to start with
  `posetag-collect --mode opencv --session SESSION --calib calib_color.yaml
  --auto-capture --dry-run`.

## Required Inputs

`posetag-collect --dry-run` now validates:

- `calib/calib_color.yaml`
- `boards/tag_registry.yaml`
- board YAMLs referenced by the face annotations and registry
- `faces/face_manifest.csv`
- annotation YAMLs listed by the face manifest
- `T_board_object.matrix` inside each annotation YAML

`objects/<object>/keypoints.json` remains optional for collection. Missing
keypoints are reported in dry-run/session metadata and trigger tag-derived
bounding-box fallback.

## Source Behavior

- `--mode opencv`: generic OpenCV webcam via `--cam`, `--width`, `--height`,
  and `--fps`.
- `--mode live`: Intel RealSense live colour stream, optional depth output
  with `--save_depth`.
- `--mode bag`: Intel RealSense `.bag` playback.
- `--mode video`: OpenCV-readable video file via `--video`.

`--mode live` and `--mode bag` fail clearly when `pyrealsense2` is unavailable.

## Output Schema

Each accepted frame writes:

- `datasets/<session>/images/*.png`
- `datasets/<session>/annotated/*.png`
- `datasets/<session>/reproj/*.png`
- optional `datasets/<session>/depth/*.npy`
- `datasets/<session>/annotations/*.json`
- `datasets/<session>/session.yaml`
- `datasets/<session>/<session>.jsonl`

Per-object annotation records now include:

- `object_name`
- `selected_face`
- `selected_board`
- normalized and pixel bounding boxes
- `transforms.T_cam_board`
- `transforms.T_board_object`
- `transforms.T_cam_object`
- `quality` fields such as tag count, selected tag, score, bbox source, and
  tag-scale diagnostics
- `6DOF_pose.position` in meters
- `6DOF_pose.orientation` as roll, pitch, yaw in degrees
- `6DOF_pose.rotation` as quaternion `[x, y, z, w]`

Per-frame records may include top-level `capture` provenance for the acceptance
policy (`manual_review`, `smart_auto`, or `continuous`). This is additive
metadata and does not change object pose semantics.

The serializer verifies that:

```text
T_cam_object == T_cam_board @ T_board_object
```

within numeric tolerance before writing an object pose record.

## Validation Run

Commands run:

```bash
python3 -m unittest tests.test_step6_collect_dataset -v
python3 -m unittest tests.test_step6_collect_dataset tests.test_workflow_status tests.test_workflow_gui -v
python3 -m unittest tests.test_workflow_gui tests.test_workflow_status tests.test_step6_collect_dataset -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q src utils tests
git diff --check
python3 -m posetag.cli.collect --help
/Users/samueladebayo/Library/Python/3.11/bin/posetag-collect --help
python3 -m posetag.cli.collect --project_root my_project --mode opencv --session run01 --dry-run --auto-capture
python3 -m posetag.cli.collect --project_root my_project --mode opencv --session run01 --dry-run --auto-capture --continuous
```

Focused collection tests:

```text
Ran 17 tests in 0.138s
OK
```

Focused Stage 8 / GUI / status tests:

```text
Ran 91 tests in 0.736s
OK
```

Full test suite:

```text
Ran 308 tests in 3.219s
OK
```

`compileall`, `git diff --check`, and both help smoke checks completed
successfully. The user-local `posetag-collect` script exists and resolves by
absolute path, but that script directory
(`/Users/samueladebayo/Library/Python/3.11/bin`) is not on this shell's `PATH`;
`python3 -m posetag.cli.collect --help` is the PATH-independent smoke check.

Local project dry-run smoke:

```text
Dataset collection dry run OK
annotations : 6
source mode : opencv
```

The intentionally invalid `--continuous --auto-capture` smoke exited with:

```text
--continuous and --auto-capture are mutually exclusive.
```

## Remaining Technical Debt

- The collector still lives mainly in `src/collect_gt_dataset.py`; the
  validation helpers are package-first, but the interactive capture loop was
  intentionally not migrated in this issue.
- `--allow-face-scan` preserves the legacy fallback that scans face YAMLs when
  `faces/face_manifest.csv` is absent. New reproducible projects should use
  the manifest.
- Bounding-box quality still depends on optional keypoint coverage when
  keypoints exist; missing keypoints fall back to tag-derived boxes.
- Smart auto-capture thresholds are conservative defaults validated with
  hardware-free policy tests; real-camera sessions should tune stability,
  cooldown, coverage, and pose-diversity thresholds.

## Manual Hardware Checks Still Needed

- Run `--mode opencv` with a real webcam at the calibration resolution.
- Run `--mode live` with an Intel RealSense device.
- Run `--mode bag` with a representative `.bag` recording.
- Run `--mode video` against a real experiment video with visible registered
  faces.
- Run `--auto-capture` on real OpenCV/RealSense/video sources and confirm it
  saves stable, diverse views without flooding near-duplicates.
- Confirm saved annotated/reprojection panels show plausible selected faces,
  axes, and bounding boxes for real hardware captures.
