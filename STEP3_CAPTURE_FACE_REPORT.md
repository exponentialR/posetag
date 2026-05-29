# Step 3 Capture-Face Validation Report

Date: 2026-05-29

Issue: #37, `[workflow] Validate Step 3: capture face shots`

Branch: `feature/37-capture-face-validation`

## Current Implementation Summary

`posetag-capture-face` is the canonical public command for Step 3. The entry
point remains a thin wrapper over the legacy top-level `src/capture_face.py`
interactive OpenCV viewer.

Step 3 consumes:

- Step 1 colour-camera calibration, usually `calib/calib_color.yaml`
- Step 2 board YAML files
- Step 2 tag registry, usually `boards/tag_registry.yaml`

Step 3 writes:

- raw face-shot PNGs
- annotated face-shot PNGs with detected tag overlays
- one metadata JSON file per saved shot
- a CSV manifest row per saved shot

The workflow records object/face identity and image provenance for later
annotation. It does not compute board-to-object transforms, dataset outputs, or
6-DoF poses.

## What Was Fixed

- `posetag-capture-face --help` now resolves from the canonical CLI wrapper
  without opening optional camera/tag dependencies.
- The canonical CLI wrapper accepts an optional `argv`, matching the tested
  pattern used by other PoseTag CLI wrappers.
- Missing `--video` with `--source video` fails clearly.
- Missing video paths fail before capture artifacts are created.
- Existing but unreadable video files fail before capture artifacts are
  created.
- Missing `pyrealsense2` for `--source realsense` fails clearly.
- Missing and malformed calibration YAML fail before camera/video preview.
- Missing, malformed, and empty tag registries fail before preview.
- Registry references to missing board YAML files fail before preview.
- Registry object/tag mismatches against board YAML files fail before preview.
- Unknown `--object_name` selections fail before preview.
- Video EOF exits cleanly instead of hanging.
- `ESC` exits cleanly without writing capture outputs.
- The optional `pyrealsense2` import in capture utilities no longer breaks
  fresh installs that do not include the RealSense extra.
- Pure validation, path, registered-face, metadata, and manifest helpers now
  live in `posetag.pipelines.capture_face`.

## Commands Tested

```bash
python3 -m unittest tests.test_step3_capture_face -v
```

Result:

```text
16 tests passed
```

```bash
python3 -m unittest discover -s tests -v
```

Result:

```text
218 tests passed
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

Result: editable install completed successfully for `posetag==0.1.0`.

```bash
$(python3 - <<'PY'
import site
from pathlib import Path
print(Path(site.getuserbase()) / 'bin' / 'posetag-capture-face')
PY
) --help
```

Result: installed `posetag-capture-face --help` resolved successfully.

## Outputs And Schema Verified

Hardware-free tests verify the default path layout:

```text
<project_root>/shots/<object_base>/side<SideLetter>/
  <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_raw.png
  <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_ann.png
  <object_base>_side<SideLetter>_<YYYYMMDD_HHMMSS>_meta.json
```

Metadata JSON tests verify:

- `object_base`
- `object_full`
- `side`
- `face_yaml`
- `expected_tag_ids`
- `detected_tag_ids`
- `validation_ok`
- `auto_face`
- `image.path_raw`
- `image.path_ann`
- `image.path_meta`
- `image.width`
- `image.height`
- `camera.fx`
- `camera.fy`
- `camera.cx`
- `camera.cy`
- `timestamp`

Manifest tests verify one CSV row per saved shot with:

- object/face identity
- raw, annotated, and metadata paths
- image size
- camera intrinsics
- detected and expected tag IDs
- validation and auto-face flags

The synthetic save test writes real small PNGs, metadata JSON, and
`shots/manifest.csv` under a temporary project root.

## Remaining Technical Debt

- The interactive viewer still lives in `src/capture_face.py`. This is an
  intentional compatibility choice for issue #37, not the final package-first
  architecture.
- A future architecture issue should move the full Step 3 viewer/runtime into
  `src/posetag/pipelines/` after workflow boundaries through annotation are
  better protected.
- The command still uses OpenCV HighGUI for the preview and object picker.
- Registry paths are still stored and emitted as string paths exactly as
  produced by the current implementation. Portable path policy should be
  handled in a later reproducibility/layout issue.

## Manual Hardware-Dependent Checks Still Needed

- Run `posetag-capture-face --source opencv --cam 0` with a real USB/laptop
  camera.
- Run `posetag-capture-face --source realsense` on a machine with
  `pyrealsense2` and a connected RealSense camera.
- Confirm live AprilTag overlays appear for the printed tag family.
- Confirm auto-side selection chooses the intended face when multiple faces of
  the same base object exist.
- Confirm the double-press force-save behavior is acceptable when expected tags
  are missing.
- Confirm saved raw images contain enough visual context for the later
  annotation step.
