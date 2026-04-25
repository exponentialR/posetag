# Step 1 Calibration Report

## Current Implementation Summary

Step 1 is exposed through the canonical PoseTag command:

```bash
posetag-calib-charuco
```

The command entry point is `posetag.cli.charuco:main`. It remains a thin
wrapper around the existing `utils.charuco_calibrate` implementation. That
legacy module still owns the interactive OpenCV capture loop and ChArUco
calibration solve. This is intentional for issue #28: no calibration maths,
coordinate conventions, units, or algorithms were changed.

Reusable validation, project IO, dictionary parsing, and YAML serialization
helpers now live in:

```text
src/posetag/pipelines/charuco_calibration.py
```

This keeps new testable behavior under the canonical `posetag` namespace while
leaving the legacy calibration loop in place as temporary technical debt.

## What Was Fixed

- `posetag-calib-charuco --help` can be exercised through the canonical CLI
  wrapper with explicit argv.
- `--source video` without `--video` fails before opening hardware or creating
  calibration output folders.
- Unreadable video paths fail before creating calibration output folders.
- Video EOF exits the capture loop with a clear message instead of hanging.
- Pressing `q` exits cleanly without solving or printing a traceback.
- `--source realsense` fails clearly when `pyrealsense2` is unavailable.
- Invalid ArUco dictionary names fail clearly and list supported dictionary
  names.
- The UI sample threshold now matches the solver threshold by default:
  `--min-corners` defaults to `4`, and smaller values are promoted to `4`.
- Direct checkout usage of `python utils/charuco_calibrate.py --help` is
  preserved as a temporary legacy-script compatibility path by adding both the
  repository root and `src/` to `sys.path` when needed.
- Project calibration IO preparation is reusable and hardware-free.
- Default latest output path is deterministic:

```text
<project_root>/calib/calib_color.yaml
```

- Each run creates:

```text
<project_root>/calib/images/set_XX/
<project_root>/calib/runs/<UTC-timestamp>/
```

- Explicit `--out` overrides the latest calibration output path.
- Calibration YAML schema generation and writing can be tested with synthetic
  calibration values.
- README Step 1 now reflects the canonical command and actual output layout.
- Added a dedicated workflow document:

```text
docs/workflows/step1_calibrate_charuco.md
```

## Commands Tested

Exact commands run during this validation:

```bash
python3 -m unittest tests.test_step1_charuco_calibration -v
python3 -m unittest discover -s tests -v
python3 -m pip install -e .
/Users/samueladebayo/Library/Python/3.11/bin/posetag-calib-charuco --help
/Users/samueladebayo/Library/Python/3.11/bin/posetag-calib-charuco --source video
/Users/samueladebayo/Library/Python/3.11/bin/posetag-calib-charuco --dict bogus
/Users/samueladebayo/Library/Python/3.11/bin/posetag-calib-charuco --source realsense
```

The installed script directory was not on this shell's `PATH`, so the installed
console script was invoked by absolute path.

Hardware-free tests added for:

- CLI help resolution.
- Dictionary parsing success and failure.
- Missing `--video` validation.
- Missing/unreadable video path validation before project artifacts are created.
- Video EOF behavior.
- Clean `q` exit behavior.
- Missing RealSense dependency validation.
- Legacy script `--help` resolution from a checkout.
- Project IO layout creation.
- Explicit `--out` override.
- Calibration YAML schema writing and `yaml.safe_load` round-trip.

Manual hardware-dependent validation remains required for actual webcam,
RealSense, and video calibration quality because this patch deliberately avoids
mocking the OpenCV solve path beyond YAML serialization.

## Outputs And Schema Verified

The generated calibration YAML round-trips with `yaml.safe_load` and contains:

- `image_width`
- `image_height`
- `camera_matrix`
- `distortion_coefficients`
- `reproj_rms`
- `model`
- `notes`

The schema remains compatible with the existing sample in the README:

```yaml
model: plumb_bob
notes: ChArUco 3x5, square=50.0mm, marker=37.0mm, dict=7X7_50
```

## Remaining Technical Debt

- The canonical CLI still delegates to `utils.charuco_calibrate`, a legacy
  top-level module.
- The interactive capture loop, OpenCV display code, and calibration solve are
  still coupled in that legacy module.
- Hardware-backed smoke tests are not automated; manual webcam, RealSense, and
  representative video validation should be recorded separately.
- The project still depends on OpenCV's installed ChArUco API shape, so
  cross-version manual checks are useful before a release.

## Next Recommended Workflow Step

Proceed to Step 2 validation only after a real calibration run has produced a
reasonable `reproj_rms`, and after confirming downstream capture will use the
same `image_width` and `image_height` recorded in `calib_color.yaml`.
