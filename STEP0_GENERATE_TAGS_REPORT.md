# STEP0 Generate Tags Report

## Current Implementation Summary

- Canonical CLI entry point: `posetag-gen-tags`
- Canonical implementation now lives in `src/posetag/pipelines/generate_tags.py`
- `src/posetag/cli/gen_tags.py` is a thin wrapper
- Legacy `src/gen_april_tags.py` remains as a temporary compatibility shim to the
  canonical implementation

## What Was Fixed

- Fixed fresh-project initialization by making config writes robust in
  `utils/project_config.py`
- Moved Step 0 generation logic onto a canonical `posetag` import path
- Kept the legacy top-level module as an explicit compatibility wrapper
- Standardized Step 0 help text and examples around `posetag-gen-tags`
- Fixed printable-sheet labels so they fit within each tag cell instead of
  overlapping neighbouring tags
- Replaced single-page truncation with deterministic multi-page output so all
  requested IDs are emitted
- Added hardware-free tests for ID parsing, project-root behavior, output
  placement, error handling, label fitting, pagination, and CLI smoke coverage
- Added focused Step 0 workflow documentation

## Commands Tested

- `python3 -m pip install -e .`
- `/Users/samueladebayo/Library/Python/3.11/bin/posetag-gen-tags --help`
- `/Users/samueladebayo/Library/Python/3.11/bin/posetag-gen-tags --project_root <tmp> --tag-size-mm 40 --ids 1-4 --dpi 150`
- `/Users/samueladebayo/Library/Python/3.11/bin/posetag-gen-tags --project_root <tmp> --tag-size-mm 40 --paper-mm 80x80 --ids 1-3 --dpi 100`
- `python3 -m unittest tests.test_step0_generate_tags`
- `python3 -m unittest tests.test_rename_phase0`

## Outputs Verified

- `boards/` is created for a new project root
- `boards/patterns/` is created when using default project-root output
- PNG output is written successfully
- Generated PNG opens with OpenCV and has non-zero dimensions
- Generated labels are compact and fit within their grid cells
- More requested IDs than one page can hold are split across page-numbered PNGs
- PDF remains optional and depends on Pillow availability

## Remaining Technical Debt

- The legacy top-level `src/gen_april_tags.py` module still exists as a
  compatibility shim and should be removed only after legacy callers are no
  longer needed
- Multi-page output currently writes one PDF per generated sheet when Pillow is
  installed; a future enhancement could also produce a single bundled PDF

## Next Recommended Workflow Step

Step 1: verify the ChArUco calibration command end-to-end from editable install,
with the same focus on one working user path, explicit output locations, tests,
and docs that match the actual CLI behavior.
