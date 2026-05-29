# Step 0: Generate Printable AprilTag Sheets

This workflow covers the first user-visible PoseTag step:

```bash
posetag-gen-tags --project_root <path> --tag-size-mm <N> --ids <ids>
```

## What The Command Does

- Resolves or initializes a PoseTag project when `--project_root` is supplied.
- Writes printable AprilTag 36h11 sheets.
- Writes PNG and PDF sheets in standard PoseTag installs.
- Splits output across multiple sheets when the requested IDs do not fit on one page.

## Inputs

- `--project_root <path>`
  Initializes or reuses the PoseTag project root.
- `--tag-size-mm <N>`
  Physical tag side length in millimetres.
- `--ids <ids>`
  AprilTag IDs as:
  - comma list: `1,2,3`
  - range: `1-4`
  - mixed: `1-3,7,9-10`
- `--dpi <N>`
  Output rendering DPI. Example: `150` for quick checks, higher for print.
- `--out_dir <path>`
  Optional explicit output directory override. When supplied with
  `--project_root`, this path is still honored instead of the project default.

## Output Locations

With `--project_root` and no `--out_dir`, outputs go to:

```text
<project_root>/boards/patterns/
```

If `--out_dir` is supplied, outputs go there instead.

Project initialization for this command creates or reuses:

```text
<project_root>/
  boards/
  shots/
  objects/
  datasets/
```

The `boards/patterns/` directory is created on demand when tags are written.

## Command Examples

Minimal example without project initialization:

```bash
posetag-gen-tags --tag-size-mm 40 --ids 1-4 --out_dir apriltags_out
```

Project-root workflow:

```bash
posetag-gen-tags --project_root my_project --tag-size-mm 40 --ids 1-4 --dpi 150
```

Explicit output override:

```bash
posetag-gen-tags --project_root my_project --tag-size-mm 40 --ids 1-3,7,9-10 --out_dir exported_patterns
```

## Guided GUI Stage 3 Flow

The calibration-first `posetag-gui` workflow presents this command as Stage 3:
Generate Object AprilTags. The guided panel is available after Stage 2 camera
calibration is complete and delegates generation to the same
`posetag-gen-tags` package workflow.

Supported guided controls mirror the existing generator:

- AprilTag family: currently `tag36h11`.
- Physical tag size in millimetres.
- ID list/ranges such as `1-4` or `1-3,7,9-10`.
- Start ID plus ID count, converted to the existing `--id_start` /
  `--id_end` CLI arguments.
- Paper preset or custom `WxH` millimetre size.
- Orientation, DPI, prefix, margin fraction, label gap fraction, Pillow text
  labels, and output folder.

With the default output folder, generated PNG sheets are written under:

```text
<project_root>/boards/patterns/
```

The dashboard previews the first generated PNG sheet, shows generated output
paths, keeps a copyable command preview, and refreshes workflow status after
generation. If you choose a custom output folder outside
`<project_root>/boards/patterns/`, the files are still written there, but the
current project status check looks for PNG sheets in the default project
folder.

## Outputs

- PNG output.
  PNG files include the requested DPI metadata.
- PDF output.
- Multiple pages: when the requested IDs exceed one sheet, PoseTag writes
  deterministic page-numbered outputs such as `page01of03`, `page02of03`, and
  `page03of03`.

## Failure Modes

The command should fail clearly for:

- malformed `--ids` values such as `1,a,3`
- descending ranges such as `4-1`
- tag IDs outside the AprilTag 36h11 dictionary
- missing required ID inputs
- non-positive `--tag-size-mm` or `--dpi` values
- invalid paper size strings such as `--paper-mm bad`
- invalid margin or label-gap fractions
- impossible page configurations where even one tag cannot fit on the selected page

## Printing Guidance

- Print at `100%` or `Actual size`.
- Do not use “Fit to page”.
- Verify the black square edge of a printed tag matches `--tag-size-mm`.
- Re-check printer scaling whenever changing paper preset, DPI, or printer driver.
