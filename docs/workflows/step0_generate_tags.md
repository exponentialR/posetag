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
