"""Generate printable AprilTag 36h11 sheets for PoseTag Step 0.

Outputs
-------
- PNG is always written.
- PDF is additionally written when Pillow is installed.

Project behavior
----------------
- With ``--project_root``, PoseTag initializes the standard project layout and,
  unless ``--out_dir`` is explicitly set, writes into
  ``<project_root>/boards/patterns/``.
- Without ``--project_root``, outputs default to ``apriltags_out/``.

Printing
--------
Print at 100% / Actual size so the black square edge matches ``--tag-size-mm``.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import textwrap
import zlib
from pathlib import Path

import cv2
import numpy as np

from posetag.utils.project_config import ensure_project_dirs, resolve_project_root

try:
    from PIL import Image, ImageDraw, ImageFont

    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


PAPER_MM = {
    "A4": (210.0, 297.0),
    "LETTER": (215.9, 279.4),
    "LEGAL": (215.9, 355.6),
}
APRILTAG_36H11_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
MARKER_BORDER_BITS = 1

DEFAULT_OUT_DIR = "apriltags_out"
DEFAULT_PREFIX = "apriltag_36h11"
DEFAULT_DPI = 600
DEFAULT_MARGIN_FRAC = 0.05
DEFAULT_TOP_GAP_FRAC = 0.08
DEFAULT_LABEL_GAP_FRAC = 0.05
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1
LABEL_MAX_FRAC = 0.9


class TagGenerationError(ValueError):
    """User-facing validation error for tag sheet generation."""


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show defaults and preserve help formatting."""


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def make_tag_bitmap(tag_id: int, side_mm: float, dpi: int) -> np.ndarray:
    side_px = mm_to_px(side_mm, dpi)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(
            APRILTAG_36H11_DICT, tag_id, side_px, MARKER_BORDER_BITS
        )
    return cv2.aruco.drawMarker(APRILTAG_36H11_DICT, tag_id, side_px)


def estimate_label_size_cv(
    text: str,
    font_scale: float = LABEL_FONT_SCALE,
    thickness: int = LABEL_THICKNESS,
) -> tuple[int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return text_w, text_h, baseline


def _fit_cv_text(text: str, max_width_px: int) -> tuple[float, int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (base_w, base_h), base_line = cv2.getTextSize(text, font, 1.0, 1)
    if base_w <= 0:
        return LABEL_FONT_SCALE, LABEL_THICKNESS, 0, 0, 0
    font_scale = min(LABEL_FONT_SCALE, max_width_px / base_w)
    font_scale = max(0.25, font_scale)
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return font_scale, thickness, text_w, text_h, baseline


def draw_label_cv(
    img: np.ndarray,
    xc: int,
    y_top: int,
    text: str,
    max_width_px: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, text_w, text_h, _ = _fit_cv_text(text, max_width_px)
    origin = (int(xc - text_w / 2), int(y_top + text_h))
    cv2.putText(
        img,
        text,
        origin,
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_label_pil(
    img_rgb: np.ndarray,
    xc: int,
    y_top: int,
    text: str,
    max_width_px: int,
) -> np.ndarray:
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_size = 24
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        while font_size > 10 and draw.textlength(text, font=font) > max_width_px:
            font_size -= 1
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()
    text_w = draw.textlength(text, font=font)
    draw.text((int(xc - text_w / 2), int(y_top)), text, fill=(0, 0, 0), font=font)
    return np.array(pil_img)


def compute_grid(
    width_px: int,
    height_px: int,
    margin_px: int,
    tag_px: int,
    label_gap_px: int,
    use_pil_text: bool,
) -> tuple[int, int, int]:
    top_gap = int(DEFAULT_TOP_GAP_FRAC * (height_px - 2 * margin_px) / 3)
    side_pad = max(8, int(0.02 * tag_px))
    bottom_pad = max(8, int(0.02 * tag_px))

    sample_text = "ID 9999 | 300 mm"
    if use_pil_text and HAVE_PIL:
        label_h = 30
    else:
        _, text_h, baseline = estimate_label_size_cv(sample_text)
        label_h = text_h + baseline

    cell_w = tag_px + 2 * side_pad
    cell_h = top_gap + tag_px + label_gap_px + label_h + bottom_pad

    cols = max(0, (width_px - 2 * margin_px) // cell_w)
    rows = max(0, (height_px - 2 * margin_px) // cell_h)
    return cols, rows, top_gap


def make_canvas(paper_w_mm: float, paper_h_mm: float, dpi: int) -> tuple[np.ndarray, int, int]:
    width_px = mm_to_px(paper_w_mm, dpi)
    height_px = mm_to_px(paper_h_mm, dpi)
    return np.full((height_px, width_px, 3), 255, dtype=np.uint8), width_px, height_px


def write_png_with_dpi(path: Path, image: np.ndarray, dpi: int) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise TagGenerationError(f"Could not encode PNG output: {path}")

    png_bytes = encoded.tobytes()
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise TagGenerationError(f"OpenCV did not produce a valid PNG stream: {path}")

    pixels_per_metre = int(round(dpi / 0.0254))
    data = struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1)
    chunk_type = b"pHYs"
    chunk = (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )

    first_chunk_length = struct.unpack(">I", png_bytes[8:12])[0]
    first_chunk_end = 8 + 4 + 4 + first_chunk_length + 4
    path.write_bytes(png_bytes[:first_chunk_end] + chunk + png_bytes[first_chunk_end:])


def parse_paper(paper: str, custom_mm: str | None, orientation: str) -> tuple[float, float]:
    if custom_mm:
        try:
            w_mm, h_mm = [
                float(x) for x in custom_mm.lower().replace("mm", "").split("x", 1)
            ]
        except Exception as exc:
            raise TagGenerationError(
                "Bad --paper-mm format. Use like: --paper-mm 210x297"
            ) from exc
    else:
        key = paper.upper()
        if key not in PAPER_MM:
            raise TagGenerationError(
                f"Unknown paper '{paper}'. Choose from: {', '.join(PAPER_MM.keys())} or use --paper-mm WxH."
            )
        w_mm, h_mm = PAPER_MM[key]

    if not (math.isfinite(w_mm) and math.isfinite(h_mm)) or w_mm <= 0 or h_mm <= 0:
        raise TagGenerationError("Paper dimensions must be positive finite millimetre values.")

    if orientation.lower() == "landscape":
        return max(w_mm, h_mm), min(w_mm, h_mm)
    return min(w_mm, h_mm), max(w_mm, h_mm)


def build_id_list(start_id: int, end_id: int) -> list[int]:
    if end_id < start_id:
        raise TagGenerationError("--id_end must be >= --id_start.")
    return list(range(start_id, end_id + 1))


def parse_ids_string(ids_str: str) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for token in (part.strip() for part in ids_str.split(",") if part.strip()):
        if "-" in token:
            try:
                start_text, end_text = token.split("-", 1)
                start_id = int(start_text)
                end_id = int(end_text)
            except ValueError as exc:
                raise TagGenerationError(
                    "Bad --ids format. Use integers and hyphen ranges, e.g. '1,2,3' or '1-4' or '1-3,7,9-10'."
                ) from exc
            if end_id < start_id:
                raise TagGenerationError(f"Bad range '{token}': end < start.")
            for tag_id in range(start_id, end_id + 1):
                if tag_id not in seen:
                    seen.add(tag_id)
                    ids.append(tag_id)
            continue

        try:
            tag_id = int(token)
        except ValueError as exc:
            raise TagGenerationError(
                "Bad --ids format. Use integers and hyphen ranges, e.g. '1,2,3' or '1-4' or '1-3,7,9-10'."
            ) from exc
        if tag_id not in seen:
            seen.add(tag_id)
            ids.append(tag_id)

    if not ids:
        raise TagGenerationError("No IDs parsed from --ids.")
    return ids


def parse_ids_arg(
    ids_str: str | None, start_id: int | None, end_id: int | None
) -> list[int]:
    if ids_str:
        return parse_ids_string(ids_str)
    if start_id is not None and end_id is not None:
        return build_id_list(start_id, end_id)
    raise TagGenerationError("Provide either --ids or both --id_start and --id_end.")


def resolve_output_dir(project_root: Path | None, out_dir: str | None) -> Path:
    if project_root is None:
        return Path(out_dir or DEFAULT_OUT_DIR)

    resolved_root = Path(resolve_project_root(project_root))
    ensure_project_dirs(resolved_root)
    if out_dir is None:
        return resolved_root / "boards" / "patterns"
    return Path(out_dir)


def validate_generation_inputs(
    *,
    tag_size_mm: float,
    dpi: int,
    ids: list[int],
    paper_w_mm: float,
    paper_h_mm: float,
    margin_frac: float,
    label_gap_frac: float,
) -> None:
    if not math.isfinite(tag_size_mm) or tag_size_mm <= 0:
        raise TagGenerationError("--tag-size-mm must be a positive millimetre value.")
    if dpi <= 0:
        raise TagGenerationError("--dpi must be a positive integer.")
    min_tag_px = APRILTAG_36H11_DICT.markerSize + 2 * MARKER_BORDER_BITS
    if mm_to_px(tag_size_mm, dpi) < min_tag_px:
        raise TagGenerationError(
            f"--tag-size-mm and --dpi must produce a tag at least {min_tag_px} pixels wide "
            "for AprilTag 36h11."
        )
    if mm_to_px(paper_w_mm, dpi) < 1 or mm_to_px(paper_h_mm, dpi) < 1:
        raise TagGenerationError("Paper dimensions and --dpi must produce a page at least 1 pixel wide and high.")
    if not (math.isfinite(margin_frac) and 0 <= margin_frac < 0.45):
        raise TagGenerationError("--margin-frac must be at least 0 and less than 0.45.")
    if not (math.isfinite(label_gap_frac) and 0 <= label_gap_frac < 1):
        raise TagGenerationError("--label-gap-frac must be at least 0 and less than 1.")
    if not (math.isfinite(paper_w_mm) and math.isfinite(paper_h_mm)):
        raise TagGenerationError("Paper dimensions must be finite millimetre values.")

    max_marker_id = APRILTAG_36H11_DICT.bytesList.shape[0] - 1
    bad_ids = [tag_id for tag_id in ids if tag_id < 0 or tag_id > max_marker_id]
    if bad_ids:
        raise TagGenerationError(
            f"AprilTag 36h11 IDs must be between 0 and {max_marker_id}; invalid ID(s): "
            + ", ".join(str(tag_id) for tag_id in bad_ids[:5])
        )


def _format_id_range(ids: list[int]) -> str:
    if len(ids) <= 10:
        return "-".join(str(tag_id) for tag_id in ids)
    return f"{ids[0]}-...-{ids[-1]}"


def layout_sheet(
    *,
    paper_w_mm: float,
    paper_h_mm: float,
    dpi: int,
    tag_size_mm: float,
    ids: list[int],
    prefix: str,
    out_dir: Path,
    use_pil_text: bool,
    margin_frac: float,
    label_gap_frac: float,
) -> tuple[list[Path], list[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    _, width_px, height_px = make_canvas(paper_w_mm, paper_h_mm, dpi)
    margin_px = int(margin_frac * min(width_px, height_px))

    tag_px = make_tag_bitmap(ids[0], tag_size_mm, dpi).shape[0]
    label_gap_px = int(label_gap_frac * ((height_px - 2 * margin_px) // 3))
    cols, rows, top_gap = compute_grid(
        width_px, height_px, margin_px, tag_px, label_gap_px, use_pil_text
    )
    capacity = cols * rows

    if capacity == 0:
        raise TagGenerationError(
            f"A single tag of {tag_size_mm:.1f} mm cannot fit on the selected paper at {dpi} dpi "
            "with current margins. Reduce tag size or margins."
        )

    cell_w = (width_px - 2 * margin_px) // max(1, cols)
    cell_h = (height_px - 2 * margin_px) // max(1, rows)
    tag_rgb_cache: dict[int, np.ndarray] = {}
    page_ids = [ids[i : i + capacity] for i in range(0, len(ids), capacity)]
    png_paths: list[Path] = []
    pdf_paths: list[Path] = []

    for page_index, ids_for_page in enumerate(page_ids, start=1):
        sheet, _, _ = make_canvas(paper_w_mm, paper_h_mm, dpi)
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= len(ids_for_page):
                    break

                tag_id = ids_for_page[idx]
                if tag_id not in tag_rgb_cache:
                    gray = make_tag_bitmap(tag_id, tag_size_mm, dpi)
                    tag_rgb_cache[tag_id] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                tag_rgb = tag_rgb_cache[tag_id]

                x0 = margin_px + col * cell_w
                y0 = margin_px + row * cell_h
                x_tag = x0 + (cell_w - tag_px) // 2
                y_tag = y0 + top_gap
                x1 = x_tag + tag_px
                y1 = y_tag + tag_px

                if x1 > width_px or y1 > height_px:
                    idx += 1
                    continue

                sheet[y_tag:y1, x_tag:x1] = tag_rgb

                label_ascii = f"ID {tag_id} | {int(tag_size_mm)} mm"
                x_center = x0 + cell_w // 2
                y_label = y_tag + tag_px + label_gap_px
                max_label_width = max(1, int(cell_w * LABEL_MAX_FRAC))

                if use_pil_text and HAVE_PIL:
                    sheet[:] = draw_label_pil(
                        sheet, x_center, y_label, label_ascii, max_label_width
                    )
                else:
                    draw_label_cv(sheet, x_center, y_label, label_ascii, max_label_width)

                idx += 1

        id_str = _format_id_range(ids_for_page)
        paper_name = f"{int(round(paper_w_mm))}x{int(round(paper_h_mm))}mm"
        base_name = f"{prefix}_IDs{id_str}_{int(tag_size_mm)}mm_{paper_name}_{dpi}dpi"
        if len(page_ids) > 1:
            base_name = f"{base_name}_page{page_index:02d}of{len(page_ids):02d}"

        png_path = out_dir / f"{base_name}.png"
        write_png_with_dpi(png_path, sheet, dpi)
        png_paths.append(png_path)

        if HAVE_PIL:
            pdf_path = out_dir / f"{base_name}.pdf"
            Image.fromarray(sheet).save(str(pdf_path), "PDF", resolution=dpi)
            pdf_paths.append(pdf_path)

    return png_paths, pdf_paths


def build_parser() -> argparse.ArgumentParser:
    description = textwrap.dedent(__doc__ or "Generate printable AprilTag 36h11 sheets.")
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=_HelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              posetag-gen-tags --project_root my_project --tag-size-mm 40 --ids 1-4 --dpi 150
              posetag-gen-tags --tag-size-mm 60 --ids 1-3,7,9-10 --out_dir sheets
            """
        ),
    )
    parser.add_argument("--tag-size-mm", type=float, required=True, help="Tag side length in millimetres.")
    parser.add_argument("--paper", type=str, default="A4", help="Paper preset: A4, LETTER, LEGAL.")
    parser.add_argument(
        "--paper-mm",
        type=str,
        default=None,
        help="Custom paper size WxH in millimetres, e.g. 210x297. Overrides --paper.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="portrait",
        choices=["portrait", "landscape"],
        help="Paper orientation.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Rendering DPI.")
    parser.add_argument(
        "--pil-text",
        action="store_true",
        help="Use Pillow for nicer text when available. Pillow also enables PDF output.",
    )
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=DEFAULT_MARGIN_FRAC,
        help="Page margin as a fraction of the smaller page dimension.",
    )
    parser.add_argument(
        "--label-gap-frac",
        type=float,
        default=DEFAULT_LABEL_GAP_FRAC,
        help="Gap between each tag and label as a fraction of the cell height.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="IDs or inclusive ranges, e.g. '1,2,3', '1-4', or '1-3,7,9-10'.",
    )
    parser.add_argument("--id_start", type=int, help="First AprilTag ID, inclusive.")
    parser.add_argument("--id_end", type=int, help="Last AprilTag ID, inclusive.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. With --project_root and no explicit override, outputs go to <project_root>/boards/patterns.",
    )
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help="Filename prefix.")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=None,
        help="Project root to initialize or reuse for PoseTag output layout.",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    if args.pil_text and not HAVE_PIL:
        print(
            "Note: Pillow not found; continuing with OpenCV text and PNG-only output.",
            file=sys.stderr,
        )

    paper_w_mm, paper_h_mm = parse_paper(args.paper, args.paper_mm, args.orientation)
    ids = parse_ids_arg(args.ids, args.id_start, args.id_end)
    validate_generation_inputs(
        tag_size_mm=args.tag_size_mm,
        dpi=args.dpi,
        ids=ids,
        paper_w_mm=paper_w_mm,
        paper_h_mm=paper_h_mm,
        margin_frac=args.margin_frac,
        label_gap_frac=args.label_gap_frac,
    )
    out_dir = resolve_output_dir(args.project_root, args.out_dir)
    return layout_sheet(
        paper_w_mm=paper_w_mm,
        paper_h_mm=paper_h_mm,
        dpi=args.dpi,
        tag_size_mm=args.tag_size_mm,
        ids=ids,
        prefix=args.prefix,
        out_dir=out_dir,
        use_pil_text=bool(args.pil_text and HAVE_PIL),
        margin_frac=args.margin_frac,
        label_gap_frac=args.label_gap_frac,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        png_paths, pdf_paths = run(args)
    except TagGenerationError as exc:
        parser.exit(2, f"Error: {exc}\n")

    if pdf_paths:
        print(f"Saved {len(png_paths)} PNG sheet(s) and {len(pdf_paths)} PDF sheet(s):")
    else:
        print(f"Saved {len(png_paths)} PNG sheet(s) (install Pillow to also produce PDF):")
    for png_path in png_paths:
        print(f"  {png_path}")
    for pdf_path in pdf_paths:
        print(f"  {pdf_path}")
    print(f"Done. Open '{png_paths[0].parent}/' and print at 100% (Actual size).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
