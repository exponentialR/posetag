"""Generate printable ChArUco calibration boards for PoseTag Step 1 setup.

Outputs
-------
- PNG is always written.
- PDF is written by standard PoseTag installs.
- YAML metadata records the board, paper, DPI, and output filenames.

Project behavior
----------------
- With ``--project_root`` and no explicit ``--out_dir``, outputs go under
  ``<project_root>/calib/boards/``.
- Without ``--project_root``, outputs default to ``charuco_boards/``.

Printing
--------
Print at 100% / Actual Size. Do not use Fit to Page. Verify one printed square
with a ruler and pass the same board arguments to ``posetag-calib-charuco``.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from posetag.pipelines.charuco_calibration import (
    CharucoCalibrationError,
    get_dictionary,
    normalize_dictionary_name,
)
from posetag.utils.project_config import ensure_project_dirs, resolve_project_root

try:
    from PIL import Image

    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


PAPER_MM = {
    "A4": (210.0, 297.0),
    "LETTER": (215.9, 279.4),
    "LEGAL": (215.9, 355.6),
}

DEFAULT_OUT_DIR = "charuco_boards"
DEFAULT_PREFIX = "charuco"
DEFAULT_DPI = 300
MARKER_BORDER_BITS = 1


class CharucoBoardGenerationError(ValueError):
    """User-facing validation error for ChArUco board generation."""


@dataclass(frozen=True)
class CharucoBoardOutputs:
    """Paths written by the ChArUco board generator."""

    png: Path
    yaml: Path
    pdf: Path | None


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show defaults and preserve help formatting."""


def mm_to_px(mm: float, dpi: int) -> int:
    """Convert a physical length in millimetres to whole pixels."""

    return int(round(mm / 25.4 * dpi))


def parse_paper(paper: str, custom_mm: str | None, orientation: str) -> tuple[str, float, float]:
    """Resolve a paper preset or custom size to oriented millimetre dimensions."""

    if custom_mm:
        try:
            w_mm, h_mm = [
                float(x) for x in custom_mm.lower().replace("mm", "").split("x", 1)
            ]
        except Exception as exc:
            raise CharucoBoardGenerationError(
                "Bad --paper-mm format. Use like: --paper-mm 210x297"
            ) from exc
        paper_name = "CUSTOM"
    else:
        paper_name = paper.upper()
        if paper_name not in PAPER_MM:
            raise CharucoBoardGenerationError(
                f"Unknown paper '{paper}'. Choose from: {', '.join(PAPER_MM.keys())} "
                "or use --paper-mm WxH."
            )
        w_mm, h_mm = PAPER_MM[paper_name]

    if not (math.isfinite(w_mm) and math.isfinite(h_mm)) or w_mm <= 0 or h_mm <= 0:
        raise CharucoBoardGenerationError(
            "Paper dimensions must be positive finite millimetre values."
        )

    if orientation.lower() == "landscape":
        return paper_name, max(w_mm, h_mm), min(w_mm, h_mm)
    return paper_name, min(w_mm, h_mm), max(w_mm, h_mm)


def resolve_output_dir(project_root: Path | None, out_dir: Path | None) -> Path:
    """Resolve the output directory without mixing ChArUco and AprilTag sheets."""

    if project_root is None:
        return Path(out_dir or DEFAULT_OUT_DIR)

    resolved_root = Path(resolve_project_root(project_root))
    ensure_project_dirs(resolved_root)
    if out_dir is None:
        return resolved_root / "calib" / "boards"
    return Path(out_dir)


def validate_generation_inputs(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dpi: int,
    paper_width_mm: float,
    paper_height_mm: float,
    dictionary: Any,
    dictionary_name: str,
) -> None:
    """Validate user inputs before asking OpenCV to render a board."""

    if squares_x < 2 or squares_y < 2:
        raise CharucoBoardGenerationError(
            "--squares-x and --squares-y must each be at least 2."
        )
    if not math.isfinite(square_length_mm) or square_length_mm <= 0:
        raise CharucoBoardGenerationError(
            "--square-length-mm must be a positive millimetre value."
        )
    if not math.isfinite(marker_length_mm) or marker_length_mm <= 0:
        raise CharucoBoardGenerationError(
            "--marker-length-mm must be a positive millimetre value."
        )
    if marker_length_mm >= square_length_mm:
        raise CharucoBoardGenerationError(
            "--marker-length-mm must be smaller than --square-length-mm."
        )
    if dpi <= 0:
        raise CharucoBoardGenerationError("--dpi must be a positive integer.")

    required_markers = required_charuco_marker_count(squares_x, squares_y)
    available_markers = dictionary_marker_count(dictionary)
    if required_markers > available_markers:
        raise CharucoBoardGenerationError(
            f"Dictionary {dictionary_name} provides {available_markers} marker IDs, "
            f"but a {squares_x}x{squares_y} ChArUco board requires "
            f"{required_markers}. Choose a larger dictionary or a smaller board."
        )

    page_w_px = mm_to_px(paper_width_mm, dpi)
    page_h_px = mm_to_px(paper_height_mm, dpi)
    board_w_px = mm_to_px(squares_x * square_length_mm, dpi)
    board_h_px = mm_to_px(squares_y * square_length_mm, dpi)
    if page_w_px < 1 or page_h_px < 1:
        raise CharucoBoardGenerationError(
            "Paper dimensions and --dpi must produce a page at least 1 pixel wide and high."
        )
    if board_w_px > page_w_px or board_h_px > page_h_px:
        raise CharucoBoardGenerationError(
            "The requested ChArUco board does not fit on the selected paper at "
            f"{dpi} dpi. Board is {squares_x * square_length_mm:g} x "
            f"{squares_y * square_length_mm:g} mm; paper is "
            f"{paper_width_mm:g} x {paper_height_mm:g} mm."
        )

    min_marker_px = int(getattr(dictionary, "markerSize", 1)) + 2 * MARKER_BORDER_BITS
    if mm_to_px(marker_length_mm, dpi) < min_marker_px:
        raise CharucoBoardGenerationError(
            f"--marker-length-mm and --dpi must produce markers at least "
            f"{min_marker_px} pixels wide for the selected dictionary."
        )


def required_charuco_marker_count(squares_x: int, squares_y: int) -> int:
    """Return the number of marker IDs OpenCV assigns to a ChArUco board."""

    return (squares_x * squares_y) // 2


def dictionary_marker_count(dictionary: Any) -> int:
    """Return how many marker IDs are available in an OpenCV ArUco dictionary."""

    bytes_list = getattr(dictionary, "bytesList", None)
    if bytes_list is None:
        raise CharucoBoardGenerationError(
            "Could not determine selected dictionary capacity from OpenCV."
        )
    return int(np.asarray(bytes_list).shape[0])


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary: Any,
):
    """Create a ChArUco board using the OpenCV API available in this build."""

    aruco = cv2.aruco
    square_m = square_length_mm / 1000.0
    marker_m = marker_length_mm / 1000.0
    try:
        if hasattr(aruco, "CharucoBoard") and callable(getattr(aruco, "CharucoBoard")):
            return aruco.CharucoBoard(
                (squares_x, squares_y),
                square_m,
                marker_m,
                dictionary,
            )
        if hasattr(aruco, "CharucoBoard_create"):
            return aruco.CharucoBoard_create(
                squares_x,
                squares_y,
                square_m,
                marker_m,
                dictionary,
            )
    except cv2.error as exc:
        raise CharucoBoardGenerationError(
            f"OpenCV could not create the ChArUco board: {exc}"
        ) from exc
    raise CharucoBoardGenerationError(
        "Installed OpenCV build lacks ChArUco board generation support."
    )


def _wrap_opencv_draw_error(exc: cv2.error) -> CharucoBoardGenerationError:
    return CharucoBoardGenerationError(
        f"OpenCV could not draw the ChArUco board: {exc}"
    )


def draw_charuco_board(board: Any, width_px: int, height_px: int) -> np.ndarray:
    """Render a board image with exact pixel dimensions."""

    out_size = (int(width_px), int(height_px))
    if hasattr(board, "generateImage"):
        try:
            return board.generateImage(
                out_size,
                marginSize=0,
                borderBits=MARKER_BORDER_BITS,
            )
        except TypeError:
            try:
                return board.generateImage(out_size, 0, MARKER_BORDER_BITS)
            except cv2.error as exc:
                raise _wrap_opencv_draw_error(exc) from exc
        except cv2.error as exc:
            raise _wrap_opencv_draw_error(exc) from exc

    if hasattr(board, "draw"):
        try:
            return board.draw(out_size, marginSize=0, borderBits=MARKER_BORDER_BITS)
        except TypeError:
            try:
                return board.draw(out_size, 0, MARKER_BORDER_BITS)
            except cv2.error as exc:
                raise _wrap_opencv_draw_error(exc) from exc
        except cv2.error as exc:
            raise _wrap_opencv_draw_error(exc) from exc

    raise CharucoBoardGenerationError(
        "Installed OpenCV build cannot draw ChArUco boards."
    )


def write_png_with_dpi(path: Path, image: np.ndarray, dpi: int) -> None:
    """Write a PNG and inject pHYs DPI metadata."""

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise CharucoBoardGenerationError(f"Could not encode PNG output: {path}")

    png_bytes = encoded.tobytes()
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CharucoBoardGenerationError(
            f"OpenCV did not produce a valid PNG stream: {path}"
        )

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


def build_metadata(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
    paper: str,
    paper_width_mm: float,
    paper_height_mm: float,
    dpi: int,
    image_width_px: int,
    image_height_px: int,
    png: Path,
    pdf: Path | None,
) -> dict[str, Any]:
    """Build the public metadata YAML schema for generated ChArUco boards."""

    metadata: dict[str, Any] = {
        "squares_x": int(squares_x),
        "squares_y": int(squares_y),
        "square_length_mm": float(square_length_mm),
        "marker_length_mm": float(marker_length_mm),
        "dictionary": dictionary_name,
        "paper": paper,
        "paper_width_mm": float(paper_width_mm),
        "paper_height_mm": float(paper_height_mm),
        "dpi": int(dpi),
        "image_width_px": int(image_width_px),
        "image_height_px": int(image_height_px),
        "png": png.name,
    }
    if pdf is not None:
        metadata["pdf"] = pdf.name
    metadata["notes"] = (
        "Print at 100% / Actual Size; do not use Fit to Page. Verify the "
        "printed square length with a ruler, then run posetag-calib-charuco "
        "with these same board parameters."
    )
    return metadata


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write board metadata using YAML safe_dump."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def _format_mm_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_output_stem(
    *,
    prefix: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
    paper: str,
    dpi: int,
) -> str:
    """Return a deterministic stem for board outputs."""

    square = _format_mm_for_filename(square_length_mm)
    marker = _format_mm_for_filename(marker_length_mm)
    return (
        f"{prefix}_{squares_x}x{squares_y}_square{square}mm_"
        f"marker{marker}mm_{dictionary_name}_{paper}_{dpi}dpi"
    )


def render_page(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    paper_width_mm: float,
    paper_height_mm: float,
    dpi: int,
    dictionary: Any,
) -> np.ndarray:
    """Render an exact-scale ChArUco board centered on a white paper canvas."""

    page_w_px = mm_to_px(paper_width_mm, dpi)
    page_h_px = mm_to_px(paper_height_mm, dpi)
    board_w_px = mm_to_px(squares_x * square_length_mm, dpi)
    board_h_px = mm_to_px(squares_y * square_length_mm, dpi)

    board = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        dictionary=dictionary,
    )
    board_image = draw_charuco_board(board, board_w_px, board_h_px)
    if board_image.ndim == 3:
        board_image = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    page = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)
    x0 = (page_w_px - board_w_px) // 2
    y0 = (page_h_px - board_h_px) // 2
    page[y0 : y0 + board_h_px, x0 : x0 + board_w_px] = board_image
    return page


def generate_charuco_board(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
    paper: str,
    paper_width_mm: float,
    paper_height_mm: float,
    dpi: int,
    out_dir: Path,
    prefix: str = DEFAULT_PREFIX,
    write_pdf: bool | None = None,
) -> CharucoBoardOutputs:
    """Generate a ChArUco board PNG/PDF and metadata YAML."""

    normalized_dict = normalize_dictionary_name(dictionary_name)
    try:
        dictionary = get_dictionary(normalized_dict)
    except CharucoCalibrationError as exc:
        raise CharucoBoardGenerationError(str(exc)) from exc

    validate_generation_inputs(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        dpi=dpi,
        paper_width_mm=paper_width_mm,
        paper_height_mm=paper_height_mm,
        dictionary=dictionary,
        dictionary_name=normalized_dict,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        paper_width_mm=paper_width_mm,
        paper_height_mm=paper_height_mm,
        dpi=dpi,
        dictionary=dictionary,
    )

    stem = build_output_stem(
        prefix=prefix,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        dictionary_name=normalized_dict,
        paper=paper,
        dpi=dpi,
    )
    png_path = out_dir / f"{stem}.png"
    should_write_pdf = HAVE_PIL if write_pdf is None else bool(write_pdf)
    pdf_path = out_dir / f"{stem}.pdf" if should_write_pdf and HAVE_PIL else None
    yaml_path = out_dir / f"{stem}.yaml"

    write_png_with_dpi(png_path, page, dpi)
    if pdf_path is not None:
        Image.fromarray(page).save(str(pdf_path), "PDF", resolution=dpi)

    metadata = build_metadata(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        dictionary_name=normalized_dict,
        paper=paper,
        paper_width_mm=paper_width_mm,
        paper_height_mm=paper_height_mm,
        dpi=dpi,
        image_width_px=page.shape[1],
        image_height_px=page.shape[0],
        png=png_path,
        pdf=pdf_path,
    )
    write_metadata(yaml_path, metadata)

    return CharucoBoardOutputs(png=png_path, yaml=yaml_path, pdf=pdf_path)


def build_parser() -> argparse.ArgumentParser:
    description = textwrap.dedent(
        __doc__ or "Generate printable ChArUco calibration boards."
    )
    parser = argparse.ArgumentParser(
        prog="posetag-gen-charuco",
        description=description,
        formatter_class=_HelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              posetag-gen-charuco --project_root my_project --squares-x 3 --squares-y 5 \\
                --square-length-mm 50 --marker-length-mm 37 --dict 7X7_50 --paper A4 --dpi 300
            """
        ),
    )
    parser.add_argument("--squares-x", type=int, default=3, help="Number of squares along board x.")
    parser.add_argument("--squares-y", type=int, default=5, help="Number of squares along board y.")
    parser.add_argument(
        "--square-length-mm",
        type=float,
        default=50.0,
        help="Physical side length of each ChArUco square in millimetres.",
    )
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=37.0,
        help="Physical side length of each ArUco marker in millimetres.",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="7X7_50",
        help="ArUco dictionary, e.g. 4X4_50, 5X5_250, 6X6_1000, 7X7_50.",
    )
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
        "--project_root",
        type=Path,
        default=None,
        help="Project root. Default outputs go to <project_root>/calib/boards/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. With --project_root and no override, outputs go to <project_root>/calib/boards/.",
    )
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help="Filename prefix.")
    return parser


def run(args: argparse.Namespace) -> CharucoBoardOutputs:
    paper_name, paper_width_mm, paper_height_mm = parse_paper(
        args.paper,
        args.paper_mm,
        args.orientation,
    )
    out_dir = resolve_output_dir(args.project_root, args.out_dir)
    return generate_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
        dictionary_name=args.dict,
        paper=paper_name,
        paper_width_mm=paper_width_mm,
        paper_height_mm=paper_height_mm,
        dpi=args.dpi,
        out_dir=out_dir,
        prefix=args.prefix,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run(args)
    except CharucoBoardGenerationError as exc:
        parser.exit(2, f"Error: {exc}\n")

    print("Saved ChArUco calibration board:")
    print(f"  PNG:  {outputs.png}")
    print(f"  YAML: {outputs.yaml}")
    if outputs.pdf is not None:
        print(f"  PDF:  {outputs.pdf}")
    else:
        print("  PDF:  unavailable in this environment")
    print("Print at 100% / Actual Size. Do not use Fit to Page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
