"""Workflow helpers for guided ChArUco board setup.

This module is GUI-independent.  It validates user-facing Step 1 setup values,
resolves PoseTag project output paths, and delegates all board rendering to
``posetag.pipelines.generate_charuco``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from posetag.pipelines.charuco_calibration import supported_dictionary_names
from posetag.pipelines.generate_charuco import (
    DEFAULT_DPI,
    DEFAULT_PREFIX,
    PAPER_MM,
    CharucoBoardGenerationError,
    parse_paper,
    resolve_output_dir,
    generate_charuco_board,
)


DEFAULT_SQUARES_X = 3
DEFAULT_SQUARES_Y = 5
DEFAULT_SQUARE_LENGTH_MM = 50.0
DEFAULT_MARKER_LENGTH_MM = 37.0
DEFAULT_DICTIONARY = "7X7_50"
PAPER_CHOICES = tuple(PAPER_MM.keys())
ORIENTATION_CHOICES = ("portrait", "landscape")
OUTPUT_FORMAT_PNG_PDF = "PNG + PDF"
OUTPUT_FORMAT_PNG = "PNG only"
OUTPUT_FORMAT_CHOICES = (OUTPUT_FORMAT_PNG_PDF, OUTPUT_FORMAT_PNG)
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_PNG_PDF
PRINT_GUIDANCE = (
    "Print at 100% / Actual Size. Do not use Fit to Page. Verify the printed "
    "square length with a ruler or calipers before calibration."
)


class CharucoBoardSetupError(ValueError):
    """User-facing validation error for guided ChArUco board setup."""


@dataclass(frozen=True)
class CharucoBoardSetupConfig:
    """User-facing ChArUco board setup values."""

    project_root: Union[Path, str]
    squares_x: int = DEFAULT_SQUARES_X
    squares_y: int = DEFAULT_SQUARES_Y
    square_length_mm: float = DEFAULT_SQUARE_LENGTH_MM
    marker_length_mm: float = DEFAULT_MARKER_LENGTH_MM
    dictionary_name: str = DEFAULT_DICTIONARY
    paper: str = "A4"
    orientation: str = "portrait"
    dpi: int = DEFAULT_DPI
    output_format: str = DEFAULT_OUTPUT_FORMAT
    out_dir: Optional[Union[Path, str]] = None
    prefix: str = DEFAULT_PREFIX


@dataclass(frozen=True)
class CharucoBoardSetupResult:
    """Paths written by the guided ChArUco setup action."""

    png: Path
    yaml: Path
    pdf: Optional[Path]
    out_dir: Path
    requested_pdf: bool

    @property
    def output_paths(self) -> tuple[Path, ...]:
        """Return generated paths in display order."""

        paths = [self.png, self.yaml]
        if self.pdf is not None:
            paths.append(self.pdf)
        return tuple(paths)


def dictionary_choices() -> tuple[str, ...]:
    """Return supported dictionary names for GUI controls."""

    names = tuple(supported_dictionary_names())
    if DEFAULT_DICTIONARY in names:
        return (
            DEFAULT_DICTIONARY,
            *(name for name in names if name != DEFAULT_DICTIONARY),
        )
    return names


def default_output_dir(project_root: Union[Path, str]) -> Path:
    """Return the default project-root output directory without creating it."""

    return Path(project_root).expanduser() / "calib" / "boards"


def generate_charuco_board_setup(
    config: CharucoBoardSetupConfig,
) -> CharucoBoardSetupResult:
    """Generate a ChArUco board from guided workflow setup values."""

    try:
        _validate_setup_config(config)
        paper_name, paper_width_mm, paper_height_mm = parse_paper(
            config.paper,
            custom_mm=None,
            orientation=config.orientation,
        )
        out_dir = resolve_output_dir(
            Path(config.project_root).expanduser(),
            _optional_path(config.out_dir),
        )
        requested_pdf = _requests_pdf(config.output_format)
        outputs = generate_charuco_board(
            squares_x=config.squares_x,
            squares_y=config.squares_y,
            square_length_mm=config.square_length_mm,
            marker_length_mm=config.marker_length_mm,
            dictionary_name=config.dictionary_name,
            paper=paper_name,
            paper_width_mm=paper_width_mm,
            paper_height_mm=paper_height_mm,
            dpi=config.dpi,
            out_dir=out_dir,
            prefix=config.prefix,
            write_pdf=requested_pdf,
        )
    except CharucoBoardGenerationError as exc:
        raise CharucoBoardSetupError(str(exc)) from exc
    except OSError as exc:
        raise CharucoBoardSetupError(
            f"Could not write ChArUco outputs: {exc}"
        ) from exc

    return CharucoBoardSetupResult(
        png=outputs.png,
        yaml=outputs.yaml,
        pdf=outputs.pdf,
        out_dir=out_dir,
        requested_pdf=requested_pdf,
    )


def _validate_setup_config(config: CharucoBoardSetupConfig) -> None:
    if not str(config.dictionary_name).strip():
        raise CharucoBoardSetupError("ChArUco dictionary must not be empty.")
    if config.paper.upper() not in PAPER_CHOICES:
        raise CharucoBoardSetupError(
            f"Paper size must be one of: {', '.join(PAPER_CHOICES)}."
        )
    if config.orientation.lower() not in ORIENTATION_CHOICES:
        raise CharucoBoardSetupError(
            f"Orientation must be one of: {', '.join(ORIENTATION_CHOICES)}."
        )
    if not str(config.prefix).strip():
        raise CharucoBoardSetupError("Output filename prefix must not be empty.")
    _normalize_output_format(config.output_format)


def _optional_path(path: Optional[Union[Path, str]]) -> Optional[Path]:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _requests_pdf(output_format: str) -> bool:
    return _normalize_output_format(output_format) == OUTPUT_FORMAT_PNG_PDF


def _normalize_output_format(output_format: str) -> str:
    text = str(output_format).strip().lower().replace("&", "+")
    collapsed = " ".join(text.split())
    compact = collapsed.replace(" ", "")
    if compact in {"png+pdf", "pdf+png"}:
        return OUTPUT_FORMAT_PNG_PDF
    if collapsed in {"png", "png only"}:
        return OUTPUT_FORMAT_PNG
    raise CharucoBoardSetupError(
        f"Output format must be one of: {', '.join(OUTPUT_FORMAT_CHOICES)}."
    )
