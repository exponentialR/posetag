"""GUI-independent helpers for guided object AprilTag generation.

This module validates Stage 3 dashboard values, builds copyable
``posetag-gen-tags`` commands, and delegates all sheet rendering to
``posetag.pipelines.generate_tags``.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from posetag.pipelines import generate_tags
from posetag.workflows.status import WorkflowStatus, inspect_camera_calibration


DEFAULT_FAMILY = "tag36h11"
FAMILY_CHOICES = (DEFAULT_FAMILY,)
PAPER_CUSTOM = "CUSTOM"
PAPER_CHOICES = (*generate_tags.PAPER_MM.keys(), PAPER_CUSTOM)
ORIENTATION_CHOICES = ("portrait", "landscape")
ID_MODE_LIST = "list"
ID_MODE_RANGE = "range"
ID_MODE_CHOICES = (ID_MODE_LIST, ID_MODE_RANGE)
ID_MODE_LABELS = {
    ID_MODE_LIST: "List / ranges",
    ID_MODE_RANGE: "Start + count",
}
DEFAULT_TAG_SIZE_MM = 40.0
DEFAULT_IDS = "1-4"
DEFAULT_ID_START = 1
DEFAULT_ID_COUNT = 4
DEFAULT_DPI = generate_tags.DEFAULT_DPI
DEFAULT_PREFIX = generate_tags.DEFAULT_PREFIX
MAX_TAG_ID = generate_tags.APRILTAG_36H11_DICT.bytesList.shape[0] - 1
PRINT_GUIDANCE = (
    "Print at 100% / Actual Size. Do not use Fit to Page. Verify the printed "
    "black-square tag edge matches the configured tag size before attaching "
    "tags to object faces."
)


class ObjectTagGenerationError(ValueError):
    """User-facing validation error for guided object tag generation."""


@dataclass(frozen=True)
class ObjectTagGenerationConfig:
    """User-facing object AprilTag sheet generation values."""

    project_root: Union[Path, str]
    family: str = DEFAULT_FAMILY
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM
    id_mode: str = ID_MODE_LIST
    ids: str = DEFAULT_IDS
    id_start: int = DEFAULT_ID_START
    id_count: int = DEFAULT_ID_COUNT
    paper: str = "A4"
    paper_mm: Optional[str] = None
    orientation: str = "portrait"
    dpi: int = DEFAULT_DPI
    pil_text: bool = False
    margin_frac: float = generate_tags.DEFAULT_MARGIN_FRAC
    label_gap_frac: float = generate_tags.DEFAULT_LABEL_GAP_FRAC
    out_dir: Optional[Union[Path, str]] = None
    prefix: str = DEFAULT_PREFIX
    require_calibration: bool = True


@dataclass(frozen=True)
class ObjectTagGenerationReadiness:
    """Display-ready readiness state for guided object tag generation."""

    ready: bool
    command_preview: str
    expected_output_dir: Path
    parsed_ids: tuple[int, ...]
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ObjectTagGenerationResult:
    """Paths written by guided object AprilTag generation."""

    png_paths: tuple[Path, ...]
    pdf_paths: tuple[Path, ...]
    out_dir: Path
    parsed_ids: tuple[int, ...]

    @property
    def output_paths(self) -> tuple[Path, ...]:
        """Return generated paths in display order."""

        return (*self.png_paths, *self.pdf_paths)


@dataclass(frozen=True)
class _ValidatedObjectTagInputs:
    family: str
    id_mode: str
    ids_text: Optional[str]
    id_start: Optional[int]
    id_end: Optional[int]
    parsed_ids: tuple[int, ...]
    paper: str
    paper_mm: Optional[str]
    orientation: str
    prefix: str


def family_choices() -> tuple[str, ...]:
    """Return supported AprilTag families for GUI controls."""

    return FAMILY_CHOICES


def default_output_dir(project_root: Union[Path, str]) -> Path:
    """Return the default Stage 3 output directory without creating it."""

    return Path(project_root).expanduser() / "boards" / "patterns"


def expected_output_dir(config: ObjectTagGenerationConfig) -> Path:
    """Return the configured output directory without creating it."""

    out_dir = _optional_path(config.out_dir)
    if out_dir is not None:
        return out_dir
    return default_output_dir(config.project_root)


def inspect_object_tag_generation(
    config: ObjectTagGenerationConfig,
) -> ObjectTagGenerationReadiness:
    """Return readiness and command state for guided object tag generation."""

    root = Path(config.project_root).expanduser()
    output_dir = expected_output_dir(config)
    checked_paths = (root / "calib" / "calib_color.yaml", output_dir)
    warnings: list[str] = []
    errors: list[str] = []
    parsed_ids: tuple[int, ...] = ()
    command_preview = ""

    if config.require_calibration:
        calibration = inspect_camera_calibration(root)
        if calibration.status != WorkflowStatus.COMPLETE:
            errors.append(
                "Complete Stage 2 camera calibration before generating object "
                "AprilTags from the guided GUI flow."
            )

    default_dir = default_output_dir(root)
    if _optional_path(config.out_dir) is not None and output_dir != default_dir:
        warnings.append(
            "Project status checks look for Stage 3 PNG sheets under "
            f"{default_dir}. Custom output folders are supported, but they may "
            "not mark Stage 3 complete."
        )

    try:
        validated = _validated_inputs(config)
        parsed_ids = validated.parsed_ids
        if not errors:
            command_preview = _command_from_validated(config, validated)
    except (ObjectTagGenerationError, generate_tags.TagGenerationError) as exc:
        errors.append(str(exc))

    return ObjectTagGenerationReadiness(
        ready=not errors,
        command_preview=command_preview,
        expected_output_dir=output_dir,
        parsed_ids=parsed_ids,
        checked_paths=checked_paths,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def build_object_tag_generation_arguments(
    config: ObjectTagGenerationConfig,
) -> tuple[str, ...]:
    """Build argv items for ``posetag-gen-tags`` without a shell."""

    return _arguments_from_validated(config, _validated_inputs(config))


def build_object_tag_generation_command(config: ObjectTagGenerationConfig) -> str:
    """Build a copyable ``posetag-gen-tags`` command from GUI state."""

    return shlex.join(
        ("posetag-gen-tags", *build_object_tag_generation_arguments(config))
    )


def generate_object_tags(
    config: ObjectTagGenerationConfig,
) -> ObjectTagGenerationResult:
    """Generate object AprilTag sheets using the existing tag generator."""

    readiness = inspect_object_tag_generation(config)
    if not readiness.ready:
        raise ObjectTagGenerationError(" ".join(readiness.errors))

    validated = _validated_inputs(config)
    args = argparse.Namespace(
        tag_size_mm=float(config.tag_size_mm),
        paper=validated.paper,
        paper_mm=validated.paper_mm,
        orientation=validated.orientation,
        dpi=int(config.dpi),
        pil_text=bool(config.pil_text),
        margin_frac=float(config.margin_frac),
        label_gap_frac=float(config.label_gap_frac),
        ids=validated.ids_text,
        id_start=validated.id_start,
        id_end=validated.id_end,
        out_dir=str(_optional_path(config.out_dir))
        if _optional_path(config.out_dir) is not None
        else None,
        prefix=validated.prefix,
        project_root=Path(config.project_root).expanduser(),
    )

    try:
        png_paths, pdf_paths = generate_tags.run(args)
    except generate_tags.TagGenerationError as exc:
        raise ObjectTagGenerationError(str(exc)) from exc
    except OSError as exc:
        raise ObjectTagGenerationError(
            f"Could not write object tag sheets: {exc}"
        ) from exc

    out_dir = png_paths[0].parent if png_paths else expected_output_dir(config)
    return ObjectTagGenerationResult(
        png_paths=tuple(png_paths),
        pdf_paths=tuple(pdf_paths),
        out_dir=out_dir,
        parsed_ids=validated.parsed_ids,
    )


def _validated_inputs(config: ObjectTagGenerationConfig) -> _ValidatedObjectTagInputs:
    family = _normalize_family(config.family)
    id_mode = _normalize_id_mode(config.id_mode)
    ids_text: Optional[str]
    id_start: Optional[int]
    id_end: Optional[int]

    if id_mode == ID_MODE_LIST:
        ids_text = str(config.ids).strip()
        if not ids_text:
            raise ObjectTagGenerationError("Object tag ID list must not be empty.")
        parsed_ids = tuple(generate_tags.parse_ids_string(ids_text))
        id_start = None
        id_end = None
    else:
        ids_text = None
        id_start = _coerce_int(config.id_start, "Start ID")
        id_count = _coerce_int(config.id_count, "ID count")
        if id_count <= 0:
            raise ObjectTagGenerationError("ID count must be a positive integer.")
        id_end = id_start + id_count - 1
        parsed_ids = tuple(generate_tags.build_id_list(id_start, id_end))

    paper, paper_mm = _normalize_paper(config.paper, config.paper_mm)
    orientation = str(config.orientation).strip().lower()
    if orientation not in ORIENTATION_CHOICES:
        raise ObjectTagGenerationError(
            f"Orientation must be one of: {', '.join(ORIENTATION_CHOICES)}."
        )
    paper_w_mm, paper_h_mm = generate_tags.parse_paper(paper, paper_mm, orientation)

    prefix = str(config.prefix).strip()
    if not prefix:
        raise ObjectTagGenerationError("Output filename prefix must not be empty.")

    generate_tags.validate_generation_inputs(
        tag_size_mm=float(config.tag_size_mm),
        dpi=int(config.dpi),
        ids=list(parsed_ids),
        paper_w_mm=paper_w_mm,
        paper_h_mm=paper_h_mm,
        margin_frac=float(config.margin_frac),
        label_gap_frac=float(config.label_gap_frac),
    )

    return _ValidatedObjectTagInputs(
        family=family,
        id_mode=id_mode,
        ids_text=ids_text,
        id_start=id_start,
        id_end=id_end,
        parsed_ids=parsed_ids,
        paper=paper,
        paper_mm=paper_mm,
        orientation=orientation,
        prefix=prefix,
    )


def _command_from_validated(
    config: ObjectTagGenerationConfig,
    validated: _ValidatedObjectTagInputs,
) -> str:
    return shlex.join(
        ("posetag-gen-tags", *_arguments_from_validated(config, validated))
    )


def _arguments_from_validated(
    config: ObjectTagGenerationConfig,
    validated: _ValidatedObjectTagInputs,
) -> tuple[str, ...]:
    parts: list[str] = [
        "--project_root",
        str(Path(config.project_root).expanduser()),
        "--tag-size-mm",
        _format_number(float(config.tag_size_mm)),
    ]
    if validated.id_mode == ID_MODE_LIST:
        assert validated.ids_text is not None
        parts.extend(["--ids", validated.ids_text])
    else:
        assert validated.id_start is not None
        assert validated.id_end is not None
        parts.extend(
            [
                "--id_start",
                str(validated.id_start),
                "--id_end",
                str(validated.id_end),
            ]
        )

    if validated.paper_mm is not None:
        parts.extend(["--paper-mm", validated.paper_mm])
    else:
        parts.extend(["--paper", validated.paper])
    parts.extend(
        [
            "--orientation",
            validated.orientation,
            "--dpi",
            str(int(config.dpi)),
            "--prefix",
            validated.prefix,
            "--margin-frac",
            _format_number(float(config.margin_frac)),
            "--label-gap-frac",
            _format_number(float(config.label_gap_frac)),
        ]
    )
    if bool(config.pil_text):
        parts.append("--pil-text")
    out_dir = _optional_path(config.out_dir)
    if out_dir is not None:
        parts.extend(["--out_dir", str(out_dir)])
    return tuple(parts)


def _normalize_family(family: str) -> str:
    text = str(family).strip().lower().replace("apriltag", "").replace("_", "")
    text = text.replace(" ", "")
    if text in {"36h11", "tag36h11"}:
        return DEFAULT_FAMILY
    raise ObjectTagGenerationError(
        "Object AprilTag family must be tag36h11. The existing generator does "
        "not support other families yet."
    )


def _normalize_id_mode(mode: str) -> str:
    text = str(mode).strip().lower().replace("_", "-")
    if text in {"list", "lists", "ids", "list-ranges", "list/ranges"}:
        return ID_MODE_LIST
    if text in {"range", "start-count", "start+count", "count"}:
        return ID_MODE_RANGE
    raise ObjectTagGenerationError(
        f"ID mode must be one of: {', '.join(ID_MODE_CHOICES)}."
    )


def _normalize_paper(paper: str, paper_mm: Optional[str]) -> tuple[str, Optional[str]]:
    paper_text = str(paper).strip().upper()
    custom_text = str(paper_mm).strip() if paper_mm is not None else ""
    if paper_text == PAPER_CUSTOM:
        if not custom_text:
            raise ObjectTagGenerationError(
                "Custom paper size requires dimensions like 210x297."
            )
        return "A4", custom_text
    if custom_text:
        return paper_text or "A4", custom_text
    if paper_text not in generate_tags.PAPER_MM:
        raise ObjectTagGenerationError(
            f"Paper size must be one of: {', '.join(PAPER_CHOICES)}."
        )
    return paper_text, None


def _coerce_int(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ObjectTagGenerationError(f"{label} must be an integer.") from exc


def _optional_path(path: Optional[Union[Path, str]]) -> Optional[Path]:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"
