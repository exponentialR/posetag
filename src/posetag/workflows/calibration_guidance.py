"""State-driven guidance for ChArUco calibration capture.

The helpers here are intentionally independent of OpenCV windows and Qt. They
summarize detected ChArUco observations, accepted samples, and coverage gaps so
the calibration UI can show deterministic guidance instead of rotating generic
tips.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import numpy as np


CELL_LABELS = {
    "top-left": "top-left corner",
    "top-center": "top edge",
    "top-right": "top-right corner",
    "middle-left": "left edge",
    "center": "center",
    "middle-right": "right edge",
    "bottom-left": "bottom-left corner",
    "bottom-center": "bottom edge",
    "bottom-right": "bottom-right corner",
}
DEFAULT_GRID_SHAPE = (3, 3)
CELL_PRIORITY = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "top-center",
    "middle-left",
    "middle-right",
    "bottom-center",
    "center",
)
SCALE_LABELS = {
    "far": "far",
    "medium": "medium distance",
    "near": "close",
}


@dataclass(frozen=True)
class CalibrationObservation:
    """Measured state for one frame containing ChArUco corners."""

    image_width: int
    image_height: int
    corner_count: int
    center_x: float
    center_y: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    area_fraction: float
    cell: str
    row: int
    col: int
    grid_shape: tuple[int, int]
    scale_bucket: str
    has_enough_corners: bool
    clipped: bool


@dataclass(frozen=True)
class CalibrationRecommendation:
    """User-facing next action derived from measured calibration state."""

    code: str
    message: str
    detail: str
    target_cell: Optional[str] = None
    ready_to_solve: bool = False


@dataclass(frozen=True)
class AutoCaptureDecision:
    """Decision for automatic capture of the current calibration frame."""

    should_capture: bool
    reason: str
    message: str


@dataclass(frozen=True)
class CalibrationGuidanceState:
    """Deterministic guidance summary for the current capture state."""

    current_observation: Optional[CalibrationObservation]
    sample_count: int
    required_samples: int
    grid_shape: tuple[int, int]
    samples_per_cell: int
    all_cells: tuple[str, ...]
    covered_cells: tuple[str, ...]
    missing_cells: tuple[str, ...]
    cell_sample_counts: tuple[tuple[str, int], ...]
    scale_buckets: tuple[str, ...]
    recommendation: CalibrationRecommendation

    @property
    def coverage_count(self) -> int:
        """Return the number of covered image-space grid cells."""

        return len(self.covered_cells)


def parse_grid_shape(value: object) -> tuple[int, int]:
    """Parse a user-facing grid shape such as ``3x3`` or ``4``."""

    text = str(value).strip().lower()
    try:
        if "x" in text:
            left, right = text.split("x", 1)
            rows = int(left)
            cols = int(right)
        else:
            rows = cols = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--coverage-grid must be an integer or ROWSxCOLS, for example 3x3."
        ) from exc
    if rows < 1 or cols < 1:
        raise ValueError("--coverage-grid values must be positive.")
    if rows > 9 or cols > 9:
        raise ValueError("--coverage-grid is limited to 9x9 for readable guidance.")
    return rows, cols


def observation_from_charuco_corners(
    corners: object,
    image_size: tuple[int, int],
    *,
    min_corners: int,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> Optional[CalibrationObservation]:
    """Summarize detected ChArUco corners as normalized frame metrics."""

    if corners is None:
        return None

    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height.")
    grid = _normalize_grid_shape(grid_shape)

    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    if points.size == 0:
        return None

    xs = points[:, 0]
    ys = points[:, 1]
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    center_x = _clamp01(float(xs.mean()) / float(width))
    center_y = _clamp01(float(ys.mean()) / float(height))
    box_width = max(0.0, max_x - min_x)
    box_height = max(0.0, max_y - min_y)
    area_fraction = _clamp01((box_width * box_height) / float(width * height))
    corner_count = int(points.shape[0])

    row, col = _row_col_for(center_x, center_y, grid)
    return CalibrationObservation(
        image_width=width,
        image_height=height,
        corner_count=corner_count,
        center_x=center_x,
        center_y=center_y,
        min_x=_clamp01(min_x / float(width)),
        min_y=_clamp01(min_y / float(height)),
        max_x=_clamp01(max_x / float(width)),
        max_y=_clamp01(max_y / float(height)),
        area_fraction=area_fraction,
        cell=_cell_id(row, col, grid),
        row=row,
        col=col,
        grid_shape=grid,
        scale_bucket=_scale_bucket(area_fraction),
        has_enough_corners=corner_count >= int(min_corners),
        clipped=_is_clipped(min_x, min_y, max_x, max_y, width, height),
    )


def analyze_calibration_guidance(
    current_observation: Optional[CalibrationObservation],
    accepted_observations: Iterable[CalibrationObservation],
    *,
    min_samples: int,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    samples_per_cell: int = 1,
    auto_capture: bool = False,
) -> CalibrationGuidanceState:
    """Return coverage state and next instruction for calibration capture."""

    grid = _normalize_grid_shape(grid_shape)
    per_cell = max(1, int(samples_per_cell))
    all_cells = cell_priority(grid)
    required_samples = max(10, int(min_samples), len(all_cells) * per_cell)
    accepted = tuple(
        observation
        for observation in accepted_observations
        if observation.has_enough_corners and not observation.clipped
    )
    counts = _cell_counts(accepted, all_cells)
    covered_cells = tuple(cell for cell, count in counts if count >= per_cell)
    missing_cells = tuple(
        cell for cell, count in counts if count < per_cell
    )
    scale_buckets = tuple(
        bucket
        for bucket in ("far", "medium", "near")
        if bucket in {obs.scale_bucket for obs in accepted}
    )

    recommendation = _recommendation(
        current_observation=current_observation,
        accepted=accepted,
        missing_cells=missing_cells,
        scale_buckets=scale_buckets,
        required_samples=required_samples,
        auto_capture=auto_capture,
    )

    return CalibrationGuidanceState(
        current_observation=current_observation,
        sample_count=len(accepted),
        required_samples=required_samples,
        grid_shape=grid,
        samples_per_cell=per_cell,
        all_cells=all_cells,
        covered_cells=covered_cells,
        missing_cells=missing_cells,
        cell_sample_counts=counts,
        scale_buckets=scale_buckets,
        recommendation=recommendation,
    )


def auto_capture_decision(
    state: CalibrationGuidanceState,
    *,
    cooldown_frames_remaining: int = 0,
) -> AutoCaptureDecision:
    """Return whether the current frame should be saved automatically."""

    observation = state.current_observation
    if observation is None:
        return AutoCaptureDecision(False, "no_detection", "No board detected.")
    if not observation.has_enough_corners:
        return AutoCaptureDecision(
            False,
            "too_few_corners",
            "Waiting for more detected ChArUco corners.",
        )
    if observation.clipped:
        return AutoCaptureDecision(
            False,
            "clipped",
            "Waiting until the board is not clipped by the frame.",
        )
    if cooldown_frames_remaining > 0:
        return AutoCaptureDecision(
            False,
            "cooldown",
            "Waiting briefly before another automatic sample.",
        )

    counts = dict(state.cell_sample_counts)
    if counts.get(observation.cell, 0) < state.samples_per_cell:
        return AutoCaptureDecision(
            True,
            "covers_missing_cell",
            f"Auto-capturing coverage for {cell_label(observation.cell, state.grid_shape)}.",
        )

    if state.missing_cells:
        return AutoCaptureDecision(
            False,
            "waiting_for_missing_cell",
            "Move to an uncovered grid cell before saving another sample.",
        )

    if observation.scale_bucket not in state.scale_buckets:
        return AutoCaptureDecision(
            True,
            "adds_scale_diversity",
            "Auto-capturing this distance to improve scale diversity.",
        )

    if state.sample_count < state.required_samples:
        balanced_target = max(
            state.samples_per_cell,
            math.ceil(state.required_samples / max(1, len(state.all_cells))),
        )
        current_count = counts.get(observation.cell, 0)
        if current_count < balanced_target or current_count <= min(counts.values()):
            return AutoCaptureDecision(
                True,
                "balances_cell_samples",
                (
                    "Auto-capturing another balanced view for "
                    f"{cell_label(observation.cell, state.grid_shape)}."
                ),
            )

    return AutoCaptureDecision(
        False,
        "no_new_coverage",
        "Move to an uncovered grid cell or change distance.",
    )


def guidance_overlay_lines(state: CalibrationGuidanceState) -> tuple[str, ...]:
    """Return compact lines suitable for an OpenCV overlay or GUI log."""

    scales = ", ".join(SCALE_LABELS[bucket] for bucket in state.scale_buckets)
    if not scales:
        scales = "none yet"
    return (
        f"Guide: {state.recommendation.message}",
        f"Coverage: {state.coverage_count}/{len(state.all_cells)} cells | "
        f"Samples: {state.sample_count}/{state.required_samples}",
        f"Scale diversity: {scales}",
    )


def cell_priority(grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE) -> tuple[str, ...]:
    """Return deterministic cell traversal order for a guidance grid."""

    grid = _normalize_grid_shape(grid_shape)
    if grid == DEFAULT_GRID_SHAPE:
        return CELL_PRIORITY

    rows, cols = grid
    cells = [_cell_id(row, col, grid) for row in range(rows) for col in range(cols)]
    return tuple(
        sorted(
            cells,
            key=lambda cell: (
                _cell_priority_ring(cell, grid),
                _cell_priority_corner_rank(cell, grid),
                cell,
            ),
        )
    )


def cell_label(cell: str, grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE) -> str:
    """Return a human-readable label for a grid cell."""

    grid = _normalize_grid_shape(grid_shape)
    if grid == DEFAULT_GRID_SHAPE and cell in CELL_LABELS:
        return CELL_LABELS[cell]
    row, col = _row_col_from_cell(cell, grid)
    return f"row {row + 1}, column {col + 1}"


def _recommendation(
    *,
    current_observation: Optional[CalibrationObservation],
    accepted: Sequence[CalibrationObservation],
    missing_cells: tuple[str, ...],
    scale_buckets: tuple[str, ...],
    required_samples: int,
    auto_capture: bool,
) -> CalibrationRecommendation:
    if current_observation is None:
        return CalibrationRecommendation(
            code="searching",
            message="Bring the ChArUco board fully into view.",
            detail="No ChArUco corners are currently detected.",
        )

    if not current_observation.has_enough_corners:
        return CalibrationRecommendation(
            code="too_few_corners",
            message="Move closer or reduce glare until more corners are detected.",
            detail=(
                f"Only {current_observation.corner_count} ChArUco corners are "
                "visible in the current frame."
            ),
        )

    if current_observation.clipped:
        return CalibrationRecommendation(
            code="clipped",
            message="Move the board slightly inward so it is not clipped.",
            detail="Detected corners are too close to the image border.",
        )

    if not accepted:
        message = (
            "Hold steady for automatic first sample."
            if auto_capture
            else "Press SPACE to save this clear first sample."
        )
        return CalibrationRecommendation(
            code="first_sample",
            message=message,
            detail="The board is detected clearly enough to begin calibration.",
            target_cell=current_observation.cell,
        )

    if current_observation.cell in missing_cells:
        label = cell_label(current_observation.cell, current_observation.grid_shape)
        message = (
            f"Hold near the {label} for automatic capture."
            if auto_capture
            else f"Hold near the {label} and press SPACE."
        )
        return CalibrationRecommendation(
            code="accept_target_cell",
            message=message,
            detail="This view covers an under-sampled part of the frame.",
            target_cell=current_observation.cell,
        )

    if missing_cells:
        target = missing_cells[0]
        label = cell_label(target, current_observation.grid_shape)
        return CalibrationRecommendation(
            code="move_to_missing_cell",
            message=f"Move the board toward the {label}.",
            detail="Calibration coverage is still missing this frame region.",
            target_cell=target,
        )

    if len(scale_buckets) < 2:
        return _scale_recommendation(current_observation)

    if len(accepted) < required_samples:
        return CalibrationRecommendation(
            code="collect_more",
            message="Collect more views while varying tilt and distance.",
            detail=(
                "Frame coverage looks balanced, but more accepted samples are "
                "needed before solving."
            ),
        )

    return CalibrationRecommendation(
        code="ready_to_solve",
        message="Coverage looks good. Press ENTER to solve.",
        detail="Accepted samples cover the frame with useful scale diversity.",
        ready_to_solve=True,
    )


def _scale_recommendation(
    current_observation: CalibrationObservation,
) -> CalibrationRecommendation:
    if current_observation.scale_bucket == "far":
        message = "Move the board closer for a larger view."
    elif current_observation.scale_bucket == "near":
        message = "Move the board farther away for a smaller view."
    else:
        message = "Change distance: capture one closer or farther view."
    return CalibrationRecommendation(
        code="change_distance",
        message=message,
        detail="Coverage is balanced, but scale diversity is still low.",
    )


def _cell_counts(
    observations: Sequence[CalibrationObservation],
    all_cells: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    counts = {cell: 0 for cell in all_cells}
    for observation in observations:
        counts[observation.cell] = counts.get(observation.cell, 0) + 1
    return tuple((cell, counts[cell]) for cell in all_cells)


def _row_col_for(
    center_x: float,
    center_y: float,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    rows, cols = grid_shape
    col = min(cols - 1, int(_clamp01(center_x) * cols))
    row = min(rows - 1, int(_clamp01(center_y) * rows))
    return row, col


def _cell_id(row: int, col: int, grid_shape: tuple[int, int]) -> str:
    if grid_shape == DEFAULT_GRID_SHAPE:
        row_name = ("top", "middle", "bottom")[row]
        col_name = ("left", "center", "right")[col]
        if row_name == "middle" and col_name == "center":
            return "center"
        return f"{row_name}-{col_name}"
    return f"r{row + 1}c{col + 1}"


def _row_col_from_cell(cell: str, grid_shape: tuple[int, int]) -> tuple[int, int]:
    if grid_shape == DEFAULT_GRID_SHAPE:
        if cell == "center":
            return 1, 1
        row_name, col_name = cell.split("-", 1)
        return (
            {"top": 0, "middle": 1, "bottom": 2}[row_name],
            {"left": 0, "center": 1, "right": 2}[col_name],
        )
    if not (cell.startswith("r") and "c" in cell):
        raise ValueError(f"Invalid grid cell {cell!r}.")
    row_text, col_text = cell[1:].split("c", 1)
    row = int(row_text) - 1
    col = int(col_text) - 1
    rows, cols = grid_shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise ValueError(f"Grid cell {cell!r} is outside {rows}x{cols}.")
    return row, col


def _cell_priority_ring(cell: str, grid_shape: tuple[int, int]) -> int:
    row, col = _row_col_from_cell(cell, grid_shape)
    rows, cols = grid_shape
    return min(row, col, rows - 1 - row, cols - 1 - col)


def _cell_priority_corner_rank(cell: str, grid_shape: tuple[int, int]) -> int:
    row, col = _row_col_from_cell(cell, grid_shape)
    rows, cols = grid_shape
    corners = ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1))
    try:
        return corners.index((row, col))
    except ValueError:
        return 4


def _normalize_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("grid_shape values must be positive.")
    return rows, cols


def _scale_bucket(area_fraction: float) -> str:
    if area_fraction < 0.04:
        return "far"
    if area_fraction >= 0.16:
        return "near"
    return "medium"


def _is_clipped(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    width: int,
    height: int,
) -> bool:
    return (
        min_x <= 1.0
        or min_y <= 1.0
        or max_x >= float(width - 2)
        or max_y >= float(height - 2)
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
