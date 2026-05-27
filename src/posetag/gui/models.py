"""Qt-independent view models for the PoseTag workflow dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from posetag.workflows.commands import command_preview
from posetag.workflows.status import StageSummary, inspect_project


@dataclass(frozen=True)
class StatusStyle:
    """Accessible colour tokens for dashboard status rendering."""

    label: str
    foreground: str
    background: str
    border: str


_STATUS_STYLES = {
    "complete": StatusStyle(
        label="Complete",
        foreground="#126c3a",
        background="#e8f5ee",
        border="#8bc8a4",
    ),
    "missing": StatusStyle(
        label="Missing",
        foreground="#59636f",
        background="#f1f3f5",
        border="#cbd3dc",
    ),
    "needs_attention": StatusStyle(
        label="Needs attention",
        foreground="#8a5a00",
        background="#fff4d7",
        border="#ddb44b",
    ),
    "not_applicable": StatusStyle(
        label="Not applicable",
        foreground="#69717d",
        background="#f1f3f5",
        border="#cbd3dc",
    ),
}
_FALLBACK_STYLE = StatusStyle(
    label="Unknown",
    foreground="#69717d",
    background="#f1f3f5",
    border="#cbd3dc",
)


@dataclass(frozen=True)
class StatusCount:
    """Display count for one workflow status bucket."""

    label: str
    status: str
    count: int

    @property
    def foreground(self) -> str:
        return _style_for_status(self.status).foreground


@dataclass(frozen=True)
class ProjectHealthViewModel:
    """Qt-independent project health summary for the dashboard."""

    status: str
    status_label: str
    status_color: str
    status_background: str
    status_border: str
    headline: str
    message: str
    counts: tuple[StatusCount, ...]
    warning_count: int
    error_count: int
    next_stage_label: str
    next_action: str


@dataclass(frozen=True)
class StageViewModel:
    """Display-ready wrapper around a workflow ``StageSummary``."""

    stage_id: int
    key: str
    name: str
    status: str
    status_label: str
    status_color: str
    message: str
    checked_paths: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    next_action: str
    command_preview: str

    @property
    def has_issues(self) -> bool:
        """Return true when the stage has warnings or errors to display."""

        return bool(self.warnings or self.errors)

    @property
    def status_background(self) -> str:
        """Return the soft background colour for this stage status."""

        return _style_for_status(self.status).background

    @property
    def status_border(self) -> str:
        """Return the border colour for this stage status."""

        return _style_for_status(self.status).border

    @classmethod
    def from_summary(
        cls,
        summary: StageSummary,
        project_root: Union[Path, str],
    ) -> "StageViewModel":
        status = str(summary.status.value)
        style = _style_for_status(status)
        return cls(
            stage_id=summary.stage_id,
            key=summary.key,
            name=summary.name,
            status=status,
            status_label=style.label,
            status_color=style.foreground,
            message=summary.message,
            checked_paths=tuple(summary.checked_paths),
            warnings=tuple(summary.warnings),
            errors=tuple(summary.errors),
            next_action=summary.next_action,
            command_preview=command_preview(summary.key, project_root),
        )


def inspect_project_view(project_root: Union[Path, str]) -> tuple[StageViewModel, ...]:
    """Inspect a project and return display models for all workflow stages."""

    root = Path(project_root).expanduser()
    return tuple(
        StageViewModel.from_summary(summary, root)
        for summary in inspect_project(root)
    )


def project_health_view(
    models: Iterable[StageViewModel],
) -> ProjectHealthViewModel:
    """Return a compact, display-ready health summary for dashboard models."""

    stage_models = tuple(models)
    counts = _status_counts(stage_models)
    complete_count = _count_status(stage_models, "complete")
    attention_count = _count_status(stage_models, "needs_attention")
    warning_count = sum(len(model.warnings) for model in stage_models)
    error_count = sum(len(model.errors) for model in stage_models)
    next_model = _first_actionable_model(stage_models)
    attention_model = _first_model_with_status(stage_models, "needs_attention")

    if not stage_models:
        return _health(
            status="needs_attention",
            headline="Inspection unavailable",
            message="Project status could not be inspected.",
            counts=counts,
            warning_count=warning_count,
            error_count=error_count,
            next_model=None,
        )

    if attention_count or error_count:
        return _health(
            status="needs_attention",
            headline="Needs attention",
            message="Resolve the reported workflow errors before continuing.",
            counts=counts,
            warning_count=warning_count,
            error_count=error_count,
            next_model=attention_model or next_model,
        )

    if complete_count == 0:
        return _health(
            status="missing",
            headline="No workflow outputs found",
            message="Start by generating AprilTag sheets for this project.",
            counts=counts,
            warning_count=warning_count,
            error_count=error_count,
            next_model=next_model,
        )

    if next_model is not None:
        return _health(
            status="complete",
            headline="Ready for next step",
            message=(
                f"{complete_count} inspected stage"
                f"{' is' if complete_count == 1 else 's are'} complete."
            ),
            counts=counts,
            warning_count=warning_count,
            error_count=error_count,
            next_model=next_model,
        )

    return _health(
        status="complete",
        headline="All inspected stages complete",
        message="No pending workflow action was reported by the status helpers.",
        counts=counts,
        warning_count=warning_count,
        error_count=error_count,
        next_model=None,
    )


def _health(
    status: str,
    headline: str,
    message: str,
    counts: tuple[StatusCount, ...],
    warning_count: int,
    error_count: int,
    next_model: Optional[StageViewModel],
) -> ProjectHealthViewModel:
    style = _style_for_status(status)
    if next_model is None:
        next_stage_label = "No pending stage"
        next_action = "All inspected stages are complete."
    else:
        next_stage_label = f"Stage {next_model.stage_id}: {next_model.name}"
        next_action = next_model.next_action

    return ProjectHealthViewModel(
        status=status,
        status_label=style.label,
        status_color=style.foreground,
        status_background=style.background,
        status_border=style.border,
        headline=headline,
        message=message,
        counts=counts,
        warning_count=warning_count,
        error_count=error_count,
        next_stage_label=next_stage_label,
        next_action=next_action,
    )


def _status_counts(models: tuple[StageViewModel, ...]) -> tuple[StatusCount, ...]:
    order = ("complete", "needs_attention", "missing", "not_applicable")
    return tuple(
        StatusCount(
            label=_style_for_status(status).label,
            status=status,
            count=_count_status(models, status),
        )
        for status in order
    )


def _count_status(models: tuple[StageViewModel, ...], status: str) -> int:
    return sum(1 for model in models if model.status == status)


def _first_model_with_status(
    models: tuple[StageViewModel, ...],
    status: str,
) -> Optional[StageViewModel]:
    for model in models:
        if model.status == status:
            return model
    return None


def _first_actionable_model(
    models: tuple[StageViewModel, ...],
) -> Optional[StageViewModel]:
    for model in models:
        if model.status not in {"complete", "not_applicable"}:
            return model
    for model in models:
        if model.status != "complete":
            return model
    return None


def _style_for_status(status: str) -> StatusStyle:
    style = _STATUS_STYLES.get(status)
    if style is not None:
        return style

    return StatusStyle(
        label=status.replace("_", " ").title() if status else _FALLBACK_STYLE.label,
        foreground=_FALLBACK_STYLE.foreground,
        background=_FALLBACK_STYLE.background,
        border=_FALLBACK_STYLE.border,
    )
