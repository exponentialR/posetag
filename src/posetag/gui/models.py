"""Qt-independent view models for the PoseTag workflow dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from posetag.workflows.commands import command_preview
from posetag.workflows.status import StageSummary, inspect_project


_STATUS_LABELS = {
    "complete": "Complete",
    "missing": "Missing",
    "needs_attention": "Needs attention",
    "not_applicable": "Not applicable",
}

_STATUS_COLORS = {
    "complete": "#2f7d4f",
    "missing": "#8a6a00",
    "needs_attention": "#b14a33",
    "not_applicable": "#69717d",
}


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

    @classmethod
    def from_summary(
        cls,
        summary: StageSummary,
        project_root: Union[Path, str],
    ) -> "StageViewModel":
        status = str(summary.status.value)
        return cls(
            stage_id=summary.stage_id,
            key=summary.key,
            name=summary.name,
            status=status,
            status_label=_STATUS_LABELS.get(
                status,
                status.replace("_", " ").title(),
            ),
            status_color=_STATUS_COLORS.get(status, "#69717d"),
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
