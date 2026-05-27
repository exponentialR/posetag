"""Main window construction for the optional PoseTag workflow dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from posetag.gui.models import (
    ProjectHealthViewModel,
    StageViewModel,
    inspect_project_view,
    project_health_view,
)


def build_main_window(qt: Any, project_root: Path) -> Any:
    """Build the PySide6 main window without importing Qt at module import time."""

    QtCore = qt.QtCore
    QtGui = qt.QtGui
    QtWidgets = qt.QtWidgets

    def _make_card(title: str) -> tuple[Any, Any]:
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel(title)
        heading.setObjectName("CardTitle")

        body = QtWidgets.QLabel()
        body.setObjectName("CardBody")
        body.setWordWrap(True)
        body.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        body.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        layout.addWidget(heading)
        layout.addWidget(body, 1)
        return frame, body

    def _divider(widgets: Any) -> Any:
        line = widgets.QFrame()
        line.setObjectName("Divider")
        line.setFrameShape(widgets.QFrame.Shape.HLine)
        line.setFrameShadow(widgets.QFrame.Shadow.Plain)
        return line

    class PoseTagMainWindow(QtWidgets.QMainWindow):
        def __init__(self, initial_project_root: Path) -> None:
            super().__init__()
            self._models: tuple[StageViewModel, ...] = ()
            self._selected_stage_id = 0

            self.setWindowTitle("PoseTag Workflow Dashboard")

            self._project_title = QtWidgets.QLabel("PoseTag Workflow Dashboard")
            self._project_title.setObjectName("AppTitle")

            self._root_input = QtWidgets.QLineEdit(str(initial_project_root))
            self._root_input.setMinimumWidth(360)
            self._root_input.returnPressed.connect(self._refresh)

            browse_button = QtWidgets.QPushButton("Browse")
            browse_button.clicked.connect(self._browse_project_root)

            refresh_button = QtWidgets.QPushButton("Refresh")
            refresh_button.clicked.connect(self._refresh)

            self._open_folder_button = QtWidgets.QPushButton("Open Folder")
            self._open_folder_button.clicked.connect(self._open_project_folder)

            self._root_status_label = QtWidgets.QLabel()
            self._root_status_label.setObjectName("MutedText")
            self._root_status_label.setWordWrap(True)

            root_bar = QtWidgets.QHBoxLayout()
            root_bar.addWidget(QtWidgets.QLabel("Project root"))
            root_bar.addWidget(self._root_input, 1)
            root_bar.addWidget(browse_button)
            root_bar.addWidget(self._open_folder_button)
            root_bar.addWidget(refresh_button)

            header_layout = QtWidgets.QVBoxLayout()
            header_layout.setSpacing(6)
            header_layout.addWidget(self._project_title)
            header_layout.addLayout(root_bar)
            header_layout.addWidget(self._root_status_label)

            rail_title = QtWidgets.QLabel("Workflow")
            rail_title.setObjectName("PanelTitle")
            self._stage_list = QtWidgets.QListWidget()
            self._stage_list.setObjectName("WorkflowRail")
            self._stage_list.setMinimumWidth(260)
            self._stage_list.setMaximumWidth(330)
            self._stage_list.setSpacing(6)
            self._stage_list.currentRowChanged.connect(self._select_stage_by_row)

            rail_panel = QtWidgets.QWidget()
            rail_layout = QtWidgets.QVBoxLayout(rail_panel)
            rail_layout.setContentsMargins(0, 0, 0, 0)
            rail_layout.setSpacing(10)
            rail_layout.addWidget(rail_title)
            rail_layout.addWidget(self._stage_list, 1)

            self._stage_title = QtWidgets.QLabel()
            self._stage_title.setObjectName("StageTitle")

            self._status_label = QtWidgets.QLabel()
            self._status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._status_label.setMinimumWidth(128)
            self._status_label.setObjectName("StatusPill")

            title_row = QtWidgets.QHBoxLayout()
            title_row.addWidget(self._stage_title, 1)
            title_row.addWidget(self._status_label)

            self._message_label = QtWidgets.QLabel()
            self._message_label.setWordWrap(True)
            self._message_label.setObjectName("StageMessage")

            status_card, self._stage_summary_label = _make_card("Stage summary")
            paths_card, self._paths_label = _make_card("Checked paths")
            warnings_card, self._warnings_label = _make_card("Warnings")
            errors_card, self._errors_label = _make_card("Errors")
            next_card, self._next_action_label = _make_card("Next action")

            cards_grid = QtWidgets.QGridLayout()
            cards_grid.setSpacing(12)
            cards_grid.addWidget(status_card, 0, 0)
            cards_grid.addWidget(next_card, 0, 1)
            cards_grid.addWidget(paths_card, 1, 0, 1, 2)
            cards_grid.addWidget(warnings_card, 2, 0)
            cards_grid.addWidget(errors_card, 2, 1)
            cards_grid.setColumnStretch(0, 1)
            cards_grid.setColumnStretch(1, 1)

            self._command_preview = QtWidgets.QLineEdit()
            self._command_preview.setReadOnly(True)
            self._command_preview.setPlaceholderText(
                "No command preview is defined for this stage."
            )
            self._copy_button = QtWidgets.QPushButton("Copy Command")
            self._copy_button.clicked.connect(self._copy_command)
            self._copy_feedback = QtWidgets.QLabel()
            self._copy_feedback.setObjectName("MutedText")
            self._copy_feedback.setWordWrap(True)

            command_card = QtWidgets.QFrame()
            command_card.setObjectName("Card")
            command_layout = QtWidgets.QVBoxLayout(command_card)
            command_layout.setContentsMargins(14, 12, 14, 12)
            command_layout.setSpacing(8)
            command_title = QtWidgets.QLabel("Command preview")
            command_title.setObjectName("CardTitle")
            command_note = QtWidgets.QLabel(
                "Copy the preview into a terminal. The dashboard stays read-only "
                "and does not run workflow commands."
            )
            command_note.setObjectName("MutedText")
            command_note.setWordWrap(True)
            command_row = QtWidgets.QHBoxLayout()
            command_row.addWidget(self._command_preview, 1)
            command_row.addWidget(self._copy_button)
            command_layout.addWidget(command_title)
            command_layout.addWidget(command_note)
            command_layout.addLayout(command_row)
            command_layout.addWidget(self._copy_feedback)

            detail_content = QtWidgets.QWidget()
            detail_content_layout = QtWidgets.QVBoxLayout(detail_content)
            detail_content_layout.setContentsMargins(0, 0, 0, 0)
            detail_content_layout.setSpacing(12)
            detail_content_layout.addLayout(title_row)
            detail_content_layout.addWidget(self._message_label)
            detail_content_layout.addLayout(cards_grid)
            detail_content_layout.addWidget(command_card)
            detail_content_layout.addStretch(1)

            detail_scroll = QtWidgets.QScrollArea()
            detail_scroll.setObjectName("DetailScroll")
            detail_scroll.setWidgetResizable(True)
            detail_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            detail_scroll.setWidget(detail_content)

            detail_panel = QtWidgets.QWidget()
            detail_layout = QtWidgets.QVBoxLayout(detail_panel)
            detail_layout.setContentsMargins(18, 0, 18, 0)
            detail_layout.addWidget(detail_scroll, 1)

            self._health_status_label = QtWidgets.QLabel()
            self._health_status_label.setObjectName("HealthStatus")
            self._health_status_label.setWordWrap(True)

            self._health_message_label = QtWidgets.QLabel()
            self._health_message_label.setObjectName("MutedText")
            self._health_message_label.setWordWrap(True)

            self._health_counts_label = QtWidgets.QLabel()
            self._health_counts_label.setObjectName("HealthCounts")
            self._health_counts_label.setWordWrap(True)

            self._health_issues_label = QtWidgets.QLabel()
            self._health_issues_label.setObjectName("CardBody")
            self._health_issues_label.setWordWrap(True)

            self._health_next_title = QtWidgets.QLabel()
            self._health_next_title.setObjectName("CardTitle")
            self._health_next_title.setWordWrap(True)

            self._health_next_body = QtWidgets.QLabel()
            self._health_next_body.setObjectName("CardBody")
            self._health_next_body.setWordWrap(True)
            self._health_next_body.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._health_panel = QtWidgets.QWidget()
            self._health_panel.setMinimumWidth(280)
            self._health_panel.setMaximumWidth(390)
            health_layout = QtWidgets.QVBoxLayout(self._health_panel)
            health_layout.setContentsMargins(0, 0, 0, 0)
            health_layout.setSpacing(12)
            health_title = QtWidgets.QLabel("Project Health")
            health_title.setObjectName("PanelTitle")
            health_layout.addWidget(health_title)
            health_layout.addWidget(self._health_status_label)
            health_layout.addWidget(self._health_message_label)
            health_layout.addWidget(_divider(QtWidgets))
            health_layout.addWidget(self._health_counts_label)
            health_layout.addWidget(_divider(QtWidgets))
            issues_title = QtWidgets.QLabel("Warnings and errors")
            issues_title.setObjectName("CardTitle")
            health_layout.addWidget(issues_title)
            health_layout.addWidget(self._health_issues_label)
            health_layout.addWidget(_divider(QtWidgets))
            health_layout.addWidget(self._health_next_title)
            health_layout.addWidget(self._health_next_body)
            health_layout.addStretch(1)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            splitter.addWidget(rail_panel)
            splitter.addWidget(detail_panel)
            splitter.addWidget(self._health_panel)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setStretchFactor(2, 0)

            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            layout.setContentsMargins(18, 16, 18, 16)
            layout.setSpacing(16)
            layout.addLayout(header_layout)
            layout.addWidget(splitter, 1)
            self.setCentralWidget(central)
            self.setStyleSheet(_style_sheet())
            self.statusBar().showMessage(
                "Read-only dashboard. Command previews are not executed."
            )

            self._refresh()

        def _browse_project_root(self) -> None:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select PoseTag project root",
                self._root_input.text() or str(Path.cwd()),
            )
            if selected:
                self._root_input.setText(selected)
                self._refresh()

        def _open_project_folder(self) -> None:
            root = Path(self._root_input.text()).expanduser()
            if not root.is_dir():
                message = "Project folder is not available to open."
                self.statusBar().showMessage(message, 4000)
                self._root_status_label.setText(_project_root_hint(root))
                return

            opened = QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(root.resolve()))
            )
            message = (
                "Opened project folder."
                if opened
                else "Could not open the project folder."
            )
            self.statusBar().showMessage(message, 4000)

        def _refresh(self) -> None:
            root = Path(self._root_input.text()).expanduser()
            self._render_project_context(root)
            try:
                self._models = inspect_project_view(root)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                self._models = ()
                self._render_error(root, exc)
                return
            self._render_stage_list()
            self._select_stage(self._selected_stage_id)
            self._render_health()

        def _render_stage_list(self) -> None:
            self._stage_list.blockSignals(True)
            self._stage_list.clear()
            for model in self._models:
                item = QtWidgets.QListWidgetItem(
                    f"Stage {model.stage_id}\n{model.name}\n{model.status_label}"
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, model.stage_id)
                item.setToolTip(model.message)
                item.setSizeHint(QtCore.QSize(250, 74))
                item.setForeground(QtGui.QBrush(QtGui.QColor(model.status_color)))
                item.setBackground(
                    QtGui.QBrush(QtGui.QColor(model.status_background))
                )
                self._stage_list.addItem(item)
            self._stage_list.blockSignals(False)

        def _select_stage_by_row(self, row: int) -> None:
            item = self._stage_list.item(row)
            if item is None:
                return
            self._select_stage(int(item.data(QtCore.Qt.ItemDataRole.UserRole)))

        def _select_stage(self, stage_id: int) -> None:
            model = self._model_by_stage_id(stage_id) or self._first_model()
            if model is None:
                return

            self._selected_stage_id = model.stage_id
            for index in range(self._stage_list.count()):
                item = self._stage_list.item(index)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) == model.stage_id:
                    if self._stage_list.currentRow() != index:
                        self._stage_list.setCurrentRow(index)
                    break

            self._stage_title.setText(f"Stage {model.stage_id}: {model.name}")
            self._status_label.setText(model.status_label)
            self._status_label.setStyleSheet(
                _pill_style(
                    foreground=model.status_color,
                    background=model.status_background,
                    border=model.status_border,
                )
            )
            self._message_label.setText(model.message)
            self._stage_summary_label.setText(
                f"{model.status_label}\n\n{model.message}"
            )
            self._paths_label.setText(
                _format_card_items(
                    model.checked_paths,
                    "No paths are checked for this stage yet.",
                )
            )
            self._warnings_label.setText(
                _format_card_items(model.warnings, "No warnings reported.")
            )
            self._errors_label.setText(
                _format_card_items(model.errors, "No errors reported.")
            )
            self._next_action_label.setText(model.next_action)
            self._command_preview.setText(model.command_preview)
            self._command_preview.setCursorPosition(0)
            self._copy_button.setEnabled(bool(model.command_preview))
            self._copy_feedback.setText(_command_ready_message(model.command_preview))

        def _render_health(self) -> None:
            health = project_health_view(self._models)
            self._health_status_label.setText(health.headline)
            self._health_status_label.setStyleSheet(
                _health_status_style(
                    foreground=health.status_color,
                    background=health.status_background,
                    border=health.status_border,
                )
            )
            self._health_message_label.setText(health.message)
            self._health_counts_label.setText(_format_health_counts(health))
            self._health_issues_label.setText(_format_health_issues(health))
            self._health_next_title.setText("Next recommended action")
            self._health_next_body.setText(
                f"{health.next_stage_label}\n\n{health.next_action}"
            )

        def _render_error(self, root: Path, exc: Exception) -> None:
            self._render_project_context(root)
            self._stage_list.clear()
            self._stage_title.setText("Project inspection failed")
            self._status_label.setText("Needs attention")
            self._status_label.setStyleSheet(
                _pill_style(
                    foreground="#8a5a00",
                    background="#fff4d7",
                    border="#ddb44b",
                )
            )
            self._message_label.setText(str(root))
            self._stage_summary_label.setText(str(exc))
            self._paths_label.setText("Project inspection stopped before path checks.")
            self._warnings_label.setText("No warnings reported.")
            self._errors_label.setText(str(exc))
            self._next_action_label.setText(
                "Check the project path and refresh the dashboard."
            )
            self._command_preview.clear()
            self._copy_button.setEnabled(False)
            self._copy_feedback.setText(_command_ready_message(""))
            self._render_health()

        def _copy_command(self) -> None:
            command = self._command_preview.text()
            if command:
                QtWidgets.QApplication.clipboard().setText(command)
            message = _copy_confirmation_message(command)
            self._copy_feedback.setText(message)
            self.statusBar().showMessage(message, 3000)

        def _render_project_context(self, root: Path) -> None:
            title = root.name or str(root)
            self._project_title.setText(f"Project: {title}")
            self._root_status_label.setText(_project_root_hint(root))
            self._open_folder_button.setEnabled(root.is_dir())

        def _model_by_stage_id(self, stage_id: int) -> Optional[StageViewModel]:
            for model in self._models:
                if model.stage_id == stage_id:
                    return model
            return None

        def _first_model(self) -> Optional[StageViewModel]:
            return self._models[0] if self._models else None

    return PoseTagMainWindow(project_root)


def _format_stage_details(model: StageViewModel) -> str:
    lines = [
        "Status",
        model.status_label,
        "",
        "Checked paths",
        *(_format_items(model.checked_paths) or ["No paths are checked yet."]),
        "",
        "Warnings",
        *(_format_items(model.warnings) or ["No warnings reported."]),
        "",
        "Errors",
        *(_format_items(model.errors) or ["No errors reported."]),
        "",
        "Next action",
        model.next_action,
    ]
    return "\n".join(lines)


def _format_card_items(items: tuple[str, ...], empty_text: str) -> str:
    rendered = _format_items(items)
    if not rendered:
        return empty_text
    return "\n".join(rendered)


def _format_items(items: tuple[str, ...]) -> list[str]:
    return [f"- {item}" for item in items]


def _format_health_counts(health: ProjectHealthViewModel) -> str:
    return "\n".join(
        f"{count.label}: {count.count}" for count in health.counts
    )


def _format_health_issues(health: ProjectHealthViewModel) -> str:
    if health.warning_count == 0 and health.error_count == 0:
        return "No warnings or errors reported."
    return (
        f"Warnings: {health.warning_count}\n"
        f"Errors: {health.error_count}"
    )


def _command_ready_message(command: str) -> str:
    if command:
        return "Ready to copy. Replace placeholder values before running it."
    return "No command preview is available for this stage."


def _copy_confirmation_message(command: str) -> str:
    if command:
        return "Copied command preview to the clipboard."
    return "No command preview is available for this stage."


def _project_root_hint(root: Path) -> str:
    if root.is_dir():
        return "Project folder found. Status checks are read-only."
    if root.exists():
        return "Selected path exists but is not a folder."
    return "Project folder not found. Status checks are read-only and created no files."


def _pill_style(foreground: str, background: str, border: str) -> str:
    return (
        f"background: {background}; color: {foreground}; "
        f"border: 1px solid {border}; border-radius: 6px; "
        "padding: 6px 10px; font-weight: 700;"
    )


def _health_status_style(foreground: str, background: str, border: str) -> str:
    return (
        f"background: {background}; color: {foreground}; "
        f"border: 1px solid {border}; border-radius: 6px; "
        "padding: 14px; font-size: 18px; font-weight: 700;"
    )


def _style_sheet() -> str:
    return """
QMainWindow, QWidget {
    background: #f6f8fb;
    color: #17202a;
    font-size: 13px;
}
QLineEdit, QListWidget {
    background: #ffffff;
    border: 1px solid #cfd6df;
    border-radius: 4px;
    padding: 6px;
}
QScrollArea#DetailScroll {
    background: transparent;
    border: none;
}
QPushButton {
    background: #22547d;
    color: #ffffff;
    border: 1px solid #1b4568;
    border-radius: 4px;
    padding: 7px 12px;
    font-weight: 600;
}
QPushButton:disabled {
    background: #b8c1ca;
    border-color: #aab4bf;
}
QFrame#Card {
    background: #ffffff;
    border: 1px solid #d7dee7;
    border-radius: 6px;
}
QFrame#Divider {
    color: #d7dee7;
    background: #d7dee7;
    max-height: 1px;
}
QLabel#AppTitle {
    font-size: 24px;
    font-weight: 700;
}
QLabel#PanelTitle {
    color: #26323f;
    font-size: 14px;
    font-weight: 700;
}
QLabel#CardTitle {
    color: #26323f;
    font-weight: 700;
}
QLabel#CardBody {
    color: #334150;
}
QLabel#MutedText {
    color: #59636f;
}
QLabel#HealthCounts {
    color: #26323f;
    background: #ffffff;
    border: 1px solid #d7dee7;
    border-radius: 6px;
    padding: 10px;
}
QStatusBar {
    background: #ffffff;
    color: #59636f;
    border-top: 1px solid #d7dee7;
}
QListWidget#WorkflowRail {
    padding: 8px;
}
QListWidget::item {
    padding: 10px;
    border: 1px solid transparent;
    border-radius: 6px;
}
QListWidget::item:selected {
    background: #e7f0fa;
    color: #0b1724;
    border: 1px solid #8ab5dd;
    border-left: 4px solid #22547d;
}
QListWidget::item:selected:!active {
    background: #e7f0fa;
    color: #0b1724;
    border: 1px solid #8ab5dd;
    border-left: 4px solid #22547d;
}
QLabel#StageTitle {
    font-size: 22px;
    font-weight: 700;
}
QLabel#StageMessage {
    color: #394653;
    padding: 2px 0 4px 0;
}
"""
