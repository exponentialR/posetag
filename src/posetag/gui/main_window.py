"""Main window construction for the optional PoseTag workflow dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from posetag.gui.models import StageViewModel, inspect_project_view


def build_main_window(qt: Any, project_root: Path) -> Any:
    """Build the PySide6 main window without importing Qt at module import time."""

    QtCore = qt.QtCore
    QtGui = qt.QtGui
    QtWidgets = qt.QtWidgets

    class PoseTagMainWindow(QtWidgets.QMainWindow):
        def __init__(self, initial_project_root: Path) -> None:
            super().__init__()
            self._models: tuple[StageViewModel, ...] = ()
            self._selected_stage_id = 0

            self.setWindowTitle("PoseTag Workflow Dashboard")
            self._root_input = QtWidgets.QLineEdit(str(initial_project_root))
            self._root_input.setMinimumWidth(360)
            self._root_input.returnPressed.connect(self._refresh)

            browse_button = QtWidgets.QPushButton("Browse")
            browse_button.clicked.connect(self._browse_project_root)

            refresh_button = QtWidgets.QPushButton("Refresh")
            refresh_button.clicked.connect(self._refresh)

            root_bar = QtWidgets.QHBoxLayout()
            root_bar.addWidget(QtWidgets.QLabel("Project root"))
            root_bar.addWidget(self._root_input, 1)
            root_bar.addWidget(browse_button)
            root_bar.addWidget(refresh_button)

            self._stage_list = QtWidgets.QListWidget()
            self._stage_list.setMinimumWidth(260)
            self._stage_list.setMaximumWidth(330)
            self._stage_list.currentRowChanged.connect(self._select_stage_by_row)

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

            self._details_text = QtWidgets.QPlainTextEdit()
            self._details_text.setReadOnly(True)
            self._details_text.setMinimumHeight(320)

            self._command_preview = QtWidgets.QLineEdit()
            self._command_preview.setReadOnly(True)
            self._copy_button = QtWidgets.QPushButton("Copy")
            self._copy_button.clicked.connect(self._copy_command)

            command_row = QtWidgets.QHBoxLayout()
            command_row.addWidget(QtWidgets.QLabel("Command preview"))
            command_row.addWidget(self._command_preview, 1)
            command_row.addWidget(self._copy_button)

            detail_panel = QtWidgets.QWidget()
            detail_layout = QtWidgets.QVBoxLayout(detail_panel)
            detail_layout.addLayout(title_row)
            detail_layout.addWidget(self._message_label)
            detail_layout.addWidget(self._details_text, 1)
            detail_layout.addLayout(command_row)

            self._health_text = QtWidgets.QPlainTextEdit()
            self._health_text.setReadOnly(True)
            self._health_text.setMinimumWidth(260)
            self._health_text.setMaximumWidth(380)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            splitter.addWidget(self._stage_list)
            splitter.addWidget(detail_panel)
            splitter.addWidget(self._health_text)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setStretchFactor(2, 0)

            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            layout.addLayout(root_bar)
            layout.addWidget(splitter, 1)
            self.setCentralWidget(central)
            self.setStyleSheet(_style_sheet())

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

        def _refresh(self) -> None:
            root = Path(self._root_input.text()).expanduser()
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
                    f"{model.stage_id}. {model.name}\n{model.status_label}"
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, model.stage_id)
                item.setForeground(QtGui.QBrush(QtGui.QColor("#17202a")))
                item.setBackground(
                    QtGui.QBrush(QtGui.QColor(_soft_color(model.status)))
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
                f"background: {model.status_color}; color: white; "
                "border-radius: 4px; padding: 5px 10px;"
            )
            self._message_label.setText(model.message)
            self._details_text.setPlainText(_format_stage_details(model))
            self._command_preview.setText(model.command_preview)
            self._copy_button.setEnabled(bool(model.command_preview))

        def _render_health(self) -> None:
            counts: dict[str, int] = {}
            for model in self._models:
                counts[model.status_label] = counts.get(model.status_label, 0) + 1

            next_model = _first_actionable_model(self._models)
            lines = [
                "Project Health",
                "",
                *(f"{label}: {count}" for label, count in sorted(counts.items())),
                "",
                "Next recommended action",
                "",
                (
                    next_model.next_action
                    if next_model
                    else "All inspected stages are complete."
                ),
            ]
            self._health_text.setPlainText("\n".join(lines))

        def _render_error(self, root: Path, exc: Exception) -> None:
            self._stage_list.clear()
            self._stage_title.setText("Project inspection failed")
            self._status_label.setText("Needs attention")
            self._message_label.setText(str(root))
            self._details_text.setPlainText(str(exc))
            self._command_preview.clear()
            self._copy_button.setEnabled(False)
            self._health_text.setPlainText("Project Health\n\nInspection failed.")

        def _copy_command(self) -> None:
            command = self._command_preview.text()
            if command:
                QtWidgets.QApplication.clipboard().setText(command)

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
        *(_format_items(model.checked_paths) or ["(none)"]),
        "",
        "Warnings",
        *(_format_items(model.warnings) or ["(none)"]),
        "",
        "Errors",
        *(_format_items(model.errors) or ["(none)"]),
        "",
        "Next action",
        model.next_action,
    ]
    return "\n".join(lines)


def _format_items(items: tuple[str, ...]) -> list[str]:
    return [f"- {item}" for item in items]


def _first_actionable_model(
    models: tuple[StageViewModel, ...],
) -> Optional[StageViewModel]:
    for model in models:
        if model.status != "complete" and model.status != "not_applicable":
            return model
    for model in models:
        if model.status != "complete":
            return model
    return None


def _soft_color(status: str) -> str:
    return {
        "complete": "#dceee4",
        "missing": "#fff0bd",
        "needs_attention": "#f8d8d0",
        "not_applicable": "#e6e9ed",
    }.get(status, "#e6e9ed")


def _style_sheet() -> str:
    return """
QMainWindow, QWidget {
    background: #f7f8fa;
    color: #17202a;
    font-size: 13px;
}
QLineEdit, QPlainTextEdit, QListWidget {
    background: #ffffff;
    border: 1px solid #cfd6df;
    border-radius: 4px;
    padding: 6px;
}
QPushButton {
    background: #244d7a;
    color: #ffffff;
    border: 1px solid #1f4268;
    border-radius: 4px;
    padding: 7px 12px;
}
QPushButton:disabled {
    background: #b8c1ca;
    border-color: #aab4bf;
}
QListWidget::item {
    padding: 10px;
    border-bottom: 1px solid #d8dee6;
}
QListWidget::item:selected {
    background: #e8f1fb;
    color: #0b1724;
    border-left: 4px solid #244d7a;
}
QListWidget::item:selected:!active {
    background: #e8f1fb;
    color: #0b1724;
    border-left: 4px solid #244d7a;
}
QLabel#StageTitle {
    font-size: 22px;
    font-weight: 700;
}
QLabel#StageMessage {
    color: #394653;
    padding: 4px 0 10px 0;
}
"""
