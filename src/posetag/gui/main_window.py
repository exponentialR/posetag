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
from posetag.workflows.charuco_setup import (
    DEFAULT_DICTIONARY,
    DEFAULT_DPI,
    DEFAULT_MARKER_LENGTH_MM,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SQUARE_LENGTH_MM,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
    ORIENTATION_CHOICES,
    OUTPUT_FORMAT_CHOICES,
    PAPER_CHOICES,
    PRINT_GUIDANCE,
    CharucoBoardSetupConfig,
    CharucoBoardSetupError,
    default_output_dir,
    dictionary_choices,
    generate_charuco_board_setup,
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

    def _make_int_spin(
        minimum: int,
        maximum: int,
        value: int,
        suffix: str = "",
    ) -> Any:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        spin.setMinimumWidth(120)
        return spin

    def _make_float_spin(
        minimum: float,
        maximum: float,
        value: float,
        suffix: str = "",
    ) -> Any:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(1.0)
        if suffix:
            spin.setSuffix(suffix)
        spin.setMinimumWidth(150)
        return spin

    def _make_field(label_text: str, field: Any) -> Any:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QtWidgets.QLabel(label_text)
        label.setObjectName("FieldLabel")
        layout.addWidget(label)
        layout.addWidget(field)
        return wrapper

    def _safe_dictionary_choices() -> tuple[str, ...]:
        try:
            names = dictionary_choices()
        except Exception:  # pragma: no cover - defensive UI boundary
            return (DEFAULT_DICTIONARY,)
        return names or (DEFAULT_DICTIONARY,)

    def _make_stage_rail_entry(model: StageViewModel, selected: bool) -> Any:
        entry = QtWidgets.QFrame()
        entry.setObjectName("RailStageEntry")
        entry.setStyleSheet(_stage_rail_entry_style(model, selected=selected))
        entry.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )

        layout = QtWidgets.QHBoxLayout(entry)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(5)

        number = QtWidgets.QLabel(str(model.stage_id))
        number.setObjectName("RailStageNumber")
        number.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        number.setFixedWidth(12)
        number.setStyleSheet(_stage_rail_number_style(model))

        dot = QtWidgets.QLabel()
        dot.setObjectName("RailStageDot")
        dot.setFixedSize(7, 7)
        dot.setStyleSheet(_stage_rail_dot_style(model))

        copy = QtWidgets.QVBoxLayout()
        copy.setContentsMargins(0, 0, 0, 0)
        copy.setSpacing(1)

        name = QtWidgets.QLabel(_stage_rail_name(model))
        name.setObjectName("RailStageName")
        name.setWordWrap(True)

        status = QtWidgets.QLabel(_stage_rail_status_label(model))
        status.setObjectName("RailStageStatus")
        status.setStyleSheet(_stage_status_chip_style(model))

        copy.addWidget(name)
        copy.addWidget(status, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(number, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(dot, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(copy, 1)
        return entry

    class _CharucoPreviewCanvas(QtWidgets.QLabel):
        def __init__(self) -> None:
            super().__init__("No generated board yet.")
            self._source_pixmap: Any = None
            self.setObjectName("CharucoPreviewCanvas")
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(300, 300)
            self.setMaximumHeight(330)
            self.setWordWrap(True)

        def clear_preview(self, message: str = "No generated board yet.") -> None:
            self._source_pixmap = None
            self.clear()
            self.setText(message)

        def show_preview(self, path: Path) -> bool:
            pixmap = QtGui.QPixmap(str(path))
            if pixmap.isNull():
                self.clear_preview("Preview unavailable.")
                return False
            self._source_pixmap = pixmap
            self.setText("")
            self._refresh_pixmap()
            return True

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._refresh_pixmap()

        def _refresh_pixmap(self) -> None:
            if self._source_pixmap is None or self._source_pixmap.isNull():
                return
            size = self.contentsRect().size()
            if size.width() <= 0 or size.height() <= 0:
                return
            scaled = self._source_pixmap.scaled(
                size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    class PoseTagMainWindow(QtWidgets.QMainWindow):
        def __init__(self, initial_project_root: Path) -> None:
            super().__init__()
            self._models: tuple[StageViewModel, ...] = ()
            self._selected_stage_id = 0
            self._stage_widgets: dict[int, Any] = {}

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

            project_root_label = QtWidgets.QLabel("Project root")
            project_root_label.setObjectName("FieldLabel")

            root_bar = QtWidgets.QHBoxLayout()
            root_bar.addWidget(project_root_label)
            root_bar.addWidget(self._root_input, 1)
            root_bar.addWidget(browse_button)
            root_bar.addWidget(self._open_folder_button)
            root_bar.addWidget(refresh_button)

            header_layout = QtWidgets.QVBoxLayout()
            header_layout.setSpacing(5)
            header_layout.addWidget(self._project_title)
            header_layout.addLayout(root_bar)
            header_layout.addWidget(self._root_status_label)

            header_panel = QtWidgets.QFrame()
            header_panel.setObjectName("HeaderPanel")
            header_panel_layout = QtWidgets.QVBoxLayout(header_panel)
            header_panel_layout.setContentsMargins(16, 12, 16, 12)
            header_panel_layout.addLayout(header_layout)

            rail_title = QtWidgets.QLabel("Workflow")
            rail_title.setObjectName("PanelTitle")

            self._stage_list = QtWidgets.QListWidget()
            self._stage_list.setObjectName("WorkflowRail")
            self._stage_list.setMinimumWidth(126)
            self._stage_list.setMaximumWidth(134)
            self._stage_list.setSpacing(4)
            self._stage_list.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._stage_list.currentRowChanged.connect(self._select_stage_by_row)

            rail_panel = QtWidgets.QFrame()
            rail_panel.setObjectName("RailPanel")
            rail_panel.setMinimumWidth(150)
            rail_panel.setMaximumWidth(158)
            rail_layout = QtWidgets.QVBoxLayout(rail_panel)
            rail_layout.setContentsMargins(10, 12, 10, 12)
            rail_layout.setSpacing(8)
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
            self._warnings_card = warnings_card
            self._errors_card = errors_card

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
            self._command_preview.setObjectName("CommandPreview")
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
            command_card.setObjectName("CommandCard")
            command_layout = QtWidgets.QVBoxLayout(command_card)
            command_layout.setContentsMargins(14, 12, 14, 12)
            command_layout.setSpacing(8)
            self._command_title = QtWidgets.QLabel("Command preview")
            self._command_title.setObjectName("CardTitle")
            self._command_note = QtWidgets.QLabel()
            self._command_note.setObjectName("MutedText")
            self._command_note.setWordWrap(True)
            command_row = QtWidgets.QHBoxLayout()
            command_row.addWidget(self._command_preview, 1)
            command_row.addWidget(self._copy_button)
            command_layout.addWidget(self._command_title)
            command_layout.addWidget(self._command_note)
            command_layout.addLayout(command_row)
            command_layout.addWidget(self._copy_feedback)

            self._last_charuco_output_dir: Optional[Path] = None
            self._last_charuco_board_file: Optional[Path] = None
            self._charuco_card = QtWidgets.QFrame()
            self._charuco_card.setObjectName("ActionCard")
            charuco_layout = QtWidgets.QVBoxLayout(self._charuco_card)
            charuco_layout.setContentsMargins(14, 12, 14, 12)
            charuco_layout.setSpacing(9)

            charuco_title = QtWidgets.QLabel("ChArUco board setup")
            charuco_title.setObjectName("CardTitle")
            charuco_note = QtWidgets.QLabel(
                "Generate the printed board for Stage 1. Calibration capture "
                "and solve stay in posetag-calib-charuco."
            )
            charuco_note.setObjectName("MutedText")
            charuco_note.setWordWrap(True)

            self._charuco_squares_x = _make_int_spin(
                2,
                99,
                DEFAULT_SQUARES_X,
            )
            self._charuco_squares_y = _make_int_spin(
                2,
                99,
                DEFAULT_SQUARES_Y,
            )
            self._charuco_square_length = _make_float_spin(
                0.1,
                1000.0,
                DEFAULT_SQUARE_LENGTH_MM,
                " mm",
            )
            self._charuco_marker_length = _make_float_spin(
                0.1,
                1000.0,
                DEFAULT_MARKER_LENGTH_MM,
                " mm",
            )
            self._charuco_dictionary = QtWidgets.QComboBox()
            self._charuco_dictionary.addItems(list(_safe_dictionary_choices()))
            self._charuco_dictionary.setCurrentText(DEFAULT_DICTIONARY)
            self._charuco_dictionary.setMinimumWidth(180)
            self._charuco_paper = QtWidgets.QComboBox()
            self._charuco_paper.addItems(list(PAPER_CHOICES))
            self._charuco_paper.setCurrentText("A4")
            self._charuco_paper.setMinimumWidth(180)
            self._charuco_orientation = QtWidgets.QComboBox()
            self._charuco_orientation.addItems(list(ORIENTATION_CHOICES))
            self._charuco_orientation.setCurrentText("portrait")
            self._charuco_orientation.setMinimumWidth(180)
            self._charuco_dpi = _make_int_spin(1, 2400, DEFAULT_DPI)
            self._charuco_output_format = QtWidgets.QComboBox()
            self._charuco_output_format.addItems(list(OUTPUT_FORMAT_CHOICES))
            self._charuco_output_format.setCurrentText(DEFAULT_OUTPUT_FORMAT)
            self._charuco_output_format.setMinimumWidth(180)

            self._charuco_out_dir = QtWidgets.QLineEdit()
            self._charuco_out_dir.setPlaceholderText(
                "Default: <project_root>/calib/boards/"
            )
            charuco_browse_button = QtWidgets.QPushButton("Browse")
            charuco_browse_button.clicked.connect(self._browse_charuco_output_dir)
            charuco_out_row = QtWidgets.QHBoxLayout()
            charuco_out_row.setContentsMargins(0, 0, 0, 0)
            charuco_out_row.addWidget(self._charuco_out_dir, 1)
            charuco_out_row.addWidget(charuco_browse_button)
            charuco_out_widget = QtWidgets.QWidget()
            charuco_out_widget.setLayout(charuco_out_row)

            charuco_grid = QtWidgets.QGridLayout()
            charuco_grid.setContentsMargins(0, 0, 0, 0)
            charuco_grid.setHorizontalSpacing(14)
            charuco_grid.setVerticalSpacing(10)
            charuco_grid.addWidget(
                _make_field("Squares X", self._charuco_squares_x),
                0,
                0,
            )
            charuco_grid.addWidget(
                _make_field("Squares Y", self._charuco_squares_y),
                0,
                1,
            )
            charuco_grid.addWidget(
                _make_field("Square length", self._charuco_square_length),
                1,
                0,
            )
            charuco_grid.addWidget(
                _make_field("Marker length", self._charuco_marker_length),
                1,
                1,
            )
            charuco_grid.addWidget(
                _make_field("Dictionary", self._charuco_dictionary),
                2,
                0,
            )
            charuco_grid.addWidget(
                _make_field("Paper size", self._charuco_paper),
                2,
                1,
            )
            charuco_grid.addWidget(
                _make_field("Orientation", self._charuco_orientation),
                3,
                0,
            )
            charuco_grid.addWidget(_make_field("DPI", self._charuco_dpi), 3, 1)
            charuco_grid.addWidget(
                _make_field("Output format", self._charuco_output_format),
                4,
                0,
            )
            charuco_grid.addWidget(
                _make_field("Output folder", charuco_out_widget),
                5,
                0,
                1,
                2,
            )
            charuco_grid.setColumnStretch(0, 1)
            charuco_grid.setColumnStretch(1, 1)

            self._charuco_generate_button = QtWidgets.QPushButton("Generate Board")
            self._charuco_generate_button.setObjectName("PrimaryActionButton")
            self._charuco_generate_button.clicked.connect(self._generate_charuco_board)
            self._charuco_open_folder_button = QtWidgets.QPushButton(
                "Open Output Folder"
            )
            self._charuco_open_folder_button.setObjectName("SecondaryActionButton")
            self._charuco_open_folder_button.setEnabled(False)
            self._charuco_open_folder_button.clicked.connect(
                self._open_charuco_output_folder
            )
            self._charuco_open_file_button = QtWidgets.QPushButton("Open Board File")
            self._charuco_open_file_button.setObjectName("SecondaryActionButton")
            self._charuco_open_file_button.setEnabled(False)
            self._charuco_open_file_button.clicked.connect(
                self._open_charuco_board_file
            )
            charuco_primary_action_row = QtWidgets.QHBoxLayout()
            charuco_primary_action_row.addWidget(self._charuco_generate_button)
            charuco_primary_action_row.addStretch(1)

            charuco_file_action_row = QtWidgets.QHBoxLayout()
            charuco_file_action_row.addWidget(self._charuco_open_file_button)
            charuco_file_action_row.addWidget(self._charuco_open_folder_button)
            charuco_file_action_row.addStretch(1)

            self._charuco_guidance = QtWidgets.QLabel(PRINT_GUIDANCE)
            self._charuco_guidance.setObjectName("GuidanceText")
            self._charuco_guidance.setWordWrap(True)

            self._charuco_outputs = QtWidgets.QLabel(
                "No generated files yet."
            )
            self._charuco_outputs.setObjectName("OutputText")
            self._charuco_outputs.setWordWrap(True)
            self._charuco_outputs.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignTop
            )
            self._charuco_outputs.setMinimumHeight(78)
            self._charuco_outputs.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._charuco_preview = _CharucoPreviewCanvas()
            charuco_preview_title = QtWidgets.QLabel("Preview")
            charuco_preview_title.setObjectName("FieldLabel")
            charuco_outputs_title = QtWidgets.QLabel("Generated files")
            charuco_outputs_title.setObjectName("FieldLabel")

            charuco_preview_column = QtWidgets.QFrame()
            charuco_preview_column.setObjectName("CharucoPreviewColumn")
            charuco_preview_layout = QtWidgets.QVBoxLayout(charuco_preview_column)
            charuco_preview_layout.setContentsMargins(10, 10, 10, 10)
            charuco_preview_layout.setSpacing(7)
            charuco_preview_layout.addWidget(charuco_preview_title)
            charuco_preview_layout.addWidget(self._charuco_preview)
            charuco_preview_layout.addWidget(charuco_outputs_title)
            charuco_preview_layout.addWidget(self._charuco_outputs)
            charuco_preview_layout.addLayout(charuco_file_action_row)
            charuco_preview_layout.addStretch(1)
            charuco_preview_column.setMinimumWidth(320)

            charuco_controls = QtWidgets.QWidget()
            charuco_controls.setObjectName("CharucoControls")
            charuco_controls_layout = QtWidgets.QVBoxLayout(charuco_controls)
            charuco_controls_layout.setContentsMargins(0, 0, 0, 0)
            charuco_controls_layout.setSpacing(9)
            charuco_controls_layout.addLayout(charuco_grid)
            charuco_controls_layout.addWidget(self._charuco_guidance)
            charuco_controls_layout.addLayout(charuco_primary_action_row)
            charuco_controls_layout.addStretch(1)

            charuco_body = QtWidgets.QHBoxLayout()
            charuco_body.setContentsMargins(0, 0, 0, 0)
            charuco_body.setSpacing(12)
            charuco_body.addWidget(
                charuco_controls,
                3,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )
            charuco_body.addWidget(
                charuco_preview_column,
                2,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )

            charuco_layout.addWidget(charuco_title)
            charuco_layout.addWidget(charuco_note)
            charuco_layout.addLayout(charuco_body)

            detail_content = QtWidgets.QWidget()
            detail_content_layout = QtWidgets.QVBoxLayout(detail_content)
            detail_content_layout.setContentsMargins(0, 0, 0, 0)
            detail_content_layout.setSpacing(10)
            detail_content_layout.addLayout(title_row)
            detail_content_layout.addWidget(self._message_label)
            detail_content_layout.addLayout(cards_grid)
            detail_content_layout.addWidget(self._charuco_card)
            detail_content_layout.addWidget(command_card)
            detail_content_layout.addStretch(1)

            detail_scroll = QtWidgets.QScrollArea()
            detail_scroll.setObjectName("DetailScroll")
            detail_scroll.setWidgetResizable(True)
            detail_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            detail_scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            detail_scroll.setWidget(detail_content)

            detail_panel = QtWidgets.QWidget()
            detail_layout = QtWidgets.QVBoxLayout(detail_panel)
            detail_layout.setContentsMargins(14, 0, 14, 0)
            detail_layout.addWidget(detail_scroll, 1)

            self._health_status_label = QtWidgets.QLabel()
            self._health_status_label.setObjectName("HealthStatus")
            self._health_status_label.setWordWrap(True)

            self._health_message_label = QtWidgets.QLabel()
            self._health_message_label.setObjectName("HealthMessage")
            self._health_message_label.setWordWrap(True)

            self._health_counts_label = QtWidgets.QLabel()
            self._health_counts_label.setObjectName("HealthCounts")
            self._health_counts_label.setTextFormat(
                QtCore.Qt.TextFormat.RichText
            )
            self._health_counts_label.setWordWrap(True)

            self._health_issues_label = QtWidgets.QLabel()
            self._health_issues_label.setObjectName("HealthSectionBody")
            self._health_issues_label.setWordWrap(True)

            self._health_issues_title = QtWidgets.QLabel("Checks")
            self._health_issues_title.setObjectName("HealthSectionTitle")
            self._health_issues_title.setWordWrap(True)

            self._health_next_title = QtWidgets.QLabel()
            self._health_next_title.setObjectName("HealthSectionTitle")
            self._health_next_title.setWordWrap(True)

            self._health_next_body = QtWidgets.QLabel()
            self._health_next_body.setObjectName("HealthSectionBody")
            self._health_next_body.setWordWrap(True)
            self._health_next_body.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._health_panel = QtWidgets.QFrame()
            self._health_panel.setObjectName("HealthPanel")
            self._health_panel.setMinimumWidth(280)
            self._health_panel.setMaximumWidth(340)
            health_layout = QtWidgets.QVBoxLayout(self._health_panel)
            health_layout.setContentsMargins(16, 16, 16, 16)
            health_layout.setSpacing(10)
            health_title = QtWidgets.QLabel("Project Health")
            health_title.setObjectName("PanelTitle")
            health_layout.addWidget(health_title)
            health_layout.addWidget(self._health_status_label)
            health_layout.addWidget(self._health_message_label)
            health_layout.addWidget(self._health_counts_label)
            health_layout.addWidget(_divider(QtWidgets))
            health_layout.addWidget(self._health_issues_title)
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
            layout.setSpacing(12)
            layout.addWidget(header_panel)
            layout.addWidget(splitter, 1)
            self.setCentralWidget(central)
            self.setStyleSheet(_style_sheet())
            self.statusBar().showMessage(
                "Dashboard can generate ChArUco board files; "
                "camera workflows are not executed."
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

        def _browse_charuco_output_dir(self) -> None:
            current = self._charuco_out_dir.text().strip()
            start_dir = current or str(self._current_project_root())
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select ChArUco board output folder",
                start_dir,
            )
            if selected:
                self._charuco_out_dir.setText(selected)

        def _generate_charuco_board(self) -> None:
            self._charuco_generate_button.setEnabled(False)
            try:
                result = generate_charuco_board_setup(self._charuco_setup_config())
            except CharucoBoardSetupError as exc:
                message = f"ChArUco board generation failed: {exc}"
                self._charuco_outputs.setText(message)
                self._last_charuco_output_dir = None
                self._last_charuco_board_file = None
                self._charuco_open_folder_button.setEnabled(False)
                self._charuco_open_file_button.setEnabled(False)
                self._charuco_preview.clear_preview("Preview unavailable.")
                self.statusBar().showMessage(message, 6000)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                message = f"ChArUco board generation failed: {exc}"
                self._charuco_outputs.setText(message)
                self._last_charuco_output_dir = None
                self._last_charuco_board_file = None
                self._charuco_open_folder_button.setEnabled(False)
                self._charuco_open_file_button.setEnabled(False)
                self._charuco_preview.clear_preview("Preview unavailable.")
                self.statusBar().showMessage(message, 6000)
            else:
                self._last_charuco_output_dir = result.out_dir
                self._last_charuco_board_file = (
                    result.pdf if result.pdf is not None else result.png
                )
                self._charuco_outputs.setText(_format_charuco_outputs(result))
                self._charuco_open_folder_button.setEnabled(result.out_dir.is_dir())
                self._charuco_open_file_button.setEnabled(
                    self._last_charuco_board_file.is_file()
                )
                self._charuco_preview.show_preview(result.png)
                self.statusBar().showMessage(
                    "Generated ChArUco board outputs.",
                    5000,
                )
                self._refresh()
            finally:
                self._charuco_generate_button.setEnabled(True)

        def _open_charuco_board_file(self) -> None:
            target = self._last_charuco_board_file
            if target is None or not target.is_file():
                message = "Generated ChArUco board file is not available to open."
                self.statusBar().showMessage(message, 4000)
                return

            opened = QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(target.resolve()))
            )
            message = (
                "Opened generated ChArUco board file."
                if opened
                else "Could not open the generated ChArUco board file."
            )
            self.statusBar().showMessage(message, 4000)

        def _open_charuco_output_folder(self) -> None:
            target = self._last_charuco_output_dir or self._charuco_output_target()
            if not target.is_dir():
                message = "ChArUco output folder is not available to open."
                self.statusBar().showMessage(message, 4000)
                return

            opened = QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(target.resolve()))
            )
            message = (
                "Opened ChArUco output folder."
                if opened
                else "Could not open the ChArUco output folder."
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
            self._stage_widgets = {}
            for model in self._models:
                item = QtWidgets.QListWidgetItem(_stage_list_label(model))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, model.stage_id)
                item.setToolTip(model.message)
                item.setSizeHint(QtCore.QSize(130, 42))
                self._stage_list.addItem(item)
                widget = _make_stage_rail_entry(
                    model,
                    selected=model.stage_id == self._selected_stage_id,
                )
                self._stage_list.setItemWidget(item, widget)
                self._stage_widgets[model.stage_id] = widget
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
            self._warnings_card.setVisible(bool(model.warnings))
            self._errors_card.setVisible(bool(model.errors))
            self._next_action_label.setText(model.next_action)
            self._command_title.setText(_command_card_title(model))
            self._command_note.setText(_command_card_note(model))
            self._command_preview.setText(model.command_preview)
            self._command_preview.setCursorPosition(0)
            self._copy_button.setText(_command_button_label(model))
            self._copy_button.setEnabled(bool(model.command_preview))
            self._copy_feedback.setText(
                _command_ready_message(
                    model.command_preview,
                    stage_complete=model.status == "complete",
                )
            )
            self._charuco_card.setVisible(model.stage_id == 1)
            self._render_stage_rail_selection()

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
            self._health_issues_title.setText(
                "Checks clear"
                if health.warning_count == 0 and health.error_count == 0
                else "Warnings and errors"
            )
            self._health_issues_label.setText(_format_health_issues(health))
            self._health_next_title.setText("Next recommended action")
            self._health_next_body.setText(
                f"{health.next_stage_label}\n\n{health.next_action}"
            )

        def _render_error(self, root: Path, exc: Exception) -> None:
            self._render_project_context(root)
            self._stage_list.clear()
            self._stage_widgets = {}
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
            self._warnings_card.setVisible(False)
            self._errors_card.setVisible(True)
            self._next_action_label.setText(
                "Check the project path and refresh the dashboard."
            )
            self._command_preview.clear()
            self._copy_button.setEnabled(False)
            self._copy_feedback.setText(_command_ready_message(""))
            self._charuco_card.setVisible(False)
            self._render_health()

        def _render_stage_rail_selection(self) -> None:
            for model in self._models:
                widget = self._stage_widgets.get(model.stage_id)
                if widget is None:
                    continue
                widget.setStyleSheet(
                    _stage_rail_entry_style(
                        model,
                        selected=model.stage_id == self._selected_stage_id,
                    )
                )

        def _copy_command(self) -> None:
            command = self._command_preview.text()
            if command:
                QtWidgets.QApplication.clipboard().setText(command)
            model = self._model_by_stage_id(self._selected_stage_id)
            message = _copy_confirmation_message(
                command,
                stage_complete=bool(model and model.status == "complete"),
            )
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

        def _current_project_root(self) -> Path:
            text = self._root_input.text().strip()
            return Path(text or ".").expanduser()

        def _charuco_setup_config(self) -> CharucoBoardSetupConfig:
            out_dir_text = self._charuco_out_dir.text().strip()
            out_dir = Path(out_dir_text).expanduser() if out_dir_text else None
            return CharucoBoardSetupConfig(
                project_root=self._current_project_root(),
                squares_x=int(self._charuco_squares_x.value()),
                squares_y=int(self._charuco_squares_y.value()),
                square_length_mm=float(self._charuco_square_length.value()),
                marker_length_mm=float(self._charuco_marker_length.value()),
                dictionary_name=self._charuco_dictionary.currentText(),
                paper=self._charuco_paper.currentText(),
                orientation=self._charuco_orientation.currentText(),
                dpi=int(self._charuco_dpi.value()),
                output_format=self._charuco_output_format.currentText(),
                out_dir=out_dir,
            )

        def _charuco_output_target(self) -> Path:
            out_dir_text = self._charuco_out_dir.text().strip()
            if out_dir_text:
                return Path(out_dir_text).expanduser()
            return default_output_dir(self._current_project_root())

    return PoseTagMainWindow(project_root)


def _stage_list_label(model: StageViewModel) -> str:
    return f"Stage {model.stage_id}\n{model.name}\n{model.status_label}"


_RAIL_STAGE_NAMES = {
    0: "Project",
    1: "ChArUco",
    2: "Calib",
    3: "Tags",
    4: "Boards",
    5: "Shots",
    6: "Annotate",
    7: "Dataset",
    8: "Export",
}

_RAIL_STATUS_LABELS = {
    "complete": "Done",
    "missing": "Missing",
    "needs_attention": "Attention",
    "not_applicable": "N/A",
}

_RAIL_STATUS_BACKGROUNDS = {
    "complete": "#fbfefd",
    "missing": "#fbfdff",
    "needs_attention": "#fffaf0",
    "not_applicable": "#fbfcfe",
}


def _stage_rail_name(model: StageViewModel) -> str:
    return _RAIL_STAGE_NAMES.get(model.stage_id, model.name)


def _stage_rail_status_label(model: StageViewModel) -> str:
    return _RAIL_STATUS_LABELS.get(model.status, model.status_label)


def _stage_rail_entry_style(model: StageViewModel, selected: bool) -> str:
    background = "#eef8ff" if selected else _RAIL_STATUS_BACKGROUNDS.get(
        model.status,
        "#fbfdff",
    )
    border = "#76b8dc" if selected else model.status_border
    accent = "#0b6f8f" if selected else model.status_color
    border_color = border if selected else "transparent"
    return (
        "QFrame#RailStageEntry { "
        f"background: {background}; border: 1px solid {border_color}; "
        f"border-left: 3px solid {accent}; border-radius: 7px; "
        "}"
    )


def _stage_rail_number_style(model: StageViewModel) -> str:
    return (
        f"background: transparent; color: {model.status_color}; "
        "border: none; font-weight: 700; font-size: 12px;"
    )


def _stage_rail_dot_style(model: StageViewModel) -> str:
    return (
        f"background: {model.status_color}; border: none; "
        "border-radius: 3px; margin-top: 4px;"
    )


def _stage_status_chip_style(model: StageViewModel) -> str:
    return (
        f"background: transparent; color: {model.status_color}; "
        "border: none; padding: 0; font-size: 9px; font-weight: 700;"
    )


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


def _format_charuco_outputs(result: Any) -> str:
    lines = [
        f"Folder: {_display_path(getattr(result, 'out_dir', result.png.parent))}",
        f"PNG:    {Path(result.png).name}",
        f"YAML:   {Path(result.yaml).name}",
    ]
    requested_pdf = getattr(result, "requested_pdf", True)
    if not requested_pdf:
        lines.append("PDF:    not requested")
    elif result.pdf is None:
        lines.append("PDF:    unavailable in this environment")
    else:
        lines.append(f"PDF:    {Path(result.pdf).name}")
    return "\n".join(lines)


def _display_path(path: Path) -> str:
    home = Path.home()
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return str(path)
    try:
        return str(Path("~") / resolved.relative_to(home))
    except ValueError:
        return str(resolved)


def _format_card_items(items: tuple[str, ...], empty_text: str) -> str:
    rendered = _format_items(items)
    if not rendered:
        return empty_text
    return "\n".join(rendered)


def _format_items(items: tuple[str, ...]) -> list[str]:
    return [f"- {item}" for item in items]


def _format_health_counts(health: ProjectHealthViewModel) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{count.label}</td>"
        f"<td align='right'><b>{count.count}</b></td>"
        "</tr>"
        for count in health.counts
    )
    return (
        "<span style='font-size:11px; font-weight:700; "
        "color:#5a6b7c;'>AT A GLANCE</span>"
        "<table width='100%' cellspacing='0' cellpadding='2'>"
        f"{rows}"
        "</table>"
    )


def _format_health_issues(health: ProjectHealthViewModel) -> str:
    if health.warning_count == 0 and health.error_count == 0:
        return "No warnings or errors reported."
    return (
        f"Warnings: {health.warning_count}\n"
        f"Errors: {health.error_count}"
    )


def _command_card_title(model: StageViewModel) -> str:
    if model.status == "complete" and model.command_preview:
        return "Rerun command"
    return "Command preview"


def _command_card_note(model: StageViewModel) -> str:
    if model.status == "complete" and model.command_preview:
        return (
            "This stage is already complete. The checked paths above are the "
            "evidence for this stage; copy this command only if you need to "
            "regenerate outputs."
        )
    if model.command_preview:
        return (
            "Copy the preview into a terminal. Camera calibration, "
            "object-board building, annotation, and dataset workflows are not "
            "executed by the dashboard."
        )
    return "No command preview is defined for this stage."


def _command_button_label(model: StageViewModel) -> str:
    if model.status == "complete" and model.command_preview:
        return "Copy Rerun"
    return "Copy Command"


def _command_ready_message(command: str, *, stage_complete: bool = False) -> str:
    if command:
        if stage_complete:
            return "Stage complete. Copy only if you need to regenerate outputs."
        return "Ready to copy. Replace placeholder values before running it."
    return "No command preview is available for this stage."


def _copy_confirmation_message(command: str, *, stage_complete: bool = False) -> str:
    if command:
        if stage_complete:
            return "Copied rerun command to the clipboard."
        return "Copied command preview to the clipboard."
    return "No command preview is available for this stage."


def _project_root_hint(root: Path) -> str:
    if root.is_dir():
        return (
            "Project folder found. Status checks are read-only except Stage 1 "
            "board generation."
        )
    if root.exists():
        return "Selected path exists but is not a folder."
    return (
        "Project folder not found. Status checks are read-only; Stage 1 board "
        "generation can create the selected project layout."
    )


def _pill_style(foreground: str, background: str, border: str) -> str:
    return (
        f"background: {background}; color: {foreground}; "
        f"border: 1px solid {border}; border-radius: 6px; "
        "padding: 6px 10px; font-weight: 700;"
    )


def _health_status_style(foreground: str, background: str, border: str) -> str:
    return (
        f"background: {background}; color: {foreground}; "
        f"border: 1px solid {border}; border-left: 4px solid {foreground}; "
        "border-radius: 7px; padding: 9px 12px; "
        "font-size: 15px; font-weight: 700;"
    )


def _style_sheet() -> str:
    return """
QMainWindow {
    background: #eef4f8;
}
QWidget {
    color: #17202a;
    font-family: "Aptos", "Segoe UI", "Helvetica Neue", "Arial", sans-serif;
    font-size: 13px;
}
QLineEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #c7d3df;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #cfe6f5;
    selection-color: #102030;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #1f7a8c;
}
QLineEdit#CommandPreview {
    background: #f7fafc;
    border-color: #b4c6d7;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QScrollArea#DetailScroll {
    background: transparent;
    border: none;
}
QPushButton {
    background: #1f5f83;
    color: #ffffff;
    border: 1px solid #194b69;
    border-radius: 6px;
    padding: 7px 13px;
    font-weight: 600;
}
QPushButton:hover {
    background: #246d96;
}
QPushButton:pressed {
    background: #174760;
}
QPushButton:disabled {
    background: #b7c0ca;
    border-color: #aab4be;
    color: #eef2f6;
}
QPushButton#PrimaryActionButton {
    background: #087966;
    border-color: #086251;
    padding: 8px 15px;
}
QPushButton#PrimaryActionButton:hover {
    background: #0b8a74;
}
QPushButton#SecondaryActionButton {
    background: #49697f;
    border-color: #365467;
}
QFrame#HeaderPanel,
QFrame#RailPanel,
QFrame#HealthPanel {
    background: #fbfdff;
    border: 1px solid #dfe9f1;
    border-radius: 8px;
}
QFrame#Card {
    background: #fbfdff;
    border: 1px solid #dfe9f1;
    border-radius: 8px;
}
QFrame#ActionCard {
    background: #ffffff;
    border: 1px solid #d7e6e1;
    border-left: 3px solid #087966;
    border-radius: 8px;
}
QFrame#CharucoPreviewColumn {
    background: #f4f8fb;
    border: 1px solid #dce7ef;
    border-radius: 8px;
}
QFrame#CommandCard {
    background: #fbfdff;
    border: 1px solid #d4e1ea;
    border-left: 3px solid #2b6f9e;
    border-radius: 8px;
}
QFrame#Divider {
    color: #cbd8e2;
    background: #cbd8e2;
    border: none;
    min-height: 1px;
    max-height: 1px;
}
QLabel#AppTitle {
    color: #102235;
    font-size: 22px;
    font-weight: 800;
}
QLabel#AppTitle,
QLabel#StageTitle,
QLabel#PanelTitle,
QLabel#CardTitle,
QLabel#HealthSectionTitle {
    font-family: "Aptos Display", "Segoe UI Variable Display", "Segoe UI",
        "Helvetica Neue", "Arial", sans-serif;
}
QLabel#PanelTitle {
    color: #102235;
    font-size: 15px;
    font-weight: 800;
}
QLabel#CardTitle {
    color: #102235;
    background: #eef5f8;
    border-left: 3px solid #54708a;
    border-radius: 4px;
    padding: 4px 7px;
    font-size: 13px;
    font-weight: 800;
}
QLabel#CardBody {
    color: #334150;
}
QLabel#MutedText {
    color: #5c6b7b;
}
QLabel#HealthMessage {
    color: #56687a;
    line-height: 120%;
}
QLabel#FieldLabel {
    color: #405367;
    font-size: 12px;
    font-weight: 700;
}
QLabel#GuidanceText {
    color: #62460b;
    background: #fffaf0;
    border: 1px solid #e5c865;
    border-radius: 8px;
    padding: 7px 9px;
}
QLabel#OutputText {
    color: #263545;
    background: #ffffff;
    border: 1px solid #d4e1ea;
    border-radius: 6px;
    padding: 7px;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 11px;
}
QLabel#CharucoPreviewCanvas {
    color: #5c6b7b;
    background: #edf4f8;
    border: 1px solid #d4e0e9;
    border-radius: 7px;
    padding: 10px;
}
QLabel#HealthCounts {
    color: #263545;
    background: #f6f9fc;
    border: 1px solid #d9e5ee;
    border-radius: 7px;
    padding: 9px 10px;
}
QLabel#HealthSectionTitle {
    color: #102235;
    font-size: 13px;
    font-weight: 800;
    padding-top: 2px;
}
QLabel#HealthSectionBody {
    color: #34495e;
    line-height: 120%;
}
QStatusBar {
    background: #ffffff;
    color: #5c6b7b;
    border-top: 1px solid #d7dee7;
}
QListWidget#WorkflowRail {
    background: transparent;
    border: none;
    padding: 0;
}
QListWidget::item {
    padding: 0;
    border: 1px solid transparent;
    border-radius: 8px;
}
QListWidget::item:selected {
    background: transparent;
    color: #0b1724;
    border: 1px solid transparent;
}
QListWidget::item:selected:!active {
    background: transparent;
    color: #0b1724;
    border: 1px solid transparent;
}
QLabel#RailStageName {
    color: #1f3042;
    font-size: 11px;
    font-weight: 700;
}
QLabel#RailStageNumber,
QLabel#RailStageDot,
QLabel#RailStageStatus {
    min-width: 0;
}
QLabel#StageTitle {
    color: #102235;
    font-size: 25px;
    font-weight: 800;
}
QLabel#StageMessage {
    color: #394653;
    padding: 2px 0 4px 0;
}
QSplitter::handle {
    background: #d7e1ea;
    margin: 0 6px;
}
"""
