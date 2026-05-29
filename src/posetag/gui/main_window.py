"""Main window construction for the optional PoseTag workflow dashboard."""

from __future__ import annotations

import html
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
from posetag.workflows.calibration_flow import (
    CALIBRATION_PROCESS_NOT_STARTED,
    CALIBRATION_SOURCE_CHOICES,
    CALIBRATION_SOURCE_LABELS,
    DEFAULT_GUIDED_AUTO_COOLDOWN,
    DEFAULT_SAMPLES_PER_CELL,
    SOURCE_OPENCV,
    SOURCE_REALSENSE,
    SOURCE_VIDEO,
    CameraCalibrationConfig,
    CameraCalibrationFlowError,
    CameraCalibrationOutputSummary,
    CameraCalibrationProcessState,
    CameraCalibrationReadiness,
    build_camera_calibration_launch,
    camera_calibration_config_from_project,
    calibration_process_failed,
    calibration_process_not_started,
    calibration_process_running,
    inspect_camera_calibration_output,
    inspect_camera_calibration_readiness,
    inspect_charuco_metadata,
    normalize_source,
    read_camera_calibration_yaml_text,
    summarize_camera_calibration_process_result,
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
            self._calibration_process: Any = None
            self._calibration_process_state = calibration_process_not_started()
            self._calibration_running_project_root: Optional[Path] = None
            self._calibration_running_expected_output: Optional[Path] = None
            self._calibration_previous_output_mtime_ns: Optional[int] = None
            self._calibration_output_existed_at_launch = False
            self._calibration_output_seen = False

            self._calibration_output_timer = QtCore.QTimer(self)
            self._calibration_output_timer.setInterval(1000)
            self._calibration_output_timer.timeout.connect(
                self._poll_calibration_output
            )

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

            self._loaded_calibration_metadata_path: Optional[Path] = None
            self._loaded_calibration_metadata_mtime: Optional[int] = None
            self._calibration_card = QtWidgets.QFrame()
            self._calibration_card.setObjectName("ActionCard")
            calibration_layout = QtWidgets.QVBoxLayout(self._calibration_card)
            calibration_layout.setContentsMargins(14, 12, 14, 12)
            calibration_layout.setSpacing(9)

            calibration_title = QtWidgets.QLabel("Camera calibration guide")
            calibration_title.setObjectName("CardTitle")
            calibration_note = QtWidgets.QLabel(
                "Use the generated ChArUco metadata to prepare a "
                "posetag-calib-charuco command. Capture and solve still run in "
                "the existing calibration window."
            )
            calibration_note.setObjectName("MutedText")
            calibration_note.setWordWrap(True)

            self._calibration_source = QtWidgets.QComboBox()
            for source in CALIBRATION_SOURCE_CHOICES:
                self._calibration_source.addItem(
                    CALIBRATION_SOURCE_LABELS[source],
                    source,
                )
            self._calibration_source.setMinimumWidth(180)
            self._calibration_source.currentIndexChanged.connect(
                self._calibration_source_changed
            )

            self._calibration_camera_index = _make_int_spin(0, 99, 0)
            self._calibration_video_path = QtWidgets.QLineEdit()
            self._calibration_video_path.setPlaceholderText(
                "Select calibration video file"
            )
            self._calibration_video_path.textChanged.connect(
                self._update_calibration_flow
            )
            calibration_video_browse_button = QtWidgets.QPushButton("Browse")
            calibration_video_browse_button.clicked.connect(
                self._browse_calibration_video
            )
            self._calibration_video_browse_button = calibration_video_browse_button
            calibration_video_row = QtWidgets.QHBoxLayout()
            calibration_video_row.setContentsMargins(0, 0, 0, 0)
            calibration_video_row.addWidget(self._calibration_video_path, 1)
            calibration_video_row.addWidget(calibration_video_browse_button)
            calibration_video_widget = QtWidgets.QWidget()
            calibration_video_widget.setLayout(calibration_video_row)

            self._calibration_squares_x = _make_int_spin(2, 99, DEFAULT_SQUARES_X)
            self._calibration_squares_y = _make_int_spin(2, 99, DEFAULT_SQUARES_Y)
            self._calibration_square_length = _make_float_spin(
                0.1,
                1000.0,
                DEFAULT_SQUARE_LENGTH_MM,
                " mm",
            )
            self._calibration_marker_length = _make_float_spin(
                0.1,
                1000.0,
                DEFAULT_MARKER_LENGTH_MM,
                " mm",
            )
            self._calibration_dictionary = QtWidgets.QComboBox()
            self._calibration_dictionary.addItems(list(_safe_dictionary_choices()))
            self._calibration_dictionary.setCurrentText(DEFAULT_DICTIONARY)
            self._calibration_dictionary.setMinimumWidth(180)
            self._calibration_grid_rows = _make_int_spin(1, 9, 3)
            self._calibration_grid_cols = _make_int_spin(1, 9, 3)
            self._calibration_samples_per_cell = _make_int_spin(
                1,
                20,
                DEFAULT_SAMPLES_PER_CELL,
            )
            self._calibration_guided_auto = QtWidgets.QCheckBox(
                "Guided auto-capture"
            )
            self._calibration_guided_auto.setChecked(True)

            for field in (
                self._calibration_camera_index,
                self._calibration_squares_x,
                self._calibration_squares_y,
                self._calibration_square_length,
                self._calibration_marker_length,
                self._calibration_grid_rows,
                self._calibration_grid_cols,
                self._calibration_samples_per_cell,
            ):
                field.valueChanged.connect(self._update_calibration_flow)
            self._calibration_dictionary.currentTextChanged.connect(
                self._update_calibration_flow
            )
            self._calibration_guided_auto.stateChanged.connect(
                self._update_calibration_flow
            )

            self._calibration_camera_field = _make_field(
                "Camera index",
                self._calibration_camera_index,
            )
            self._calibration_video_field = _make_field(
                "Video path",
                calibration_video_widget,
            )

            calibration_grid = QtWidgets.QGridLayout()
            calibration_grid.setContentsMargins(0, 0, 0, 0)
            calibration_grid.setHorizontalSpacing(14)
            calibration_grid.setVerticalSpacing(10)
            calibration_grid.addWidget(
                _make_field("Source", self._calibration_source),
                0,
                0,
            )
            calibration_grid.addWidget(self._calibration_camera_field, 0, 1)
            calibration_grid.addWidget(self._calibration_video_field, 1, 0, 1, 2)
            calibration_grid.addWidget(
                _make_field("Squares X", self._calibration_squares_x),
                2,
                0,
            )
            calibration_grid.addWidget(
                _make_field("Squares Y", self._calibration_squares_y),
                2,
                1,
            )
            calibration_grid.addWidget(
                _make_field("Square length", self._calibration_square_length),
                3,
                0,
            )
            calibration_grid.addWidget(
                _make_field("Marker length", self._calibration_marker_length),
                3,
                1,
            )
            calibration_grid.addWidget(
                _make_field("Dictionary", self._calibration_dictionary),
                4,
                0,
                1,
                2,
            )
            calibration_grid.addWidget(
                _make_field("Coverage rows", self._calibration_grid_rows),
                5,
                0,
            )
            calibration_grid.addWidget(
                _make_field("Coverage cols", self._calibration_grid_cols),
                5,
                1,
            )
            calibration_grid.addWidget(
                _make_field("Samples / cell", self._calibration_samples_per_cell),
                6,
                0,
            )
            calibration_grid.addWidget(self._calibration_guided_auto, 6, 1)
            calibration_grid.setColumnStretch(0, 1)
            calibration_grid.setColumnStretch(1, 1)

            self._calibration_metadata = QtWidgets.QLabel(
                "No ChArUco metadata inspected yet."
            )
            self._calibration_metadata.setObjectName("OutputText")
            self._calibration_metadata.setWordWrap(True)
            self._calibration_metadata.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._calibration_expected_output = QtWidgets.QLabel()
            self._calibration_expected_output.setObjectName("OutputText")
            self._calibration_expected_output.setWordWrap(True)
            self._calibration_expected_output.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._calibration_readiness = QtWidgets.QLabel()
            self._calibration_readiness.setObjectName("OutputText")
            self._calibration_readiness.setWordWrap(True)
            self._calibration_readiness.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._calibration_command_preview = QtWidgets.QLineEdit()
            self._calibration_command_preview.setObjectName("CommandPreview")
            self._calibration_command_preview.setReadOnly(True)
            self._calibration_command_preview.setPlaceholderText(
                "Resolve readiness messages before copying a calibration command."
            )
            self._calibration_copy_button = QtWidgets.QPushButton("Copy Command")
            self._calibration_copy_button.clicked.connect(
                self._copy_calibration_command
            )
            self._calibration_run_button = QtWidgets.QPushButton("Run Calibration")
            self._calibration_run_button.setObjectName("PrimaryActionButton")
            self._calibration_run_button.clicked.connect(self._run_calibration)
            self._calibration_copy_feedback = QtWidgets.QLabel()
            self._calibration_copy_feedback.setObjectName("MutedText")
            self._calibration_copy_feedback.setWordWrap(True)
            calibration_command_row = QtWidgets.QHBoxLayout()
            calibration_command_row.setContentsMargins(0, 0, 0, 0)
            calibration_command_row.addWidget(self._calibration_command_preview, 1)
            calibration_command_row.addWidget(self._calibration_copy_button)
            calibration_command_widget = QtWidgets.QWidget()
            calibration_command_widget.setLayout(calibration_command_row)

            calibration_controls = QtWidgets.QLabel(
                "Controls: guided auto-capture saves useful samples; SPACE "
                "manually adds a sample; ENTER solves calibration; q quits "
                "without writing calibration output."
            )
            calibration_controls.setObjectName("GuidanceText")
            calibration_controls.setWordWrap(True)

            calibration_refresh_button = QtWidgets.QPushButton("Refresh Status")
            calibration_refresh_button.setObjectName("SecondaryActionButton")
            calibration_refresh_button.clicked.connect(self._refresh)
            self._calibration_process_state_label = QtWidgets.QLabel()
            self._calibration_process_state_label.setObjectName("OutputText")
            self._calibration_process_state_label.setWordWrap(True)
            self._calibration_process_state_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._calibration_log = QtWidgets.QPlainTextEdit()
            self._calibration_log.setObjectName("CalibrationLog")
            self._calibration_log.setReadOnly(True)
            self._calibration_log.setMaximumHeight(118)
            self._calibration_log.setPlaceholderText(
                "Calibration stdout/stderr will appear here after launch."
            )
            self._calibration_log.document().setMaximumBlockCount(250)
            calibration_action_row = QtWidgets.QHBoxLayout()
            calibration_action_row.addWidget(self._calibration_run_button)
            calibration_action_row.addWidget(calibration_refresh_button)
            calibration_action_row.addStretch(1)

            calibration_status_grid = QtWidgets.QGridLayout()
            calibration_status_grid.setContentsMargins(0, 0, 0, 0)
            calibration_status_grid.setHorizontalSpacing(12)
            calibration_status_grid.setVerticalSpacing(8)
            calibration_status_grid.addWidget(
                _make_field("Metadata", self._calibration_metadata),
                0,
                0,
            )
            calibration_status_grid.addWidget(
                _make_field("Expected output", self._calibration_expected_output),
                0,
                1,
            )
            calibration_status_grid.addWidget(
                _make_field("Readiness", self._calibration_readiness),
                1,
                0,
                1,
                2,
            )
            calibration_status_grid.addWidget(
                _make_field("Command preview", calibration_command_widget),
                2,
                0,
                1,
                2,
            )
            calibration_status_grid.addWidget(
                _make_field("Process state", self._calibration_process_state_label),
                3,
                0,
                1,
                2,
            )
            calibration_status_grid.addWidget(
                _make_field("Process log", self._calibration_log),
                4,
                0,
                1,
                2,
            )
            calibration_status_grid.setColumnStretch(0, 1)
            calibration_status_grid.setColumnStretch(1, 1)

            calibration_layout.addWidget(calibration_title)
            calibration_layout.addWidget(calibration_note)
            calibration_layout.addLayout(calibration_grid)
            calibration_layout.addWidget(calibration_controls)
            calibration_layout.addLayout(calibration_status_grid)
            calibration_layout.addWidget(self._calibration_copy_feedback)
            calibration_layout.addLayout(calibration_action_row)

            detail_content = QtWidgets.QWidget()
            detail_content_layout = QtWidgets.QVBoxLayout(detail_content)
            detail_content_layout.setContentsMargins(0, 0, 0, 0)
            detail_content_layout.setSpacing(10)
            detail_content_layout.addLayout(title_row)
            detail_content_layout.addWidget(self._message_label)
            detail_content_layout.addLayout(cards_grid)
            detail_content_layout.addWidget(self._charuco_card)
            detail_content_layout.addWidget(self._calibration_card)
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

            self._calibration_result_label = QtWidgets.QLabel()
            self._calibration_result_label.setObjectName("HealthSectionBody")
            self._calibration_result_label.setTextFormat(
                QtCore.Qt.TextFormat.RichText
            )
            self._calibration_result_label.setWordWrap(True)
            self._calibration_result_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._open_calibration_yaml_button = QtWidgets.QPushButton("Open YAML")
            self._open_calibration_yaml_button.setObjectName("SecondaryActionButton")
            self._open_calibration_yaml_button.clicked.connect(
                self._open_calibration_yaml
            )
            self._view_calibration_yaml_button = QtWidgets.QPushButton("View Raw")
            self._view_calibration_yaml_button.setObjectName("SecondaryActionButton")
            self._view_calibration_yaml_button.clicked.connect(
                self._view_calibration_yaml
            )
            self._copy_calibration_summary_button = QtWidgets.QPushButton(
                "Copy Summary"
            )
            self._copy_calibration_summary_button.setObjectName(
                "SecondaryActionButton"
            )
            self._copy_calibration_summary_button.clicked.connect(
                self._copy_calibration_summary
            )
            self._calibration_result_actions = QtWidgets.QWidget()
            calibration_result_actions_layout = QtWidgets.QVBoxLayout(
                self._calibration_result_actions
            )
            calibration_result_actions_layout.setContentsMargins(0, 0, 0, 0)
            calibration_result_actions_layout.setSpacing(6)
            calibration_result_actions_layout.addWidget(
                self._open_calibration_yaml_button
            )
            calibration_result_actions_layout.addWidget(
                self._view_calibration_yaml_button
            )
            calibration_result_actions_layout.addWidget(
                self._copy_calibration_summary_button
            )
            self._calibration_result_divider = _divider(QtWidgets)
            self._project_health_title = QtWidgets.QLabel("Project Health")
            self._project_health_title.setObjectName("HealthSectionTitle")

            health_content = QtWidgets.QFrame()
            health_content.setObjectName("HealthPanel")
            health_layout = QtWidgets.QVBoxLayout(health_content)
            health_layout.setContentsMargins(16, 16, 16, 16)
            health_layout.setSpacing(10)

            self._health_panel = QtWidgets.QScrollArea()
            self._health_panel.setObjectName("HealthScroll")
            self._health_panel.setMinimumWidth(280)
            self._health_panel.setMaximumWidth(340)
            self._health_panel.setWidgetResizable(True)
            self._health_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self._health_panel.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._health_panel.setWidget(health_content)
            self._side_panel_title = QtWidgets.QLabel("Project Health")
            self._side_panel_title.setObjectName("PanelTitle")
            health_layout.addWidget(self._side_panel_title)
            health_layout.addWidget(self._calibration_result_label)
            health_layout.addWidget(self._calibration_result_actions)
            health_layout.addWidget(self._calibration_result_divider)
            health_layout.addWidget(self._project_health_title)
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
                "Dashboard can generate ChArUco board files and prepare "
                "camera-calibration commands."
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

        def _browse_calibration_video(self) -> None:
            current = self._calibration_video_path.text().strip()
            start_path = current or str(self._current_project_root())
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select calibration video",
                start_path,
                "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
            )
            if selected:
                self._calibration_video_path.setText(selected)

        def _calibration_source_changed(self) -> None:
            self._render_calibration_source_fields()
            self._update_calibration_flow()

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

        def _open_calibration_yaml(self) -> None:
            summary = inspect_camera_calibration_output(self._current_project_root())
            if not summary.exists:
                message = "Calibration YAML is not available to open."
                self.statusBar().showMessage(message, 4000)
                return

            opened = QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(summary.path.resolve()))
            )
            message = "Opened calibration YAML." if opened else "Could not open YAML."
            self.statusBar().showMessage(message, 4000)

        def _view_calibration_yaml(self) -> None:
            summary = inspect_camera_calibration_output(self._current_project_root())
            if not summary.exists:
                message = "Calibration YAML is not available to view."
                self.statusBar().showMessage(message, 4000)
                return
            try:
                raw_text = read_camera_calibration_yaml_text(summary.path)
            except CameraCalibrationFlowError as exc:
                self.statusBar().showMessage(str(exc), 5000)
                return

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Calibration YAML")
            dialog.resize(760, 560)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)
            path_label = QtWidgets.QLabel(_display_path(summary.path))
            path_label.setObjectName("MutedText")
            path_label.setWordWrap(True)
            editor = QtWidgets.QPlainTextEdit()
            editor.setObjectName("CalibrationYamlView")
            editor.setReadOnly(True)
            editor.setPlainText(raw_text)
            close_button = QtWidgets.QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            button_row = QtWidgets.QHBoxLayout()
            button_row.addStretch(1)
            button_row.addWidget(close_button)
            layout.addWidget(path_label)
            layout.addWidget(editor, 1)
            layout.addLayout(button_row)
            dialog.exec()

        def _copy_calibration_summary(self) -> None:
            summary = inspect_camera_calibration_output(self._current_project_root())
            if not summary.exists:
                message = "Calibration summary is not available to copy."
                self.statusBar().showMessage(message, 4000)
                return
            QtWidgets.QApplication.clipboard().setText(
                _format_calibration_result_summary(summary)
            )
            self.statusBar().showMessage(
                "Copied calibration summary to the clipboard.",
                3000,
            )

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
            self._calibration_card.setVisible(model.stage_id == 2)
            if model.stage_id == 2:
                self._sync_calibration_from_project_metadata()
                self._update_calibration_flow()
            self._render_side_panel_calibration_result()
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
            self._render_side_panel_calibration_result()

        def _render_side_panel_calibration_result(self) -> None:
            show_calibration = self._selected_stage_id == 2
            self._side_panel_title.setText(
                "Calibration Result" if show_calibration else "Project Health"
            )
            self._calibration_result_label.setVisible(show_calibration)
            self._calibration_result_actions.setVisible(show_calibration)
            self._calibration_result_divider.setVisible(show_calibration)
            self._project_health_title.setVisible(show_calibration)
            if not show_calibration:
                return

            summary = inspect_camera_calibration_output(self._current_project_root())
            self._calibration_result_label.setText(
                _format_calibration_result_summary_html(summary)
            )
            for button in (
                self._open_calibration_yaml_button,
                self._view_calibration_yaml_button,
                self._copy_calibration_summary_button,
            ):
                button.setEnabled(summary.exists)

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
            self._calibration_card.setVisible(False)
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

        def _sync_calibration_from_project_metadata(self) -> None:
            metadata_state = inspect_charuco_metadata(self._current_project_root())
            selected = metadata_state.selected_path
            mtime = _path_mtime_ns(selected)
            if (
                selected == self._loaded_calibration_metadata_path
                and mtime == self._loaded_calibration_metadata_mtime
            ):
                return

            if metadata_state.metadata is None:
                self._loaded_calibration_metadata_path = selected
                self._loaded_calibration_metadata_mtime = mtime
                return

            config = camera_calibration_config_from_project(
                self._current_project_root(),
                source=self._calibration_source_value(),
                camera_index=int(self._calibration_camera_index.value()),
                video_path=self._calibration_video_path.text().strip() or None,
            )
            self._apply_calibration_config(config)
            self._loaded_calibration_metadata_path = selected
            self._loaded_calibration_metadata_mtime = mtime

        def _apply_calibration_config(
            self,
            config: CameraCalibrationConfig,
        ) -> None:
            fields = (
                self._calibration_squares_x,
                self._calibration_squares_y,
                self._calibration_square_length,
                self._calibration_marker_length,
                self._calibration_dictionary,
            )
            for field in fields:
                field.blockSignals(True)
            try:
                self._calibration_squares_x.setValue(int(config.squares_x))
                self._calibration_squares_y.setValue(int(config.squares_y))
                self._calibration_square_length.setValue(
                    float(config.square_length_mm)
                )
                self._calibration_marker_length.setValue(
                    float(config.marker_length_mm)
                )
                dictionary_name = str(config.dictionary_name)
                if self._calibration_dictionary.findText(dictionary_name) < 0:
                    self._calibration_dictionary.addItem(dictionary_name)
                self._calibration_dictionary.setCurrentText(dictionary_name)
            finally:
                for field in fields:
                    field.blockSignals(False)

        def _update_calibration_flow(self) -> None:
            if not hasattr(self, "_calibration_readiness"):
                return
            self._render_calibration_source_fields()
            readiness = inspect_camera_calibration_readiness(
                self._calibration_config()
            )
            self._calibration_metadata.setText(
                _format_calibration_metadata(readiness)
            )
            self._calibration_expected_output.setText(
                _format_calibration_expected_output(readiness)
            )
            self._calibration_readiness.setText(
                _format_calibration_readiness(readiness)
            )

            model = self._model_by_stage_id(self._selected_stage_id)
            stage_complete = bool(
                model and model.stage_id == 2 and model.status == "complete"
            )
            self._calibration_command_preview.setText(readiness.command_preview)
            self._calibration_command_preview.setCursorPosition(0)
            self._calibration_copy_button.setEnabled(bool(readiness.command_preview))
            self._calibration_copy_feedback.setText(
                _calibration_command_ready_message(
                    readiness,
                    stage_complete=stage_complete,
                )
            )
            if (
                self._calibration_process_state.state
                == CALIBRATION_PROCESS_NOT_STARTED
            ):
                self._set_calibration_process_state(
                    calibration_process_not_started(readiness.expected_output)
                )
            process_running = self._calibration_process_is_running()
            self._calibration_run_button.setEnabled(
                readiness.ready and not process_running
            )
            self._calibration_run_button.setText(
                _calibration_run_button_label(self._calibration_process_state)
            )

            if model is None or model.stage_id != 2:
                return

            self._command_preview.setText(readiness.command_preview)
            self._command_preview.setCursorPosition(0)
            self._copy_button.setEnabled(bool(readiness.command_preview))
            self._command_note.setText(
                _calibration_command_card_note(readiness, model)
            )
            self._copy_feedback.setText(
                _calibration_command_ready_message(
                    readiness,
                    stage_complete=model.status == "complete",
                )
            )

        def _copy_calibration_command(self) -> None:
            command = self._calibration_command_preview.text()
            if command:
                QtWidgets.QApplication.clipboard().setText(command)
            model = self._model_by_stage_id(self._selected_stage_id)
            message = _copy_confirmation_message(
                command,
                stage_complete=bool(
                    model and model.stage_id == 2 and model.status == "complete"
                ),
            )
            self._calibration_copy_feedback.setText(message)
            self.statusBar().showMessage(message, 3000)

        def _run_calibration(self) -> None:
            if self._calibration_process_is_running():
                message = "Calibration is already running."
                self.statusBar().showMessage(message, 3000)
                return

            config = self._calibration_config()
            readiness = inspect_camera_calibration_readiness(config)
            if not readiness.ready:
                message = "Resolve calibration readiness messages before running."
                self._calibration_copy_feedback.setText(message)
                self.statusBar().showMessage(message, 5000)
                self._update_calibration_flow()
                return

            try:
                launch = build_camera_calibration_launch(config)
            except Exception as exc:
                message = f"Could not prepare calibration launch: {exc}"
                self._set_calibration_process_state(
                    calibration_process_failed(
                        message,
                        expected_output=readiness.expected_output,
                    )
                )
                self._append_calibration_log(f"[gui] {message}")
                self.statusBar().showMessage(message, 6000)
                self._update_calibration_flow()
                return

            process = QtCore.QProcess(self)
            process.setProgram(launch.program)
            process.setArguments(list(launch.arguments))
            process.setProcessChannelMode(
                QtCore.QProcess.ProcessChannelMode.SeparateChannels
            )
            if hasattr(QtCore, "QProcessEnvironment"):
                env = QtCore.QProcessEnvironment.systemEnvironment()
                env.insert("PYTHONUNBUFFERED", "1")
                process.setProcessEnvironment(env)
            process.readyReadStandardOutput.connect(self._read_calibration_stdout)
            process.readyReadStandardError.connect(self._read_calibration_stderr)
            process.finished.connect(self._calibration_process_finished)
            process.errorOccurred.connect(self._calibration_process_error)

            self._calibration_process = process
            self._calibration_running_project_root = self._current_project_root()
            self._calibration_running_expected_output = launch.expected_output
            self._calibration_output_existed_at_launch = (
                launch.expected_output.exists()
            )
            self._calibration_previous_output_mtime_ns = _path_mtime_ns(
                launch.expected_output
            )
            self._calibration_output_seen = self._calibration_output_existed_at_launch
            self._calibration_log.clear()
            self._append_calibration_log(f"$ {launch.display_command}")
            self._append_calibration_log(
                "[gui] Launching with the current Python interpreter."
            )
            self._set_calibration_process_state(
                calibration_process_running(launch.expected_output)
            )
            self._calibration_output_timer.start()
            self._update_calibration_flow()
            process.start()
            self.statusBar().showMessage("Calibration process started.", 4000)

        def _read_calibration_stdout(self) -> None:
            process = self._calibration_process
            if process is None:
                return
            self._append_calibration_output(process.readAllStandardOutput(), "")

        def _read_calibration_stderr(self) -> None:
            process = self._calibration_process
            if process is None:
                return
            self._append_calibration_output(process.readAllStandardError(), "stderr")

        def _calibration_process_finished(
            self,
            exit_code: int,
            exit_status: Any,
        ) -> None:
            self._read_calibration_stdout()
            self._read_calibration_stderr()
            root = (
                self._calibration_running_project_root
                or self._current_project_root()
            )
            expected_output = self._calibration_running_expected_output
            crashed = exit_status == QtCore.QProcess.ExitStatus.CrashExit
            state = summarize_camera_calibration_process_result(
                root,
                exit_code=int(exit_code),
                crashed=crashed,
                expected_output=expected_output,
                previous_output_mtime_ns=self._calibration_previous_output_mtime_ns,
                require_output_update=self._calibration_output_existed_at_launch,
            )
            self._calibration_process = None
            self._calibration_output_timer.stop()
            self._set_calibration_process_state(state)
            self._append_calibration_log(f"[gui] {state.message}")
            self._refresh()
            self.statusBar().showMessage(state.message, 7000)

        def _calibration_process_error(self, error: Any) -> None:
            process = self._calibration_process
            error_name = _qt_enum_name(error)
            detail = process.errorString() if process is not None else error_name
            self._append_calibration_log(f"[gui] Process error: {detail}")
            failed_to_start = QtCore.QProcess.ProcessError.FailedToStart
            if error != failed_to_start:
                return

            state = calibration_process_failed(
                f"Calibration process failed to start: {detail}",
                expected_output=self._calibration_running_expected_output,
            )
            self._calibration_process = None
            self._calibration_output_timer.stop()
            self._set_calibration_process_state(state)
            self._update_calibration_flow()
            self.statusBar().showMessage(state.message, 7000)

        def _poll_calibration_output(self) -> None:
            target = self._calibration_running_expected_output
            if target is None or self._calibration_output_seen:
                return
            if not target.exists():
                return
            self._calibration_output_seen = True
            self._append_calibration_log(
                f"[gui] Detected calibration output: {_display_path(target)}"
            )
            self._refresh()

        def _append_calibration_output(self, data: Any, prefix: str) -> None:
            text = bytes(data).decode("utf-8", errors="replace")
            if not text:
                return
            if prefix:
                for line in text.rstrip().splitlines():
                    self._append_calibration_log(f"[{prefix}] {line}")
            else:
                self._append_calibration_log(text.rstrip())

        def _append_calibration_log(self, text: str) -> None:
            if not text:
                return
            self._calibration_log.appendPlainText(text)
            scrollbar = self._calibration_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_calibration_process_state(
            self,
            state: CameraCalibrationProcessState,
        ) -> None:
            self._calibration_process_state = state
            if hasattr(self, "_calibration_process_state_label"):
                self._calibration_process_state_label.setText(
                    _format_calibration_process_state(state)
                )
            if hasattr(self, "_calibration_run_button"):
                self._calibration_run_button.setText(
                    _calibration_run_button_label(state)
                )

        def _calibration_process_is_running(self) -> bool:
            process = self._calibration_process
            if process is None:
                return False
            return process.state() != QtCore.QProcess.ProcessState.NotRunning

        def _render_calibration_source_fields(self) -> None:
            source = self._calibration_source_value()
            is_webcam = source == SOURCE_OPENCV
            is_video = source == SOURCE_VIDEO
            self._calibration_camera_field.setVisible(is_webcam)
            self._calibration_video_field.setVisible(is_video)
            self._calibration_video_path.setEnabled(is_video)
            self._calibration_video_browse_button.setEnabled(is_video)

        def _calibration_config(self) -> CameraCalibrationConfig:
            return CameraCalibrationConfig(
                project_root=self._current_project_root(),
                source=self._calibration_source_value(),
                camera_index=int(self._calibration_camera_index.value()),
                video_path=self._calibration_video_path.text().strip() or None,
                squares_x=int(self._calibration_squares_x.value()),
                squares_y=int(self._calibration_squares_y.value()),
                square_length_mm=float(self._calibration_square_length.value()),
                marker_length_mm=float(self._calibration_marker_length.value()),
                dictionary_name=self._calibration_dictionary.currentText(),
                coverage_grid=(
                    f"{int(self._calibration_grid_rows.value())}x"
                    f"{int(self._calibration_grid_cols.value())}"
                ),
                samples_per_cell=int(self._calibration_samples_per_cell.value()),
                guided_auto=bool(self._calibration_guided_auto.isChecked()),
                guided_auto_cooldown=DEFAULT_GUIDED_AUTO_COOLDOWN,
            )

        def _calibration_source_value(self) -> str:
            data = self._calibration_source.currentData()
            raw = str(
                data if data is not None else self._calibration_source.currentText()
            )
            try:
                return normalize_source(raw)
            except Exception:
                return SOURCE_OPENCV

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


def _format_calibration_metadata(readiness: CameraCalibrationReadiness) -> str:
    metadata = readiness.metadata
    if metadata is None:
        if readiness.metadata_path is None:
            return "Missing: <project_root>/calib/boards/charuco_*.yaml"
        return (
            f"Found: {_display_path(readiness.metadata_path)}\n"
            "Could not autofill board parameters from this metadata."
        )

    return "\n".join(
        [
            f"Using: {_display_path(metadata.path)}",
            f"Board: {metadata.squares_x} x {metadata.squares_y}",
            f"Square: {metadata.square_length_mm:g} mm",
            f"Marker: {metadata.marker_length_mm:g} mm",
            f"Dictionary: {metadata.dictionary_name}",
        ]
    )


def _format_calibration_expected_output(
    readiness: CameraCalibrationReadiness,
) -> str:
    state = "present" if readiness.output_exists else "missing"
    return f"{_display_path(readiness.expected_output)}\nStatus: {state}"


def _format_calibration_readiness(
    readiness: CameraCalibrationReadiness,
) -> str:
    lines = ["Ready to run." if readiness.ready else "Not ready yet."]
    if readiness.errors:
        lines.extend(["", "Errors", *_format_items(readiness.errors)])
    if readiness.warnings:
        lines.extend(["", "Warnings", *_format_items(readiness.warnings)])
    return "\n".join(lines)


def _format_calibration_process_state(
    state: CameraCalibrationProcessState,
) -> str:
    lines = [state.label, "", state.message]
    if state.expected_output is not None:
        lines.extend(["", f"Expected output: {_display_path(state.expected_output)}"])
    if state.exit_code is not None:
        lines.append(f"Exit code: {state.exit_code}")
    return "\n".join(lines)


def _format_calibration_result_summary(
    summary: CameraCalibrationOutputSummary,
) -> str:
    if not summary.exists:
        return "\n".join(
            [
                "No calibration YAML yet.",
                "",
                "Expected output",
                _display_path(summary.path),
            ]
        )

    lines = [
        "Calibration valid" if summary.valid else "Calibration needs attention",
        summary.message,
        "",
        "Output",
        _display_path(summary.path),
    ]
    if summary.image_width is not None and summary.image_height is not None:
        lines.extend(
            [
                "",
                "Image size",
                f"{summary.image_width} x {summary.image_height}",
            ]
        )
    if summary.model:
        lines.extend(["", "Model", summary.model])
    if summary.reproj_rms is not None:
        lines.extend(["", "RMS", f"{summary.reproj_rms:.3f} px"])
    if summary.camera_params is not None:
        fx, fy, cx, cy = summary.camera_params
        lines.extend(
            [
                "",
                "Intrinsics",
                f"fx={fx:.3f}",
                f"fy={fy:.3f}",
                f"cx={cx:.3f}",
                f"cy={cy:.3f}",
            ]
        )
    if summary.distortion_coefficients:
        distortion = "  ".join(
            f"{name}={value:.6g}"
            for name, value in summary.distortion_coefficients
        )
        lines.extend(["", "Distortion", distortion])
    if summary.latest_run_dir is not None:
        lines.extend(["", "Run snapshot", _display_path(summary.latest_run_dir)])
    return "\n".join(lines)


def _format_calibration_result_summary_html(
    summary: CameraCalibrationOutputSummary,
) -> str:
    if not summary.exists:
        return "".join(
            [
                "<div>No calibration YAML yet.</div>",
                _result_block_html("Expected output", _display_path(summary.path)),
            ]
        )

    lines = [
        (
            "<div><b>"
            + html.escape(
                "Calibration valid"
                if summary.valid
                else "Calibration needs attention"
            )
            + "</b></div>"
        ),
        f"<div>{html.escape(summary.message)}</div>",
        _result_block_html("Output", _display_path(summary.path)),
    ]
    if summary.image_width is not None and summary.image_height is not None:
        lines.append(
            _result_block_html(
                "Image size",
                f"{summary.image_width} x {summary.image_height}",
            )
        )
    if summary.model:
        lines.append(_result_block_html("Model", summary.model))
    if summary.reproj_rms is not None:
        lines.append(_result_block_html("RMS", f"{summary.reproj_rms:.3f} px"))
    if summary.camera_params is not None:
        fx, fy, cx, cy = summary.camera_params
        lines.append(
            _result_block_html(
                "Intrinsics",
                "<br>".join(
                    html.escape(value)
                    for value in (
                        f"fx={fx:.3f}",
                        f"fy={fy:.3f}",
                        f"cx={cx:.3f}",
                        f"cy={cy:.3f}",
                    )
                ),
                already_escaped=True,
            )
        )
    if summary.distortion_coefficients:
        distortion = "  ".join(
            f"{name}={value:.6g}"
            for name, value in summary.distortion_coefficients
        )
        lines.append(_result_block_html("Distortion", distortion))
    if summary.latest_run_dir is not None:
        lines.append(
            _result_block_html("Run snapshot", _display_path(summary.latest_run_dir))
        )
    return "".join(lines)


def _result_block_html(
    label: str,
    value: str,
    *,
    already_escaped: bool = False,
) -> str:
    rendered_value = value if already_escaped else html.escape(value)
    return (
        "<p style='margin:10px 0 0 0;'>"
        "<span style='color:#405367; font-weight:600;'>"
        f"{html.escape(label)}</span><br>"
        "<span style='font-family:\"Menlo\", \"Consolas\", "
        "\"Courier New\", monospace;'>"
        f"{rendered_value}</span></p>"
    )


def _calibration_run_button_label(state: CameraCalibrationProcessState) -> str:
    if state.running:
        return "Calibration Running..."
    if state.success:
        return "Run Calibration Again"
    return "Run Calibration"


def _qt_enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


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


def _calibration_command_card_note(
    readiness: CameraCalibrationReadiness,
    model: StageViewModel,
) -> str:
    if not readiness.command_preview:
        return (
            "Resolve the calibration readiness messages above before copying a "
            "posetag-calib-charuco command."
        )
    if model.status == "complete":
        return (
            "Calibration output already exists. Copy this command only if you "
            "need to rerun the existing posetag-calib-charuco workflow."
        )
    return (
        "Run Calibration starts the existing calibration workflow; Copy Command "
        "keeps the terminal fallback. Guided auto-capture saves useful samples; "
        "SPACE manually adds a sample, ENTER solves, and q quits in the OpenCV "
        "window."
    )


def _calibration_command_ready_message(
    readiness: CameraCalibrationReadiness,
    *,
    stage_complete: bool = False,
) -> str:
    if not readiness.command_preview:
        return "No runnable calibration command is available yet."
    if stage_complete:
        return "Calibration output exists. Copy only if you need to rerun it."
    return "Ready to run or copy a calibration command."


def _project_root_hint(root: Path) -> str:
    if root.is_dir():
        return (
            "Project folder found. Status checks are read-only except Stage 1 "
            "board generation and Stage 2 command preparation."
        )
    if root.exists():
        return "Selected path exists but is not a folder."
    return (
        "Project folder not found. Status checks are read-only; Stage 1 board "
        "generation can create the selected project layout."
    )


def _path_mtime_ns(path: Optional[Path]) -> Optional[int]:
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


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
QPlainTextEdit#CalibrationLog,
QPlainTextEdit#CalibrationYamlView {
    color: #263545;
    background: #f7fafc;
    border: 1px solid #d4e1ea;
    border-radius: 6px;
    padding: 7px;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 11px;
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
QFrame#HeaderPanel {
    background: #fbfdff;
    border: 1px solid #dfe9f1;
    border-radius: 8px;
}
QFrame#RailPanel {
    background: #f2f7fb;
    border: 1px solid #d8e5ee;
    border-radius: 8px;
}
QFrame#HealthPanel {
    background: #f6fafc;
    border: 1px solid #d8e5ee;
    border-radius: 8px;
}
QScrollArea#HealthScroll {
    background: transparent;
    border: none;
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
    background: #eef5f9;
    border: 1px solid #d3e1ea;
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
