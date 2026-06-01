"""Main window construction for the optional PoseTag workflow dashboard."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    PRINT_GUIDANCE as CHARUCO_PRINT_GUIDANCE,
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
from posetag.workflows.board_building import (
    BOARD_BUILDING_GUIDANCE,
    BOARD_NATIVE_CAPTURE_GUIDANCE,
    BOARD_PROCESS_NOT_STARTED,
    BOARD_SOURCE_CHOICES,
    BOARD_SOURCE_LABELS,
    DEFAULT_CAMERA_INDEX as DEFAULT_BOARD_CAMERA_INDEX,
    DEFAULT_FAMILY as DEFAULT_BOARD_FAMILY,
    DEFAULT_FPS as DEFAULT_BOARD_FPS,
    DEFAULT_HEIGHT as DEFAULT_BOARD_HEIGHT,
    DEFAULT_OBJECT_LABEL as DEFAULT_BOARD_OBJECT_LABEL,
    DEFAULT_OBJECT_NAME as DEFAULT_BOARD_OBJECT_NAME,
    DEFAULT_SIDE_LABEL as DEFAULT_BOARD_SIDE_LABEL,
    DEFAULT_SIDE_LABELS as DEFAULT_BOARD_SIDE_LABELS,
    DEFAULT_TAG_SIZE_MM as DEFAULT_BOARD_TAG_SIZE_MM,
    DEFAULT_WIDTH as DEFAULT_BOARD_WIDTH,
    DEFAULT_Z_THRESHOLD_M as DEFAULT_BOARD_Z_THRESHOLD_M,
    SOURCE_OPENCV as BOARD_SOURCE_OPENCV,
    SOURCE_REALSENSE as BOARD_SOURCE_REALSENSE,
    SOURCE_VIDEO as BOARD_SOURCE_VIDEO,
    BoardBuildingConfig,
    BoardBuildingProcessState,
    BoardBuildingReadiness,
    BoardBatchItem,
    BoardBatchRow,
    BoardBatchDraft,
    NativeBoardCaptureObservation,
    NativeBoardCaptureResult,
    NativeBoardCaptureSession,
    board_building_process_failed,
    board_building_process_not_started,
    board_building_process_running,
    build_board_batch_items,
    build_board_building_launch,
    compose_board_object_name,
    default_boards_dir,
    default_board_batch_draft_path,
    default_calibration_path,
    delete_board_batch_draft,
    inspect_board_building,
    infer_latest_object_tag_size_mm,
    load_board_batch_draft,
    next_default_side_label,
    normalize_source as normalize_board_source,
    parse_board_batch_rows,
    save_board_batch_draft,
    summarize_board_building_process_result,
)
from posetag.workflows.capture_face import (
    CAPTURE_FACE_GUIDANCE,
    CAPTURE_FACE_PROCESS_NOT_STARTED,
    CAPTURE_SOURCE_CHOICES,
    CAPTURE_SOURCE_LABELS,
    DEFAULT_CAMERA_INDEX as DEFAULT_CAPTURE_CAMERA_INDEX,
    DEFAULT_FAMILY as DEFAULT_CAPTURE_FAMILY,
    DEFAULT_FPS as DEFAULT_CAPTURE_FPS,
    DEFAULT_HEIGHT as DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_MIN_EXPECTED as DEFAULT_CAPTURE_MIN_EXPECTED,
    DEFAULT_AUTO_CAPTURE_COOLDOWN as DEFAULT_CAPTURE_AUTO_COOLDOWN,
    DEFAULT_AUTO_CAPTURE_FRAMES as DEFAULT_CAPTURE_AUTO_FRAMES,
    DEFAULT_PANEL_WIDTH as DEFAULT_CAPTURE_PANEL_WIDTH,
    DEFAULT_RECENT_WIDTH as DEFAULT_CAPTURE_RECENT_WIDTH,
    DEFAULT_WIDTH as DEFAULT_CAPTURE_WIDTH,
    LAYOUT_CHOICES as CAPTURE_LAYOUT_CHOICES,
    SOURCE_OPENCV as CAPTURE_SOURCE_OPENCV,
    SOURCE_REALSENSE as CAPTURE_SOURCE_REALSENSE,
    SOURCE_VIDEO as CAPTURE_SOURCE_VIDEO,
    CaptureFaceConfig,
    NativeFaceCaptureObservation,
    NativeFaceCaptureResult,
    NativeFaceCaptureSession,
    CaptureFaceProcessState,
    CaptureFaceReadiness,
    CaptureFaceSavedShot,
    build_capture_face_launch,
    capture_face_process_failed,
    capture_face_process_not_started,
    capture_face_process_running,
    default_calibration_path as default_capture_calibration_path,
    default_manifest_path as default_capture_manifest_path,
    default_registry_path as default_capture_registry_path,
    default_shots_dir as default_capture_shots_dir,
    inspect_capture_face_readiness,
    normalize_source as normalize_capture_source,
    summarize_capture_face_process_result,
)
from posetag.workflows.object_tags import (
    DEFAULT_DPI as DEFAULT_OBJECT_TAG_DPI,
    DEFAULT_FAMILY as DEFAULT_OBJECT_TAG_FAMILY,
    DEFAULT_ID_COUNT as DEFAULT_OBJECT_TAG_ID_COUNT,
    DEFAULT_ID_START as DEFAULT_OBJECT_TAG_ID_START,
    DEFAULT_IDS as DEFAULT_OBJECT_TAG_IDS,
    DEFAULT_PREFIX as DEFAULT_OBJECT_TAG_PREFIX,
    DEFAULT_TAG_SIZE_MM as DEFAULT_OBJECT_TAG_SIZE_MM,
    FAMILY_CHOICES as OBJECT_TAG_FAMILY_CHOICES,
    ID_MODE_CHOICES as OBJECT_TAG_ID_MODE_CHOICES,
    ID_MODE_LABELS as OBJECT_TAG_ID_MODE_LABELS,
    ID_MODE_LIST as OBJECT_TAG_ID_MODE_LIST,
    ID_MODE_RANGE as OBJECT_TAG_ID_MODE_RANGE,
    MAX_TAG_ID as OBJECT_TAG_MAX_ID,
    ORIENTATION_CHOICES as OBJECT_TAG_ORIENTATION_CHOICES,
    PAPER_CHOICES as OBJECT_TAG_PAPER_CHOICES,
    PAPER_CUSTOM as OBJECT_TAG_PAPER_CUSTOM,
    PRINT_GUIDANCE as OBJECT_TAG_PRINT_GUIDANCE,
    ObjectTagGenerationConfig,
    ObjectTagGenerationError,
    ObjectTagGenerationReadiness,
    default_output_dir as default_object_tag_output_dir,
    generate_object_tags,
    inspect_object_tag_generation,
)

CAPTURE_FACE_QUEUE_LABEL = "All missing registered faces"


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

    def _make_collapsible_group(
        title: str,
        content: Any,
        *,
        checked: bool = False,
    ) -> Any:
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("CollapsibleGroup")
        group.setCheckable(True)
        group.setChecked(checked)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(content)
        content.setVisible(checked)
        group.toggled.connect(content.setVisible)
        return group

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

    class _ImagePreviewCanvas(QtWidgets.QLabel):
        def __init__(
            self,
            empty_message: str,
            *,
            object_name: str = "ImagePreviewCanvas",
        ) -> None:
            super().__init__(empty_message)
            self._empty_message = empty_message
            self._source_pixmap: Any = None
            self.setObjectName(object_name)
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(300, 300)
            self.setMaximumHeight(330)
            self.setWordWrap(True)

        def clear_preview(self, message: Optional[str] = None) -> None:
            self._source_pixmap = None
            self.clear()
            self.setText(message or self._empty_message)

        def show_preview(self, path: Path) -> bool:
            pixmap = QtGui.QPixmap(str(path))
            if pixmap.isNull():
                self.clear_preview("Preview unavailable.")
                return False
            self._source_pixmap = pixmap
            self.setText("")
            self._refresh_pixmap()
            return True

        def show_pixmap(self, pixmap: Any) -> bool:
            if pixmap is None or pixmap.isNull():
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

    def _pixmap_from_bgr_frame(frame: Any) -> Any:
        if frame is None:
            return QtGui.QPixmap()
        if len(frame.shape) != 3 or frame.shape[2] < 3:
            return QtGui.QPixmap()
        rgb = frame[:, :, :3][:, :, ::-1].copy()
        height, width, channels = rgb.shape
        try:
            image_format = QtGui.QImage.Format.Format_RGB888
        except AttributeError:  # pragma: no cover - Qt compatibility
            image_format = QtGui.QImage.Format_RGB888
        image = QtGui.QImage(
            rgb.data,
            width,
            height,
            channels * width,
            image_format,
        ).copy()
        return QtGui.QPixmap.fromImage(image)

    def _face_record_name(face: Mapping[str, Any]) -> str:
        return str(face.get("object", "")).strip() or Path(
            str(face.get("yaml", ""))
        ).stem

    def _capture_face_saved_shot_icon(
        shots: Sequence[CaptureFaceSavedShot],
        size: Any,
    ) -> Any:
        width = max(24, int(size.width()))
        height = max(24, int(size.height()))
        canvas = QtGui.QPixmap(width, height)
        canvas.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(canvas)
        try:
            visible = list(shots[:4])
            for offset, shot in enumerate(reversed(visible)):
                path = _capture_face_shot_preview_path(shot)
                if path is None:
                    continue
                source = QtGui.QPixmap(str(path))
                if source.isNull():
                    continue
                dx = offset * 8
                dy = offset * 6
                target = QtCore.QRect(dx, dy, width - 24, height - 18)
                scaled = source.scaled(
                    target.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(target, scaled)
                painter.setPen(QtGui.QColor("#ffffff"))
                painter.drawRect(target)
        finally:
            painter.end()
        return QtGui.QIcon(canvas)

    def _capture_face_saved_shots_from_item(
        item: Any,
    ) -> tuple[CaptureFaceSavedShot, ...]:
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(payload, CaptureFaceSavedShot):
            return (payload,)
        if isinstance(payload, tuple) and all(
            isinstance(shot, CaptureFaceSavedShot) for shot in payload
        ):
            return payload
        return ()

    def _capture_face_saved_shot_from_item(
        item: Any,
    ) -> Optional[CaptureFaceSavedShot]:
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(payload, CaptureFaceSavedShot):
            return payload
        return None

    def _make_capture_face_saved_shot_item(
        shot: CaptureFaceSavedShot,
        icon_size: Any,
    ) -> Any:
        item = QtWidgets.QListWidgetItem(shot.timestamp or f"row {shot.row_index}")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, shot)
        item.setToolTip(_format_capture_face_saved_shot_details(shot))
        preview_path = _capture_face_shot_preview_path(shot)
        if preview_path is not None:
            pixmap = QtGui.QPixmap(str(preview_path))
            if not pixmap.isNull():
                item.setIcon(
                    QtGui.QIcon(
                        pixmap.scaled(
                            icon_size,
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )
        item.setForeground(
            QtGui.QBrush(QtGui.QColor("#087a3d" if shot.coverage_ok else "#8a5b00"))
        )
        return item

    class _NativeBoardCaptureDialog(QtWidgets.QDialog):
        def __init__(self, parent: Any, config: BoardBuildingConfig) -> None:
            super().__init__(parent)
            self.setWindowTitle("Guided Board Capture")
            self.setModal(True)
            self.resize(980, 720)
            self._config = config
            self._session: Optional[NativeBoardCaptureSession] = None
            self._last_observation: Optional[NativeBoardCaptureObservation] = None
            self.result_payload: Optional[NativeBoardCaptureResult] = None

            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(80)
            self._timer.timeout.connect(self._poll_frame)

            self._preview = _ImagePreviewCanvas(
                "Starting camera...",
                object_name="NativeBoardCapturePreview",
            )
            self._preview.setMinimumSize(540, 380)
            self._preview.setMaximumHeight(520)

            title = QtWidgets.QLabel("Guided object board capture")
            title.setObjectName("CardTitle")
            note = QtWidgets.QLabel(BOARD_NATIVE_CAPTURE_GUIDANCE)
            note.setObjectName("MutedText")
            note.setWordWrap(True)

            self._guidance = QtWidgets.QLabel("Starting detector...")
            self._guidance.setObjectName("GuidanceText")
            self._guidance.setWordWrap(True)

            self._ids = QtWidgets.QListWidget()
            self._ids.setMinimumHeight(130)
            self._origin = QtWidgets.QComboBox()
            self._auto_capture = QtWidgets.QCheckBox("Auto-capture stable view")
            self._auto_capture.setChecked(True)

            self._status = QtWidgets.QLabel("No capture yet.")
            self._status.setObjectName("OutputText")
            self._status.setWordWrap(True)
            self._status.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._capture_button = QtWidgets.QPushButton("Save Next")
            self._capture_button.setObjectName("SecondaryActionButton")
            self._capture_button.clicked.connect(self._capture_now)
            self._save_button = QtWidgets.QPushButton("Save Board Definition")
            self._save_button.setObjectName("PrimaryActionButton")
            self._save_button.setEnabled(False)
            self._save_button.clicked.connect(self._save_capture)
            self._restart_button = QtWidgets.QPushButton("Resume Live View")
            self._restart_button.setObjectName("SecondaryActionButton")
            self._restart_button.setEnabled(False)
            self._restart_button.clicked.connect(self._resume_live_view)
            close_button = QtWidgets.QPushButton("Close")
            close_button.setObjectName("SecondaryActionButton")
            close_button.clicked.connect(self.reject)

            controls = QtWidgets.QVBoxLayout()
            controls.setSpacing(8)
            controls.addWidget(_make_field("Guidance", self._guidance))
            controls.addWidget(_make_field("Detected tag IDs", self._ids))
            controls.addWidget(_make_field("Origin tag", self._origin))
            controls.addWidget(self._auto_capture)
            controls.addWidget(_make_field("Capture status", self._status))
            controls.addStretch(1)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(self._capture_button)
            buttons.addWidget(self._restart_button)
            buttons.addStretch(1)
            buttons.addWidget(close_button)
            buttons.addWidget(self._save_button)

            body = QtWidgets.QHBoxLayout()
            body.setSpacing(14)
            body.addWidget(self._preview, 3)
            controls_widget = QtWidgets.QWidget()
            controls_widget.setLayout(controls)
            controls_widget.setMinimumWidth(300)
            body.addWidget(controls_widget, 2)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(10)
            layout.addWidget(title)
            layout.addWidget(note)
            layout.addLayout(body, 1)
            layout.addLayout(buttons)

            self._start_session()

        def _start_session(self) -> None:
            try:
                self._session = NativeBoardCaptureSession(self._config)
            except Exception as exc:
                self._guidance.setText(f"Could not start guided capture: {exc}")
                self._status.setText("Guided capture did not start.")
                self._capture_button.setEnabled(False)
                return
            self._timer.start()
            self._status.setText("Live detection is running.")

        def _poll_frame(self) -> None:
            if self._session is None:
                return
            try:
                observation = self._session.read_observation()
            except Exception as exc:
                self._timer.stop()
                self._guidance.setText(f"Guided capture failed: {exc}")
                self._status.setText("Capture stopped before a board was saved.")
                return

            self._last_observation = observation
            self._guidance.setText(observation.guidance)
            if observation.annotated_frame_bgr is not None:
                self._preview.show_pixmap(
                    _pixmap_from_bgr_frame(observation.annotated_frame_bgr)
                )
            self._update_detected_ids(observation.pose_ready_ids)
            if observation.source_exhausted:
                self._timer.stop()
                self._status.setText("Video ended before auto-capture.")
                return
            if (
                observation.ready_to_capture
                and self._auto_capture.isChecked()
                and self._session.captured_observation is None
            ):
                self._capture_observation(observation, automatic=True)

        def _update_detected_ids(
            self,
            ids: tuple[int, ...],
            *,
            force_all_checked: bool = False,
        ) -> None:
            checked = _checked_tag_ids_for_update(
                ids,
                self._checked_ids(),
                force_all=force_all_checked,
            )
            origin_text = self._origin.currentText()
            self._ids.clear()
            self._origin.clear()
            for tag_id in ids:
                item = QtWidgets.QListWidgetItem(str(tag_id))
                item.setFlags(
                    item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if tag_id in checked
                    else QtCore.Qt.CheckState.Unchecked
                )
                self._ids.addItem(item)
                self._origin.addItem(str(tag_id), tag_id)
            if origin_text:
                index = self._origin.findText(origin_text)
                if index >= 0:
                    self._origin.setCurrentIndex(index)

        def _checked_ids(self) -> set[int]:
            selected: set[int] = set()
            for row in range(self._ids.count()):
                item = self._ids.item(row)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    selected.add(int(item.text()))
            return selected

        def _capture_now(self) -> None:
            if self._last_observation is None:
                self._status.setText("No frame has been detected yet.")
                return
            self._capture_observation(self._last_observation, automatic=False)

        def _capture_observation(
            self,
            observation: NativeBoardCaptureObservation,
            *,
            automatic: bool,
        ) -> None:
            if self._session is None:
                return
            try:
                captured = self._session.capture_current(observation)
            except Exception as exc:
                self._status.setText(f"Could not capture board view: {exc}")
                return
            self._timer.stop()
            self._update_detected_ids(
                captured.pose_ready_ids,
                force_all_checked=True,
            )
            self._save_button.setEnabled(True)
            self._restart_button.setEnabled(True)
            prefix = "Auto-captured" if automatic else "Captured"
            self._status.setText(
                f"{prefix} stable board view with IDs "
                f"{_format_id_summary(captured.pose_ready_ids)}. Confirm IDs "
                "and choose the origin tag, then save."
            )

        def _resume_live_view(self) -> None:
            if self._session is None:
                return
            self._session.clear_capture()
            self._save_button.setEnabled(False)
            self._restart_button.setEnabled(False)
            self._status.setText("Live detection resumed.")
            self._timer.start()

        def _save_capture(self) -> None:
            if self._session is None:
                return
            selected = tuple(sorted(self._checked_ids()))
            if not selected:
                self._status.setText("Select at least one detected tag ID.")
                return
            origin_data = self._origin.currentData()
            if origin_data is None:
                self._status.setText("Choose an origin tag.")
                return
            origin_id = int(origin_data)
            try:
                self.result_payload = self._session.save_capture(
                    selected_ids=selected,
                    origin_id=origin_id,
                )
            except Exception as exc:
                self._status.setText(f"Could not save board definition: {exc}")
                return
            self.accept()

        def done(self, result: int) -> None:
            self._timer.stop()
            if self._session is not None:
                self._session.close()
            super().done(result)

    class _NativeBoardBatchCaptureDialog(QtWidgets.QDialog):
        def __init__(
            self,
            parent: Any,
            base_config: BoardBuildingConfig,
            items: tuple[BoardBatchItem, ...],
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Batch Guided Board Capture")
            self.setModal(True)
            self.resize(1100, 760)
            self._base_config = base_config
            self._items = items
            self._index = 0
            self._session: Optional[NativeBoardCaptureSession] = None
            self._last_observation: Optional[NativeBoardCaptureObservation] = None
            self.saved_results: dict[str, NativeBoardCaptureResult] = {}
            self.skipped_items: dict[str, BoardBatchItem] = {}
            self._syncing_queue_selection = False

            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(80)
            self._timer.timeout.connect(self._poll_frame)

            title = QtWidgets.QLabel("Batch guided board capture")
            title.setObjectName("CardTitle")
            note = QtWidgets.QLabel(
                "Keep the camera open and present each queued object side. "
                "PoseTag auto-captures a stable tag view, then you confirm IDs "
                "and the origin before saving and moving to the next board. "
                "Revisiting a saved board replaces that same board definition."
            )
            note.setObjectName("MutedText")
            note.setWordWrap(True)

            self._current_item = QtWidgets.QLabel()
            self._current_item.setObjectName("StageTitle")
            self._progress = QtWidgets.QLabel()
            self._progress.setObjectName("MutedText")
            self._progress.setWordWrap(True)

            self._preview = _ImagePreviewCanvas(
                "Starting camera...",
                object_name="NativeBoardBatchPreview",
            )
            self._preview.setMinimumSize(540, 380)
            self._preview.setMaximumHeight(520)
            self._guidance = QtWidgets.QLabel("Starting detector...")
            self._guidance.setObjectName("GuidanceText")
            self._guidance.setWordWrap(True)
            self._ids = QtWidgets.QListWidget()
            self._ids.setMinimumHeight(130)
            self._origin = QtWidgets.QComboBox()
            self._auto_capture = QtWidgets.QCheckBox("Auto-capture stable view")
            self._auto_capture.setChecked(True)
            self._status = QtWidgets.QLabel("No capture yet.")
            self._status.setObjectName("OutputText")
            self._status.setWordWrap(True)
            self._status.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._queue = QtWidgets.QListWidget()
            self._queue.setObjectName("BatchCaptureQueue")
            self._queue.setMinimumWidth(250)
            self._queue.setMaximumWidth(330)
            self._queue.setSpacing(3)
            self._queue.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            for index, item in enumerate(items, start=1):
                tag_size = _format_batch_item_tag_size(item, base_config)
                queue_item = QtWidgets.QListWidgetItem(
                    f"{index}. {item.object_name} ({tag_size} mm)"
                )
                queue_item.setToolTip(f"{item.object_name} | tag size {tag_size} mm")
                self._queue.addItem(queue_item)
            self._queue.currentRowChanged.connect(self._select_queue_row)

            self._capture_button = QtWidgets.QPushButton("Capture Now")
            self._capture_button.setObjectName("SecondaryActionButton")
            self._capture_button.clicked.connect(self._capture_now)
            self._resume_button = QtWidgets.QPushButton("Resume Live View")
            self._resume_button.setObjectName("SecondaryActionButton")
            self._resume_button.setEnabled(False)
            self._resume_button.clicked.connect(self._resume_live_view)
            self._skip_button = QtWidgets.QPushButton("Skip")
            self._skip_button.setObjectName("SecondaryActionButton")
            self._skip_button.clicked.connect(self._skip_current)
            self._previous_button = QtWidgets.QPushButton("Previous")
            self._previous_button.setObjectName("SecondaryActionButton")
            self._previous_button.clicked.connect(self._previous_item)
            self._save_next_button = QtWidgets.QPushButton("Save & Next")
            self._save_next_button.setObjectName("PrimaryActionButton")
            self._save_next_button.setEnabled(False)
            self._save_next_button.clicked.connect(self._save_current)
            finish_button = QtWidgets.QPushButton("Finish Batch")
            finish_button.setObjectName("SecondaryActionButton")
            finish_button.clicked.connect(self.accept)

            controls = QtWidgets.QVBoxLayout()
            controls.setSpacing(8)
            controls.addWidget(_make_field("Guidance", self._guidance))
            controls.addWidget(_make_field("Detected tag IDs", self._ids))
            controls.addWidget(_make_field("Origin tag", self._origin))
            controls.addWidget(self._auto_capture)
            controls.addWidget(_make_field("Capture status", self._status))
            controls.addStretch(1)

            body = QtWidgets.QHBoxLayout()
            body.setSpacing(14)
            body.addWidget(_make_field("Queue", self._queue), 1)
            body.addWidget(self._preview, 3)
            controls_widget = QtWidgets.QWidget()
            controls_widget.setLayout(controls)
            controls_widget.setMinimumWidth(300)
            body.addWidget(controls_widget, 2)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(self._capture_button)
            buttons.addWidget(self._resume_button)
            buttons.addWidget(self._previous_button)
            buttons.addWidget(self._skip_button)
            buttons.addStretch(1)
            buttons.addWidget(finish_button)
            buttons.addWidget(self._save_next_button)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(10)
            layout.addWidget(title)
            layout.addWidget(note)
            layout.addWidget(self._current_item)
            layout.addWidget(self._progress)
            layout.addLayout(body, 1)
            layout.addLayout(buttons)

            self._start_session()
            self._render_current_item()

        def _start_session(self) -> None:
            try:
                self._session = NativeBoardCaptureSession(
                    self._config_for_item(self._items[0])
                )
            except Exception as exc:
                self._guidance.setText(f"Could not start batch capture: {exc}")
                self._status.setText("Batch capture did not start.")
                self._capture_button.setEnabled(False)
                self._save_next_button.setEnabled(False)
                return
            self._timer.start()
            self._status.setText("Live detection is running.")

        def _current(self) -> BoardBatchItem:
            return self._items[self._index]

        def _config_for_item(self, item: BoardBatchItem) -> BoardBuildingConfig:
            return BoardBuildingConfig(
                project_root=self._base_config.project_root,
                object_name=item.object_name,
                tag_size_mm=(
                    item.tag_size_mm
                    if item.tag_size_mm is not None
                    else self._base_config.tag_size_mm
                ),
                family=self._base_config.family,
                calibration_path=self._base_config.calibration_path,
                source=self._base_config.source,
                camera_index=self._base_config.camera_index,
                video_path=self._base_config.video_path,
                width=self._base_config.width,
                height=self._base_config.height,
                fps=self._base_config.fps,
                out_dir=self._base_config.out_dir,
                registry_path=self._base_config.registry_path,
                save_shot=self._base_config.save_shot,
                shots_dir=self._base_config.shots_dir,
                z_threshold_m=self._base_config.z_threshold_m,
                allow_nonplanar=self._base_config.allow_nonplanar,
                require_object_tags=self._base_config.require_object_tags,
            )

        def _render_current_item(self) -> None:
            item = self._current()
            self._syncing_queue_selection = True
            self._queue.setCurrentRow(self._index)
            self._syncing_queue_selection = False
            tag_size = _format_batch_item_tag_size(item, self._base_config)
            self._current_item.setText(
                f"Current board: {item.object_name} | tag size {tag_size} mm"
            )
            action = (
                "Replace & Next"
                if item.object_name in self.saved_results
                else "Save & Next"
            )
            self._save_next_button.setText(action)
            self._progress.setText(
                f"{self._index + 1} / {len(self._items)} queued | "
                f"{len(self.saved_results)} saved | "
                f"{len(self.skipped_items)} skipped"
            )
            self._previous_button.setEnabled(self._index > 0)

        def _select_queue_row(self, row: int) -> None:
            if self._syncing_queue_selection:
                return
            if row < 0 or row >= len(self._items):
                return
            self._index = row
            self._reset_for_current_item(
                "Selected queued board.",
                inspect_saved=True,
            )

        def _poll_frame(self) -> None:
            if self._session is None:
                return
            try:
                observation = self._session.read_observation()
            except Exception as exc:
                self._timer.stop()
                self._guidance.setText(f"Batch capture failed: {exc}")
                self._status.setText("Capture stopped before the batch finished.")
                return
            self._last_observation = observation
            self._guidance.setText(observation.guidance)
            if observation.annotated_frame_bgr is not None:
                self._preview.show_pixmap(
                    _pixmap_from_bgr_frame(observation.annotated_frame_bgr)
                )
            self._update_detected_ids(observation.pose_ready_ids)
            if observation.source_exhausted:
                self._timer.stop()
                self._status.setText("Video ended before the batch finished.")
                return
            if (
                observation.ready_to_capture
                and self._auto_capture.isChecked()
                and self._session.captured_observation is None
            ):
                self._capture_observation(observation, automatic=True)

        def _update_detected_ids(
            self,
            ids: tuple[int, ...],
            *,
            force_all_checked: bool = False,
        ) -> None:
            checked = _checked_tag_ids_for_update(
                ids,
                self._checked_ids(),
                force_all=force_all_checked,
            )
            origin_text = self._origin.currentText()
            self._ids.clear()
            self._origin.clear()
            for tag_id in ids:
                item = QtWidgets.QListWidgetItem(str(tag_id))
                item.setFlags(
                    item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if tag_id in checked
                    else QtCore.Qt.CheckState.Unchecked
                )
                self._ids.addItem(item)
                self._origin.addItem(str(tag_id), tag_id)
            if origin_text:
                index = self._origin.findText(origin_text)
                if index >= 0:
                    self._origin.setCurrentIndex(index)

        def _select_origin_tag(self, origin_id: int) -> None:
            for index in range(self._origin.count()):
                data = self._origin.itemData(index)
                if data is not None and int(data) == int(origin_id):
                    self._origin.setCurrentIndex(index)
                    return

        def _checked_ids(self) -> set[int]:
            selected: set[int] = set()
            for row in range(self._ids.count()):
                item = self._ids.item(row)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    selected.add(int(item.text()))
            return selected

        def _capture_now(self) -> None:
            if self._last_observation is None:
                self._status.setText("No frame has been detected yet.")
                return
            self._capture_observation(self._last_observation, automatic=False)

        def _capture_observation(
            self,
            observation: NativeBoardCaptureObservation,
            *,
            automatic: bool,
        ) -> None:
            if self._session is None:
                return
            try:
                captured = self._session.capture_current(observation)
            except Exception as exc:
                self._status.setText(f"Could not capture board view: {exc}")
                return
            self._timer.stop()
            self._update_detected_ids(
                captured.pose_ready_ids,
                force_all_checked=True,
            )
            self._save_next_button.setEnabled(True)
            self._resume_button.setEnabled(True)
            self._resume_button.setText("Resume Live View")
            prefix = "Auto-captured" if automatic else "Captured"
            self._status.setText(
                f"{prefix} {self._current().object_name} with IDs "
                f"{_format_id_summary(captured.pose_ready_ids)}. Confirm IDs "
                "and origin, then save and continue."
            )

        def _resume_live_view(self) -> None:
            if self._session is None:
                return
            self._session.clear_capture()
            self._save_next_button.setEnabled(False)
            self._resume_button.setEnabled(False)
            self._resume_button.setText("Resume Live View")
            self._status.setText("Live detection resumed.")
            self._timer.start()

        def _save_current(self) -> None:
            if self._session is None:
                return
            selected = tuple(sorted(self._checked_ids()))
            if not selected:
                self._status.setText("Select at least one detected tag ID.")
                return
            origin_data = self._origin.currentData()
            if origin_data is None:
                self._status.setText("Choose an origin tag.")
                return
            item = self._current()
            try:
                result = self._session.save_capture(
                    selected_ids=selected,
                    origin_id=int(origin_data),
                    config=self._config_for_item(item),
                )
            except Exception as exc:
                self._status.setText(f"Could not save {item.object_name}: {exc}")
                return
            replaced = item.object_name in self.saved_results
            self.saved_results[item.object_name] = result
            self.skipped_items.pop(item.object_name, None)
            conflict_note = (
                f" ({len(result.registry_conflicts)} registry conflict"
                f"{'s' if len(result.registry_conflicts) != 1 else ''})"
                if result.registry_conflicts
                else ""
            )
            action = "replaced" if replaced else "saved"
            action_title = "Replaced" if replaced else "Saved"
            queue_item = self._queue.item(self._index)
            queue_item.setText(
                f"{self._index + 1}. {action}: {item.object_name}{conflict_note}"
            )
            queue_item.setToolTip(
                f"{item.object_name}\n"
                f"Selected IDs: {_format_id_summary(result.selected_ids)}\n"
                f"Origin tag: {result.origin_id}\n"
                f"Board YAML: {_display_path(result.board_yaml_path)}"
            )
            self._advance_or_finish(
                f"{action_title} {item.object_name}{conflict_note}."
            )

        def _skip_current(self) -> None:
            item = self._current()
            if item.object_name in self.saved_results:
                self._advance_or_finish(f"Kept saved {item.object_name}.")
                return
            self.skipped_items[item.object_name] = item
            self._queue.item(self._index).setText(
                f"{self._index + 1}. skipped: {item.object_name}"
            )
            self._advance_or_finish(f"Skipped {item.object_name}.")

        def _previous_item(self) -> None:
            if self._index <= 0:
                return
            self._index -= 1
            self._reset_for_current_item(
                "Moved to previous board.",
                inspect_saved=True,
            )

        def _advance_or_finish(self, message: str) -> None:
            if self._index + 1 >= len(self._items):
                self._timer.stop()
                self._save_next_button.setEnabled(False)
                self._resume_button.setEnabled(False)
                self._status.setText(
                    f"{message} Batch complete: {len(self.saved_results)} saved, "
                    f"{len(self.skipped_items)} skipped."
                )
                self._render_current_item()
                return
            self._index += 1
            self._reset_for_current_item(message)

        def _reset_for_current_item(
            self,
            message: str,
            *,
            inspect_saved: bool = False,
        ) -> None:
            if self._session is not None:
                try:
                    self._session.set_capture_config(
                        self._config_for_item(self._current())
                    )
                except Exception as exc:
                    self._timer.stop()
                    self._status.setText(
                        f"Could not switch to {self._current().object_name}: {exc}"
                    )
                    return
            self._ids.clear()
            self._origin.clear()
            self._save_next_button.setEnabled(False)
            self._resume_button.setEnabled(False)
            self._resume_button.setText("Resume Live View")
            self._last_observation = None
            self._render_current_item()
            saved = self.saved_results.get(self._current().object_name)
            if inspect_saved and saved is not None:
                self._show_saved_result(saved, message)
                return
            self._status.setText(
                f"{message} Present {self._current().object_name} to the camera."
            )
            if self._session is not None:
                self._timer.start()

        def _show_saved_result(
            self,
            saved: NativeBoardCaptureResult,
            message: str,
        ) -> None:
            self._timer.stop()
            if self._session is not None:
                self._session.clear_capture()
            self._update_detected_ids(
                saved.selected_ids,
                force_all_checked=True,
            )
            self._select_origin_tag(saved.origin_id)
            self._save_next_button.setEnabled(False)
            self._resume_button.setEnabled(True)
            self._resume_button.setText("Retake Live View")
            self._status.setText(
                f"{message} Saved {self._current().object_name}.\n"
                f"Selected IDs: {_format_id_summary(saved.selected_ids)}\n"
                f"Origin tag: {saved.origin_id}\n"
                f"Board YAML: {_display_path(saved.board_yaml_path)}\n"
                "Use Retake Live View to replace this board definition."
            )

        def done(self, result: int) -> None:
            self._timer.stop()
            if self._session is not None:
                self._session.close()
            super().done(result)

    class _NativeFaceBatchCaptureDialog(QtWidgets.QDialog):
        def __init__(self, parent: Any, config: CaptureFaceConfig) -> None:
            super().__init__(parent)
            self.setWindowTitle("Batch Guided Face-Shot Capture")
            self.setModal(True)
            self.resize(1120, 760)
            self._config = config
            self._session: Optional[NativeFaceCaptureSession] = None
            self._last_observation: Optional[NativeFaceCaptureObservation] = None
            self._saved_results: list[NativeFaceCaptureResult] = []
            self._saved_shots: tuple[CaptureFaceSavedShot, ...] = ()
            self._syncing_queue_selection = False

            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(80)
            self._timer.timeout.connect(self._poll_frame)

            title = QtWidgets.QLabel("Batch guided face-shot capture")
            title.setObjectName("CardTitle")
            note = QtWidgets.QLabel(
                "Keep the camera open and present each registered face. "
                "PoseTag auto-captures missing faces when tags are stable; "
                "select an already captured face and press Retake Now to add "
                "a new raw and annotated reference shot."
            )
            note.setObjectName("MutedText")
            note.setWordWrap(True)

            self._current_item = QtWidgets.QLabel()
            self._current_item.setObjectName("StageTitle")
            self._progress = QtWidgets.QLabel()
            self._progress.setObjectName("MutedText")
            self._progress.setWordWrap(True)

            self._queue = QtWidgets.QListWidget()
            self._queue.setObjectName("BatchCaptureQueue")
            self._queue.setMinimumWidth(270)
            self._queue.setMaximumWidth(360)
            self._queue.setSpacing(3)
            self._queue.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self._queue.currentRowChanged.connect(self._select_queue_row)

            self._preview = _ImagePreviewCanvas(
                "Starting camera...",
                object_name="NativeFaceBatchPreview",
            )
            self._preview.setMinimumSize(560, 380)
            self._preview.setMaximumHeight(520)
            self._guidance = QtWidgets.QLabel("Starting detector...")
            self._guidance.setObjectName("GuidanceText")
            self._guidance.setWordWrap(True)
            self._ids = QtWidgets.QListWidget()
            self._ids.setMinimumHeight(96)
            self._expected = QtWidgets.QLabel("Expected tags: none")
            self._expected.setObjectName("MutedText")
            self._expected.setWordWrap(True)
            self._status = QtWidgets.QLabel("No capture yet.")
            self._status.setObjectName("OutputText")
            self._status.setWordWrap(True)
            self._status.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._auto_capture = QtWidgets.QCheckBox("Auto-capture missing faces")
            self._auto_capture.setChecked(True)

            self._saved_gallery = QtWidgets.QListWidget()
            self._saved_gallery.setObjectName("FaceShotGallery")
            self._saved_gallery.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
            self._saved_gallery.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
            self._saved_gallery.setMovement(QtWidgets.QListView.Movement.Static)
            self._saved_gallery.setIconSize(QtCore.QSize(150, 96))
            self._saved_gallery.setGridSize(QtCore.QSize(178, 132))
            self._saved_gallery.setMinimumHeight(150)
            self._saved_gallery.itemClicked.connect(self._preview_saved_shot)
            self._saved_gallery.itemDoubleClicked.connect(self._preview_saved_shot)

            self._saved_stack_gallery = QtWidgets.QListWidget()
            self._saved_stack_gallery.setObjectName("FaceShotGallery")
            self._saved_stack_gallery.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
            self._saved_stack_gallery.setResizeMode(
                QtWidgets.QListView.ResizeMode.Adjust
            )
            self._saved_stack_gallery.setMovement(QtWidgets.QListView.Movement.Static)
            self._saved_stack_gallery.setIconSize(QtCore.QSize(104, 72))
            self._saved_stack_gallery.setGridSize(QtCore.QSize(132, 104))
            self._saved_stack_gallery.setMaximumHeight(118)
            self._saved_stack_gallery.itemClicked.connect(self._preview_stack_shot)
            self._saved_stack_field = _make_field(
                "Shots in selected stack",
                self._saved_stack_gallery,
            )
            self._saved_stack_field.setVisible(False)

            self._capture_button = QtWidgets.QPushButton("Save Next")
            self._capture_button.setObjectName("SecondaryActionButton")
            self._capture_button.clicked.connect(self._capture_now)
            self._resume_button = QtWidgets.QPushButton("Resume Live View")
            self._resume_button.setObjectName("SecondaryActionButton")
            self._resume_button.setEnabled(False)
            self._resume_button.clicked.connect(self._resume_live_view)
            self._previous_button = QtWidgets.QPushButton("Previous")
            self._previous_button.setObjectName("SecondaryActionButton")
            self._previous_button.clicked.connect(self._previous_item)
            finish_button = QtWidgets.QPushButton("Finish Batch")
            finish_button.setObjectName("PrimaryActionButton")
            finish_button.clicked.connect(self.accept)

            controls = QtWidgets.QVBoxLayout()
            controls.setSpacing(8)
            controls.addWidget(_make_field("Guidance", self._guidance))
            controls.addWidget(_make_field("Detected tag IDs", self._ids))
            controls.addWidget(_make_field("Expected tag IDs", self._expected))
            controls.addWidget(self._auto_capture)
            controls.addWidget(_make_field("Saved shots", self._saved_gallery))
            controls.addWidget(self._saved_stack_field)
            controls.addWidget(_make_field("Capture status", self._status))
            controls.addStretch(1)

            body = QtWidgets.QHBoxLayout()
            body.setSpacing(14)
            body.addWidget(_make_field("Queue", self._queue), 1)
            body.addWidget(self._preview, 3)
            controls_widget = QtWidgets.QWidget()
            controls_widget.setLayout(controls)
            controls_widget.setMinimumWidth(330)
            body.addWidget(controls_widget, 2)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(self._capture_button)
            buttons.addWidget(self._resume_button)
            buttons.addWidget(self._previous_button)
            buttons.addStretch(1)
            buttons.addWidget(finish_button)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(10)
            layout.addWidget(title)
            layout.addWidget(note)
            layout.addWidget(self._current_item)
            layout.addWidget(self._progress)
            layout.addLayout(body, 1)
            layout.addLayout(buttons)

            self._start_session()

        def _start_session(self) -> None:
            try:
                self._session = NativeFaceCaptureSession(self._config)
            except Exception as exc:
                self._guidance.setText(f"Could not start face-shot capture: {exc}")
                self._status.setText("Face-shot capture did not start.")
                self._capture_button.setEnabled(False)
                self._resume_button.setEnabled(False)
                self._previous_button.setEnabled(False)
                return
            for index, face in enumerate(self._session.faces, start=1):
                name = _face_record_name(face)
                item = QtWidgets.QListWidgetItem(f"{index}. {name}")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setToolTip(name)
                self._queue.addItem(item)
            self._refresh_saved_shots()
            self._render_current_item()
            self._timer.start()
            self._status.setText("Live detection is running.")

        def _select_queue_row(self, row: int) -> None:
            if self._syncing_queue_selection or self._session is None:
                return
            if row < 0 or row >= len(self._session.faces):
                return
            self._session.select_face_index(row)
            self._last_observation = None
            self._resume_live_view(message="Selected queued face.")

        def _poll_frame(self) -> None:
            if self._session is None:
                return
            try:
                observation = self._session.read_observation()
            except Exception as exc:
                self._timer.stop()
                self._guidance.setText(f"Face-shot capture failed: {exc}")
                self._status.setText("Capture stopped before the batch finished.")
                return
            self._last_observation = observation
            self._guidance.setText(observation.guidance)
            self._update_detected_ids(observation.detected_ids)
            self._expected.setText(
                f"Expected tags: {_format_id_summary(observation.expected_ids)}"
            )
            if observation.annotated_frame_bgr is not None:
                self._preview.show_pixmap(
                    _pixmap_from_bgr_frame(observation.annotated_frame_bgr)
                )
            if observation.source_exhausted:
                self._timer.stop()
                self._status.setText("Video ended before the batch finished.")
                return
            self._render_current_item()
            if (
                observation.validation_ok
                and self._auto_capture.isChecked()
                and self._session.should_auto_capture(observation)
            ):
                self._save_observation(observation, automatic=True)

        def _update_detected_ids(self, ids: tuple[int, ...]) -> None:
            self._ids.clear()
            for tag_id in ids:
                self._ids.addItem(str(tag_id))

        def _capture_now(self) -> None:
            if self._last_observation is None:
                self._status.setText("No live frame has been detected yet.")
                return
            self._save_observation(self._last_observation, automatic=False)

        def _save_observation(
            self,
            observation: NativeFaceCaptureObservation,
            *,
            automatic: bool,
        ) -> None:
            if self._session is None:
                return
            face_was_captured = (
                _face_record_name(observation.face)
                in self._session.captured_faces
                if observation.face is not None
                else False
            )
            try:
                result = self._session.save_observation(observation)
            except Exception as exc:
                self._status.setText(f"Could not save face shot: {exc}")
                return
            self._saved_results.append(result)
            self._refresh_saved_shots()
            self._render_queue()
            prefix = (
                "Auto-saved"
                if automatic
                else "Retook"
                if face_was_captured
                else "Saved"
            )
            self._status.setText(
                f"{prefix} {result.object_full}.\n"
                f"Raw: {_display_path(result.raw_path)}\n"
                f"Annotated: {_display_path(result.annotated_path)}"
            )
            next_index = self._session.next_uncaptured_index(
                start=self._session.face_idx + 1
            )
            if (automatic or not face_was_captured) and next_index is not None:
                self._session.select_face_index(next_index)
                self._last_observation = None
            self._render_current_item()

        def _resume_live_view(self, message: str = "Live detection resumed.") -> None:
            self._resume_button.setEnabled(False)
            self._capture_button.setEnabled(self._session is not None)
            self._status.setText(message)
            self._timer.start()
            self._render_current_item()

        def _previous_item(self) -> None:
            if self._session is None:
                return
            self._session.select_face_index(self._session.face_idx - 1)
            self._resume_live_view(message="Moved to previous face.")

        def _render_current_item(self) -> None:
            if self._session is None:
                return
            face = self._session.current_face
            name = _face_record_name(face)
            captured = name in self._session.captured_faces
            self._syncing_queue_selection = True
            self._queue.setCurrentRow(self._session.face_idx)
            self._syncing_queue_selection = False
            self._current_item.setText(f"Current face: {name}")
            self._progress.setText(
                f"{self._session.face_idx + 1} / {len(self._session.faces)} queued | "
                f"{len(self._session.captured_faces)} captured | "
                f"{len(self._saved_results)} saved this session"
            )
            self._capture_button.setText("Retake Now" if captured else "Save Next")
            self._previous_button.setEnabled(self._session.face_idx > 0)

        def _render_queue(self) -> None:
            if self._session is None:
                return
            for row, face in enumerate(self._session.faces):
                item = self._queue.item(row)
                if item is None:
                    continue
                name = _face_record_name(face)
                captured = name in self._session.captured_faces
                marker = "[x]" if captured else "[ ]"
                item.setText(f"{row + 1}. {marker} {name}")
                item.setForeground(
                    QtGui.QBrush(
                        QtGui.QColor("#087a3d" if captured else "#8a5b00")
                    )
                )

        def _refresh_saved_shots(self) -> None:
            if self._session is None:
                return
            outputs = inspect_capture_face_readiness(self._config).output_status
            self._saved_shots = tuple(reversed(outputs.saved_shots))
            self._saved_gallery.clear()
            if self._saved_shots:
                item = QtWidgets.QListWidgetItem(
                    _capture_face_saved_shot_collection_label(self._saved_shots)
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, self._saved_shots)
                item.setToolTip("Click to inspect this saved-shot stack.")
                item.setIcon(
                    _capture_face_saved_shot_icon(
                        self._saved_shots,
                        self._saved_gallery.iconSize(),
                    )
                )
                self._saved_gallery.addItem(item)
            self._render_queue()

        def _preview_saved_shot(self, item: Any) -> None:
            shots = _capture_face_saved_shots_from_item(item)
            if not shots:
                return
            self._populate_saved_stack(shots)
            self._preview_saved_stack_shot(shots[0])
            self._saved_stack_field.setVisible(True)

        def _populate_saved_stack(
            self,
            shots: tuple[CaptureFaceSavedShot, ...],
        ) -> None:
            self._saved_stack_gallery.clear()
            for shot in shots:
                self._saved_stack_gallery.addItem(
                    _make_capture_face_saved_shot_item(
                        shot,
                        self._saved_stack_gallery.iconSize(),
                    )
                )
            if self._saved_stack_gallery.count() > 0:
                self._saved_stack_gallery.setCurrentRow(0)

        def _preview_stack_shot(self, item: Any) -> None:
            shot = _capture_face_saved_shot_from_item(item)
            if shot is not None:
                self._preview_saved_stack_shot(shot)

        def _preview_saved_stack_shot(self, shot: CaptureFaceSavedShot) -> None:
            preview_path = _capture_face_shot_preview_path(shot)
            if preview_path is not None:
                self._preview.show_preview(preview_path)
            self._resume_button.setEnabled(True)
            self._capture_button.setEnabled(False)
            self._timer.stop()
            self._status.setText(
                "Previewing saved face shot.\n"
                + _format_capture_face_saved_shot_details(shot)
                + "\n\nUse Resume Live View to continue capture."
            )

        def done(self, result: int) -> None:
            self._timer.stop()
            if self._session is not None:
                self._session.close()
            super().done(result)

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
            self._board_process: Any = None
            self._board_process_state = board_building_process_not_started()
            self._board_running_expected_yaml: Optional[Path] = None
            self._board_running_expected_registry: Optional[Path] = None
            self._board_previous_yaml_mtime_ns: Optional[int] = None
            self._board_previous_registry_mtime_ns: Optional[int] = None
            self._board_outputs_existed_at_launch = False
            self._board_output_seen = False
            self._capture_face_process: Any = None
            self._capture_face_process_state = capture_face_process_not_started()
            self._capture_running_expected_manifest: Optional[Path] = None
            self._capture_previous_manifest_mtime_ns: Optional[int] = None
            self._capture_manifest_existed_at_launch = False
            self._capture_manifest_seen = False

            self._calibration_output_timer = QtCore.QTimer(self)
            self._calibration_output_timer.setInterval(1000)
            self._calibration_output_timer.timeout.connect(
                self._poll_calibration_output
            )
            self._board_output_timer = QtCore.QTimer(self)
            self._board_output_timer.setInterval(1000)
            self._board_output_timer.timeout.connect(self._poll_board_output)
            self._capture_output_timer = QtCore.QTimer(self)
            self._capture_output_timer.setInterval(1000)
            self._capture_output_timer.timeout.connect(self._poll_capture_face_output)

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

            self._charuco_guidance = QtWidgets.QLabel(CHARUCO_PRINT_GUIDANCE)
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

            self._charuco_preview = _ImagePreviewCanvas(
                "No generated board yet.",
                object_name="CharucoPreviewCanvas",
            )
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

            self._last_object_tag_output_dir: Optional[Path] = None
            self._last_object_tag_sheet_file: Optional[Path] = None
            self._object_tags_card = QtWidgets.QFrame()
            self._object_tags_card.setObjectName("ActionCard")
            object_tags_layout = QtWidgets.QVBoxLayout(self._object_tags_card)
            object_tags_layout.setContentsMargins(14, 12, 14, 12)
            object_tags_layout.setSpacing(9)

            object_tags_title = QtWidgets.QLabel("Object AprilTag sheets")
            object_tags_title.setObjectName("CardTitle")
            object_tags_note = QtWidgets.QLabel(
                "Generate the printable tags for object faces using the same "
                "package workflow as posetag-gen-tags."
            )
            object_tags_note.setObjectName("MutedText")
            object_tags_note.setWordWrap(True)

            self._object_tags_family = QtWidgets.QComboBox()
            self._object_tags_family.addItems(list(OBJECT_TAG_FAMILY_CHOICES))
            self._object_tags_family.setCurrentText(DEFAULT_OBJECT_TAG_FAMILY)
            self._object_tags_family.setMinimumWidth(180)

            self._object_tags_tag_size = _make_float_spin(
                0.1,
                1000.0,
                DEFAULT_OBJECT_TAG_SIZE_MM,
                " mm",
            )

            self._object_tags_id_mode = QtWidgets.QComboBox()
            for mode in OBJECT_TAG_ID_MODE_CHOICES:
                self._object_tags_id_mode.addItem(
                    OBJECT_TAG_ID_MODE_LABELS[mode],
                    mode,
                )
            self._object_tags_id_mode.setMinimumWidth(180)
            self._object_tags_id_mode.currentIndexChanged.connect(
                self._object_tag_mode_changed
            )

            self._object_tags_ids = QtWidgets.QLineEdit(DEFAULT_OBJECT_TAG_IDS)
            self._object_tags_ids.setPlaceholderText("Examples: 1-4 or 1-3,7,9-10")
            self._object_tags_id_start = _make_int_spin(
                0,
                OBJECT_TAG_MAX_ID,
                DEFAULT_OBJECT_TAG_ID_START,
            )
            self._object_tags_id_count = _make_int_spin(
                1,
                OBJECT_TAG_MAX_ID + 1,
                DEFAULT_OBJECT_TAG_ID_COUNT,
            )

            self._object_tags_paper = QtWidgets.QComboBox()
            self._object_tags_paper.addItems(list(OBJECT_TAG_PAPER_CHOICES))
            self._object_tags_paper.setCurrentText("A4")
            self._object_tags_paper.setMinimumWidth(180)
            self._object_tags_paper.currentTextChanged.connect(
                self._object_tag_paper_changed
            )
            self._object_tags_paper_mm = QtWidgets.QLineEdit()
            self._object_tags_paper_mm.setPlaceholderText("Custom paper, e.g. 210x297")
            self._object_tags_orientation = QtWidgets.QComboBox()
            self._object_tags_orientation.addItems(
                list(OBJECT_TAG_ORIENTATION_CHOICES)
            )
            self._object_tags_orientation.setCurrentText("portrait")
            self._object_tags_orientation.setMinimumWidth(180)
            self._object_tags_dpi = _make_int_spin(
                1,
                2400,
                DEFAULT_OBJECT_TAG_DPI,
            )
            self._object_tags_prefix = QtWidgets.QLineEdit(
                DEFAULT_OBJECT_TAG_PREFIX
            )
            self._object_tags_margin_frac = _make_float_spin(
                0.0,
                0.44,
                0.05,
            )
            self._object_tags_margin_frac.setSingleStep(0.01)
            self._object_tags_label_gap_frac = _make_float_spin(
                0.0,
                0.99,
                0.05,
            )
            self._object_tags_label_gap_frac.setSingleStep(0.01)
            self._object_tags_pil_text = QtWidgets.QCheckBox("Use Pillow labels")

            self._object_tags_out_dir = QtWidgets.QLineEdit()
            self._object_tags_out_dir.setPlaceholderText(
                "Default: <project_root>/boards/patterns/"
            )
            object_tags_browse_button = QtWidgets.QPushButton("Browse")
            object_tags_browse_button.clicked.connect(
                self._browse_object_tags_output_dir
            )
            object_tags_out_row = QtWidgets.QHBoxLayout()
            object_tags_out_row.setContentsMargins(0, 0, 0, 0)
            object_tags_out_row.addWidget(self._object_tags_out_dir, 1)
            object_tags_out_row.addWidget(object_tags_browse_button)
            object_tags_out_widget = QtWidgets.QWidget()
            object_tags_out_widget.setLayout(object_tags_out_row)

            self._object_tags_list_field = _make_field(
                "IDs / ranges",
                self._object_tags_ids,
            )
            self._object_tags_start_field = _make_field(
                "Start ID",
                self._object_tags_id_start,
            )
            self._object_tags_count_field = _make_field(
                "ID count",
                self._object_tags_id_count,
            )
            self._object_tags_paper_mm_field = _make_field(
                "Custom paper",
                self._object_tags_paper_mm,
            )

            object_tags_grid = QtWidgets.QGridLayout()
            object_tags_grid.setContentsMargins(0, 0, 0, 0)
            object_tags_grid.setHorizontalSpacing(14)
            object_tags_grid.setVerticalSpacing(10)
            object_tags_grid.addWidget(
                _make_field("Family", self._object_tags_family),
                0,
                0,
            )
            object_tags_grid.addWidget(
                _make_field("Tag size", self._object_tags_tag_size),
                0,
                1,
            )
            object_tags_grid.addWidget(
                _make_field("ID entry", self._object_tags_id_mode),
                1,
                0,
            )
            object_tags_grid.addWidget(self._object_tags_list_field, 1, 1)
            object_tags_grid.addWidget(self._object_tags_start_field, 2, 0)
            object_tags_grid.addWidget(self._object_tags_count_field, 2, 1)
            object_tags_grid.addWidget(
                _make_field("Paper size", self._object_tags_paper),
                3,
                0,
            )
            object_tags_grid.addWidget(self._object_tags_paper_mm_field, 3, 1)
            object_tags_grid.addWidget(
                _make_field("Orientation", self._object_tags_orientation),
                4,
                0,
            )
            object_tags_grid.addWidget(
                _make_field("DPI", self._object_tags_dpi),
                4,
                1,
            )
            object_tags_grid.addWidget(
                _make_field("Filename prefix", self._object_tags_prefix),
                5,
                0,
                1,
                2,
            )
            object_tags_grid.addWidget(
                _make_field("Margin fraction", self._object_tags_margin_frac),
                6,
                0,
            )
            object_tags_grid.addWidget(
                _make_field("Label gap fraction", self._object_tags_label_gap_frac),
                6,
                1,
            )
            object_tags_grid.addWidget(self._object_tags_pil_text, 7, 0)
            object_tags_grid.addWidget(
                _make_field("Output folder", object_tags_out_widget),
                8,
                0,
                1,
                2,
            )
            object_tags_grid.setColumnStretch(0, 1)
            object_tags_grid.setColumnStretch(1, 1)

            for field in (
                self._object_tags_tag_size,
                self._object_tags_id_start,
                self._object_tags_id_count,
                self._object_tags_dpi,
                self._object_tags_margin_frac,
                self._object_tags_label_gap_frac,
            ):
                field.valueChanged.connect(self._update_object_tag_flow)
            for field in (
                self._object_tags_family,
                self._object_tags_orientation,
            ):
                field.currentTextChanged.connect(self._update_object_tag_flow)
            for field in (
                self._object_tags_ids,
                self._object_tags_paper_mm,
                self._object_tags_prefix,
                self._object_tags_out_dir,
            ):
                field.textChanged.connect(self._update_object_tag_flow)
            self._object_tags_pil_text.stateChanged.connect(
                self._update_object_tag_flow
            )

            self._object_tags_guidance = QtWidgets.QLabel(
                OBJECT_TAG_PRINT_GUIDANCE
            )
            self._object_tags_guidance.setObjectName("GuidanceText")
            self._object_tags_guidance.setWordWrap(True)

            self._object_tags_readiness = QtWidgets.QLabel()
            self._object_tags_readiness.setObjectName("OutputText")
            self._object_tags_readiness.setWordWrap(True)
            self._object_tags_readiness.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._object_tags_expected_output = QtWidgets.QLabel()
            self._object_tags_expected_output.setObjectName("OutputText")
            self._object_tags_expected_output.setWordWrap(True)
            self._object_tags_expected_output.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._object_tags_outputs = QtWidgets.QLabel("No generated files yet.")
            self._object_tags_outputs.setObjectName("OutputText")
            self._object_tags_outputs.setWordWrap(True)
            self._object_tags_outputs.setMinimumHeight(96)
            self._object_tags_outputs.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignTop
            )
            self._object_tags_outputs.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self._object_tags_generate_button = QtWidgets.QPushButton(
                "Generate Object Tags"
            )
            self._object_tags_generate_button.setObjectName("PrimaryActionButton")
            self._object_tags_generate_button.clicked.connect(
                self._generate_object_tags
            )
            object_tags_refresh_button = QtWidgets.QPushButton("Refresh Status")
            object_tags_refresh_button.setObjectName("SecondaryActionButton")
            object_tags_refresh_button.clicked.connect(self._refresh)
            self._object_tags_open_file_button = QtWidgets.QPushButton(
                "Open First Sheet"
            )
            self._object_tags_open_file_button.setObjectName("SecondaryActionButton")
            self._object_tags_open_file_button.setEnabled(False)
            self._object_tags_open_file_button.clicked.connect(
                self._open_object_tags_sheet_file
            )
            self._object_tags_open_folder_button = QtWidgets.QPushButton(
                "Open Output Folder"
            )
            self._object_tags_open_folder_button.setObjectName("SecondaryActionButton")
            self._object_tags_open_folder_button.setEnabled(False)
            self._object_tags_open_folder_button.clicked.connect(
                self._open_object_tags_output_folder
            )

            object_tags_action_row = QtWidgets.QHBoxLayout()
            object_tags_action_row.addWidget(self._object_tags_generate_button)
            object_tags_action_row.addWidget(object_tags_refresh_button)
            object_tags_action_row.addStretch(1)

            object_tags_file_action_row = QtWidgets.QHBoxLayout()
            object_tags_file_action_row.addWidget(self._object_tags_open_file_button)
            object_tags_file_action_row.addWidget(
                self._object_tags_open_folder_button
            )
            object_tags_file_action_row.addStretch(1)

            object_tags_status_grid = QtWidgets.QGridLayout()
            object_tags_status_grid.setContentsMargins(0, 0, 0, 0)
            object_tags_status_grid.setHorizontalSpacing(12)
            object_tags_status_grid.setVerticalSpacing(8)
            object_tags_status_grid.addWidget(
                _make_field("Readiness", self._object_tags_readiness),
                0,
                0,
            )
            object_tags_status_grid.addWidget(
                _make_field("Expected output", self._object_tags_expected_output),
                0,
                1,
            )
            object_tags_status_grid.addWidget(
                _make_field("Generated files", self._object_tags_outputs),
                1,
                0,
                1,
                2,
            )
            object_tags_status_grid.setColumnStretch(0, 1)
            object_tags_status_grid.setColumnStretch(1, 1)

            self._object_tags_preview = _ImagePreviewCanvas(
                "No generated tag sheet yet.",
                object_name="ObjectTagPreviewCanvas",
            )
            object_tags_preview_title = QtWidgets.QLabel("Preview")
            object_tags_preview_title.setObjectName("FieldLabel")
            object_tags_preview_note = QtWidgets.QLabel(
                "First generated PNG sheet."
            )
            object_tags_preview_note.setObjectName("MutedText")
            object_tags_preview_note.setWordWrap(True)
            object_tags_preview_column = QtWidgets.QFrame()
            object_tags_preview_column.setObjectName("ObjectTagPreviewColumn")
            object_tags_preview_layout = QtWidgets.QVBoxLayout(
                object_tags_preview_column
            )
            object_tags_preview_layout.setContentsMargins(10, 10, 10, 10)
            object_tags_preview_layout.setSpacing(7)
            object_tags_preview_layout.addWidget(object_tags_preview_title)
            object_tags_preview_layout.addWidget(self._object_tags_preview)
            object_tags_preview_layout.addWidget(object_tags_preview_note)
            object_tags_preview_layout.addStretch(1)
            object_tags_preview_column.setMinimumWidth(320)

            object_tags_controls = QtWidgets.QWidget()
            object_tags_controls_layout = QtWidgets.QVBoxLayout(
                object_tags_controls
            )
            object_tags_controls_layout.setContentsMargins(0, 0, 0, 0)
            object_tags_controls_layout.setSpacing(9)
            object_tags_controls_layout.addLayout(object_tags_grid)
            object_tags_controls_layout.addWidget(self._object_tags_guidance)
            object_tags_controls_layout.addLayout(object_tags_status_grid)
            object_tags_controls_layout.addLayout(object_tags_file_action_row)
            object_tags_controls_layout.addLayout(object_tags_action_row)
            object_tags_controls_layout.addStretch(1)

            object_tags_body = QtWidgets.QHBoxLayout()
            object_tags_body.setContentsMargins(0, 0, 0, 0)
            object_tags_body.setSpacing(12)
            object_tags_body.addWidget(
                object_tags_controls,
                3,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )
            object_tags_body.addWidget(
                object_tags_preview_column,
                2,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )

            object_tags_layout.addWidget(object_tags_title)
            object_tags_layout.addWidget(object_tags_note)
            object_tags_layout.addLayout(object_tags_body)

            self._board_building_card = QtWidgets.QFrame()
            self._board_building_card.setObjectName("ActionCard")
            board_building_layout = QtWidgets.QVBoxLayout(
                self._board_building_card
            )
            board_building_layout.setContentsMargins(14, 12, 14, 12)
            board_building_layout.setSpacing(9)

            board_building_title = QtWidgets.QLabel("Object board definitions")
            board_building_title.setObjectName("CardTitle")
            board_building_note = QtWidgets.QLabel(
                "Prepare the existing posetag-make-board workflow after "
                "printing and attaching object AprilTags. Name each physical "
                "object and side/face before capture."
            )
            board_building_note.setObjectName("MutedText")
            board_building_note.setWordWrap(True)

            self._board_object_label = QtWidgets.QLineEdit(
                DEFAULT_BOARD_OBJECT_LABEL
            )
            self._board_object_label.setPlaceholderText(
                "Example: connection_plate_white"
            )
            self._board_side_label = QtWidgets.QComboBox()
            self._board_side_label.setEditable(True)
            self._board_side_label.addItems(DEFAULT_BOARD_SIDE_LABELS)
            self._board_side_label.setCurrentText(DEFAULT_BOARD_SIDE_LABEL)
            self._board_side_label.setMinimumWidth(150)
            if self._board_side_label.lineEdit() is not None:
                self._board_side_label.lineEdit().setPlaceholderText(
                    "sideA, front, top"
                )
            self._board_next_side_button = QtWidgets.QPushButton("Next Side")
            self._board_next_side_button.setObjectName("SecondaryActionButton")
            self._board_next_side_button.clicked.connect(
                self._set_next_board_side_label
            )
            board_side_row = QtWidgets.QHBoxLayout()
            board_side_row.setContentsMargins(0, 0, 0, 0)
            board_side_row.addWidget(self._board_side_label, 1)
            board_side_row.addWidget(self._board_next_side_button)
            board_side_widget = QtWidgets.QWidget()
            board_side_widget.setLayout(board_side_row)
            self._board_object_name = QtWidgets.QLineEdit(DEFAULT_BOARD_OBJECT_NAME)
            self._board_object_name.setPlaceholderText(
                "Example: connection_plate_white_sideA"
            )
            self._board_object_name.setReadOnly(True)
            self._board_object_name.setToolTip(
                "Board definition name passed to posetag-make-board --object_name."
            )
            self._board_batch_instances = QtWidgets.QLineEdit()
            self._board_batch_instances.setPlaceholderText(
                "Optional: 01-08 or 01,02"
            )
            self._board_batch_sides = QtWidgets.QLineEdit("sideA, sideB")
            self._board_batch_sides.setPlaceholderText("sideA, sideB or sideA-sideD")
            self._board_batch_rows_model: list[BoardBatchRow] = []
            self._board_batch_draft_loading = False
            self._loaded_board_batch_draft_path: Optional[Path] = None
            self._loaded_board_batch_draft_mtime_ns: Optional[int] = None
            self._board_batch_row_list = QtWidgets.QListWidget()
            self._board_batch_row_list.setObjectName("BoardObjectRows")
            self._board_batch_row_list.setMaximumHeight(96)
            self._board_batch_row_list.setMinimumHeight(72)
            self._board_batch_row_list.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self._board_batch_row_list.currentRowChanged.connect(
                self._select_board_batch_row
            )
            self._board_add_batch_row_button = QtWidgets.QPushButton("Add Object")
            self._board_add_batch_row_button.setObjectName("SecondaryActionButton")
            self._board_add_batch_row_button.clicked.connect(
                self._add_board_batch_row
            )
            self._board_remove_batch_row_button = QtWidgets.QPushButton("Remove")
            self._board_remove_batch_row_button.setObjectName("SecondaryActionButton")
            self._board_remove_batch_row_button.clicked.connect(
                self._remove_selected_board_batch_row
            )
            self._board_clear_batch_rows_button = QtWidgets.QPushButton("Clear")
            self._board_clear_batch_rows_button.setObjectName("SecondaryActionButton")
            self._board_clear_batch_rows_button.clicked.connect(
                self._clear_board_batch_rows
            )
            self._board_batch_rows = QtWidgets.QPlainTextEdit()
            self._board_batch_rows.setPlaceholderText(
                "Optional multi-object batch rows:\n"
                "connection_plate | 01-08 | sideA, sideB | 80\n"
                "column | 01-04 | sideA-sideB | 40\n"
                "column | 01-04 | sideC-sideD | 80"
            )
            self._board_batch_rows.setMaximumHeight(82)
            self._board_batch_preview = QtWidgets.QLabel()
            self._board_batch_preview.setObjectName("OutputText")
            self._board_batch_preview.setWordWrap(True)
            self._board_batch_preview.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._board_batch_preview.setMinimumHeight(72)
            self._board_batch_preview.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignTop
            )
            self._board_tag_size = _make_float_spin(
                0.1,
                1000.0,
                DEFAULT_BOARD_TAG_SIZE_MM,
                " mm",
            )
            self._board_tag_size_autofill_mm: Optional[float] = None
            self._board_tag_size_user_edited = False
            self._board_tag_size_autofilling = False
            board_batch_buttons = QtWidgets.QHBoxLayout()
            board_batch_buttons.setContentsMargins(0, 0, 0, 0)
            board_batch_buttons.setSpacing(6)
            board_batch_buttons.addWidget(self._board_add_batch_row_button)
            board_batch_buttons.addWidget(self._board_remove_batch_row_button)
            board_batch_buttons.addWidget(self._board_clear_batch_rows_button)

            board_batch_widget = QtWidgets.QWidget()
            board_batch_grid = QtWidgets.QGridLayout(board_batch_widget)
            board_batch_grid.setContentsMargins(0, 0, 0, 0)
            board_batch_grid.setHorizontalSpacing(10)
            board_batch_grid.setVerticalSpacing(8)
            board_batch_grid.addWidget(
                _make_field("Object", self._board_object_label),
                0,
                0,
            )
            board_batch_grid.addWidget(
                _make_field("Instances", self._board_batch_instances),
                0,
                1,
            )
            board_batch_grid.addWidget(
                _make_field("Sides / faces", self._board_batch_sides),
                0,
                2,
            )
            board_batch_grid.addWidget(
                _make_field("Tag size", self._board_tag_size),
                0,
                3,
            )
            board_batch_grid.addLayout(board_batch_buttons, 1, 0)
            board_batch_grid.addWidget(
                _make_field("Objects", self._board_batch_row_list),
                1,
                1,
            )
            board_batch_grid.addWidget(
                _make_field("Queue", self._board_batch_preview),
                1,
                2,
                1,
                2,
            )
            board_batch_grid.setColumnStretch(0, 2)
            board_batch_grid.setColumnStretch(1, 1)
            board_batch_grid.setColumnStretch(2, 1)
            board_batch_grid.setColumnStretch(3, 1)

            board_bulk_widget = QtWidgets.QWidget()
            board_bulk_layout = QtWidgets.QVBoxLayout(board_bulk_widget)
            board_bulk_layout.setContentsMargins(0, 0, 0, 0)
            board_bulk_layout.setSpacing(6)
            board_bulk_layout.addWidget(
                _make_field("Rows", self._board_batch_rows),
            )
            board_bulk_group = _make_collapsible_group(
                "Bulk Paste",
                board_bulk_widget,
                checked=False,
            )
            self._board_family = QtWidgets.QLineEdit(DEFAULT_BOARD_FAMILY)
            self._board_family.setPlaceholderText("tag36h11")
            self._board_source = QtWidgets.QComboBox()
            for source in BOARD_SOURCE_CHOICES:
                self._board_source.addItem(BOARD_SOURCE_LABELS[source], source)
            self._board_source.setMinimumWidth(180)
            self._board_source.currentIndexChanged.connect(
                self._board_source_changed
            )
            self._board_camera_index = _make_int_spin(
                0,
                99,
                DEFAULT_BOARD_CAMERA_INDEX,
            )
            self._board_video_path = QtWidgets.QLineEdit()
            self._board_video_path.setPlaceholderText("Select board-building video")
            board_video_browse_button = QtWidgets.QPushButton("Browse")
            board_video_browse_button.clicked.connect(self._browse_board_video)
            self._board_video_browse_button = board_video_browse_button
            board_video_row = QtWidgets.QHBoxLayout()
            board_video_row.setContentsMargins(0, 0, 0, 0)
            board_video_row.addWidget(self._board_video_path, 1)
            board_video_row.addWidget(board_video_browse_button)
            board_video_widget = QtWidgets.QWidget()
            board_video_widget.setLayout(board_video_row)

            self._board_width = _make_int_spin(1, 10000, DEFAULT_BOARD_WIDTH)
            self._board_height = _make_int_spin(1, 10000, DEFAULT_BOARD_HEIGHT)
            self._board_fps = _make_int_spin(1, 240, DEFAULT_BOARD_FPS)
            self._board_z_threshold = _make_float_spin(
                0.001,
                10.0,
                DEFAULT_BOARD_Z_THRESHOLD_M,
                " m",
            )
            self._board_z_threshold.setSingleStep(0.001)
            self._board_save_shot = QtWidgets.QCheckBox("Save audit shot")
            self._board_save_shot.stateChanged.connect(
                self._board_save_shot_changed
            )
            self._board_allow_nonplanar = QtWidgets.QCheckBox(
                "Allow non-planar warning"
            )

            self._board_calibration_path = QtWidgets.QLineEdit()
            self._board_calibration_path.setPlaceholderText(
                "Default: <project_root>/calib/calib_color.yaml"
            )
            board_calib_browse_button = QtWidgets.QPushButton("Browse")
            board_calib_browse_button.clicked.connect(
                self._browse_board_calibration
            )
            board_calib_row = QtWidgets.QHBoxLayout()
            board_calib_row.setContentsMargins(0, 0, 0, 0)
            board_calib_row.addWidget(self._board_calibration_path, 1)
            board_calib_row.addWidget(board_calib_browse_button)
            board_calib_widget = QtWidgets.QWidget()
            board_calib_widget.setLayout(board_calib_row)

            self._board_out_dir = QtWidgets.QLineEdit()
            self._board_out_dir.setPlaceholderText(
                "Default: <project_root>/boards/"
            )
            board_out_browse_button = QtWidgets.QPushButton("Browse")
            board_out_browse_button.clicked.connect(self._browse_board_output_dir)
            board_out_row = QtWidgets.QHBoxLayout()
            board_out_row.setContentsMargins(0, 0, 0, 0)
            board_out_row.addWidget(self._board_out_dir, 1)
            board_out_row.addWidget(board_out_browse_button)
            board_out_widget = QtWidgets.QWidget()
            board_out_widget.setLayout(board_out_row)

            self._board_registry_path = QtWidgets.QLineEdit()
            self._board_registry_path.setPlaceholderText(
                "Default: <project_root>/boards/tag_registry.yaml"
            )
            board_registry_browse_button = QtWidgets.QPushButton("Browse")
            board_registry_browse_button.clicked.connect(
                self._browse_board_registry
            )
            board_registry_row = QtWidgets.QHBoxLayout()
            board_registry_row.setContentsMargins(0, 0, 0, 0)
            board_registry_row.addWidget(self._board_registry_path, 1)
            board_registry_row.addWidget(board_registry_browse_button)
            board_registry_widget = QtWidgets.QWidget()
            board_registry_widget.setLayout(board_registry_row)

            self._board_shots_dir = QtWidgets.QLineEdit()
            self._board_shots_dir.setPlaceholderText(
                "Default: <project_root>/boards/shots/"
            )
            board_shots_browse_button = QtWidgets.QPushButton("Browse")
            board_shots_browse_button.clicked.connect(self._browse_board_shots_dir)
            board_shots_row = QtWidgets.QHBoxLayout()
            board_shots_row.setContentsMargins(0, 0, 0, 0)
            board_shots_row.addWidget(self._board_shots_dir, 1)
            board_shots_row.addWidget(board_shots_browse_button)
            board_shots_widget = QtWidgets.QWidget()
            board_shots_widget.setLayout(board_shots_row)

            self._board_camera_field = _make_field(
                "Camera index",
                self._board_camera_index,
            )
            self._board_video_field = _make_field("Video path", board_video_widget)
            self._board_shots_field = _make_field(
                "Audit-shot folder",
                board_shots_widget,
            )

            board_capture_widget = QtWidgets.QWidget()
            board_capture_grid = QtWidgets.QGridLayout(board_capture_widget)
            board_capture_grid.setContentsMargins(0, 0, 0, 0)
            board_capture_grid.setHorizontalSpacing(10)
            board_capture_grid.setVerticalSpacing(8)
            board_capture_grid.addWidget(
                _make_field("Source", self._board_source),
                0,
                0,
            )
            board_capture_grid.addWidget(
                _make_field("Calibration YAML", board_calib_widget),
                1,
                0,
                1,
                2,
            )
            board_capture_grid.addWidget(self._board_camera_field, 2, 0)
            board_capture_grid.addWidget(self._board_video_field, 2, 1)
            board_capture_grid.setColumnStretch(0, 1)
            board_capture_grid.setColumnStretch(1, 1)

            board_single_widget = QtWidgets.QWidget()
            board_single_grid = QtWidgets.QGridLayout(board_single_widget)
            board_single_grid.setContentsMargins(0, 0, 0, 0)
            board_single_grid.setHorizontalSpacing(10)
            board_single_grid.setVerticalSpacing(8)
            board_single_grid.addWidget(
                _make_field("Side / face", board_side_widget),
                0,
                0,
            )
            board_single_grid.addWidget(
                _make_field("Board definition", self._board_object_name),
                0,
                1,
            )
            board_single_grid.setColumnStretch(0, 1)
            board_single_grid.setColumnStretch(1, 1)
            board_single_group = _make_collapsible_group(
                "Single Board",
                board_single_widget,
                checked=False,
            )

            board_advanced_widget = QtWidgets.QWidget()
            board_advanced_grid = QtWidgets.QGridLayout(board_advanced_widget)
            board_advanced_grid.setContentsMargins(0, 0, 0, 0)
            board_advanced_grid.setHorizontalSpacing(10)
            board_advanced_grid.setVerticalSpacing(8)
            board_advanced_grid.addWidget(
                _make_field("AprilTag family", self._board_family),
                0,
                0,
                1,
                2,
            )
            board_advanced_grid.addWidget(
                _make_field("Width", self._board_width),
                1,
                0,
            )
            board_advanced_grid.addWidget(
                _make_field("Height", self._board_height),
                1,
                1,
            )
            board_advanced_grid.addWidget(
                _make_field("FPS", self._board_fps),
                2,
                0,
            )
            board_advanced_grid.addWidget(
                _make_field("Planarity threshold", self._board_z_threshold),
                2,
                1,
            )
            board_advanced_grid.addWidget(
                _make_field("Board output folder", board_out_widget),
                3,
                0,
                1,
                2,
            )
            board_advanced_grid.addWidget(
                _make_field("Tag registry path", board_registry_widget),
                4,
                0,
                1,
                2,
            )
            board_advanced_grid.addWidget(self._board_save_shot, 5, 0)
            board_advanced_grid.addWidget(self._board_allow_nonplanar, 5, 1)
            board_advanced_grid.addWidget(self._board_shots_field, 6, 0, 1, 2)
            board_advanced_grid.setColumnStretch(0, 1)
            board_advanced_grid.setColumnStretch(1, 1)
            board_advanced_group = _make_collapsible_group(
                "Advanced Capture And Outputs",
                board_advanced_widget,
                checked=False,
            )

            for field in (
                self._board_camera_index,
                self._board_width,
                self._board_height,
                self._board_fps,
                self._board_z_threshold,
            ):
                field.valueChanged.connect(self._update_board_building_flow)
            self._board_tag_size.valueChanged.connect(
                self._board_tag_size_changed
            )
            self._board_object_label.textChanged.connect(
                self._board_identity_changed
            )
            self._board_side_label.currentTextChanged.connect(
                self._board_identity_changed
            )
            self._board_batch_instances.textChanged.connect(
                self._board_batch_fields_changed
            )
            self._board_batch_sides.textChanged.connect(
                self._board_batch_fields_changed
            )
            self._board_batch_rows.textChanged.connect(
                self._board_batch_fields_changed
            )
            for field in (
                self._board_family,
                self._board_video_path,
                self._board_calibration_path,
                self._board_out_dir,
                self._board_registry_path,
                self._board_shots_dir,
            ):
                field.textChanged.connect(self._update_board_building_flow)
            self._board_allow_nonplanar.stateChanged.connect(
                self._update_board_building_flow
            )

            self._board_guidance = QtWidgets.QLabel(BOARD_BUILDING_GUIDANCE)
            self._board_guidance.setObjectName("GuidanceText")
            self._board_guidance.setWordWrap(True)
            self._board_controls = QtWidgets.QLabel(
                "After ENTER captures a frame, the terminal asks for the "
                "detected tag IDs to include, then asks which selected tag is "
                "the board origin."
            )
            self._board_controls.setObjectName("GuidanceText")
            self._board_controls.setWordWrap(True)
            self._board_readiness = QtWidgets.QLabel()
            self._board_readiness.setObjectName("OutputText")
            self._board_readiness.setWordWrap(True)
            self._board_readiness.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._board_outputs = QtWidgets.QLabel()
            self._board_outputs.setObjectName("OutputText")
            self._board_outputs.setWordWrap(True)
            self._board_outputs.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._board_process_state_label = QtWidgets.QLabel()
            self._board_process_state_label.setObjectName("OutputText")
            self._board_process_state_label.setWordWrap(True)
            self._board_process_state_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._board_log = QtWidgets.QPlainTextEdit()
            self._board_log.setObjectName("CalibrationLog")
            self._board_log.setReadOnly(True)
            self._board_log.setMaximumHeight(118)
            self._board_log.setPlaceholderText(
                "Board-builder stdout/stderr will appear here after launch."
            )
            self._board_log.document().setMaximumBlockCount(250)
            self._board_prompt_input = QtWidgets.QLineEdit()
            self._board_prompt_input.setPlaceholderText(
                "Prompt response, e.g. 12,37 or 12"
            )
            self._board_prompt_input.returnPressed.connect(
                self._send_board_prompt_response
            )
            self._board_send_prompt_button = QtWidgets.QPushButton("Send Response")
            self._board_send_prompt_button.setObjectName("SecondaryActionButton")
            self._board_send_prompt_button.clicked.connect(
                self._send_board_prompt_response
            )
            board_prompt_row = QtWidgets.QHBoxLayout()
            board_prompt_row.setContentsMargins(0, 0, 0, 0)
            board_prompt_row.addWidget(self._board_prompt_input, 1)
            board_prompt_row.addWidget(self._board_send_prompt_button)
            board_prompt_widget = QtWidgets.QWidget()
            board_prompt_widget.setLayout(board_prompt_row)
            board_refresh_button = QtWidgets.QPushButton("Refresh Status")
            board_refresh_button.setObjectName("SecondaryActionButton")
            board_refresh_button.clicked.connect(self._refresh)
            self._board_batch_capture_button = QtWidgets.QPushButton(
                "Start Batch"
            )
            self._board_batch_capture_button.setObjectName("PrimaryActionButton")
            self._board_batch_capture_button.clicked.connect(
                self._open_native_board_batch_capture
            )
            self._board_guided_capture_button = QtWidgets.QPushButton(
                "Guided Capture"
            )
            self._board_guided_capture_button.setObjectName("SecondaryActionButton")
            self._board_guided_capture_button.clicked.connect(
                self._open_native_board_capture
            )
            self._board_run_button = QtWidgets.QPushButton("Run Board Builder")
            self._board_run_button.setObjectName("SecondaryActionButton")
            self._board_run_button.clicked.connect(self._run_board_building)

            board_summary_grid = QtWidgets.QGridLayout()
            board_summary_grid.setContentsMargins(0, 0, 0, 0)
            board_summary_grid.setHorizontalSpacing(10)
            board_summary_grid.setVerticalSpacing(8)
            board_summary_grid.addWidget(
                _make_field("Readiness", self._board_readiness),
                0,
                0,
            )
            board_summary_grid.addWidget(
                _make_field("Expected outputs", self._board_outputs),
                0,
                1,
            )
            board_summary_grid.setColumnStretch(0, 1)
            board_summary_grid.setColumnStretch(1, 1)

            board_cli_grid = QtWidgets.QGridLayout()
            board_cli_grid.setContentsMargins(0, 0, 0, 0)
            board_cli_grid.setHorizontalSpacing(10)
            board_cli_grid.setVerticalSpacing(8)
            board_cli_grid.addWidget(
                _make_field("Process state", self._board_process_state_label),
                0,
                0,
                1,
                2,
            )
            board_cli_grid.addWidget(
                _make_field("Prompt response", board_prompt_widget),
                1,
                0,
                1,
                2,
            )
            board_cli_grid.addWidget(
                _make_field("Process log", self._board_log),
                2,
                0,
                1,
                2,
            )
            board_cli_grid.setColumnStretch(0, 1)
            board_cli_grid.setColumnStretch(1, 1)
            board_cli_action_row = QtWidgets.QHBoxLayout()
            board_cli_action_row.addWidget(self._board_run_button)
            board_cli_action_row.addStretch(1)
            board_cli_widget = QtWidgets.QWidget()
            board_cli_layout = QtWidgets.QVBoxLayout(board_cli_widget)
            board_cli_layout.setContentsMargins(0, 0, 0, 0)
            board_cli_layout.setSpacing(8)
            board_cli_layout.addWidget(self._board_guidance)
            board_cli_layout.addWidget(self._board_controls)
            board_cli_layout.addLayout(board_cli_grid)
            board_cli_layout.addLayout(board_cli_action_row)
            board_cli_group = _make_collapsible_group(
                "CLI Fallback And Logs",
                board_cli_widget,
                checked=False,
            )

            board_action_row = QtWidgets.QHBoxLayout()
            board_action_row.addWidget(self._board_batch_capture_button)
            board_action_row.addWidget(self._board_guided_capture_button)
            board_action_row.addWidget(board_refresh_button)
            board_action_row.addStretch(1)

            board_building_layout.addWidget(board_building_title)
            board_building_layout.addWidget(board_building_note)
            board_building_layout.addWidget(board_batch_widget)
            board_building_layout.addWidget(board_bulk_group)
            board_building_layout.addWidget(board_capture_widget)
            board_building_layout.addWidget(board_single_group)
            board_building_layout.addWidget(board_advanced_group)
            board_building_layout.addLayout(board_summary_grid)
            board_building_layout.addLayout(board_action_row)
            board_building_layout.addWidget(board_cli_group)

            self._capture_face_card = QtWidgets.QFrame()
            self._capture_face_card.setObjectName("ActionCard")
            capture_face_layout = QtWidgets.QVBoxLayout(self._capture_face_card)
            capture_face_layout.setContentsMargins(14, 12, 14, 12)
            capture_face_layout.setSpacing(9)

            capture_face_title = QtWidgets.QLabel("Face-shot capture")
            capture_face_title.setObjectName("CardTitle")
            capture_face_note = QtWidgets.QLabel(
                "Capture the registered board faces carried forward from "
                "Stage 4. Missing faces are queued first."
            )
            capture_face_note.setObjectName("MutedText")
            capture_face_note.setWordWrap(True)

            self._capture_face_object = QtWidgets.QComboBox()
            self._capture_face_object.setEditable(True)
            self._capture_face_object.setMinimumWidth(240)
            if self._capture_face_object.lineEdit() is not None:
                self._capture_face_object.lineEdit().setPlaceholderText(
                    "All missing faces, one object, or one face"
                )
            self._capture_face_object.currentTextChanged.connect(
                self._update_capture_face_flow
            )

            self._capture_face_queue = QtWidgets.QListWidget()
            self._capture_face_queue.setObjectName("BatchCaptureQueue")
            self._capture_face_queue.setMinimumHeight(120)
            self._capture_face_queue.setMaximumHeight(190)
            self._capture_face_queue.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            )
            self._capture_face_queue.itemSelectionChanged.connect(
                self._update_capture_face_action_buttons
            )

            self._capture_face_source = QtWidgets.QComboBox()
            for source in CAPTURE_SOURCE_CHOICES:
                self._capture_face_source.addItem(
                    CAPTURE_SOURCE_LABELS[source],
                    source,
                )
            self._capture_face_source.setMinimumWidth(180)
            self._capture_face_source.currentIndexChanged.connect(
                self._capture_face_source_changed
            )
            self._capture_face_camera_index = _make_int_spin(
                0,
                99,
                DEFAULT_CAPTURE_CAMERA_INDEX,
            )
            self._capture_face_video_path = QtWidgets.QLineEdit()
            self._capture_face_video_path.setPlaceholderText(
                "Select face-shot video"
            )
            capture_video_browse_button = QtWidgets.QPushButton("Browse")
            capture_video_browse_button.clicked.connect(
                self._browse_capture_face_video
            )
            self._capture_face_video_browse_button = capture_video_browse_button
            capture_video_row = QtWidgets.QHBoxLayout()
            capture_video_row.setContentsMargins(0, 0, 0, 0)
            capture_video_row.addWidget(self._capture_face_video_path, 1)
            capture_video_row.addWidget(capture_video_browse_button)
            capture_video_widget = QtWidgets.QWidget()
            capture_video_widget.setLayout(capture_video_row)

            self._capture_face_width = _make_int_spin(
                1,
                10000,
                DEFAULT_CAPTURE_WIDTH,
            )
            self._capture_face_height = _make_int_spin(
                1,
                10000,
                DEFAULT_CAPTURE_HEIGHT,
            )
            self._capture_face_fps = _make_int_spin(1, 240, DEFAULT_CAPTURE_FPS)
            self._capture_face_min_expected = _make_int_spin(
                1,
                1000,
                DEFAULT_CAPTURE_MIN_EXPECTED,
            )
            self._capture_face_panel_width = _make_int_spin(
                0,
                2000,
                DEFAULT_CAPTURE_PANEL_WIDTH,
            )
            self._capture_face_recent_width = _make_int_spin(
                0,
                2000,
                DEFAULT_CAPTURE_RECENT_WIDTH,
            )
            self._capture_face_layout = QtWidgets.QComboBox()
            self._capture_face_layout.addItems(CAPTURE_LAYOUT_CHOICES)
            self._capture_face_layout.setCurrentText("by_object_side")

            self._capture_face_calibration_path = QtWidgets.QLineEdit()
            self._capture_face_calibration_path.setPlaceholderText(
                "Default: <project_root>/calib/calib_color.yaml"
            )
            capture_calib_browse_button = QtWidgets.QPushButton("Browse")
            capture_calib_browse_button.clicked.connect(
                self._browse_capture_face_calibration
            )
            capture_calib_row = QtWidgets.QHBoxLayout()
            capture_calib_row.setContentsMargins(0, 0, 0, 0)
            capture_calib_row.addWidget(self._capture_face_calibration_path, 1)
            capture_calib_row.addWidget(capture_calib_browse_button)
            capture_calib_widget = QtWidgets.QWidget()
            capture_calib_widget.setLayout(capture_calib_row)

            self._capture_face_registry_path = QtWidgets.QLineEdit()
            self._capture_face_registry_path.setPlaceholderText(
                "Default: <project_root>/boards/tag_registry.yaml"
            )
            capture_registry_browse_button = QtWidgets.QPushButton("Browse")
            capture_registry_browse_button.clicked.connect(
                self._browse_capture_face_registry
            )
            capture_registry_row = QtWidgets.QHBoxLayout()
            capture_registry_row.setContentsMargins(0, 0, 0, 0)
            capture_registry_row.addWidget(self._capture_face_registry_path, 1)
            capture_registry_row.addWidget(capture_registry_browse_button)
            capture_registry_widget = QtWidgets.QWidget()
            capture_registry_widget.setLayout(capture_registry_row)

            self._capture_face_out_dir = QtWidgets.QLineEdit()
            self._capture_face_out_dir.setPlaceholderText(
                "Default: <project_root>/shots/"
            )
            capture_out_browse_button = QtWidgets.QPushButton("Browse")
            capture_out_browse_button.clicked.connect(
                self._browse_capture_face_output_dir
            )
            capture_out_row = QtWidgets.QHBoxLayout()
            capture_out_row.setContentsMargins(0, 0, 0, 0)
            capture_out_row.addWidget(self._capture_face_out_dir, 1)
            capture_out_row.addWidget(capture_out_browse_button)
            capture_out_widget = QtWidgets.QWidget()
            capture_out_widget.setLayout(capture_out_row)

            self._capture_face_manifest_path = QtWidgets.QLineEdit()
            self._capture_face_manifest_path.setPlaceholderText(
                "Default: <project_root>/shots/manifest.csv"
            )
            capture_manifest_browse_button = QtWidgets.QPushButton("Browse")
            capture_manifest_browse_button.clicked.connect(
                self._browse_capture_face_manifest
            )
            capture_manifest_row = QtWidgets.QHBoxLayout()
            capture_manifest_row.setContentsMargins(0, 0, 0, 0)
            capture_manifest_row.addWidget(self._capture_face_manifest_path, 1)
            capture_manifest_row.addWidget(capture_manifest_browse_button)
            capture_manifest_widget = QtWidgets.QWidget()
            capture_manifest_widget.setLayout(capture_manifest_row)

            self._capture_face_raw_dir = QtWidgets.QLineEdit()
            self._capture_face_raw_dir.setPlaceholderText(
                "Default split: <project_root>/shots/images"
            )
            self._capture_face_ann_dir = QtWidgets.QLineEdit()
            self._capture_face_ann_dir.setPlaceholderText(
                "Default split: <project_root>/shots/ann"
            )
            self._capture_face_meta_dir = QtWidgets.QLineEdit()
            self._capture_face_meta_dir.setPlaceholderText(
                "Default split: <project_root>/shots/meta"
            )
            self._capture_face_family = QtWidgets.QLineEdit(DEFAULT_CAPTURE_FAMILY)
            self._capture_face_family.setPlaceholderText("tag36h11")

            capture_grid = QtWidgets.QGridLayout()
            capture_grid.setContentsMargins(0, 0, 0, 0)
            capture_grid.setHorizontalSpacing(10)
            capture_grid.setVerticalSpacing(8)
            capture_grid.addWidget(
                _make_field("Object / face", self._capture_face_object),
                0,
                0,
            )
            capture_grid.addWidget(
                _make_field("Source", self._capture_face_source),
                0,
                1,
            )
            self._capture_face_camera_field = _make_field(
                "Camera index",
                self._capture_face_camera_index,
            )
            self._capture_face_video_field = _make_field(
                "Video path",
                capture_video_widget,
            )
            capture_grid.addWidget(self._capture_face_camera_field, 1, 0)
            capture_grid.addWidget(self._capture_face_video_field, 1, 1)
            capture_grid.addWidget(
                _make_field("Calibration YAML", capture_calib_widget),
                2,
                0,
                1,
                2,
            )
            capture_grid.addWidget(
                _make_field("Tag registry", capture_registry_widget),
                3,
                0,
                1,
                2,
            )
            capture_grid.setColumnStretch(0, 1)
            capture_grid.setColumnStretch(1, 1)

            capture_advanced_widget = QtWidgets.QWidget()
            capture_advanced_grid = QtWidgets.QGridLayout(capture_advanced_widget)
            capture_advanced_grid.setContentsMargins(0, 0, 0, 0)
            capture_advanced_grid.setHorizontalSpacing(10)
            capture_advanced_grid.setVerticalSpacing(8)
            capture_advanced_grid.addWidget(
                _make_field("AprilTag family", self._capture_face_family),
                0,
                0,
                1,
                2,
            )
            capture_advanced_grid.addWidget(
                _make_field("Width", self._capture_face_width),
                1,
                0,
            )
            capture_advanced_grid.addWidget(
                _make_field("Height", self._capture_face_height),
                1,
                1,
            )
            capture_advanced_grid.addWidget(
                _make_field("FPS", self._capture_face_fps),
                2,
                0,
            )
            capture_advanced_grid.addWidget(
                _make_field("Minimum expected tags", self._capture_face_min_expected),
                2,
                1,
            )
            capture_advanced_grid.addWidget(
                _make_field("Layout", self._capture_face_layout),
                3,
                0,
            )
            capture_advanced_grid.addWidget(
                _make_field("Output folder", capture_out_widget),
                3,
                1,
            )
            capture_advanced_grid.addWidget(
                _make_field("Manifest CSV", capture_manifest_widget),
                4,
                0,
                1,
                2,
            )
            capture_advanced_grid.addWidget(
                _make_field("Raw images folder", self._capture_face_raw_dir),
                5,
                0,
            )
            capture_advanced_grid.addWidget(
                _make_field("Annotated images folder", self._capture_face_ann_dir),
                5,
                1,
            )
            capture_advanced_grid.addWidget(
                _make_field("Metadata folder", self._capture_face_meta_dir),
                6,
                0,
                1,
                2,
            )
            capture_advanced_grid.addWidget(
                _make_field("Info panel width", self._capture_face_panel_width),
                7,
                0,
            )
            capture_advanced_grid.addWidget(
                _make_field("Recent panel width", self._capture_face_recent_width),
                7,
                1,
            )
            capture_advanced_grid.setColumnStretch(0, 1)
            capture_advanced_grid.setColumnStretch(1, 1)
            capture_advanced_group = _make_collapsible_group(
                "Advanced Capture And Outputs",
                capture_advanced_widget,
                checked=False,
            )

            for field in (
                self._capture_face_camera_index,
                self._capture_face_width,
                self._capture_face_height,
                self._capture_face_fps,
                self._capture_face_min_expected,
                self._capture_face_panel_width,
                self._capture_face_recent_width,
            ):
                field.valueChanged.connect(self._update_capture_face_flow)
            self._capture_face_layout.currentTextChanged.connect(
                self._update_capture_face_flow
            )
            for field in (
                self._capture_face_family,
                self._capture_face_video_path,
                self._capture_face_calibration_path,
                self._capture_face_registry_path,
                self._capture_face_out_dir,
                self._capture_face_manifest_path,
                self._capture_face_raw_dir,
                self._capture_face_ann_dir,
                self._capture_face_meta_dir,
            ):
                field.textChanged.connect(self._update_capture_face_flow)

            self._capture_face_guidance = QtWidgets.QLabel(CAPTURE_FACE_GUIDANCE)
            self._capture_face_guidance.setObjectName("GuidanceText")
            self._capture_face_guidance.setWordWrap(True)
            self._capture_face_readiness = QtWidgets.QLabel()
            self._capture_face_readiness.setObjectName("OutputText")
            self._capture_face_readiness.setWordWrap(True)
            self._capture_face_readiness.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._capture_face_outputs = QtWidgets.QLabel()
            self._capture_face_outputs.setObjectName("OutputText")
            self._capture_face_outputs.setWordWrap(True)
            self._capture_face_outputs.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._capture_face_gallery = QtWidgets.QListWidget()
            self._capture_face_gallery.setObjectName("FaceShotGallery")
            self._capture_face_gallery.setViewMode(
                QtWidgets.QListView.ViewMode.IconMode
            )
            self._capture_face_gallery.setResizeMode(
                QtWidgets.QListView.ResizeMode.Adjust
            )
            self._capture_face_gallery.setMovement(
                QtWidgets.QListView.Movement.Static
            )
            self._capture_face_gallery.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self._capture_face_gallery.setIconSize(QtCore.QSize(128, 88))
            self._capture_face_gallery.setGridSize(QtCore.QSize(176, 128))
            self._capture_face_gallery.setMinimumHeight(156)
            self._capture_face_gallery.setMaximumHeight(240)
            self._capture_face_gallery.setSpacing(8)
            self._capture_face_gallery.itemClicked.connect(
                self._select_capture_face_gallery_item
            )
            self._capture_face_gallery.itemSelectionChanged.connect(
                self._select_capture_face_gallery_item
            )
            self._capture_face_gallery_preview = _ImagePreviewCanvas(
                "Select a saved face shot.",
                object_name="FaceShotPreview",
            )
            self._capture_face_gallery_preview.setMinimumSize(260, 170)
            self._capture_face_gallery_preview.setMaximumHeight(240)
            self._capture_face_gallery_details = QtWidgets.QLabel(
                "Saved face-shot previews will appear after capture."
            )
            self._capture_face_gallery_details.setObjectName("OutputText")
            self._capture_face_gallery_details.setWordWrap(True)
            self._capture_face_gallery_details.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._capture_face_stack_gallery = QtWidgets.QListWidget()
            self._capture_face_stack_gallery.setObjectName("FaceShotGallery")
            self._capture_face_stack_gallery.setViewMode(
                QtWidgets.QListView.ViewMode.IconMode
            )
            self._capture_face_stack_gallery.setResizeMode(
                QtWidgets.QListView.ResizeMode.Adjust
            )
            self._capture_face_stack_gallery.setMovement(
                QtWidgets.QListView.Movement.Static
            )
            self._capture_face_stack_gallery.setIconSize(QtCore.QSize(104, 72))
            self._capture_face_stack_gallery.setGridSize(QtCore.QSize(132, 104))
            self._capture_face_stack_gallery.setMaximumHeight(118)
            self._capture_face_stack_gallery.itemClicked.connect(
                self._select_capture_face_stack_item
            )
            self._capture_face_gallery_side = QtWidgets.QWidget()
            capture_gallery_side = QtWidgets.QVBoxLayout(
                self._capture_face_gallery_side
            )
            capture_gallery_side.setContentsMargins(0, 0, 0, 0)
            capture_gallery_side.setSpacing(8)
            capture_gallery_side.addWidget(self._capture_face_stack_gallery)
            capture_gallery_side.addWidget(self._capture_face_gallery_preview, 1)
            capture_gallery_side.addWidget(self._capture_face_gallery_details)
            self._capture_face_gallery_side.setVisible(False)
            capture_gallery_layout = QtWidgets.QHBoxLayout()
            capture_gallery_layout.setContentsMargins(0, 0, 0, 0)
            capture_gallery_layout.setSpacing(10)
            capture_gallery_layout.addWidget(self._capture_face_gallery, 2)
            capture_gallery_layout.addWidget(self._capture_face_gallery_side, 1)
            capture_gallery_widget = QtWidgets.QWidget()
            capture_gallery_widget.setLayout(capture_gallery_layout)
            self._capture_face_process_state_label = QtWidgets.QLabel()
            self._capture_face_process_state_label.setObjectName("OutputText")
            self._capture_face_process_state_label.setWordWrap(True)
            self._capture_face_process_state_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._capture_face_log = QtWidgets.QPlainTextEdit()
            self._capture_face_log.setObjectName("CalibrationLog")
            self._capture_face_log.setReadOnly(True)
            self._capture_face_log.setMaximumHeight(118)
            self._capture_face_log.setPlaceholderText(
                "Face-shot capture stdout/stderr will appear here after launch."
            )
            self._capture_face_log.document().setMaximumBlockCount(250)

            capture_summary_grid = QtWidgets.QGridLayout()
            capture_summary_grid.setContentsMargins(0, 0, 0, 0)
            capture_summary_grid.setHorizontalSpacing(10)
            capture_summary_grid.setVerticalSpacing(8)
            capture_summary_grid.addWidget(
                _make_field("Readiness", self._capture_face_readiness),
                0,
                0,
            )
            capture_summary_grid.addWidget(
                _make_field("Coverage and outputs", self._capture_face_outputs),
                0,
                1,
            )
            capture_summary_grid.addWidget(
                _make_field("Saved face-shot gallery", capture_gallery_widget),
                1,
                0,
                1,
                2,
            )
            capture_summary_grid.addWidget(
                _make_field("Process state", self._capture_face_process_state_label),
                2,
                0,
                1,
                2,
            )
            capture_summary_grid.addWidget(
                _make_field("Process log", self._capture_face_log),
                3,
                0,
                1,
                2,
            )
            capture_summary_grid.setColumnStretch(0, 1)
            capture_summary_grid.setColumnStretch(1, 1)

            capture_refresh_button = QtWidgets.QPushButton("Refresh Status")
            capture_refresh_button.setObjectName("SecondaryActionButton")
            capture_refresh_button.clicked.connect(self._refresh)
            self._capture_face_run_button = QtWidgets.QPushButton("Start Batch")
            self._capture_face_run_button.setObjectName("PrimaryActionButton")
            self._capture_face_run_button.clicked.connect(
                lambda: self._open_native_capture_face("batch")
            )
            self._capture_face_selected_button = QtWidgets.QPushButton(
                "Capture Selected"
            )
            self._capture_face_selected_button.setObjectName("SecondaryActionButton")
            self._capture_face_selected_button.clicked.connect(
                lambda: self._open_native_capture_face("selected")
            )
            self._capture_face_current_button = QtWidgets.QPushButton(
                "Capture Current"
            )
            self._capture_face_current_button.setObjectName("SecondaryActionButton")
            self._capture_face_current_button.clicked.connect(
                lambda: self._open_native_capture_face("current")
            )
            capture_action_row = QtWidgets.QHBoxLayout()
            capture_action_row.addWidget(self._capture_face_run_button)
            capture_action_row.addWidget(self._capture_face_selected_button)
            capture_action_row.addWidget(self._capture_face_current_button)
            capture_action_row.addWidget(capture_refresh_button)
            capture_action_row.addStretch(1)

            capture_face_layout.addWidget(capture_face_title)
            capture_face_layout.addWidget(capture_face_note)
            capture_face_layout.addWidget(
                _make_field("Registered face queue", self._capture_face_queue)
            )
            capture_face_layout.addLayout(capture_grid)
            capture_face_layout.addWidget(capture_advanced_group)
            capture_face_layout.addWidget(self._capture_face_guidance)
            capture_face_layout.addLayout(capture_summary_grid)
            capture_face_layout.addLayout(capture_action_row)

            detail_content = QtWidgets.QWidget()
            detail_content_layout = QtWidgets.QVBoxLayout(detail_content)
            detail_content_layout.setContentsMargins(0, 0, 0, 0)
            detail_content_layout.setSpacing(10)
            detail_content_layout.addLayout(title_row)
            detail_content_layout.addWidget(self._message_label)
            detail_content_layout.addLayout(cards_grid)
            detail_content_layout.addWidget(self._charuco_card)
            detail_content_layout.addWidget(self._calibration_card)
            detail_content_layout.addWidget(self._object_tags_card)
            detail_content_layout.addWidget(self._board_building_card)
            detail_content_layout.addWidget(self._capture_face_card)
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
            self._health_counts_label.setOpenExternalLinks(False)
            self._health_counts_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            self._health_counts_label.linkActivated.connect(
                self._activate_health_count
            )

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
                "camera-calibration, object-tag, board-building, and "
                "face-shot capture workflows."
            )

            self._render_board_batch_rows()
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

        def _browse_object_tags_output_dir(self) -> None:
            current = self._object_tags_out_dir.text().strip()
            start_dir = current or str(self._current_project_root())
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select object AprilTag output folder",
                start_dir,
            )
            if selected:
                self._object_tags_out_dir.setText(selected)

        def _browse_board_calibration(self) -> None:
            current = self._board_calibration_path.text().strip()
            start_path = current or str(
                default_calibration_path(self._current_project_root())
            )
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select board-building calibration YAML",
                start_path,
                "YAML files (*.yaml *.yml);;All files (*)",
            )
            if selected:
                self._board_calibration_path.setText(selected)

        def _browse_board_video(self) -> None:
            current = self._board_video_path.text().strip()
            start_path = current or str(self._current_project_root())
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select board-building video",
                start_path,
                "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
            )
            if selected:
                self._board_video_path.setText(selected)

        def _browse_board_output_dir(self) -> None:
            current = self._board_out_dir.text().strip()
            start_dir = current or str(default_boards_dir(self._current_project_root()))
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select board YAML output folder",
                start_dir,
            )
            if selected:
                self._board_out_dir.setText(selected)

        def _browse_board_registry(self) -> None:
            current = self._board_registry_path.text().strip()
            start_path = current or str(
                default_boards_dir(self._current_project_root())
                / "tag_registry.yaml"
            )
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select tag registry YAML",
                start_path,
                "YAML files (*.yaml *.yml);;All files (*)",
            )
            if selected:
                self._board_registry_path.setText(selected)

        def _browse_board_shots_dir(self) -> None:
            current = self._board_shots_dir.text().strip()
            start_dir = current or str(
                default_boards_dir(self._current_project_root()) / "shots"
            )
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select audit-shot output folder",
                start_dir,
            )
            if selected:
                self._board_shots_dir.setText(selected)

        def _browse_capture_face_calibration(self) -> None:
            current = self._capture_face_calibration_path.text().strip()
            start_path = current or str(
                default_capture_calibration_path(self._current_project_root())
            )
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select face-shot calibration YAML",
                start_path,
                "YAML files (*.yaml *.yml);;All files (*)",
            )
            if selected:
                self._capture_face_calibration_path.setText(selected)

        def _browse_capture_face_registry(self) -> None:
            current = self._capture_face_registry_path.text().strip()
            start_path = current or str(
                default_capture_registry_path(self._current_project_root())
            )
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select face-shot tag registry",
                start_path,
                "YAML files (*.yaml *.yml);;All files (*)",
            )
            if selected:
                self._capture_face_registry_path.setText(selected)

        def _browse_capture_face_video(self) -> None:
            current = self._capture_face_video_path.text().strip()
            start_path = current or str(self._current_project_root())
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select face-shot video",
                start_path,
                "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
            )
            if selected:
                self._capture_face_video_path.setText(selected)

        def _browse_capture_face_output_dir(self) -> None:
            current = self._capture_face_out_dir.text().strip()
            start_dir = current or str(default_capture_shots_dir(self._current_project_root()))
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select face-shot output folder",
                start_dir,
            )
            if selected:
                self._capture_face_out_dir.setText(selected)

        def _browse_capture_face_manifest(self) -> None:
            current = self._capture_face_manifest_path.text().strip()
            start_path = current or str(
                default_capture_manifest_path(self._current_project_root())
            )
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select face-shot manifest CSV",
                start_path,
                "CSV files (*.csv);;All files (*)",
            )
            if selected:
                self._capture_face_manifest_path.setText(selected)

        def _board_identity_changed(self) -> None:
            self._sync_board_definition_name()
            self._update_board_batch_preview()
            self._update_board_building_flow()

        def _board_tag_size_changed(self) -> None:
            if not getattr(self, "_board_tag_size_autofilling", False):
                self._board_tag_size_user_edited = True
            self._update_board_building_flow()

        def _board_batch_fields_changed(self) -> None:
            self._update_board_batch_preview()
            self._save_board_batch_draft_if_needed(show_errors=False)

        def _sync_board_definition_name(self) -> None:
            try:
                object_name = compose_board_object_name(
                    self._board_object_label.text(),
                    self._board_side_label.currentText(),
                )
            except Exception:
                object_name = ""
            self._board_object_name.setText(object_name)

        def _set_next_board_side_label(self) -> None:
            try:
                side_label = next_default_side_label(
                    self._current_project_root(),
                    self._board_object_label.text(),
                    out_dir=Path(self._board_out_dir.text()).expanduser()
                    if self._board_out_dir.text().strip()
                    else None,
                )
            except Exception as exc:
                message = f"Could not choose next side label: {exc}"
                self.statusBar().showMessage(message, 5000)
                self._update_board_building_flow()
                return
            self._board_side_label.setCurrentText(side_label)
            self._board_identity_changed()

        def _sync_board_tag_size_from_object_tags(self) -> None:
            if not hasattr(self, "_board_tag_size"):
                return
            configured = float(self._object_tags_tag_size.value())
            inferred = infer_latest_object_tag_size_mm(self._current_project_root())
            if abs(configured - DEFAULT_OBJECT_TAG_SIZE_MM) > 1e-9:
                candidate = configured
            elif inferred is not None:
                candidate = inferred
            else:
                candidate = configured

            current = float(self._board_tag_size.value())
            should_update = not getattr(self, "_board_tag_size_user_edited", False)
            if not should_update or abs(current - candidate) <= 1e-9:
                return
            self._board_tag_size_autofilling = True
            self._board_tag_size.blockSignals(True)
            self._board_tag_size.setValue(candidate)
            self._board_tag_size.blockSignals(False)
            self._board_tag_size_autofilling = False
            self._board_tag_size_autofill_mm = candidate

        def _board_batch_rows_for_persistence(self) -> tuple[BoardBatchRow, ...]:
            rows_text = self._board_batch_rows.toPlainText().strip()
            if rows_text:
                return parse_board_batch_rows(rows_text)
            if self._board_batch_rows_model:
                return tuple(self._board_batch_rows_model)
            return ()

        def _select_board_batch_row(self, row_index: int) -> None:
            if row_index < 0 or row_index >= len(self._board_batch_rows_model):
                return
            row = self._board_batch_rows_model[row_index]
            try:
                first_item = build_board_batch_items((row,))[0]
            except Exception:
                first_item = None

            fields = (
                self._board_object_label,
                self._board_batch_instances,
                self._board_batch_sides,
                self._board_side_label,
                self._board_tag_size,
            )
            for field in fields:
                field.blockSignals(True)
            try:
                self._board_object_label.setText(row.object_label)
                self._board_batch_instances.setText(row.instances)
                self._board_batch_sides.setText(row.sides)
                if first_item is not None:
                    self._board_side_label.setCurrentText(first_item.side_label)
                if row.tag_size_mm is not None:
                    self._board_tag_size.setValue(float(row.tag_size_mm))
                    self._board_tag_size_user_edited = True
            finally:
                for field in fields:
                    field.blockSignals(False)

            self._sync_board_definition_name()
            self._update_board_batch_preview()
            self._update_board_building_flow()

        def _save_board_batch_draft_if_needed(
            self,
            *,
            show_errors: bool = True,
        ) -> None:
            if getattr(self, "_board_batch_draft_loading", False):
                return
            try:
                rows = self._board_batch_rows_for_persistence()
            except Exception as exc:
                if show_errors:
                    message = f"Could not save board queue draft: {exc}"
                    self.statusBar().showMessage(message, 5000)
                return
            if not rows:
                return

            try:
                path = save_board_batch_draft(
                    self._current_project_root(),
                    rows,
                    self._board_building_config(),
                )
            except Exception as exc:
                if show_errors:
                    message = f"Could not save board queue draft: {exc}"
                    self.statusBar().showMessage(message, 5000)
                return
            self._loaded_board_batch_draft_path = path
            self._loaded_board_batch_draft_mtime_ns = _path_mtime_ns(path)

        def _delete_board_batch_draft(self) -> None:
            if getattr(self, "_board_batch_draft_loading", False):
                return
            path = default_board_batch_draft_path(self._current_project_root())
            try:
                delete_board_batch_draft(self._current_project_root())
            except Exception as exc:
                message = f"Could not remove board queue draft: {exc}"
                self.statusBar().showMessage(message, 5000)
                return
            self._loaded_board_batch_draft_path = path
            self._loaded_board_batch_draft_mtime_ns = None

        def _load_board_batch_draft(self) -> None:
            if not hasattr(self, "_board_batch_row_list"):
                return
            root = self._current_project_root()
            path = default_board_batch_draft_path(root)
            mtime = _path_mtime_ns(path)
            if (
                path == self._loaded_board_batch_draft_path
                and mtime == self._loaded_board_batch_draft_mtime_ns
            ):
                return
            if mtime is None:
                if self._loaded_board_batch_draft_path != path:
                    self._board_batch_rows_model = []
                    self._board_batch_rows.blockSignals(True)
                    self._board_batch_rows.clear()
                    self._board_batch_rows.blockSignals(False)
                    self._render_board_batch_rows()
                    self._update_board_batch_preview()
                self._loaded_board_batch_draft_path = path
                self._loaded_board_batch_draft_mtime_ns = None
                return

            try:
                draft = load_board_batch_draft(root)
            except Exception as exc:
                message = f"Could not load board queue draft: {exc}"
                self._board_batch_preview.setText(message)
                self.statusBar().showMessage(message, 6000)
                self._loaded_board_batch_draft_path = path
                self._loaded_board_batch_draft_mtime_ns = mtime
                return
            if draft is None:
                self._loaded_board_batch_draft_path = path
                self._loaded_board_batch_draft_mtime_ns = None
                return

            self._apply_board_batch_draft(draft)
            self._loaded_board_batch_draft_path = draft.path
            self._loaded_board_batch_draft_mtime_ns = _path_mtime_ns(draft.path)

        def _apply_board_batch_draft(self, draft: BoardBatchDraft) -> None:
            self._board_batch_draft_loading = True
            widgets = (
                self._board_family,
                self._board_tag_size,
                self._board_source,
                self._board_camera_index,
                self._board_video_path,
                self._board_width,
                self._board_height,
                self._board_fps,
                self._board_calibration_path,
                self._board_out_dir,
                self._board_registry_path,
                self._board_save_shot,
                self._board_shots_dir,
                self._board_z_threshold,
                self._board_allow_nonplanar,
                self._board_batch_rows,
                self._board_batch_row_list,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                self._board_batch_rows_model = list(draft.rows)
                self._board_batch_rows.clear()
                self._board_family.setText(draft.family)
                self._board_tag_size.setValue(float(draft.tag_size_mm))
                source_index = self._board_source.findData(draft.source)
                if source_index >= 0:
                    self._board_source.setCurrentIndex(source_index)
                self._board_camera_index.setValue(int(draft.camera_index))
                self._board_video_path.setText(draft.video_path)
                self._board_width.setValue(int(draft.width))
                self._board_height.setValue(int(draft.height))
                self._board_fps.setValue(int(draft.fps))
                self._board_calibration_path.setText(draft.calibration_path)
                self._board_out_dir.setText(draft.out_dir)
                self._board_registry_path.setText(draft.registry_path)
                self._board_save_shot.setChecked(bool(draft.save_shot))
                self._board_shots_dir.setText(draft.shots_dir)
                self._board_z_threshold.setValue(float(draft.z_threshold_m))
                self._board_allow_nonplanar.setChecked(bool(draft.allow_nonplanar))
                self._board_tag_size_user_edited = True
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
                self._board_batch_draft_loading = False

            self._render_board_batch_rows()
            if self._board_batch_rows_model:
                self._board_batch_row_list.setCurrentRow(0)
                if self._board_batch_row_list.currentRow() != 0:
                    self._select_board_batch_row(0)
            self._render_board_source_fields()
            self._render_board_save_shot_fields()
            self._update_board_batch_preview()
            self._update_board_building_flow()

        def _board_batch_items(self) -> tuple[BoardBatchItem, ...]:
            rows_text = self._board_batch_rows.toPlainText().strip()
            if rows_text:
                rows = parse_board_batch_rows(rows_text)
            elif self._board_batch_rows_model:
                rows = tuple(self._board_batch_rows_model)
            else:
                rows = (
                    BoardBatchRow(
                        object_label=self._board_object_label.text(),
                        instances=self._board_batch_instances.text(),
                        sides=self._board_batch_sides.text(),
                    ),
                )
            return build_board_batch_items(rows)

        def _add_board_batch_row(self) -> None:
            row = BoardBatchRow(
                object_label=self._board_object_label.text(),
                instances=self._board_batch_instances.text(),
                sides=self._board_batch_sides.text(),
                tag_size_mm=float(self._board_tag_size.value()),
            )
            candidate_rows = (*self._board_batch_rows_model, row)
            try:
                build_board_batch_items(candidate_rows)
            except Exception as exc:
                message = f"Could not add object row: {exc}"
                self._board_batch_preview.setText(message)
                self.statusBar().showMessage(message, 5000)
                return
            self._board_batch_rows_model.append(row)
            self._render_board_batch_rows()
            self._board_batch_row_list.setCurrentRow(
                len(self._board_batch_rows_model) - 1
            )
            self._update_board_batch_preview()
            self._save_board_batch_draft_if_needed()
            self._update_board_building_flow()
            self.statusBar().showMessage("Added object row to board batch.", 3000)

        def _remove_selected_board_batch_row(self) -> None:
            row_index = self._board_batch_row_list.currentRow()
            if row_index < 0 or row_index >= len(self._board_batch_rows_model):
                self.statusBar().showMessage("Select an object row to remove.", 3000)
                return
            del self._board_batch_rows_model[row_index]
            self._render_board_batch_rows()
            if self._board_batch_rows_model:
                self._board_batch_row_list.setCurrentRow(
                    min(row_index, len(self._board_batch_rows_model) - 1)
                )
                self._save_board_batch_draft_if_needed()
            else:
                self._delete_board_batch_draft()
            self._update_board_batch_preview()
            self._update_board_building_flow()
            self.statusBar().showMessage("Removed object row from board batch.", 3000)

        def _clear_board_batch_rows(self) -> None:
            if not self._board_batch_rows_model:
                self.statusBar().showMessage("No object rows to clear.", 3000)
                return
            self._board_batch_rows_model.clear()
            self._render_board_batch_rows()
            self._delete_board_batch_draft()
            self._update_board_batch_preview()
            self._update_board_building_flow()
            self.statusBar().showMessage("Cleared object rows from board batch.", 3000)

        def _render_board_batch_rows(self) -> None:
            if not hasattr(self, "_board_batch_row_list"):
                return
            self._board_batch_row_list.blockSignals(True)
            self._board_batch_row_list.clear()
            if not self._board_batch_rows_model:
                item = QtWidgets.QListWidgetItem(
                    "No saved object rows; current fields define the queue."
                )
                item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                self._board_batch_row_list.addItem(item)
                self._board_batch_row_list.blockSignals(False)
                return
            for row in self._board_batch_rows_model:
                instances = row.instances.strip() or "single"
                sides = row.sides.strip() or DEFAULT_BOARD_SIDE_LABEL
                tag_size = (
                    f"{row.tag_size_mm:g}"
                    if row.tag_size_mm is not None
                    else "default"
                )
                self._board_batch_row_list.addItem(
                    f"{row.object_label.strip()} | {instances} | {sides} | {tag_size} mm"
                )
            self._board_batch_row_list.blockSignals(False)

        def _update_board_batch_preview(self) -> None:
            if not hasattr(self, "_board_batch_preview"):
                return
            try:
                items = self._board_batch_items()
            except Exception as exc:
                self._board_batch_preview.setText(f"Batch setup error: {exc}")
                return
            self._board_batch_preview.setText(_format_board_batch_preview(items))

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

        def _object_tag_mode_changed(self) -> None:
            self._render_object_tag_mode_fields()
            self._update_object_tag_flow()

        def _object_tag_paper_changed(self) -> None:
            self._render_object_tag_paper_fields()
            self._update_object_tag_flow()

        def _board_source_changed(self) -> None:
            self._render_board_source_fields()
            self._update_board_building_flow()

        def _capture_face_source_changed(self) -> None:
            self._render_capture_face_source_fields()
            self._update_capture_face_flow()

        def _board_save_shot_changed(self) -> None:
            self._render_board_save_shot_fields()
            self._update_board_building_flow()

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

        def _generate_object_tags(self) -> None:
            self._object_tags_generate_button.setEnabled(False)
            try:
                result = generate_object_tags(self._object_tag_config())
            except ObjectTagGenerationError as exc:
                message = f"Object AprilTag generation failed: {exc}"
                self._object_tags_outputs.setText(message)
                self._last_object_tag_output_dir = None
                self._last_object_tag_sheet_file = None
                self._object_tags_open_folder_button.setEnabled(False)
                self._object_tags_open_file_button.setEnabled(False)
                self._object_tags_preview.clear_preview("Preview unavailable.")
                self.statusBar().showMessage(message, 6000)
                self._update_object_tag_flow()
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                message = f"Object AprilTag generation failed: {exc}"
                self._object_tags_outputs.setText(message)
                self._last_object_tag_output_dir = None
                self._last_object_tag_sheet_file = None
                self._object_tags_open_folder_button.setEnabled(False)
                self._object_tags_open_file_button.setEnabled(False)
                self._object_tags_preview.clear_preview("Preview unavailable.")
                self.statusBar().showMessage(message, 6000)
                self._update_object_tag_flow()
            else:
                first_sheet = result.pdf_paths[0] if result.pdf_paths else None
                if first_sheet is None and result.png_paths:
                    first_sheet = result.png_paths[0]
                self._last_object_tag_output_dir = result.out_dir
                self._last_object_tag_sheet_file = first_sheet
                self._object_tags_outputs.setText(_format_object_tag_outputs(result))
                self._object_tags_open_folder_button.setEnabled(
                    result.out_dir.is_dir()
                )
                self._object_tags_open_file_button.setEnabled(
                    bool(first_sheet and first_sheet.is_file())
                )
                if result.png_paths:
                    self._object_tags_preview.show_preview(result.png_paths[0])
                else:
                    self._object_tags_preview.clear_preview("Preview unavailable.")
                self.statusBar().showMessage(
                    "Generated object AprilTag sheet outputs.",
                    5000,
                )
                self._refresh()
            finally:
                self._update_object_tag_flow()

        def _open_object_tags_sheet_file(self) -> None:
            target = self._last_object_tag_sheet_file
            if target is None or not target.is_file():
                message = "Generated object AprilTag sheet is not available to open."
                self.statusBar().showMessage(message, 4000)
                return

            opened = QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(target.resolve()))
            )
            message = (
                "Opened generated object AprilTag sheet."
                if opened
                else "Could not open the generated object AprilTag sheet."
            )
            self.statusBar().showMessage(message, 4000)

        def _open_object_tags_output_folder(self) -> None:
            target = (
                self._last_object_tag_output_dir
                or self._object_tag_output_target()
            )
            if not target.is_dir():
                message = "Object AprilTag output folder is not available to open."
                self.statusBar().showMessage(message, 4000)
                return

            opened = QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(target.resolve()))
            )
            message = (
                "Opened object AprilTag output folder."
                if opened
                else "Could not open the object AprilTag output folder."
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
            self._load_board_batch_draft()
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
            self._object_tags_card.setVisible(model.stage_id == 3)
            self._board_building_card.setVisible(model.stage_id == 4)
            self._capture_face_card.setVisible(model.stage_id == 5)
            if model.stage_id == 2:
                self._sync_calibration_from_project_metadata()
                self._update_calibration_flow()
            if model.stage_id == 3:
                self._update_object_tag_flow()
            if model.stage_id == 4:
                self._sync_board_tag_size_from_object_tags()
                self._update_board_building_flow()
            if model.stage_id == 5:
                self._update_capture_face_flow()
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

        def _activate_health_count(self, href: str) -> None:
            prefix = "stage-status:"
            if not href.startswith(prefix):
                return
            status = href[len(prefix) :]
            matches = [model for model in self._models if model.status == status]
            if not matches:
                return
            stage_ids = [model.stage_id for model in matches]
            try:
                current_index = stage_ids.index(self._selected_stage_id)
            except ValueError:
                target = matches[0]
            else:
                target = matches[(current_index + 1) % len(matches)]
            self._select_stage(target.stage_id)
            self.statusBar().showMessage(
                f"Showing {target.status_label}: Stage {target.stage_id} {target.name}",
                3500,
            )

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
            self._object_tags_card.setVisible(False)
            self._board_building_card.setVisible(False)
            self._capture_face_card.setVisible(False)
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

        def _update_object_tag_flow(self) -> None:
            if not hasattr(self, "_object_tags_readiness"):
                return
            self._render_object_tag_mode_fields()
            self._render_object_tag_paper_fields()
            readiness = inspect_object_tag_generation(self._object_tag_config())
            self._object_tags_readiness.setText(
                _format_object_tag_readiness(readiness)
            )
            self._object_tags_expected_output.setText(
                _format_object_tag_expected_output(readiness)
            )
            self._object_tags_generate_button.setEnabled(readiness.ready)

            model = self._model_by_stage_id(self._selected_stage_id)
            if model is None or model.stage_id != 3:
                return

            self._command_preview.setText(readiness.command_preview)
            self._command_preview.setCursorPosition(0)
            self._copy_button.setEnabled(bool(readiness.command_preview))
            self._command_note.setText(
                _object_tag_command_card_note(readiness, model)
            )
            self._copy_feedback.setText(
                _object_tag_command_ready_message(
                    readiness,
                    stage_complete=model.status == "complete",
                )
            )

        def _update_board_building_flow(self) -> None:
            if not hasattr(self, "_board_readiness"):
                return
            self._render_board_source_fields()
            self._render_board_save_shot_fields()
            self._update_board_batch_preview()
            self._save_board_batch_draft_if_needed(show_errors=False)
            readiness = inspect_board_building(self._board_building_config())
            self._board_readiness.setText(
                _format_board_building_readiness(readiness)
            )
            self._board_outputs.setText(
                _format_board_building_outputs(readiness)
            )
            if self._board_process_state.state == BOARD_PROCESS_NOT_STARTED:
                self._set_board_process_state(
                    board_building_process_not_started(
                        readiness.expected_board_yaml,
                        readiness.expected_registry,
                    )
                )
            process_running = self._board_process_is_running()
            try:
                batch_ready = bool(self._board_batch_items())
            except Exception:
                batch_ready = False
            self._board_batch_capture_button.setEnabled(
                readiness.ready and batch_ready and not process_running
            )
            self._board_guided_capture_button.setEnabled(
                readiness.ready and not process_running
            )
            self._board_run_button.setEnabled(readiness.ready and not process_running)
            self._board_run_button.setText(
                _board_building_run_button_label(self._board_process_state)
            )
            self._board_prompt_input.setEnabled(process_running)
            self._board_send_prompt_button.setEnabled(process_running)

            model = self._model_by_stage_id(self._selected_stage_id)
            if model is None or model.stage_id != 4:
                return

            self._command_preview.setText(readiness.command_preview)
            self._command_preview.setCursorPosition(0)
            self._copy_button.setEnabled(bool(readiness.command_preview))
            self._command_note.setText(
                _board_building_command_card_note(readiness, model)
            )
            self._copy_feedback.setText(
                _board_building_command_ready_message(
                    readiness,
                    stage_complete=model.status == "complete",
                )
            )

        def _update_capture_face_flow(self) -> None:
            if not hasattr(self, "_capture_face_readiness"):
                return
            self._render_capture_face_source_fields()
            readiness = inspect_capture_face_readiness(
                self._capture_face_config()
            )
            if self._sync_capture_face_choices(readiness):
                readiness = inspect_capture_face_readiness(
                    self._capture_face_config()
                )
            self._update_capture_face_queue(readiness)
            self._capture_face_readiness.setText(
                _format_capture_face_readiness(readiness)
            )
            self._capture_face_outputs.setText(
                _format_capture_face_outputs(readiness)
            )
            self._update_capture_face_gallery(readiness)
            if (
                self._capture_face_process_state.state
                == CAPTURE_FACE_PROCESS_NOT_STARTED
            ):
                self._set_capture_face_process_state(
                    capture_face_process_not_started(
                        readiness.expected_manifest,
                    )
                )
            process_running = self._capture_face_process_is_running()
            self._update_capture_face_action_buttons(
                readiness.ready,
                process_running,
            )

            model = self._model_by_stage_id(self._selected_stage_id)
            if model is None or model.stage_id != 5:
                return

            self._command_preview.setText(readiness.command_preview)
            self._command_preview.setCursorPosition(0)
            self._copy_button.setEnabled(bool(readiness.command_preview))
            self._command_note.setText(
                _capture_face_command_card_note(readiness, model)
            )
            self._copy_feedback.setText(
                _capture_face_command_ready_message(
                    readiness,
                    stage_complete=model.status == "complete",
                )
            )

        def _sync_capture_face_choices(
            self,
            readiness: CaptureFaceReadiness,
        ) -> bool:
            choices = tuple(
                dict.fromkeys(
                    (
                        CAPTURE_FACE_QUEUE_LABEL,
                        *readiness.registered_bases,
                        *readiness.registered_faces,
                    )
                )
            )
            current = self._capture_face_object.currentText().strip()
            updated = False
            self._capture_face_object.blockSignals(True)
            try:
                self._capture_face_object.clear()
                self._capture_face_object.addItems(choices)
                if current:
                    if self._capture_face_object.findText(current) < 0:
                        self._capture_face_object.addItem(current)
                    self._capture_face_object.setCurrentText(current)
                elif choices:
                    self._capture_face_object.setCurrentText(
                        CAPTURE_FACE_QUEUE_LABEL
                    )
                    updated = True
            finally:
                self._capture_face_object.blockSignals(False)
            return updated

        def _update_capture_face_queue(
            self,
            readiness: CaptureFaceReadiness,
        ) -> None:
            if not hasattr(self, "_capture_face_queue"):
                return
            outputs = readiness.output_status
            covered = set(outputs.covered_faces)
            missing = set(outputs.missing_faces)
            self._capture_face_queue.blockSignals(True)
            try:
                self._capture_face_queue.clear()
                for face in outputs.registered_faces:
                    status = "missing" if face in missing else "captured"
                    marker = "[ ]" if face in missing else "[x]"
                    item = QtWidgets.QListWidgetItem(f"{marker} {face}")
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, face)
                    if face in covered:
                        item.setForeground(QtGui.QBrush(QtGui.QColor("#087a3d")))
                    elif face in missing:
                        item.setForeground(QtGui.QBrush(QtGui.QColor("#8a5b00")))
                    item.setToolTip(status)
                    self._capture_face_queue.addItem(item)
                if self._capture_face_queue.count() == 0:
                    item = QtWidgets.QListWidgetItem(
                        "No registered faces found in boards/tag_registry.yaml"
                    )
                    self._capture_face_queue.addItem(item)
            finally:
                self._capture_face_queue.blockSignals(False)

        def _update_capture_face_gallery(
            self,
            readiness: CaptureFaceReadiness,
        ) -> None:
            if not hasattr(self, "_capture_face_gallery"):
                return
            selected_meta = self._selected_capture_face_gallery_metadata_path()
            shots = tuple(reversed(readiness.output_status.saved_shots))
            selected_row: Optional[int] = None
            self._capture_face_gallery.blockSignals(True)
            try:
                self._capture_face_gallery.clear()
                if shots:
                    item = QtWidgets.QListWidgetItem(
                        _capture_face_saved_shot_collection_label(shots)
                    )
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, shots)
                    item.setToolTip("Click to inspect this saved-shot stack.")
                    if any(shot.coverage_ok for shot in shots):
                        item.setForeground(QtGui.QBrush(QtGui.QColor("#087a3d")))
                    else:
                        item.setForeground(QtGui.QBrush(QtGui.QColor("#8a5b00")))
                    item.setIcon(
                        _capture_face_saved_shot_icon(
                            shots,
                            self._capture_face_gallery.iconSize(),
                        )
                    )
                    self._capture_face_gallery.addItem(item)
                    if (
                        selected_meta is not None
                        and any(shot.metadata_path == selected_meta for shot in shots)
                    ):
                        selected_row = 0
                if selected_row is not None:
                    self._capture_face_gallery.setCurrentRow(selected_row)
                else:
                    self._capture_face_gallery.clearSelection()
                    self._capture_face_gallery.setCurrentRow(-1)
            finally:
                self._capture_face_gallery.blockSignals(False)
            if selected_row is None:
                self._collapse_capture_face_gallery()
            else:
                self._select_capture_face_gallery_item()

        def _selected_capture_face_gallery_metadata_path(self) -> Optional[Path]:
            if not hasattr(self, "_capture_face_gallery"):
                return None
            item = self._capture_face_gallery.currentItem()
            if item is None:
                return None
            shots = _capture_face_saved_shots_from_item(item)
            if shots:
                return shots[0].metadata_path
            return None

        def _select_capture_face_gallery_item(self) -> None:
            if not hasattr(self, "_capture_face_gallery"):
                return
            item = self._capture_face_gallery.currentItem()
            if item is None:
                self._collapse_capture_face_gallery()
                return
            shots = _capture_face_saved_shots_from_item(item)
            if not shots:
                self._collapse_capture_face_gallery("Preview unavailable.")
                return
            self._capture_face_gallery_side.setVisible(True)
            self._populate_capture_face_stack_gallery(shots)
            self._preview_capture_face_stack_shot(shots[0])

        def _collapse_capture_face_gallery(
            self,
            message: str = "Click a saved-shot stack to inspect its images.",
        ) -> None:
            if not hasattr(self, "_capture_face_gallery_side"):
                return
            self._capture_face_stack_gallery.clear()
            self._capture_face_gallery_preview.clear_preview(message)
            self._capture_face_gallery_details.setText(
                "Saved shots stay collapsed until you select a stack."
            )
            self._capture_face_gallery_side.setVisible(False)

        def _populate_capture_face_stack_gallery(
            self,
            shots: tuple[CaptureFaceSavedShot, ...],
        ) -> None:
            self._capture_face_stack_gallery.blockSignals(True)
            try:
                self._capture_face_stack_gallery.clear()
                for shot in shots:
                    self._capture_face_stack_gallery.addItem(
                        _make_capture_face_saved_shot_item(
                            shot,
                            self._capture_face_stack_gallery.iconSize(),
                        )
                    )
                if self._capture_face_stack_gallery.count() > 0:
                    self._capture_face_stack_gallery.setCurrentRow(0)
            finally:
                self._capture_face_stack_gallery.blockSignals(False)

        def _select_capture_face_stack_item(self, item: Any) -> None:
            shot = _capture_face_saved_shot_from_item(item)
            if shot is not None:
                self._preview_capture_face_stack_shot(shot)

        def _preview_capture_face_stack_shot(
            self,
            shot: CaptureFaceSavedShot,
        ) -> None:
            preview_path = _capture_face_shot_preview_path(shot)
            preview_loaded = (
                preview_path is not None
                and self._capture_face_gallery_preview.show_preview(preview_path)
            )
            if not preview_loaded:
                self._capture_face_gallery_preview.clear_preview(
                    "Preview image was not found."
                )
            self._capture_face_gallery_details.setText(
                _format_capture_face_saved_shot_details(shot)
            )

        def _update_capture_face_action_buttons(
            self,
            ready: Optional[bool] = None,
            process_running: Optional[bool] = None,
        ) -> None:
            if not hasattr(self, "_capture_face_run_button"):
                return
            if ready is None:
                ready = bool(
                    getattr(self, "_latest_capture_face_readiness_ready", False)
                )
            else:
                self._latest_capture_face_readiness_ready = bool(ready)
            if process_running is None:
                process_running = self._capture_face_process_is_running()
            enabled = bool(ready) and not process_running
            selected_faces = self._selected_capture_face_queue_faces()
            current_face = self._current_capture_face_queue_face()
            object_text = self._capture_face_object.currentText().strip()
            current_available = bool(
                current_face
                or (object_text and object_text != CAPTURE_FACE_QUEUE_LABEL)
            )

            self._capture_face_run_button.setEnabled(enabled)
            self._capture_face_run_button.setText(
                _capture_face_run_button_label(self._capture_face_process_state)
            )
            if hasattr(self, "_capture_face_selected_button"):
                self._capture_face_selected_button.setEnabled(
                    enabled and bool(selected_faces)
                )
            if hasattr(self, "_capture_face_current_button"):
                self._capture_face_current_button.setEnabled(
                    enabled and current_available
                )

        def _selected_capture_face_queue_faces(self) -> tuple[str, ...]:
            if not hasattr(self, "_capture_face_queue"):
                return ()
            faces: list[str] = []
            for item in self._capture_face_queue.selectedItems():
                face = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if face:
                    faces.append(str(face))
            return tuple(dict.fromkeys(faces))

        def _current_capture_face_queue_face(self) -> str:
            if not hasattr(self, "_capture_face_queue"):
                return ""
            item = self._capture_face_queue.currentItem()
            if item is None:
                return ""
            face = item.data(QtCore.Qt.ItemDataRole.UserRole)
            return str(face) if face else ""

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

        def _open_native_board_capture(self) -> None:
            if self._board_process_is_running():
                message = "Stop the CLI board builder before guided capture."
                self.statusBar().showMessage(message, 4000)
                return

            config = self._board_building_config()
            readiness = inspect_board_building(config)
            if not readiness.ready:
                message = "Resolve board-building readiness messages before guided capture."
                self._copy_feedback.setText(message)
                self.statusBar().showMessage(message, 5000)
                self._update_board_building_flow()
                return

            dialog = _NativeBoardCaptureDialog(self, config)
            result = dialog.exec()
            if (
                result == QtWidgets.QDialog.DialogCode.Accepted
                and dialog.result_payload is not None
            ):
                saved = dialog.result_payload
                message = (
                    "Saved board definition: "
                    f"{_display_path(saved.board_yaml_path)}"
                )
                self._append_board_log(f"[gui] {message}")
                self.statusBar().showMessage(message, 6000)
                self._refresh()
                return
            self._update_board_building_flow()

        def _open_native_board_batch_capture(self) -> None:
            if self._board_process_is_running():
                message = "Stop the CLI board builder before batch capture."
                self.statusBar().showMessage(message, 4000)
                return

            config = self._board_building_config()
            readiness = inspect_board_building(config)
            if not readiness.ready:
                message = "Resolve board-building readiness messages before batch capture."
                self._copy_feedback.setText(message)
                self.statusBar().showMessage(message, 5000)
                self._update_board_building_flow()
                return
            try:
                items = self._board_batch_items()
            except Exception as exc:
                message = f"Resolve batch setup before starting: {exc}"
                self._board_batch_preview.setText(message)
                self.statusBar().showMessage(message, 5000)
                return

            dialog = _NativeBoardBatchCaptureDialog(self, config, items)
            result = dialog.exec()
            if result == QtWidgets.QDialog.DialogCode.Accepted:
                message = (
                    f"Batch capture saved {len(dialog.saved_results)} board "
                    f"definition{'s' if len(dialog.saved_results) != 1 else ''}."
                )
                if dialog.skipped_items:
                    message += f" Skipped {len(dialog.skipped_items)}."
                self._append_board_log(f"[gui] {message}")
                self.statusBar().showMessage(message, 6000)
                self._refresh()
                return
            self._update_board_building_flow()

        def _run_board_building(self) -> None:
            if self._board_process_is_running():
                message = "Board builder is already running."
                self.statusBar().showMessage(message, 3000)
                return

            config = self._board_building_config()
            readiness = inspect_board_building(config)
            if not readiness.ready:
                message = "Resolve board-building readiness messages before running."
                self._copy_feedback.setText(message)
                self.statusBar().showMessage(message, 5000)
                self._update_board_building_flow()
                return

            try:
                launch = build_board_building_launch(config)
            except Exception as exc:
                message = f"Could not prepare board-builder launch: {exc}"
                self._set_board_process_state(
                    board_building_process_failed(
                        message,
                        expected_board_yaml=readiness.expected_board_yaml,
                        expected_registry=readiness.expected_registry,
                    )
                )
                self._append_board_log(f"[gui] {message}")
                self.statusBar().showMessage(message, 6000)
                self._update_board_building_flow()
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
            process.readyReadStandardOutput.connect(self._read_board_stdout)
            process.readyReadStandardError.connect(self._read_board_stderr)
            process.finished.connect(self._board_process_finished)
            process.errorOccurred.connect(self._board_process_error)

            self._board_process = process
            self._board_running_expected_yaml = launch.expected_board_yaml
            self._board_running_expected_registry = launch.expected_registry
            self._board_outputs_existed_at_launch = (
                launch.expected_board_yaml.exists()
                and launch.expected_registry.exists()
            )
            self._board_previous_yaml_mtime_ns = _path_mtime_ns(
                launch.expected_board_yaml
            )
            self._board_previous_registry_mtime_ns = _path_mtime_ns(
                launch.expected_registry
            )
            self._board_output_seen = self._board_outputs_existed_at_launch
            self._board_log.clear()
            self._append_board_log(f"$ {launch.display_command}")
            self._append_board_log(
                "[gui] Launching with the current Python interpreter."
            )
            self._append_board_log(
                "[gui] Use the OpenCV window for ENTER/ESC, then send "
                "terminal prompt responses here."
            )
            self._set_board_process_state(
                board_building_process_running(
                    launch.expected_board_yaml,
                    launch.expected_registry,
                )
            )
            self._board_output_timer.start()
            self._update_board_building_flow()
            process.start()
            self.statusBar().showMessage("Board builder process started.", 4000)

        def _send_board_prompt_response(self) -> None:
            process = self._board_process
            response = self._board_prompt_input.text()
            if process is None or not self._board_process_is_running():
                message = "Board builder is not running."
                self.statusBar().showMessage(message, 3000)
                return
            if not response.strip():
                message = "Enter a response before sending it to the board builder."
                self.statusBar().showMessage(message, 3000)
                return
            process.write((response + "\n").encode("utf-8"))
            self._append_board_log(f"[gui stdin] {response}")
            self._board_prompt_input.clear()

        def _read_board_stdout(self) -> None:
            process = self._board_process
            if process is None:
                return
            self._append_board_output(process.readAllStandardOutput(), "")

        def _read_board_stderr(self) -> None:
            process = self._board_process
            if process is None:
                return
            self._append_board_output(process.readAllStandardError(), "stderr")

        def _board_process_finished(
            self,
            exit_code: int,
            exit_status: Any,
        ) -> None:
            self._read_board_stdout()
            self._read_board_stderr()
            crashed = exit_status == QtCore.QProcess.ExitStatus.CrashExit
            state = summarize_board_building_process_result(
                exit_code=int(exit_code),
                crashed=crashed,
                expected_board_yaml=self._board_running_expected_yaml,
                expected_registry=self._board_running_expected_registry,
                previous_board_mtime_ns=self._board_previous_yaml_mtime_ns,
                previous_registry_mtime_ns=self._board_previous_registry_mtime_ns,
                require_output_update=self._board_outputs_existed_at_launch,
            )
            self._board_process = None
            self._board_output_timer.stop()
            self._set_board_process_state(state)
            self._append_board_log(f"[gui] {state.message}")
            self._refresh()
            self.statusBar().showMessage(state.message, 7000)

        def _board_process_error(self, error: Any) -> None:
            process = self._board_process
            error_name = _qt_enum_name(error)
            detail = process.errorString() if process is not None else error_name
            self._append_board_log(f"[gui] Process error: {detail}")
            failed_to_start = QtCore.QProcess.ProcessError.FailedToStart
            if error != failed_to_start:
                return

            state = board_building_process_failed(
                f"Board-builder process failed to start: {detail}",
                expected_board_yaml=self._board_running_expected_yaml,
                expected_registry=self._board_running_expected_registry,
            )
            self._board_process = None
            self._board_output_timer.stop()
            self._set_board_process_state(state)
            self._update_board_building_flow()
            self.statusBar().showMessage(state.message, 7000)

        def _poll_board_output(self) -> None:
            board_yaml = self._board_running_expected_yaml
            registry = self._board_running_expected_registry
            if board_yaml is None or registry is None or self._board_output_seen:
                return
            if not (board_yaml.exists() and registry.exists()):
                return
            self._board_output_seen = True
            self._append_board_log(
                f"[gui] Detected board outputs: {_display_path(board_yaml)} "
                f"and {_display_path(registry)}"
            )
            self._refresh()

        def _append_board_output(self, data: Any, prefix: str) -> None:
            text = bytes(data).decode("utf-8", errors="replace")
            if not text:
                return
            if prefix:
                for line in text.rstrip().splitlines():
                    self._append_board_log(f"[{prefix}] {line}")
            else:
                self._append_board_log(text.rstrip())

        def _append_board_log(self, text: str) -> None:
            if not text:
                return
            self._board_log.appendPlainText(text)
            scrollbar = self._board_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_board_process_state(
            self,
            state: BoardBuildingProcessState,
        ) -> None:
            self._board_process_state = state
            if hasattr(self, "_board_process_state_label"):
                self._board_process_state_label.setText(
                    _format_board_building_process_state(state)
                )
            if hasattr(self, "_board_run_button"):
                self._board_run_button.setText(
                    _board_building_run_button_label(state)
                )

        def _board_process_is_running(self) -> bool:
            process = self._board_process
            if process is None:
                return False
            return process.state() != QtCore.QProcess.ProcessState.NotRunning

        def _open_native_capture_face(self, mode: str = "batch") -> None:
            if self._capture_face_process_is_running():
                message = "Stop the CLI face-shot process before guided capture."
                self.statusBar().showMessage(message, 4000)
                return
            if mode == "selected" and not self._selected_capture_face_queue_faces():
                message = "Select one or more registered faces in the queue first."
                self.statusBar().showMessage(message, 4000)
                return

            config = self._capture_face_config(mode)
            readiness = inspect_capture_face_readiness(config)
            if not readiness.ready:
                message = "Resolve face-shot readiness messages before guided capture."
                self._copy_feedback.setText(message)
                self.statusBar().showMessage(message, 5000)
                self._update_capture_face_flow()
                return

            dialog = _NativeFaceBatchCaptureDialog(self, config)
            result = dialog.exec()
            if result == QtWidgets.QDialog.DialogCode.Accepted:
                message = "Face-shot capture window closed."
                self._append_capture_face_log(f"[gui] {message}")
                self.statusBar().showMessage(message, 5000)
                self._refresh()
                return
            self._update_capture_face_flow()

        def _run_capture_face(self, mode: str = "batch") -> None:
            if self._capture_face_process_is_running():
                message = "Face-shot capture is already running."
                self.statusBar().showMessage(message, 3000)
                return

            if mode == "selected" and not self._selected_capture_face_queue_faces():
                message = "Select one or more registered faces in the queue first."
                self.statusBar().showMessage(message, 4000)
                return

            config = self._capture_face_config(mode)
            readiness = inspect_capture_face_readiness(config)
            if not readiness.ready:
                message = "Resolve face-shot readiness messages before running."
                self._copy_feedback.setText(message)
                self.statusBar().showMessage(message, 5000)
                self._update_capture_face_flow()
                return

            try:
                launch = build_capture_face_launch(config)
            except Exception as exc:
                message = f"Could not prepare face-shot capture launch: {exc}"
                self._set_capture_face_process_state(
                    capture_face_process_failed(
                        message,
                        expected_manifest=readiness.expected_manifest,
                    )
                )
                self._append_capture_face_log(f"[gui] {message}")
                self.statusBar().showMessage(message, 6000)
                self._update_capture_face_flow()
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
            process.readyReadStandardOutput.connect(self._read_capture_face_stdout)
            process.readyReadStandardError.connect(self._read_capture_face_stderr)
            process.finished.connect(self._capture_face_process_finished)
            process.errorOccurred.connect(self._capture_face_process_error)

            self._capture_face_process = process
            self._capture_running_expected_manifest = launch.expected_manifest
            self._capture_manifest_existed_at_launch = (
                launch.expected_manifest.exists()
            )
            self._capture_previous_manifest_mtime_ns = _path_mtime_ns(
                launch.expected_manifest
            )
            self._capture_manifest_seen = False
            self._capture_face_log.clear()
            self._append_capture_face_log(f"$ {launch.display_command}")
            self._append_capture_face_log(
                "[gui] Launching with the current Python interpreter."
            )
            self._append_capture_face_log(
                "[gui] Use the OpenCV face queue; auto-capture runs when tags "
                "are stable, ENTER saves manually, and q or ESC quits."
            )
            self._set_capture_face_process_state(
                capture_face_process_running(launch.expected_manifest)
            )
            self._capture_output_timer.start()
            self._update_capture_face_flow()
            process.start()
            self.statusBar().showMessage("Face-shot capture process started.", 4000)

        def _read_capture_face_stdout(self) -> None:
            process = self._capture_face_process
            if process is None:
                return
            self._append_capture_face_output(process.readAllStandardOutput(), "")

        def _read_capture_face_stderr(self) -> None:
            process = self._capture_face_process
            if process is None:
                return
            self._append_capture_face_output(process.readAllStandardError(), "stderr")

        def _capture_face_process_finished(
            self,
            exit_code: int,
            exit_status: Any,
        ) -> None:
            self._read_capture_face_stdout()
            self._read_capture_face_stderr()
            crashed = exit_status == QtCore.QProcess.ExitStatus.CrashExit
            state = summarize_capture_face_process_result(
                exit_code=int(exit_code),
                crashed=crashed,
                expected_manifest=self._capture_running_expected_manifest,
                previous_manifest_mtime_ns=self._capture_previous_manifest_mtime_ns,
                require_output_update=self._capture_manifest_existed_at_launch,
            )
            self._capture_face_process = None
            self._capture_output_timer.stop()
            self._set_capture_face_process_state(state)
            self._append_capture_face_log(f"[gui] {state.message}")
            self._refresh()
            self.statusBar().showMessage(state.message, 7000)

        def _capture_face_process_error(self, error: Any) -> None:
            process = self._capture_face_process
            error_name = _qt_enum_name(error)
            detail = process.errorString() if process is not None else error_name
            self._append_capture_face_log(f"[gui] Process error: {detail}")
            failed_to_start = QtCore.QProcess.ProcessError.FailedToStart
            if error != failed_to_start:
                return

            state = capture_face_process_failed(
                f"Face-shot capture process failed to start: {detail}",
                expected_manifest=self._capture_running_expected_manifest,
            )
            self._capture_face_process = None
            self._capture_output_timer.stop()
            self._set_capture_face_process_state(state)
            self._update_capture_face_flow()
            self.statusBar().showMessage(state.message, 7000)

        def _poll_capture_face_output(self) -> None:
            manifest = self._capture_running_expected_manifest
            if manifest is None or self._capture_manifest_seen:
                return
            if not manifest.exists():
                return
            current_mtime = _path_mtime_ns(manifest)
            if (
                self._capture_manifest_existed_at_launch
                and current_mtime == self._capture_previous_manifest_mtime_ns
            ):
                return
            self._capture_manifest_seen = True
            self._append_capture_face_log(
                f"[gui] Detected face-shot manifest update: "
                f"{_display_path(manifest)}"
            )
            self._refresh()

        def _append_capture_face_output(self, data: Any, prefix: str) -> None:
            text = bytes(data).decode("utf-8", errors="replace")
            if not text:
                return
            if prefix:
                for line in text.rstrip().splitlines():
                    self._append_capture_face_log(f"[{prefix}] {line}")
            else:
                self._append_capture_face_log(text.rstrip())

        def _append_capture_face_log(self, text: str) -> None:
            if not text:
                return
            self._capture_face_log.appendPlainText(text)
            scrollbar = self._capture_face_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_capture_face_process_state(
            self,
            state: CaptureFaceProcessState,
        ) -> None:
            self._capture_face_process_state = state
            if hasattr(self, "_capture_face_process_state_label"):
                self._capture_face_process_state_label.setText(
                    _format_capture_face_process_state(state)
                )
            if hasattr(self, "_capture_face_run_button"):
                self._capture_face_run_button.setText(
                    _capture_face_run_button_label(state)
                )
                self._update_capture_face_action_buttons()

        def _capture_face_process_is_running(self) -> bool:
            process = self._capture_face_process
            if process is None:
                return False
            return process.state() != QtCore.QProcess.ProcessState.NotRunning

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

        def _render_object_tag_mode_fields(self) -> None:
            mode = self._object_tag_id_mode_value()
            using_list = mode == OBJECT_TAG_ID_MODE_LIST
            self._object_tags_list_field.setVisible(using_list)
            self._object_tags_start_field.setVisible(not using_list)
            self._object_tags_count_field.setVisible(not using_list)

        def _render_object_tag_paper_fields(self) -> None:
            custom = (
                self._object_tags_paper.currentText().strip().upper()
                == OBJECT_TAG_PAPER_CUSTOM
            )
            self._object_tags_paper_mm_field.setVisible(custom)

        def _render_board_source_fields(self) -> None:
            source = self._board_source_value()
            is_webcam = source == BOARD_SOURCE_OPENCV
            is_video = source == BOARD_SOURCE_VIDEO
            uses_capture_size = source in {
                BOARD_SOURCE_OPENCV,
                BOARD_SOURCE_REALSENSE,
            }
            self._board_camera_field.setVisible(is_webcam)
            self._board_video_field.setVisible(is_video)
            self._board_video_path.setEnabled(is_video)
            self._board_video_browse_button.setEnabled(is_video)
            for field in (self._board_width, self._board_height, self._board_fps):
                field.setEnabled(uses_capture_size)

        def _render_capture_face_source_fields(self) -> None:
            source = self._capture_face_source_value()
            is_webcam = source == CAPTURE_SOURCE_OPENCV
            is_video = source == CAPTURE_SOURCE_VIDEO
            uses_capture_size = source in {
                CAPTURE_SOURCE_OPENCV,
                CAPTURE_SOURCE_REALSENSE,
            }
            self._capture_face_camera_field.setVisible(is_webcam)
            self._capture_face_video_field.setVisible(is_video)
            self._capture_face_video_path.setEnabled(is_video)
            self._capture_face_video_browse_button.setEnabled(is_video)
            for field in (
                self._capture_face_width,
                self._capture_face_height,
                self._capture_face_fps,
            ):
                field.setEnabled(uses_capture_size)

        def _render_board_save_shot_fields(self) -> None:
            save_shot = bool(self._board_save_shot.isChecked())
            self._board_shots_field.setVisible(save_shot)
            self._board_shots_dir.setEnabled(save_shot)

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

        def _object_tag_config(self) -> ObjectTagGenerationConfig:
            out_dir_text = self._object_tags_out_dir.text().strip()
            out_dir = Path(out_dir_text).expanduser() if out_dir_text else None
            paper_mm_text = self._object_tags_paper_mm.text().strip()
            paper_mm = paper_mm_text or None
            return ObjectTagGenerationConfig(
                project_root=self._current_project_root(),
                family=self._object_tags_family.currentText(),
                tag_size_mm=float(self._object_tags_tag_size.value()),
                id_mode=self._object_tag_id_mode_value(),
                ids=self._object_tags_ids.text(),
                id_start=int(self._object_tags_id_start.value()),
                id_count=int(self._object_tags_id_count.value()),
                paper=self._object_tags_paper.currentText(),
                paper_mm=paper_mm,
                orientation=self._object_tags_orientation.currentText(),
                dpi=int(self._object_tags_dpi.value()),
                pil_text=bool(self._object_tags_pil_text.isChecked()),
                margin_frac=float(self._object_tags_margin_frac.value()),
                label_gap_frac=float(self._object_tags_label_gap_frac.value()),
                out_dir=out_dir,
                prefix=self._object_tags_prefix.text(),
            )

        def _board_building_config(self) -> BoardBuildingConfig:
            self._sync_board_definition_name()
            calib_text = self._board_calibration_path.text().strip()
            video_text = self._board_video_path.text().strip()
            out_dir_text = self._board_out_dir.text().strip()
            registry_text = self._board_registry_path.text().strip()
            shots_text = self._board_shots_dir.text().strip()
            return BoardBuildingConfig(
                project_root=self._current_project_root(),
                object_name=self._board_object_name.text(),
                tag_size_mm=float(self._board_tag_size.value()),
                family=self._board_family.text(),
                calibration_path=Path(calib_text).expanduser()
                if calib_text
                else None,
                source=self._board_source_value(),
                camera_index=int(self._board_camera_index.value()),
                video_path=Path(video_text).expanduser() if video_text else None,
                width=int(self._board_width.value()),
                height=int(self._board_height.value()),
                fps=int(self._board_fps.value()),
                out_dir=Path(out_dir_text).expanduser() if out_dir_text else None,
                registry_path=Path(registry_text).expanduser()
                if registry_text
                else None,
                save_shot=bool(self._board_save_shot.isChecked()),
                shots_dir=Path(shots_text).expanduser() if shots_text else None,
                z_threshold_m=float(self._board_z_threshold.value()),
                allow_nonplanar=bool(self._board_allow_nonplanar.isChecked()),
            )

        def _capture_face_config(self, mode: str = "batch") -> CaptureFaceConfig:
            calib_text = self._capture_face_calibration_path.text().strip()
            registry_text = self._capture_face_registry_path.text().strip()
            video_text = self._capture_face_video_path.text().strip()
            out_dir_text = self._capture_face_out_dir.text().strip()
            manifest_text = self._capture_face_manifest_path.text().strip()
            raw_dir_text = self._capture_face_raw_dir.text().strip()
            ann_dir_text = self._capture_face_ann_dir.text().strip()
            meta_dir_text = self._capture_face_meta_dir.text().strip()
            object_text = self._capture_face_object.currentText().strip()
            queue_faces: tuple[str, ...] = ()
            capture_all = False
            object_name = ""
            if mode == "selected":
                queue_faces = self._selected_capture_face_queue_faces()
            elif mode == "current":
                object_name = self._current_capture_face_queue_face()
                if not object_name and object_text != CAPTURE_FACE_QUEUE_LABEL:
                    object_name = object_text
                if not object_name:
                    capture_all = True
            else:
                capture_all = True
            return CaptureFaceConfig(
                project_root=self._current_project_root(),
                object_name=object_name,
                queue_faces=queue_faces,
                family=self._capture_face_family.text(),
                calibration_path=Path(calib_text).expanduser()
                if calib_text
                else None,
                registry_path=Path(registry_text).expanduser()
                if registry_text
                else None,
                source=self._capture_face_source_value(),
                camera_index=int(self._capture_face_camera_index.value()),
                video_path=Path(video_text).expanduser() if video_text else None,
                width=int(self._capture_face_width.value()),
                height=int(self._capture_face_height.value()),
                fps=int(self._capture_face_fps.value()),
                min_expected=int(self._capture_face_min_expected.value()),
                layout=self._capture_face_layout.currentText(),
                out_dir=Path(out_dir_text).expanduser() if out_dir_text else None,
                manifest_path=Path(manifest_text).expanduser()
                if manifest_text
                else None,
                raw_dir=Path(raw_dir_text).expanduser() if raw_dir_text else None,
                ann_dir=Path(ann_dir_text).expanduser() if ann_dir_text else None,
                meta_dir=Path(meta_dir_text).expanduser() if meta_dir_text else None,
                panel_width=int(self._capture_face_panel_width.value()),
                recent_width=int(self._capture_face_recent_width.value()),
                capture_all=capture_all,
                auto_capture=True,
                auto_capture_frames=DEFAULT_CAPTURE_AUTO_FRAMES,
                auto_capture_cooldown=DEFAULT_CAPTURE_AUTO_COOLDOWN,
                exit_when_complete=capture_all or bool(queue_faces),
            )

        def _object_tag_id_mode_value(self) -> str:
            data = self._object_tags_id_mode.currentData()
            raw = str(data if data is not None else self._object_tags_id_mode.currentText())
            if raw in OBJECT_TAG_ID_MODE_CHOICES:
                return raw
            return OBJECT_TAG_ID_MODE_RANGE if "count" in raw.lower() else OBJECT_TAG_ID_MODE_LIST

        def _board_source_value(self) -> str:
            data = self._board_source.currentData()
            raw = str(data if data is not None else self._board_source.currentText())
            try:
                return normalize_board_source(raw)
            except Exception:
                return BOARD_SOURCE_OPENCV

        def _capture_face_source_value(self) -> str:
            data = self._capture_face_source.currentData()
            raw = str(
                data
                if data is not None
                else self._capture_face_source.currentText()
            )
            try:
                return normalize_capture_source(raw)
            except Exception:
                return CAPTURE_SOURCE_OPENCV

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

        def _object_tag_output_target(self) -> Path:
            out_dir_text = self._object_tags_out_dir.text().strip()
            if out_dir_text:
                return Path(out_dir_text).expanduser()
            return default_object_tag_output_dir(self._current_project_root())

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


def _format_object_tag_outputs(result: Any) -> str:
    png_paths = tuple(getattr(result, "png_paths", ()))
    pdf_paths = tuple(getattr(result, "pdf_paths", ()))
    parsed_ids = tuple(getattr(result, "parsed_ids", ()))
    lines = [
        f"Folder: {_display_path(getattr(result, 'out_dir', Path('.')))}",
        f"IDs:    {_format_id_summary(parsed_ids)}",
        f"PNG:    {len(png_paths)} sheet{'s' if len(png_paths) != 1 else ''}",
    ]
    lines.extend(f"  - {Path(path).name}" for path in png_paths)
    if pdf_paths:
        lines.append(
            f"PDF:    {len(pdf_paths)} sheet{'s' if len(pdf_paths) != 1 else ''}"
        )
        lines.extend(f"  - {Path(path).name}" for path in pdf_paths)
    else:
        lines.append("PDF:    unavailable in this environment")
    return "\n".join(lines)


def _format_object_tag_readiness(
    readiness: ObjectTagGenerationReadiness,
) -> str:
    lines = ["Ready to generate." if readiness.ready else "Not ready yet."]
    if readiness.parsed_ids:
        lines.extend(["", f"IDs: {_format_id_summary(readiness.parsed_ids)}"])
    if readiness.errors:
        lines.extend(["", "Errors", *_format_items(readiness.errors)])
    if readiness.warnings:
        lines.extend(["", "Warnings", *_format_items(readiness.warnings)])
    return "\n".join(lines)


def _format_object_tag_expected_output(
    readiness: ObjectTagGenerationReadiness,
) -> str:
    state = "present" if readiness.expected_output_dir.exists() else "missing"
    return f"{_display_path(readiness.expected_output_dir)}\nStatus: {state}"


def _format_board_building_readiness(
    readiness: BoardBuildingReadiness,
) -> str:
    lines = [
        (
            "Ready to capture board definitions."
            if readiness.ready
            else "Not ready yet."
        )
    ]
    if readiness.errors:
        lines.extend(["", "Errors", *_format_items(readiness.errors)])
    if readiness.warnings:
        lines.extend(["", "Warnings", *_format_items(readiness.warnings)])
    return "\n".join(lines)


def _format_board_building_outputs(readiness: BoardBuildingReadiness) -> str:
    outputs = readiness.output_status
    lines = [
        "Calibration YAML",
        f"{_display_path(readiness.calibration_path)}",
        f"Status: {'present' if readiness.calibration_path.exists() else 'missing'}",
        "",
        "Board YAML",
        f"{_display_path(outputs.board_yaml_path)}",
        f"Status: {'present' if outputs.board_yaml_exists else 'missing'}",
        "",
        "Tag registry",
        f"{_display_path(outputs.registry_path)}",
        f"Status: {'present' if outputs.registry_exists else 'missing'}",
    ]
    if outputs.shots_dir in outputs.checked_paths:
        lines.extend(
            [
                "",
                "Audit shots",
                f"{_display_path(outputs.shots_dir)}",
                f"Status: {'present' if outputs.shots_dir_exists else 'missing'}",
            ]
        )
    return "\n".join(lines)


def _format_board_building_process_state(
    state: BoardBuildingProcessState,
) -> str:
    lines = [state.label, "", state.message]
    if state.expected_board_yaml is not None:
        lines.extend(
            ["", f"Expected board: {_display_path(state.expected_board_yaml)}"]
        )
    if state.expected_registry is not None:
        lines.append(f"Expected registry: {_display_path(state.expected_registry)}")
    if state.exit_code is not None:
        lines.append(f"Exit code: {state.exit_code}")
    return "\n".join(lines)


def _format_capture_face_readiness(
    readiness: CaptureFaceReadiness,
) -> str:
    lines = [
        (
            "Ready to capture face shots."
            if readiness.ready
            else "Not ready yet."
        )
    ]
    if readiness.registered_bases:
        lines.extend(
            [
                "",
                "Registered objects",
                *_format_items(readiness.registered_bases),
            ]
        )
    if readiness.selected_faces:
        lines.extend(
            [
                "",
                "Selected faces",
                *_format_items(readiness.selected_faces),
            ]
        )
    if readiness.errors:
        lines.extend(["", "Errors", *_format_items(readiness.errors)])
    if readiness.warnings:
        lines.extend(["", "Warnings", *_format_items(readiness.warnings)])
    return "\n".join(lines)


def _format_capture_face_outputs(readiness: CaptureFaceReadiness) -> str:
    outputs = readiness.output_status
    lines = [
        "Calibration YAML",
        f"{_display_path(readiness.calibration_path)}",
        f"Status: {'present' if readiness.calibration_path.exists() else 'missing'}",
        "",
        "Tag registry",
        f"{_display_path(outputs.registry_path)}",
        f"Status: {'present' if outputs.registry_path.exists() else 'missing'}",
        "",
        "Manifest",
        f"{_display_path(outputs.manifest_path)}",
        f"Status: {'present' if outputs.manifest_exists else 'missing'}",
        "",
        "Coverage",
        (
            f"{outputs.covered_face_count}/{outputs.registered_face_count} "
            "registered faces"
        ),
    ]
    if outputs.missing_faces:
        lines.extend(["", "Missing faces", *_format_items(outputs.missing_faces[:8])])
        if len(outputs.missing_faces) > 8:
            lines.append(f"- ... {len(outputs.missing_faces) - 8} more")
    if outputs.invalid_shot_count:
        lines.extend(["", "Invalid rows", str(outputs.invalid_shot_count)])
    if outputs.saved_shots:
        lines.extend(["", "Saved shots", str(len(outputs.saved_shots))])
    return "\n".join(lines)


def _capture_face_saved_shot_label(shot: CaptureFaceSavedShot) -> str:
    status = "ok" if shot.coverage_ok else "check"
    timestamp = shot.timestamp or f"row {shot.row_index}"
    return f"{shot.object_full}\n{timestamp}\n{status}"


def _group_capture_face_saved_shots(
    shots: Sequence[CaptureFaceSavedShot],
) -> tuple[tuple[str, tuple[CaptureFaceSavedShot, ...]], ...]:
    grouped: dict[str, list[CaptureFaceSavedShot]] = {}
    for shot in shots:
        grouped.setdefault(shot.object_full or "(unknown face)", []).append(shot)
    return tuple((face_name, tuple(group)) for face_name, group in grouped.items())


def _capture_face_saved_shot_group_label(
    shots: Sequence[CaptureFaceSavedShot],
) -> str:
    if not shots:
        return "No saved shots"
    latest = shots[0]
    count = len(shots)
    status = "ok" if any(shot.coverage_ok for shot in shots) else "check"
    timestamp = latest.timestamp or f"row {latest.row_index}"
    return (
        f"{latest.object_full or '(unknown face)'}\n"
        f"{count} shot{'s' if count != 1 else ''}\n"
        f"latest {timestamp} | {status}"
    )


def _capture_face_saved_shot_collection_label(
    shots: Sequence[CaptureFaceSavedShot],
) -> str:
    if not shots:
        return "No saved face shots"
    face_count = len({shot.object_full for shot in shots if shot.object_full})
    valid_count = sum(1 for shot in shots if shot.coverage_ok)
    return (
        "Saved face shots\n"
        f"{len(shots)} shot{'s' if len(shots) != 1 else ''}"
        f" | {face_count} face{'s' if face_count != 1 else ''}\n"
        f"{valid_count} valid"
    )


def _capture_face_shot_preview_path(shot: CaptureFaceSavedShot) -> Optional[Path]:
    if shot.annotated_path.exists():
        return shot.annotated_path
    if shot.raw_path.exists():
        return shot.raw_path
    return None


def _format_capture_face_saved_shot_details(
    shot: CaptureFaceSavedShot,
) -> str:
    lines = [
        shot.object_full or "(unknown face)",
        "",
        f"Status: {'valid coverage shot' if shot.coverage_ok else 'needs attention'}",
        f"Timestamp: {shot.timestamp or '(missing)'}",
        f"Side: {shot.side or '?'}",
        f"Expected tags: {_format_id_summary(shot.expected_tag_ids)}",
        f"Detected tags: {_format_id_summary(shot.detected_tag_ids)}",
        "",
        f"Annotated: {_display_path(shot.annotated_path)}",
        f"Raw: {_display_path(shot.raw_path)}",
        f"Metadata: {_display_path(shot.metadata_path)}",
    ]
    if shot.warnings:
        lines.extend(["", "Warnings", *_format_items(shot.warnings[:4])])
        if len(shot.warnings) > 4:
            lines.append(f"- ... {len(shot.warnings) - 4} more")
    return "\n".join(lines)


def _format_capture_face_saved_shot_group_details(
    shots: Sequence[CaptureFaceSavedShot],
) -> str:
    if not shots:
        return "No saved face shots."
    if len(shots) == 1:
        return _format_capture_face_saved_shot_details(shots[0])

    latest = shots[0]
    lines = [
        latest.object_full or "(unknown face)",
        "",
        f"Saved shots: {len(shots)}",
        f"Valid coverage shots: {sum(1 for shot in shots if shot.coverage_ok)}",
        "",
        "Newest shot",
        _format_capture_face_saved_shot_details(latest),
    ]
    previous = shots[1:6]
    if previous:
        lines.extend(["", "Previous shots"])
        lines.extend(
            f"- {shot.timestamp or f'row {shot.row_index}'} | "
            f"Raw: {_display_path(shot.raw_path)} | "
            f"Annotated: {_display_path(shot.annotated_path)}"
            for shot in previous
        )
    if len(shots) > len(previous) + 1:
        lines.append(f"- ... {len(shots) - len(previous) - 1} more")
    return "\n".join(lines)


def _format_capture_face_process_state(
    state: CaptureFaceProcessState,
) -> str:
    lines = [state.label, "", state.message]
    if state.expected_manifest is not None:
        lines.extend(
            ["", f"Expected manifest: {_display_path(state.expected_manifest)}"]
        )
    if state.exit_code is not None:
        lines.append(f"Exit code: {state.exit_code}")
    return "\n".join(lines)


def _format_id_summary(ids: tuple[int, ...]) -> str:
    if not ids:
        return "none"
    if len(ids) <= 12:
        return ", ".join(str(tag_id) for tag_id in ids)
    return f"{ids[0]}-{ids[-1]} ({len(ids)} IDs)"


def _checked_tag_ids_for_update(
    ids: tuple[int, ...],
    current_checked: set[int],
    *,
    force_all: bool = False,
) -> set[int]:
    id_set = set(ids)
    if force_all or not current_checked:
        return id_set
    retained = current_checked & id_set
    return retained if retained else id_set


def _format_board_batch_preview(items: tuple[BoardBatchItem, ...]) -> str:
    if not items:
        return "No boards queued."
    shown = [
        (
            f"{item.object_name} ({item.tag_size_mm:g} mm)"
            if item.tag_size_mm is not None
            else item.object_name
        )
        for item in items[:8]
    ]
    lines = [f"{len(items)} board{'s' if len(items) != 1 else ''} queued."]
    lines.extend(f"- {name}" for name in shown)
    if len(items) > len(shown):
        lines.append(f"- ... {len(items) - len(shown)} more")
    return "\n".join(lines)


def _format_batch_item_tag_size(
    item: BoardBatchItem,
    base_config: BoardBuildingConfig,
) -> str:
    value = item.tag_size_mm if item.tag_size_mm is not None else base_config.tag_size_mm
    return f"{value:g}"


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


def _board_building_run_button_label(state: BoardBuildingProcessState) -> str:
    if state.running:
        return "Board Builder Running..."
    if state.success:
        return "Run Board Builder Again"
    return "Run Board Builder"


def _capture_face_run_button_label(state: CaptureFaceProcessState) -> str:
    if state.running:
        return "Capture Running..."
    if state.success:
        return "Start Batch Again"
    return "Start Batch"


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
    rows = "".join(_format_health_count_row(count) for count in health.counts)
    return (
        "<span style='font-size:11px; font-weight:700; "
        "color:#5a6b7c;'>AT A GLANCE</span>"
        "<table width='100%' cellspacing='0' cellpadding='2'>"
        f"{rows}"
        "</table>"
    )


def _format_health_count_row(count: Any) -> str:
    label = html.escape(str(count.label))
    status = html.escape(str(count.status), quote=True)
    count_text = html.escape(str(count.count))
    if count.count:
        label_cell = f"<a href='stage-status:{status}'>{label}</a>"
        count_cell = f"<a href='stage-status:{status}'><b>{count_text}</b></a>"
    else:
        label_cell = label
        count_cell = f"<b>{count_text}</b>"
    return (
        "<tr>"
        f"<td>{label_cell}</td>"
        f"<td align='right'>{count_cell}</td>"
        "</tr>"
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


def _object_tag_command_card_note(
    readiness: ObjectTagGenerationReadiness,
    model: StageViewModel,
) -> str:
    if not readiness.command_preview:
        return (
            "Resolve the object AprilTag readiness messages above before "
            "generating sheets or copying a posetag-gen-tags command."
        )
    if model.status == "complete":
        return (
            "Object AprilTag sheets already exist in the checked paths. "
            "Generate again only if you need a replacement print set."
        )
    return (
        "Generate Object Tags runs the existing posetag-gen-tags workflow; "
        "Copy Command keeps the terminal fallback with the same arguments."
    )


def _object_tag_command_ready_message(
    readiness: ObjectTagGenerationReadiness,
    *,
    stage_complete: bool = False,
) -> str:
    if not readiness.command_preview:
        return "No runnable object AprilTag command is available yet."
    if stage_complete:
        return "Object tag sheets exist. Copy only if you need to regenerate them."
    return "Ready to generate or copy an object AprilTag command."


def _board_building_command_card_note(
    readiness: BoardBuildingReadiness,
    model: StageViewModel,
) -> str:
    if not readiness.command_preview:
        return (
            "Resolve the board-building readiness messages above before "
            "copying a posetag-make-board command."
        )
    if model.status == "complete":
        return (
            "Board YAML and tag registry outputs already pass the current "
            "checks. Copy this command only if you need to rebuild a board."
        )
    return (
        "Start Batch opens the native guided capture flow for every queued "
        "object side. Guided Capture handles the current board only. Run "
        "Board Builder starts the existing posetag-make-board CLI fallback, "
        "and Copy Command keeps the terminal fallback. In the CLI path, press "
        "ENTER in the OpenCV preview to capture, then answer the selected-ID "
        "and origin-ID prompts in the Stage 4 prompt box."
    )


def _board_building_command_ready_message(
    readiness: BoardBuildingReadiness,
    *,
    stage_complete: bool = False,
) -> str:
    if not readiness.command_preview:
        return "No runnable board-building command is available yet."
    if stage_complete:
        return "Board definition outputs exist. Copy only if you need to rerun it."
    return "Ready for batch capture, single-board capture, CLI launch, or copy."


def _capture_face_command_card_note(
    readiness: CaptureFaceReadiness,
    model: StageViewModel,
) -> str:
    if not readiness.command_preview:
        return (
            "Resolve the face-shot readiness messages above before copying a "
            "posetag-capture-face command."
        )
    if model.status == "complete":
        return (
            "Face-shot coverage already includes every registered face. Run "
            "capture again only if you need replacement reference images."
        )
    return (
        "Start Batch opens the native guided face-shot capture window for the "
        "registered face queue; Capture Selected and Capture Current narrow "
        "that same flow. Copy Command keeps the posetag-capture-face/OpenCV "
        "CLI fallback. Both paths write the same raw image, annotated image, "
        "metadata JSON, and manifest row."
    )


def _capture_face_command_ready_message(
    readiness: CaptureFaceReadiness,
    *,
    stage_complete: bool = False,
) -> str:
    if not readiness.command_preview:
        return "No runnable face-shot capture command is available yet."
    if stage_complete:
        return "Face-shot coverage exists. Copy only if you need to rerun capture."
    return "Ready to open guided capture or copy a face-shot capture command."


def _project_root_hint(root: Path) -> str:
    if root.is_dir():
        return (
            "Project folder found. Status checks are read-only except guided "
            "Stage 1 board generation, Stage 2 calibration launch, and Stage "
            "3 object tag generation. Stage 4 can launch the existing "
            "board-building workflow, and Stage 5 can open guided face-shot "
            "capture."
        )
    if root.exists():
        return "Selected path exists but is not a folder."
    return (
        "Project folder not found. Status checks are read-only; guided Stage "
        "1 and Stage 3 generation can create the selected project layout."
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
QGroupBox#CollapsibleGroup {
    background: #fbfdff;
    border: 1px solid #d7e4ed;
    border-radius: 7px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: 700;
}
QGroupBox#CollapsibleGroup::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #203245;
}
QFrame#CharucoPreviewColumn,
QFrame#ObjectTagPreviewColumn {
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
QLabel#ObjectTagPreviewCanvas {
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
QListWidget#BatchCaptureQueue {
    background: #ffffff;
    border: 1px solid #b8ccda;
    border-radius: 7px;
    padding: 5px;
}
QListWidget#BatchCaptureQueue::item {
    padding: 7px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
}
QListWidget#BatchCaptureQueue::item:hover {
    background: #eff7fb;
    border: 1px solid #bad7e7;
}
QListWidget#BatchCaptureQueue::item:selected,
QListWidget#BatchCaptureQueue::item:selected:!active {
    background: #d9efff;
    color: #08253d;
    border: 2px solid #0b6f8f;
}
QListWidget#FaceShotGallery {
    background: #ffffff;
    border: 1px solid #b8ccda;
    border-radius: 7px;
    padding: 6px;
}
QListWidget#FaceShotGallery::item {
    padding: 6px;
    border: 1px solid transparent;
    border-radius: 6px;
}
QListWidget#FaceShotGallery::item:hover {
    background: #eff7fb;
    border: 1px solid #bad7e7;
}
QListWidget#FaceShotGallery::item:selected,
QListWidget#FaceShotGallery::item:selected:!active {
    background: #d9efff;
    color: #08253d;
    border: 2px solid #0b6f8f;
}
QListWidget#BoardObjectRows {
    background: #ffffff;
    border: 1px solid #c7d3df;
    border-radius: 6px;
}
QListWidget#BoardObjectRows::item {
    padding: 5px 7px;
    border: 1px solid transparent;
    border-radius: 5px;
}
QListWidget#BoardObjectRows::item:selected,
QListWidget#BoardObjectRows::item:selected:!active {
    background: #d9efff;
    color: #08253d;
    border: 1px solid #0b6f8f;
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
