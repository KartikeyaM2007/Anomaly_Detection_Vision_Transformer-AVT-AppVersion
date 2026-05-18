from __future__ import annotations

import base64
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSize, QTimer, Qt, QThread, QUrl, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:  # QtMultimedia lives in PySide6 Addons on some installs.
    QAudioOutput = None
    QMediaPlayer = None

from vad_platform.config import RuntimeConfig, default_config
from vad_platform.detector import ViolenceDetectionService

from desktop_app.workers import AnalysisWorker, CameraWorker, RuntimeLoadWorker


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.base_config = default_config(project_root)
        self.service = ViolenceDetectionService(project_root=project_root, config=self.base_config)
        self.video_path: Path | None = None
        self.video_capture: cv2.VideoCapture | None = None
        self.video_frame_count = 0
        self.video_fps = 25.0
        self.video_playing = False
        self.video_slider_dragging = False
        self.media_player = None
        self.audio_output = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._render_next_video_frame)
        self.analysis_thread: QThread | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.runtime_thread: QThread | None = None
        self.runtime_worker: RuntimeLoadWorker | None = None
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None
        self.timeline_data: list[dict[str, Any]] = []
        self.score_history: list[float] = []
        self.live_metric_labels: dict[str, QLabel] = {}
        self.theme_mode = "night"
        self.terminal_started_at: float | None = None
        self.live_terminal_started_at: float | None = None
        self.last_live_terminal_status = ""
        self.analysis_started_at: float | None = None
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self._tick_analysis_timer)
        self.model_progress_timer = QTimer(self)
        self.model_progress_timer.timeout.connect(self._pulse_model_progress)
        self.workflow_steps: dict[str, QFrame] = {}
        self.workflow_order = ["queued", "runtime", "frames", "features", "scoring", "completed"]
        self.workflow_index = 0

        self.setWindowTitle("AnomalyGuard - Offline Video Anomaly Detection")
        self.setMinimumSize(1180, 760)
        self._build_ui()
        self._apply_theme()
        self._refresh_device_status()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(12)

        header = QGridLayout()
        header.setHorizontalSpacing(12)
        header.setVerticalSpacing(10)
        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("AnomalyGuard")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Local ViTMAE + Transformer surveillance analytics")
        subtitle.setObjectName("Subtitle")
        brand_layout.addWidget(title)
        brand_layout.addWidget(subtitle)
        brand_layout.addStretch(1)
        header.addWidget(brand, 0, 0, 1, 2)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        sensitivity_box = QHBoxLayout()
        sensitivity_label = QLabel("Sensitivity")
        sensitivity_label.setToolTip("Sensitivity is the anomaly threshold. Lower sensitivity values make alerts easier to trigger; higher values require stronger anomaly scores.")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 99)
        self.sensitivity_slider.setValue(6)
        self.sensitivity_slider.setFixedWidth(150)
        self.sensitivity_slider.valueChanged.connect(self._sensitivity_slider_changed)
        sensitivity_box.addWidget(sensitivity_label)
        sensitivity_box.addWidget(self.sensitivity_slider)
        sensitivity_box.addWidget(self._info_button("Sensitivity", "Sensitivity controls the anomaly decision threshold. Lower values flag more segments as anomaly; higher values are stricter and reduce alerts."))
        controls.addLayout(sensitivity_box)

        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.01, 0.99)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(0.06)
        self.threshold_input.setDecimals(2)
        self.threshold_input.setPrefix("Threshold ")
        self.threshold_input.setMinimumWidth(145)
        self.threshold_input.setToolTip("Numeric anomaly threshold used for final NORMAL/ANOMALY decisions.")
        self.threshold_input.valueChanged.connect(self._threshold_spin_changed)
        controls.addWidget(self.threshold_input)
        controls.addWidget(self._info_button("Threshold", "Threshold is the exact probability cutoff. If anomaly probability is greater than or equal to this number, that segment is marked ANOMALY."))

        self.device_mode = QComboBox()
        self.device_mode.addItems(["Auto", "GPU", "CPU"])
        self.device_mode.currentTextChanged.connect(self._device_mode_changed)
        controls.addWidget(self.device_mode)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self._load_model_only)
        controls.addWidget(self.load_model_button)

        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self._choose_video)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self._start_analysis)
        self.analyze_button.setEnabled(False)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_dashboard)
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self._toggle_camera)
        self.info_button = QPushButton("Info")
        self.info_button.clicked.connect(self._show_info_dialog)
        self.theme_button = QPushButton("Day")
        self.theme_button.setToolTip("Switch between black/white night mode and white/black day mode.")
        self.theme_button.clicked.connect(self._toggle_theme)
        controls.addWidget(self.upload_button)
        controls.addWidget(self.analyze_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.camera_button)
        controls.addWidget(self.info_button)
        controls.addWidget(self.theme_button)
        controls.addStretch(1)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        header.addWidget(controls_widget, 1, 0, 1, 2)
        outer.addLayout(header)

        tabs = QTabWidget()
        outer.addWidget(tabs, 1)
        tabs.addTab(self._build_dashboard_tab(), "Video Analysis")
        tabs.addTab(self._build_live_tab(), "Real Time")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _build_dashboard_tab(self) -> QWidget:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page.verticalScrollBar().setSingleStep(16)
        content = QWidget()
        page.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(14)

        top = QSplitter(Qt.Horizontal)
        layout.addWidget(top)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        self.video_label = QLabel("Upload a video to preview it here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("VideoSurface")
        self.video_label.setMinimumHeight(360)
        self.video_label.setMaximumHeight(520)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label)

        player_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_video_playback)
        self.video_progress = QSlider(Qt.Horizontal)
        self.video_progress.setRange(0, 0)
        self.video_progress.sliderPressed.connect(lambda: setattr(self, "video_slider_dragging", True))
        self.video_progress.sliderReleased.connect(self._seek_video_from_slider)
        self.video_time_label = QLabel("00:00 / 00:00")
        self.video_time_label.setMinimumWidth(110)
        player_row.addWidget(self.play_button)
        player_row.addWidget(self.video_progress, 1)
        player_row.addWidget(self.video_time_label)
        left_layout.addLayout(player_row)

        self.audio_status_label = QLabel("Audio: optional QtMultimedia")
        self.audio_status_label.setObjectName("MutedHelp")
        left_layout.addWidget(self.audio_status_label)
        top.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        self.metric_grid = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        metrics = [
            ("device", "Device"),
            ("status", "Status"),
            ("fps", "Video FPS"),
            ("confidence", "Confidence"),
            ("peak", "Peak Score"),
            ("coverage", "Anomaly Coverage"),
            ("latency", "Processing Time"),
            ("segments", "Anomaly Segments"),
            ("frames", "Frames"),
            ("duration", "Duration"),
        ]
        metric_help = {
            "device": "The active runtime device. GPU means CUDA inference is used; CPU forces local processor inference.",
            "confidence": "The model confidence for the final operational prediction.",
            "peak": "Highest anomaly score found in any timeline segment.",
            "coverage": "Percent of video duration covered by anomaly segments above the sensitivity threshold.",
            "latency": "Total time spent reading, extracting features, and scoring the video.",
            "segments": "Number of timeline windows classified as anomaly.",
        }
        for index, (key, label) in enumerate(metrics):
            card = self._metric_card(label, "--")
            card.setToolTip(metric_help.get(key, label))
            self.metric_labels[key] = card.findChild(QLabel, "MetricValue")
            self.metric_grid.addWidget(card, index // 2, index % 2)
        right_layout.addLayout(self.metric_grid)
        right_layout.addStretch(1)
        top.addWidget(right)
        top.setSizes([760, 560])

        self.workflow_box = self._build_workflow_panel()
        layout.addWidget(self.workflow_box)

        self.terminal_output = QTextEdit()
        self.terminal_output.setObjectName("AnalysisTerminal")
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setMinimumHeight(320)
        self._reset_terminal()
        terminal_group = QGroupBox("Analysis Terminal")
        terminal_layout = QVBoxLayout(terminal_group)
        clear_row = QHBoxLayout()
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setFormat("Model load %p%")
        clear_terminal = QPushButton("Clear")
        clear_terminal.clicked.connect(self._reset_terminal)
        clear_row.addWidget(self.model_progress, 1)
        clear_row.addWidget(clear_terminal)
        terminal_layout.addLayout(clear_row)
        terminal_layout.addWidget(self.terminal_output)
        layout.addWidget(terminal_group)

        graphs = QSplitter(Qt.Horizontal)
        self.timeline_plot = pg.PlotWidget()
        self.timeline_plot.setMinimumHeight(240)
        self.timeline_plot.setBackground("#000000")
        self.timeline_plot.setLabel("left", "Anomaly")
        self.timeline_plot.setLabel("bottom", "Seconds")
        self.timeline_plot.setMouseEnabled(x=True, y=False)
        self.timeline_plot.showGrid(x=True, y=True, alpha=0.2)
        graphs.addWidget(
            self._plot_group(
                "Timeline Graph",
                self.timeline_plot,
                "Shows anomaly probability across video time. Values near 1.0 are stronger anomaly signals; the dashed line is the selected sensitivity threshold.",
            )
        )

        self.worm_plot = pg.PlotWidget()
        self.worm_plot.setMinimumHeight(240)
        self.worm_plot.setBackground("#000000")
        self.worm_plot.setLabel("left", "Trend")
        self.worm_plot.setLabel("bottom", "Segment")
        self.worm_plot.setMouseEnabled(x=True, y=False)
        self.worm_plot.showGrid(x=True, y=True, alpha=0.2)
        graphs.addWidget(
            self._plot_group(
                "Worm Graph",
                self.worm_plot,
                "Shows the smoothed anomaly trend across consecutive timeline segments. It is useful for seeing whether risk is rising, falling, or sustained.",
            )
        )
        graphs.setSizes([640, 640])
        layout.addWidget(graphs)

        bottom = QSplitter(Qt.Horizontal)
        self.frames_list = QListWidget()
        self.frames_list.setObjectName("FrameList")
        self.frames_list.setViewMode(QListWidget.IconMode)
        self.frames_list.setResizeMode(QListWidget.Adjust)
        self.frames_list.setMovement(QListWidget.Static)
        self.frames_list.setSpacing(14)
        self.frames_list.setIconSize(QSize(260, 165))
        self.frames_list.setGridSize(QSize(300, 250))
        self.frames_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.frames_list.setMinimumHeight(480)
        self.frames_list.itemDoubleClicked.connect(self._show_frame_detail)
        bottom.addWidget(self._list_group("Detected Frames", self.frames_list))
        event_panel = QWidget()
        event_layout = QVBoxLayout(event_panel)
        event_layout.setContentsMargins(0, 0, 0, 0)
        self.event_bar = QProgressBar()
        self.event_bar.setObjectName("AnomalyCoverageBar")
        self.event_bar.setRange(0, 100)
        self.event_bar.setFormat("Anomaly coverage %p%")
        self.score_plot = pg.PlotWidget()
        self.score_plot.setMinimumHeight(180)
        self.score_plot.setBackground("#000000")
        self.score_plot.setMouseEnabled(x=False, y=False)
        self.score_plot.showGrid(x=True, y=True, alpha=0.15)
        self.event_plot = pg.PlotWidget()
        self.event_plot.setMinimumHeight(190)
        self.event_plot.setBackground("#000000")
        self.event_plot.setMouseEnabled(x=False, y=False)
        self.event_plot.showGrid(x=True, y=True, alpha=0.15)
        self.events_list = QListWidget()
        self.events_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.events_list.setMinimumHeight(160)
        event_layout.addWidget(self.event_bar)
        event_layout.addWidget(self.score_plot)
        event_layout.addWidget(self.event_plot)
        event_layout.addWidget(self.events_list)
        bottom.addWidget(self._list_group("Event Summary", event_panel))
        bottom.setSizes([640, 640])
        layout.addWidget(bottom)
        return page

    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(14)
        self.live_video_label = QLabel("Start camera for live local detection")
        self.live_video_label.setObjectName("VideoSurface")
        self.live_video_label.setAlignment(Qt.AlignCenter)
        self.live_video_label.setMinimumHeight(560)
        self.live_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.live_video_label, 2)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setMinimumWidth(420)
        side = QWidget()
        side_scroll.setWidget(side)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 8, 0)
        side_layout.setSpacing(10)
        self.live_status = QLabel("Status: idle")
        self.live_status.setObjectName("LiveStatus")
        self.live_score = QLabel("Score: --")
        self.live_score.setObjectName("LiveScore")
        self.live_score_bar = QProgressBar()
        self.live_score_bar.setRange(0, 100)
        self.live_score_bar.setValue(0)
        self.live_score_bar.setFormat("Anomaly %p%")
        self.live_score_bar.setObjectName("LiveScoreBar")
        self.live_fps = QLabel("Camera FPS: --")
        self.live_features = QLabel("Feature history: --")
        self.screen_focus_input = QCheckBox("Screen Focus")
        self.screen_focus_input.setToolTip("Crop to the likely screen/monitor region before inference. Leave off for webcam/CCTV scenes.")
        self.live_reset_button = QPushButton("Reset Live Buffers")
        self.live_reset_button.clicked.connect(self._reset_live_state)
        side_layout.addWidget(self.live_status)
        side_layout.addWidget(self.live_score)
        side_layout.addWidget(self.live_score_bar)

        self.live_metric_grid = QGridLayout()
        live_metrics = [
            ("cam_fps", "Camera FPS"),
            ("ai_fps", "AI FPS"),
            ("latency", "Latency"),
            ("people", "People"),
            ("threshold", "Threshold"),
            ("alerts", "Alerts"),
            ("motion", "Motion"),
            ("model", "Model"),
        ]
        self.live_metric_labels.clear()
        for index, (key, label) in enumerate(live_metrics):
            card = self._metric_card(label, "--")
            card.setMinimumHeight(58)
            card.setMaximumHeight(68)
            self.live_metric_labels[key] = card.findChild(QLabel, "MetricValue")
            self.live_metric_grid.addWidget(card, index // 2, index % 2)
        side_layout.addLayout(self.live_metric_grid)

        side_layout.addWidget(self.screen_focus_input)
        side_layout.addWidget(self.live_reset_button)
        self.live_people = QListWidget()
        self.live_people.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.live_people.setMinimumHeight(130)
        side_layout.addWidget(self._list_group("Tracked People", self.live_people))
        self.live_plot = pg.PlotWidget()
        self.live_plot.setMinimumHeight(230)
        self.live_plot.setBackground("#000000")
        self.live_plot.setYRange(0, 1)
        self.live_plot.setMouseEnabled(x=False, y=False)
        self.live_plot.setLabel("left", "Anomaly")
        self.live_plot.setLabel("bottom", "Recent live scores")
        self.live_plot.showGrid(x=True, y=True, alpha=0.2)
        side_layout.addWidget(
            self._plot_group(
                "Live Score Trend",
                self.live_plot,
                "Realtime anomaly probability. Green points are below threshold; red points are above threshold. The dashed line is the current threshold.",
            ),
            1,
        )
        self.live_events = QListWidget()
        self.live_events.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        side_layout.addWidget(self._list_group("Alert Log", self.live_events), 1)
        self.live_terminal = QTextEdit()
        self.live_terminal.setObjectName("AnalysisTerminal")
        self.live_terminal.setReadOnly(True)
        self.live_terminal.setMinimumHeight(180)
        side_layout.addWidget(self._terminal_group("Live Terminal", self.live_terminal))
        side_layout.addStretch(1)
        layout.addWidget(side_scroll, 1)
        self._reset_live_widgets()
        return page

    def _build_workflow_panel(self) -> QGroupBox:
        group = QGroupBox("Analysis Workflow")
        layout = QVBoxLayout(group)
        header = QHBoxLayout()
        self.analysis_state = QLabel("Waiting for upload")
        self.analysis_state.setObjectName("WorkflowState")
        self.analysis_elapsed = QLabel("00:00.0")
        self.analysis_elapsed.setObjectName("WorkflowTimer")
        header.addWidget(self.analysis_state, 1)
        header.addWidget(QLabel("Elapsed"))
        header.addWidget(self.analysis_elapsed)
        layout.addLayout(header)

        steps = QGridLayout()
        labels = [
            ("queued", "1", "File queued", "Ready to analyze"),
            ("runtime", "2", "Runtime ready", "Load weights and device"),
            ("frames", "3", "Frames ready", "Read FPS and duration"),
            ("features", "4", "Feature extracting", "Build VideoMAE clips"),
            ("scoring", "5", "Scoring", "Run anomaly model"),
            ("completed", "6", "Completed", "Metrics ready"),
        ]
        for column, (key, index, title, detail) in enumerate(labels):
            card = QFrame()
            card.setObjectName("StepPending")
            card_layout = QVBoxLayout(card)
            badge = QLabel(index)
            badge.setObjectName("StepBadge")
            name = QLabel(title)
            name.setObjectName("StepTitle")
            small = QLabel(detail)
            small.setObjectName("StepDetail")
            small.setWordWrap(True)
            card_layout.addWidget(badge)
            card_layout.addWidget(name)
            card_layout.addWidget(small)
            self.workflow_steps[key] = card
            steps.addWidget(card, 0, column)
        layout.addLayout(steps)
        self._set_workflow_step("queued", "Waiting for upload")
        return group

    def _metric_card(self, title: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        label = QLabel(title)
        label.setObjectName("MetricLabel")
        metric = QLabel(value)
        metric.setObjectName("MetricValue")
        metric.setWordWrap(True)
        metric.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label)
        layout.addWidget(metric)
        return card

    def _plot_group(self, title: str, plot: pg.PlotWidget, help_text: str = "") -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("PanelTitle")
        header.addWidget(title_label)
        header.addStretch(1)
        if help_text:
            info = QPushButton("?")
            info.setObjectName("InfoChip")
            info.setFixedSize(28, 28)
            info.clicked.connect(lambda: QMessageBox.information(self, title, help_text))
            header.addWidget(info)
        layout.addLayout(header)
        layout.addWidget(plot)
        return group

    def _info_button(self, title: str, help_text: str) -> QPushButton:
        info = QPushButton("?")
        info.setObjectName("InfoChip")
        info.setFixedSize(28, 28)
        info.clicked.connect(lambda: QMessageBox.information(self, title, help_text))
        return info

    def _list_group(self, title: str, widget: QListWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _terminal_group(self, title: str, widget: QTextEdit) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _apply_theme(self) -> None:
        colors = self._theme_colors()
        self.setStyleSheet(
            f"""
            QWidget {{ background: {colors["bg"]}; color: {colors["text"]}; font-family: Segoe UI; font-size: 13px; }}
            #AppTitle {{ font-size: 28px; font-weight: 800; color: {colors["strong"]}; }}
            #Subtitle, #MutedHelp {{ color: {colors["muted"]}; padding-left: 10px; }}
            QPushButton {{ background: {colors["button_bg"]}; color: {colors["button_text"]}; border: 0; border-radius: 8px; padding: 9px 14px; font-weight: 700; }}
            QPushButton:hover {{ background: {colors["button_hover"]}; }}
            QPushButton:disabled {{ background: {colors["disabled_bg"]}; color: {colors["muted"]}; }}
            #InfoChip {{ background: {colors["panel"]}; color: {colors["text"]}; border: 1px solid {colors["border"]}; border-radius: 14px; padding: 0; }}
            #FrameCard {{ background: {colors["card"]}; border: 1px solid {colors["border"]}; border-radius: 8px; }}
            #FrameCard[prediction="ANOMALY"] {{ border-color: #f4212e; }}
            #FrameCard[prediction="NORMAL"] {{ border-color: #00ba7c; }}
            #FrameImage {{ background: {colors["bg"]}; border-radius: 6px; }}
            #FrameMeta {{ color: {colors["strong"]}; font-weight: 800; }}
            #FrameSubtle {{ color: {colors["muted"]}; }}
            QDoubleSpinBox, QComboBox {{ background: {colors["bg"]}; border: 1px solid {colors["border"]}; border-radius: 8px; padding: 7px; min-width: 96px; color: {colors["text"]}; }}
            QSlider::groove:horizontal {{ height: 5px; background: {colors["border"]}; border-radius: 3px; }}
            QSlider::handle:horizontal {{ width: 18px; margin: -7px 0; border-radius: 9px; background: {colors["strong"]}; }}
            QSlider::sub-page:horizontal {{ background: {colors["strong"]}; border-radius: 3px; }}
            QProgressBar {{ background: {colors["panel"]}; border: 1px solid {colors["border"]}; border-radius: 7px; color: {colors["strong"]}; text-align: center; min-height: 18px; }}
            QProgressBar::chunk {{ background: {colors["strong"]}; border-radius: 6px; }}
            #AnomalyCoverageBar::chunk {{ background: {colors["strong"]}; border-radius: 6px; }}
            #VideoSurface {{ background: {colors["bg"]}; border: 1px solid {colors["border"]}; border-radius: 8px; color: {colors["muted"]}; font-size: 18px; }}
            #MetricCard {{ background: {colors["card"]}; border: 1px solid {colors["border"]}; border-radius: 8px; min-height: 82px; }}
            #MetricLabel {{ color: {colors["muted"]}; font-size: 12px; }}
            #MetricValue {{ color: {colors["strong"]}; font-size: 16px; font-weight: 800; }}
            QGroupBox {{ border: 1px solid {colors["border"]}; border-radius: 8px; margin-top: 12px; padding: 10px; font-weight: 800; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {colors["text"]}; }}
            #PanelTitle {{ color: {colors["strong"]}; font-weight: 800; font-size: 15px; }}
            QTextEdit, QListWidget {{ background: {colors["bg"]}; border: 1px solid {colors["border"]}; border-radius: 8px; color: {colors["text"]}; }}
            #AnalysisTerminal {{ background: {colors["bg"]}; color: {colors["text"]}; font-family: Consolas, Cascadia Mono, Courier New; font-size: 14px; }}
            #WorkflowState {{ color: {colors["text"]}; font-weight: 800; }}
            #WorkflowTimer {{ color: {colors["strong"]}; font-size: 18px; font-weight: 900; }}
            #StepPending, #StepRunning, #StepDone, #StepFailed {{ border: 1px solid {colors["border"]}; border-radius: 8px; padding: 4px; min-height: 106px; }}
            #StepPending {{ background: {colors["card"]}; }}
            #StepRunning {{ background: {colors["panel"]}; border-color: {colors["strong"]}; }}
            #StepDone {{ background: #07170d; border-color: #00ba7c; }}
            #StepFailed {{ background: #210b0e; border-color: #f4212e; }}
            #StepBadge {{ background: {colors["bg"]}; border: 1px solid {colors["border"]}; border-radius: 15px; min-width: 30px; max-width: 30px; min-height: 30px; qproperty-alignment: AlignCenter; font-weight: 900; color: {colors["strong"]}; }}
            #StepTitle {{ font-weight: 900; color: {colors["strong"]}; }}
            #StepDetail {{ color: {colors["muted"]}; font-size: 11px; }}
            QTabWidget::pane {{ border: 0; }}
            QTabBar::tab {{ background: {colors["bg"]}; color: {colors["muted"]}; padding: 9px 18px; border: 1px solid {colors["border"]}; border-bottom: 0; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
            QTabBar::tab:selected {{ background: {colors["panel"]}; color: {colors["strong"]}; }}
            #LiveStatus {{ font-size: 24px; font-weight: 800; }}
            #LiveScore {{ font-size: 22px; font-weight: 800; color: {colors["strong"]}; }}
            QScrollBar:vertical {{ background: {colors["bg"]}; width: 12px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {colors["border"]}; border-radius: 6px; min-height: 36px; }}
            QScrollBar::handle:vertical:hover {{ background: {colors["muted"]}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{ background: {colors["bg"]}; height: 12px; margin: 0; }}
            QScrollBar::handle:horizontal {{ background: {colors["border"]}; border-radius: 6px; min-width: 36px; }}
            QScrollBar::handle:horizontal:hover {{ background: {colors["muted"]}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
            """
        )
        if hasattr(self, "theme_button"):
            self.theme_button.setText("Night" if self.theme_mode == "day" else "Day")
        self._apply_plot_theme()

    def _theme_colors(self) -> dict[str, str]:
        if self.theme_mode == "day":
            return {
                "bg": "#ffffff",
                "card": "#f7f9f9",
                "panel": "#eff3f4",
                "border": "#d7dbdc",
                "text": "#0f1419",
                "strong": "#000000",
                "muted": "#536471",
                "button_bg": "#0f1419",
                "button_text": "#ffffff",
                "button_hover": "#272c30",
                "disabled_bg": "#d7dbdc",
            }
        return {
            "bg": "#000000",
            "card": "#080808",
            "panel": "#16181c",
            "border": "#2f3336",
            "text": "#e7e9ea",
            "strong": "#ffffff",
            "muted": "#71767b",
            "button_bg": "#eff3f4",
            "button_text": "#0f1419",
            "button_hover": "#d7dbdc",
            "disabled_bg": "#202327",
        }

    def _toggle_theme(self) -> None:
        self.theme_mode = "day" if self.theme_mode == "night" else "night"
        self._apply_theme()
        if self.timeline_data:
            self._draw_timeline(self.timeline_data, float(self.threshold_input.value()))
            self._draw_worm(self.timeline_data)
        self._update_live_plot()
        self.status.showMessage(f"{self.theme_mode.title()} theme enabled")

    def _apply_plot_theme(self) -> None:
        colors = self._theme_colors()
        for attr in ("timeline_plot", "worm_plot", "score_plot", "event_plot", "live_plot"):
            plot = getattr(self, attr, None)
            if plot is None:
                continue
            plot.setBackground(colors["bg"])
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(pg.mkPen(colors["border"]))
                axis.setTextPen(pg.mkPen(colors["muted"]))

    def _plot_line_color(self) -> str:
        return "#000000" if self.theme_mode == "day" else "#ffffff"

    def _plot_fill_brush(self) -> tuple[int, int, int, int]:
        return (0, 0, 0, 22) if self.theme_mode == "day" else (255, 255, 255, 28)

    def _refresh_device_status(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                device = f"GPU: {torch.cuda.get_device_name(0)}"
            else:
                device = "CPU"
        except Exception as exc:
            device = f"Unknown ({exc})"
        self._set_metric("device", device)

    def _sensitivity_slider_changed(self, value: int) -> None:
        target = value / 100.0
        if abs(self.threshold_input.value() - target) > 0.001:
            self.threshold_input.blockSignals(True)
            self.threshold_input.setValue(target)
            self.threshold_input.blockSignals(False)

    def _threshold_spin_changed(self, value: float) -> None:
        target = int(round(value * 100))
        if self.sensitivity_slider.value() != target:
            self.sensitivity_slider.blockSignals(True)
            self.sensitivity_slider.setValue(target)
            self.sensitivity_slider.blockSignals(False)
        if hasattr(self, "live_metric_labels"):
            self._set_live_metric("threshold", f"{value:.2f}")
            self._update_live_plot()

    def _device_mode_changed(self) -> None:
        preference = self.device_mode.currentText().lower()
        if preference == "auto":
            preference = "auto"
        elif preference == "gpu":
            preference = "cuda"
        else:
            preference = "cpu"

        self.base_config = replace(self.base_config, device_preference=preference)
        self.service = ViolenceDetectionService(project_root=self.project_root, config=self.base_config)
        self.runtime_worker = None
        self.analysis_worker = None
        self.model_progress.setValue(0)
        self._append_terminal(f"[runtime] device mode changed to {self.device_mode.currentText()}")
        self._refresh_device_status()

    def _reset_terminal(self) -> None:
        self.terminal_started_at = time.perf_counter()
        text = "\n".join(
            [
                "Windows PowerShell",
                "Copyright (C) Microsoft Corporation. All rights reserved.",
                "",
                "PS avt> waiting for analysis...",
            ]
        )
        if hasattr(self, "terminal_output"):
            self.terminal_output.setPlainText(text)

    def _reset_dashboard(self) -> None:
        self.service.reset()
        self.timeline_data.clear()
        self.score_history.clear()
        self.model_progress.setValue(0)
        self.analysis_elapsed.setText("00:00.0")
        self._set_workflow_step("queued", "Waiting for upload")
        self._reset_terminal()
        for key in ("status", "fps", "confidence", "peak", "coverage", "latency", "segments", "frames", "duration"):
            self._set_metric(key, "--")
        self.timeline_plot.clear()
        self.worm_plot.clear()
        self.event_plot.clear()
        self.score_plot.clear()
        self.event_bar.setValue(0)
        self.frames_list.clear()
        self.events_list.clear()
        self.status.showMessage("Reset complete")

    def _terminal_time(self) -> str:
        if self.terminal_started_at is None:
            return "00:00.000"
        elapsed = max(0.0, time.perf_counter() - self.terminal_started_at)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        millis = int((elapsed - int(elapsed)) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _append_terminal(self, message: str, level: str = "info") -> None:
        if not hasattr(self, "terminal_output"):
            return
        prefix = "ERR" if level == "error" else "OK " if level == "complete" else "   "
        self.terminal_output.append(f"[{self._terminal_time()}] {prefix} {message}")
        self.terminal_output.verticalScrollBar().setValue(self.terminal_output.verticalScrollBar().maximum())

    def _reset_live_terminal(self) -> None:
        self.live_terminal_started_at = time.perf_counter()
        self.last_live_terminal_status = ""
        if hasattr(self, "live_terminal"):
            self.live_terminal.setPlainText("PS avt-live> waiting for camera and loaded model...")

    def _live_terminal_time(self) -> str:
        if self.live_terminal_started_at is None:
            return "00:00.000"
        elapsed = max(0.0, time.perf_counter() - self.live_terminal_started_at)
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}.{int((elapsed - int(elapsed)) * 1000):03d}"

    def _append_live_terminal(self, message: str, level: str = "info") -> None:
        if not hasattr(self, "live_terminal"):
            return
        prefix = "ERR" if level == "error" else "OK " if level == "complete" else "   "
        self.live_terminal.append(f"[{self._live_terminal_time()}] {prefix} {message}")
        self.live_terminal.verticalScrollBar().setValue(self.live_terminal.verticalScrollBar().maximum())

    def _set_workflow_step(self, step: str, detail: str = "", failed: bool = False) -> None:
        if step in self.workflow_order:
            self.workflow_index = self.workflow_order.index(step)
        self.analysis_state.setText(detail or step.replace("_", " ").title())
        for key, card in self.workflow_steps.items():
            index = self.workflow_order.index(key)
            if failed and index == self.workflow_index:
                card.setObjectName("StepFailed")
            elif index < self.workflow_index or step == "completed":
                card.setObjectName("StepDone")
            elif index == self.workflow_index:
                card.setObjectName("StepRunning")
            else:
                card.setObjectName("StepPending")
            card.style().unpolish(card)
            card.style().polish(card)

    def _sync_workflow_from_message(self, message: str) -> None:
        progress = self._model_progress_from_message(message)
        if progress is not None:
            self.model_progress.setValue(max(self.model_progress.value(), progress))
        if message.startswith("[runtime]") or message.startswith("[weights]"):
            self._set_workflow_step("runtime", message.replace("[runtime]", "").replace("[weights]", "").strip())
        elif message.startswith("[video]") or message.startswith("[frames]"):
            self._set_workflow_step("frames", message.split("]", 1)[-1].strip())
        elif message.startswith("[clips]") or message.startswith("[features]") or message.startswith("[videomae]"):
            self._set_workflow_step("features", message.split("]", 1)[-1].strip())
        elif message.startswith("[model]") or message.startswith("[timeline]") or message.startswith("[calc]"):
            self._set_workflow_step("scoring", message.split("]", 1)[-1].strip())
        elif message.startswith("[done]"):
            self.model_progress.setValue(100)
            self._set_workflow_step("completed", "Analysis completed")

    def _model_progress_from_message(self, message: str) -> int | None:
        mapping = [
            ("[runtime] importing", 5),
            ("[runtime] device", 12),
            ("[model] building", 20),
            ("[weights] loading", 32),
            ("[weights] loaded", 48),
            ("[videomae] loading processor", 62),
            ("[videomae] loading model", 82),
            ("[runtime] ready", 100),
            ("[runtime] cached", 100),
        ]
        for prefix, value in mapping:
            if message.startswith(prefix):
                return value
        return None

    def _tick_analysis_timer(self) -> None:
        if self.analysis_started_at is None:
            return
        elapsed = time.perf_counter() - self.analysis_started_at
        mins = int(elapsed // 60)
        secs = elapsed % 60
        self.analysis_elapsed.setText(f"{mins:02d}:{secs:04.1f}")

    def _pulse_model_progress(self) -> None:
        value = self.model_progress.value()
        if 0 < value < 95:
            self.model_progress.setValue(value + 1)

    @Slot()
    def _load_model_only(self) -> None:
        self.load_model_button.setEnabled(False)
        self._reset_terminal()
        self._append_terminal("$ avt load-model")
        self._set_workflow_step("runtime", "Loading model runtime")
        self.model_progress.setValue(1)
        self.model_progress_timer.start(650)

        self.runtime_thread = QThread(self)
        self.runtime_worker = RuntimeLoadWorker(self.service)
        self.runtime_worker.moveToThread(self.runtime_thread)
        self.runtime_thread.started.connect(self.runtime_worker.run)
        self.runtime_worker.progress.connect(self._append_log)
        self.runtime_worker.finished.connect(self._runtime_loaded)
        self.runtime_worker.failed.connect(self._runtime_failed)
        self.runtime_worker.finished.connect(self.runtime_thread.quit)
        self.runtime_worker.failed.connect(self.runtime_thread.quit)
        self.runtime_thread.finished.connect(self.runtime_worker.deleteLater)
        self.runtime_thread.finished.connect(self.runtime_thread.deleteLater)
        self.runtime_thread.finished.connect(lambda: setattr(self, "runtime_thread", None))
        self.runtime_thread.start()

    @Slot(dict)
    def _runtime_loaded(self, health: dict[str, Any]) -> None:
        self.load_model_button.setEnabled(True)
        self.model_progress_timer.stop()
        self.model_progress.setValue(100)
        self._append_terminal("[runtime] ready", "complete")
        self._append_live_terminal(f"model ready on {health.get('device') or 'device'}", "complete")
        self._set_metric("device", str(health.get("device") or "ready"))
        self.status.showMessage("Model runtime loaded")

    @Slot(str)
    def _runtime_failed(self, error: str) -> None:
        self.load_model_button.setEnabled(True)
        self.model_progress_timer.stop()
        self._append_terminal(error, "error")
        self._set_workflow_step("runtime", error, failed=True)
        QMessageBox.critical(self, "Model load failed", error)

    @Slot()
    def _choose_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose video",
            str(self.project_root),
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if not filename:
            return
        self.video_path = Path(filename)
        self.analyze_button.setEnabled(True)
        self.status.showMessage(f"Loaded {self.video_path.name}")
        self._reset_terminal()
        self._append_terminal(f"file queued: {self.video_path.name}")
        self._set_workflow_step("queued", f"{self.video_path.name} queued")
        self._open_video_preview(self.video_path)

    def _open_video_preview(self, video_path: Path) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = cv2.VideoCapture(str(video_path))
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_progress.setRange(0, max(0, self.video_frame_count - 1))
        self.video_progress.setValue(0)
        self.video_playing = True
        self.play_button.setText("Pause")
        self._update_video_time(0)
        self._setup_audio(video_path)
        self.video_timer.start(33)

    def _render_next_video_frame(self) -> None:
        if self.video_capture is None or not self.video_playing:
            return
        ok, frame = self.video_capture.read()
        if not ok:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_progress.setValue(0)
            self._seek_audio(0)
            return
        frame_index = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if not self.video_slider_dragging:
            self.video_progress.setValue(max(0, frame_index - 1))
            self._update_video_time(frame_index - 1)
        self._set_image(self.video_label, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _toggle_video_playback(self) -> None:
        self.video_playing = not self.video_playing
        self.play_button.setText("Pause" if self.video_playing else "Play")
        if self.media_player is not None:
            if self.video_playing:
                self.media_player.play()
            else:
                self.media_player.pause()

    def _seek_video_from_slider(self) -> None:
        self.video_slider_dragging = False
        frame_index = self.video_progress.value()
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._seek_audio(frame_index)
        self._update_video_time(frame_index)

    def _update_video_time(self, frame_index: int) -> None:
        current = frame_index / max(self.video_fps, 1.0)
        total = self.video_frame_count / max(self.video_fps, 1.0) if self.video_frame_count else 0.0
        self.video_time_label.setText(f"{_format_clock(current)} / {_format_clock(total)}")

    def _setup_audio(self, video_path: Path) -> None:
        if QMediaPlayer is None or QAudioOutput is None:
            self.audio_status_label.setText("Audio unavailable: install PySide6-Addons for sound playback")
            return
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.70)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.media_player.play()
        self.audio_status_label.setText("Audio: on")

    def _seek_audio(self, frame_index: int) -> None:
        if self.media_player is not None:
            self.media_player.setPosition(int((frame_index / max(self.video_fps, 1.0)) * 1000))

    @Slot()
    def _start_analysis(self) -> None:
        if not self.video_path:
            return
        self.analyze_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.model_progress.setValue(max(self.model_progress.value(), 1))
        self.model_progress_timer.start(650)
        self.analysis_started_at = time.perf_counter()
        self.analysis_timer.start(100)
        self._set_metric("status", "Analyzing")
        self._set_workflow_step("runtime", "Loading runtime")
        self.status.showMessage("Analyzing video locally...")
        self._append_terminal("$ desktop analysis started")

        self.analysis_thread = QThread(self)
        self.analysis_worker = AnalysisWorker(self.service, self.video_path, float(self.threshold_input.value()))
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self._append_log)
        self.analysis_worker.finished.connect(self._analysis_finished)
        self.analysis_worker.failed.connect(self._analysis_failed)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        self.analysis_thread.finished.connect(lambda: setattr(self, "analysis_thread", None))
        self.analysis_thread.start()

    @Slot(str)
    def _append_log(self, message: str) -> None:
        self._append_terminal(message)
        self._sync_workflow_from_message(message)

    @Slot(dict)
    def _analysis_finished(self, result: dict[str, Any]) -> None:
        self.analyze_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.analysis_timer.stop()
        self.model_progress_timer.stop()
        self.model_progress.setValue(100)
        self._populate_result(result)
        prediction = result.get("operational", {}).get("prediction", "DONE")
        self._append_terminal("analysis complete", "complete")
        self._set_workflow_step("completed", f"Analysis complete: {prediction}")
        self.status.showMessage(f"Analysis complete: {prediction}")

    @Slot(str)
    def _analysis_failed(self, error: str) -> None:
        self.analyze_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.analysis_timer.stop()
        self.model_progress_timer.stop()
        self._set_metric("status", "Error")
        self._append_terminal(error, "error")
        self._set_workflow_step(self.workflow_order[self.workflow_index], error, failed=True)
        self.status.showMessage("Analysis failed")
        QMessageBox.critical(self, "Analysis failed", error)

    def _populate_result(self, result: dict[str, Any]) -> None:
        metrics = result.get("metrics", {})
        operational = result.get("operational", {})
        self.timeline_data = list(result.get("timeline") or [])
        self._set_metric("status", operational.get("prediction", "--"))
        self._set_metric("fps", f"{metrics.get('fps', 0):.2f}")
        self._set_metric("confidence", _percent(operational.get("confidence")))
        self._set_metric("peak", _percent(metrics.get("peak_score")))
        self._set_metric("coverage", _percent(metrics.get("anomaly_coverage")))
        self._set_metric("latency", f"{metrics.get('processing_seconds', 0):.2f}s")
        self._set_metric("segments", str(metrics.get("anomaly_segments", 0)))
        self._set_metric("frames", str(metrics.get("frames", 0)))
        self._set_metric("duration", f"{metrics.get('duration_seconds', 0):.2f}s")

        self._draw_timeline(self.timeline_data, float(result.get("threshold", self.threshold_input.value())))
        self._draw_worm(self.timeline_data)
        self._populate_frames(result.get("frame_samples") or [])
        self._populate_events(result)

    def _draw_timeline(self, timeline: list[dict[str, Any]], threshold: float) -> None:
        self.timeline_plot.clear()
        if not timeline:
            return
        x = [(float(item["start"]) + float(item["end"])) / 2 for item in timeline]
        y = [float(item["prob_anomaly"]) for item in timeline]
        duration = max(float(timeline[-1].get("end", max(x, default=1.0))), 1.0)
        normal_x = [point for point, score in zip(x, y) if score < threshold]
        normal_y = [score for score in y if score < threshold]
        anomaly_x = [point for point, score in zip(x, y) if score >= threshold]
        anomaly_y = [score for score in y if score >= threshold]
        self.timeline_plot.plot(x, y, pen=pg.mkPen(self._plot_line_color(), width=2))
        if normal_x:
            self.timeline_plot.plot(normal_x, normal_y, pen=None, symbol="o", symbolBrush="#00ba7c", symbolPen="#00ba7c", symbolSize=8)
        if anomaly_x:
            self.timeline_plot.plot(anomaly_x, anomaly_y, pen=None, symbol="o", symbolBrush="#f4212e", symbolPen="#f4212e", symbolSize=8)
        self.timeline_plot.addLine(y=threshold, pen=pg.mkPen("#71767b", width=1, style=Qt.DashLine))
        self.timeline_plot.setYRange(0, 1.0, padding=0)
        self.timeline_plot.setXRange(0, duration, padding=0.02)
        self.timeline_plot.setLimits(xMin=0, xMax=duration, yMin=0, yMax=1.0, minYRange=0.2, maxYRange=1.0)

    def _draw_worm(self, timeline: list[dict[str, Any]]) -> None:
        self.worm_plot.clear()
        if not timeline:
            return
        raw = [float(item["prob_anomaly"]) for item in timeline]
        y = _moving_average(raw, window=4)
        x = list(range(1, len(y) + 1))
        threshold = float(self.threshold_input.value())
        self.worm_plot.plot(x, y, pen=pg.mkPen(self._plot_line_color(), width=2), fillLevel=0, brush=self._plot_fill_brush())
        normal_x = [point for point, score in zip(x, raw) if score < threshold]
        normal_y = [score for score, raw_score in zip(y, raw) if raw_score < threshold]
        anomaly_x = [point for point, score in zip(x, raw) if score >= threshold]
        anomaly_y = [score for score, raw_score in zip(y, raw) if raw_score >= threshold]
        if normal_x:
            self.worm_plot.plot(normal_x, normal_y, pen=None, symbol="o", symbolBrush="#00ba7c", symbolPen="#00ba7c", symbolSize=7)
        if anomaly_x:
            self.worm_plot.plot(anomaly_x, anomaly_y, pen=None, symbol="o", symbolBrush="#f4212e", symbolPen="#f4212e", symbolSize=7)
        self.worm_plot.addLine(y=threshold, pen=pg.mkPen("#71767b", width=1, style=Qt.DashLine))
        max_x = max(1, len(y))
        self.worm_plot.setYRange(0, 1.0, padding=0)
        self.worm_plot.setXRange(1, max_x, padding=0.02)
        self.worm_plot.setLimits(xMin=1, xMax=max_x, yMin=0, yMax=1.0, minYRange=0.2, maxYRange=1.0)

    def _populate_frames(self, samples: list[dict[str, Any]]) -> None:
        self.frames_list.clear()
        for sample in samples:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, sample)
            item.setSizeHint(QSize(300, 255))
            self.frames_list.addItem(item)
            self.frames_list.setItemWidget(item, self._frame_card(sample))

    def _frame_card(self, sample: dict[str, Any]) -> QWidget:
        score = float(sample.get("score", 0.0))
        prediction = str(sample.get("prediction", "NORMAL"))
        card = QFrame()
        card.setObjectName("FrameCard")
        card.setProperty("prediction", prediction)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        image = QLabel()
        image.setObjectName("FrameImage")
        image.setAlignment(Qt.AlignCenter)
        image.setMinimumSize(260, 150)
        pixmap = _pixmap_from_data_url(str(sample.get("image", "")))
        if pixmap:
            image.setPixmap(pixmap.scaled(260, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        meta = QLabel(f"{sample.get('time', 0):.2f}s  |  {prediction}  |  {score * 100:.1f}%")
        meta.setObjectName("FrameMeta")
        meta.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Double-click to inspect")
        subtitle.setObjectName("FrameSubtle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(image)
        layout.addWidget(meta)
        layout.addWidget(subtitle)
        return card

    def _populate_events(self, result: dict[str, Any]) -> None:
        self.events_list.clear()
        metrics = result.get("metrics", {})
        coverage = float(metrics.get("anomaly_coverage") or 0.0)
        self.event_bar.setValue(int(round(coverage * 100)))
        self._draw_score_summary(result)
        self._draw_event_summary(result)
        events = [
            f"Prediction: {result.get('operational', {}).get('prediction', '--')}",
            f"Peak score: {_percent(metrics.get('peak_score'))}",
            f"Average score: {_percent(metrics.get('average_score'))}",
            f"Anomaly seconds: {metrics.get('anomaly_seconds', 0):.2f}s",
            f"Feature vectors: {metrics.get('features', 0)} x {metrics.get('feature_dim', 0)}",
            f"Clips scored: {metrics.get('clips', 0)}",
        ]
        for segment in result.get("anomaly_segments") or []:
            events.append(f"Anomaly {segment['start']:.2f}s-{segment['end']:.2f}s score={segment['prob_anomaly']:.3f}")
        for text in events:
            self.events_list.addItem(text)
        self.events_list.addItem("Suggestions")
        for suggestion in self._suggestions_for_result(result):
            self.events_list.addItem(f"- {suggestion}")

    def _suggestions_for_result(self, result: dict[str, Any]) -> list[str]:
        metrics = result.get("metrics", {})
        coverage = float(metrics.get("anomaly_coverage") or 0.0)
        processing = float(metrics.get("processing_seconds") or 0.0)
        suggestions = [
            "Calibrate the sensitivity threshold on a validation set before using the app operationally.",
            "Export ONNX/TensorRT later for lower latency and smaller desktop builds.",
        ]
        if coverage > 0.65:
            suggestions.append("High anomaly coverage: review threshold and sample false positives on normal clips.")
        if processing > 10:
            suggestions.append("For speed, cache VideoMAE features for uploaded videos or process fewer overlapping clips.")
        suggestions.append("For UI polish, add report export with the frame gallery, timeline, and final score chart.")
        return suggestions

    def _draw_event_summary(self, result: dict[str, Any]) -> None:
        self.event_plot.clear()
        timeline = result.get("timeline") or []
        if not timeline:
            return
        x = list(range(len(timeline)))
        scores = [float(item.get("prob_anomaly", 0.0)) for item in timeline]
        threshold = float(result.get("threshold", self.threshold_input.value()))
        brushes = [_status_brush(score, threshold, alpha=150) for score in scores]
        bar = pg.BarGraphItem(x=x, height=scores, width=0.46, brushes=brushes)
        self.event_plot.addItem(bar)
        self.event_plot.plot(x, scores, pen=pg.mkPen(self._plot_line_color(), width=1))
        self.event_plot.addLine(y=threshold, pen=pg.mkPen("#71767b", width=1, style=Qt.DashLine))
        self.event_plot.setYRange(0, 1, padding=0)
        self.event_plot.setXRange(0, max(1, len(timeline) - 1), padding=0.02)
        self.event_plot.setLimits(xMin=0, xMax=max(1, len(timeline) - 1), yMin=0, yMax=1, minYRange=0.2, maxYRange=1)

    def _draw_score_summary(self, result: dict[str, Any]) -> None:
        self.score_plot.clear()
        metrics = result.get("metrics", {})
        operational = result.get("operational", {}) or result.get("overall", {})
        values = [
            float(operational.get("prob_anomaly", 0.0)),
            float(metrics.get("peak_score") or 0.0),
            float(metrics.get("average_score") or 0.0),
        ]
        labels = ["Prediction", "Peak", "Average"]
        threshold = float(result.get("threshold", self.threshold_input.value()))
        brushes = [_status_brush(value, threshold, alpha=180) for value in values]
        self.score_plot.addItem(pg.BarGraphItem(x=[0, 1, 2], height=values, width=0.28, brushes=brushes))
        axis = self.score_plot.getAxis("bottom")
        axis.setTicks([list(enumerate(labels))])
        for index, value in enumerate(values):
            text = pg.TextItem(f"{value * 100:.1f}%", color=self._plot_line_color(), anchor=(0.5, 1.2))
            text.setPos(index, value)
            self.score_plot.addItem(text)
        self.score_plot.addLine(y=threshold, pen=pg.mkPen("#71767b", width=1, style=Qt.DashLine))
        self.score_plot.setYRange(0, 1, padding=0)
        self.score_plot.setXRange(-0.6, 2.6, padding=0)
        self.score_plot.setLimits(xMin=-0.6, xMax=2.6, yMin=0, yMax=1, minYRange=0.2, maxYRange=1)

    def _show_frame_detail(self, item: QListWidgetItem) -> None:
        sample = item.data(Qt.UserRole) or {}
        pixmap = _pixmap_from_data_url(str(sample.get("image", "")))
        dialog = QDialog(self)
        dialog.setWindowTitle("Detected Frame")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        if pixmap:
            image.setPixmap(pixmap.scaled(720, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image, 1)
        score = float(sample.get("score", 0.0))
        layout.addWidget(QLabel(f"Time: {sample.get('time', 0):.2f}s    Prediction: {sample.get('prediction', '--')}    Score: {score * 100:.1f}%"))
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    @Slot()
    def _toggle_camera(self) -> None:
        if self.camera_thread is not None:
            self.camera_button.setEnabled(False)
            if self.camera_worker:
                self.camera_worker.stop()
            return
        if getattr(self.service, "_runtime", None) is None:
            self._reset_live_terminal()
            self._append_live_terminal("load the model first; live anomaly scoring uses the trained VideoMAE + Transformer model", "error")
            self.live_status.setText("Status: load model first")
            self.live_score.setText("Score: model required")
            self.live_events.clear()
            self.live_events.addItem("Click Load Model before starting live anomaly detection.")
            QMessageBox.information(
                self,
                "Load model first",
                "Live anomaly detection uses the trained VideoMAE + Transformer model. Click Load Model first, then start the camera.",
            )
            return
        self.camera_button.setText("Stop Camera")
        self.score_history.clear()
        self.live_events.clear()
        self.live_events.addItem("Warming up trained live model...")
        self.live_status.setText("Status: starting")
        self.live_score.setText("Score: --")
        self._set_live_metric("threshold", f"{float(self.threshold_input.value()):.2f}")
        self._reset_live_terminal()
        self._append_live_terminal("$ avt-live camera --model videomae-transformer")
        self._append_live_terminal("trained model is loaded; collecting sliding window frames")
        self._append_terminal("[runtime] live camera starting")
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(
            self.service,
            float(self.threshold_input.value()),
            focus_screen=self.screen_focus_input.isChecked(),
        )
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(lambda frame: self._set_image(self.live_video_label, frame))
        self.camera_worker.result_ready.connect(self._live_result)
        self.camera_worker.failed.connect(self._live_failed)
        self.camera_worker.stopped.connect(self._camera_stopped)
        self.camera_worker.stopped.connect(self.camera_thread.quit)
        self.camera_thread.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()

    def _show_info_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Model & Training Info")
        dialog.resize(1220, 840)
        layout = QVBoxLayout(dialog)

        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        overview = QWidget()
        overview_layout = QGridLayout(overview)
        info_sections = [
            (
                "Model Used",
                [
                    ("Feature extractor", "VideoMAE / ViT-Base features, 768 dimensions"),
                    ("Classifier", "AnomalyTransformer temporal encoder"),
                    ("Architecture", "768 -> 512 projection, 4 Transformer layers, 8 heads, FF 1024, dropout 0.3"),
                    ("Runtime checkpoint", "artifacts/checkpoints/best_model.pt"),
                ],
            ),
            (
                "Training Setup",
                [
                    ("Epoch budget", "50 epochs with early stopping patience 10"),
                    ("Best epoch", "Epoch 21, validation AUC 93.25%, validation accuracy 85.16%"),
                    ("Optimizer", "AdamW, learning rate 3e-4, weight decay 1e-4"),
                    ("Loss", "Weighted CrossEntropy with label smoothing 0.1"),
                ],
            ),
            (
                "Data Used",
                [
                    ("Dataset", "UCF-Crime VideoMAE feature dataset"),
                    ("Feature files", "1,890 .npy files"),
                    ("Split", "70% train, 15% validation, 15% test"),
                    ("Train / Val / Test", "1,323 / 283 / 284 videos"),
                ],
            ),
            (
                "Notebook Result",
                [
                    ("Test Accuracy", "92.25%"),
                    ("Test ROC AUC", "94.96%"),
                    ("Average Precision", "94.49%"),
                    ("Best Val AUC", "93.25%"),
                ],
            ),
        ]
        for index, (title, rows) in enumerate(info_sections):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            for label, value in rows:
                row = QLabel(f"<b>{label}</b><br>{value}")
                row.setWordWrap(True)
                box_layout.addWidget(row)
            overview_layout.addWidget(box, index // 2, index % 2)
        tabs.addTab(overview, "Overview")

        categories = QListWidget()
        for text in [
            "Normal 940",
            "RoadAccidents 150",
            "Robbery 150",
            "Burglary 100",
            "Stealing 100",
            "Abuse 50",
            "Arrest 50",
            "Assault 50",
            "Explosion 50",
            "Fighting 50",
            "Shooting 50",
            "Shoplifting 50",
            "Vandalism 50",
            "Arson 49",
        ]:
            categories.addItem(text)
        tabs.addTab(categories, "Categories")

        screenshot_scroll = QScrollArea()
        screenshot_scroll.setWidgetResizable(True)
        screenshot_scroll.setFrameShape(QFrame.NoFrame)
        screenshots = QWidget()
        screenshot_scroll.setWidget(screenshots)
        screenshot_layout = QVBoxLayout(screenshots)
        screenshot_layout.setContentsMargins(10, 10, 10, 10)
        screenshot_layout.setSpacing(18)
        images = [
            ("Training progress", self.project_root / "docs" / "screenshots" / "training-notebook" / "training-progress.png"),
            ("Test evaluation", self.project_root / "docs" / "screenshots" / "training-notebook" / "test-set-evaluation.png"),
            ("Per-category accuracy", self.project_root / "docs" / "screenshots" / "training-notebook" / "per-category-accuracy.png"),
        ]
        for title, path in images:
            card = QGroupBox(title)
            card_layout = QVBoxLayout(card)
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(960, 360)
            if path.exists():
                image_label.setPixmap(QPixmap(str(path)).scaled(980, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                image_label.setText("Screenshot missing")
            image_label.setToolTip(str(path))
            card_layout.addWidget(image_label)
            screenshot_layout.addWidget(card)
        screenshot_layout.addStretch(1)
        tabs.addTab(screenshot_scroll, "Screenshots")

        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    @Slot(dict)
    def _live_result(self, payload: dict[str, Any]) -> None:
        result = payload.get("result") or payload
        score = float(result.get("prob_anomaly", 0.0))
        prediction = result.get("prediction") or payload.get("status", "warming")
        status = payload.get("status", "scored")
        if status == "warming":
            needed = int(payload.get("needed_frames", 0))
            self.live_status.setText(f"Status: model warming ({needed} frames)")
        elif status == "model_required":
            self.live_status.setText("Status: load model first")
        else:
            self.live_status.setText(f"Status: {prediction}")
        self.live_score.setText(f"Score: {score * 100:.1f}%")
        self.live_score_bar.setValue(int(round(score * 100)))
        self._set_live_score_bar_color(prediction == "ANOMALY")
        self.live_fps.setText(f"Camera FPS: {payload.get('camera_fps', 0):.1f}")
        self.live_features.setText(f"Feature history: {payload.get('feature_count', '--')}")
        self._set_live_metric("cam_fps", f"{float(payload.get('camera_fps', 0.0)):.1f}")
        self._set_live_metric("ai_fps", f"{float(payload.get('processed_fps', 0.0)):.1f}")
        self._set_live_metric("latency", f"{float(payload.get('latency_ms', 0.0)):.0f} ms")
        self._set_live_metric("people", str(payload.get("person_count", 0)))
        self._set_live_metric("threshold", f"{float(self.threshold_input.value()):.2f}")
        self._set_live_metric("alerts", str(len(payload.get("events") or [])))
        self._set_live_metric("motion", f"{float(payload.get('motion_score', 0.0)) * 100:.0f}%")
        self._set_live_metric("model", str(payload.get("model_status", "waiting")))
        self._update_live_people(payload.get("people") or [])
        terminal_status = (
            f"{status}|{prediction}|{float(score):.3f}|{payload.get('model_status')}|"
            f"{payload.get('person_count')}|{payload.get('activity_regions')}"
        )
        if terminal_status != self.last_live_terminal_status:
            self.last_live_terminal_status = terminal_status
            self._append_live_terminal(
                f"model={payload.get('model_status')} status={status} prediction={prediction} "
                f"score={score * 100:.1f}% fast={float(payload.get('fast_score', 0.0)) * 100:.1f}% "
                f"people={payload.get('person_count', 0)} motion={float(payload.get('motion_score', 0.0)) * 100:.0f}%"
            )
        self.score_history = (self.score_history + [score])[-120:]
        self._update_live_plot()
        self.live_events.clear()
        events = payload.get("events") or []
        if not events:
            self.live_events.addItem("No alerts")
        for event in events:
            self.live_events.addItem(
                f"{event.get('time', '--')}  ANOMALY {float(event.get('probability', 0)) * 100:.1f}%  "
                f"confidence {float(event.get('confidence', 0)) * 100:.1f}%"
            )

    def _live_failed(self, error: str) -> None:
        self.status.showMessage(error)
        self.live_status.setText("Status: error")
        self._append_live_terminal(error, "error")
        self._append_terminal(f"[runtime] live error={error}")

    @Slot()
    def _reset_live_state(self) -> None:
        self.service.reset()
        self.score_history.clear()
        self._reset_live_widgets()
        self._append_live_terminal("live buffers reset")
        self._append_terminal("[runtime] live buffers reset")

    @Slot()
    def _camera_stopped(self) -> None:
        self.camera_thread = None
        self.camera_worker = None
        self.camera_button.setEnabled(True)
        self.camera_button.setText("Start Camera")
        self.live_status.setText("Status: idle")
        self._append_live_terminal("live camera stopped", "complete")
        self._append_terminal("[runtime] live camera stopped")

    def _reset_live_widgets(self) -> None:
        self.live_plot.clear()
        self.live_events.clear()
        self.live_events.addItem("No alerts")
        self.live_status.setText("Status: idle")
        self.live_score.setText("Score: --")
        self.live_score_bar.setValue(0)
        self.live_fps.setText("Camera FPS: --")
        self.live_features.setText("Feature history: 0")
        self.live_people.clear()
        self.live_people.addItem("No tracked people")
        self._reset_live_terminal()
        for key, value in {
            "cam_fps": "--",
            "ai_fps": "--",
            "latency": "--",
            "people": "0",
            "threshold": f"{float(self.threshold_input.value()):.2f}",
            "alerts": "0",
            "motion": "--",
            "model": "waiting",
        }.items():
            self._set_live_metric(key, value)
        self._set_live_score_bar_color(False)
        self._update_live_plot()

    def _set_live_metric(self, key: str, value: str) -> None:
        label = self.live_metric_labels.get(key)
        if label:
            label.setText(value)

    def _set_live_score_bar_color(self, anomaly: bool) -> None:
        color = "#f4212e" if anomaly else "#00ba7c"
        self.live_score_bar.setStyleSheet(
            f"#LiveScoreBar::chunk {{ background: {color}; border-radius: 6px; }}"
        )

    def _update_live_people(self, people: list[dict[str, Any]]) -> None:
        self.live_people.clear()
        if not people:
            self.live_people.addItem("No tracked people")
            return
        for person in people:
            self.live_people.addItem(
                f"{person.get('label', 'Person')}  risk {float(person.get('risk', 0.0)) * 100:.0f}%  "
                f"emotion {person.get('emotion', 'unknown')}  motion {float(person.get('motion', 0.0)):.0f}"
            )

    def _update_live_plot(self) -> None:
        self.live_plot.clear()
        threshold = float(self.threshold_input.value())
        self.live_plot.addLine(y=threshold, pen=pg.mkPen("#71767b", width=1, style=Qt.DashLine))
        if self.score_history:
            xs = list(range(len(self.score_history)))
            self.live_plot.plot(xs, self.score_history, pen=pg.mkPen(self._plot_line_color(), width=2))
            normal_x = [x for x, score in zip(xs, self.score_history) if score < threshold]
            normal_y = [score for score in self.score_history if score < threshold]
            anomaly_x = [x for x, score in zip(xs, self.score_history) if score >= threshold]
            anomaly_y = [score for score in self.score_history if score >= threshold]
            if normal_x:
                self.live_plot.plot(normal_x, normal_y, pen=None, symbol="o", symbolSize=7, symbolBrush="#00ba7c")
            if anomaly_x:
                self.live_plot.plot(anomaly_x, anomaly_y, pen=None, symbol="o", symbolSize=7, symbolBrush="#f4212e")
            self.live_plot.setXRange(max(0, len(self.score_history) - 60), max(60, len(self.score_history)), padding=0)
        else:
            self.live_plot.setXRange(0, 60, padding=0)
        self.live_plot.setYRange(0, 1, padding=0)
        self.live_plot.setLimits(yMin=0, yMax=1, minYRange=0.2, maxYRange=1, xMin=0, maxXRange=120)

    def _set_metric(self, key: str, value: str) -> None:
        label = self.metric_labels.get(key)
        if label:
            label.setText(value)

    def _set_image(self, target: QLabel, frame_rgb: np.ndarray) -> None:
        height, width, channels = frame_rgb.shape
        image = QImage(frame_rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        target.setPixmap(pixmap.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
        if self.media_player is not None:
            self.media_player.stop()
        if self.camera_worker:
            self.camera_worker.stop()
        super().closeEvent(event)


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"


def _format_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _status_brush(value: float, threshold: float, alpha: int = 255) -> QColor:
    color = QColor("#f4212e" if value >= threshold else "#00ba7c")
    color.setAlpha(alpha)
    return color


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(sum(values[start : index + 1]) / (index - start + 1))
    return averaged


def _pixmap_from_data_url(data_url: str) -> QPixmap | None:
    if not data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(encoded)
        pixmap = QPixmap()
        pixmap.loadFromData(raw)
        return pixmap
    except Exception:
        return None
