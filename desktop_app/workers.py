from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

from vad_platform.detector import ViolenceDetectionService


class AnalysisWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, service: ViolenceDetectionService, video_path: Path, threshold: float):
        super().__init__()
        self.service = service
        self.video_path = video_path
        self.threshold = threshold

    @Slot()
    def run(self) -> None:
        try:
            result = self.service.analyze_video_path(
                self.video_path,
                filename=self.video_path.name,
                threshold=self.threshold,
                progress=self.progress.emit,
            )
            if "error" in result:
                self.failed.emit(str(result["error"]))
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class RuntimeLoadWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, service: ViolenceDetectionService):
        super().__init__()
        self.service = service

    @Slot()
    def run(self) -> None:
        try:
            ready = self.service._ensure_runtime(progress=self.progress.emit)
            health = self.service.health()
            if ready:
                self.finished.emit(health)
            else:
                self.failed.emit(str(health.get("error") or "Runtime is not ready"))
        except Exception as exc:
            self.failed.emit(str(exc))


class CameraWorker(QObject):
    frame_ready = Signal(object)
    result_ready = Signal(dict)
    progress = Signal(str)
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        service: ViolenceDetectionService,
        threshold: float,
        camera_index: int = 0,
        focus_screen: bool = False,
    ):
        super().__init__()
        self.service = service
        self.threshold = threshold
        self.camera_index = camera_index
        self.focus_screen = focus_screen
        self._running = False

    @Slot()
    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.failed.emit("Could not open camera.")
            self.stopped.emit()
            return

        last_score_at = 0.0
        frame_count = 0
        started_at = time.perf_counter()
        try:
            while self._running:
                ok, frame_bgr = cap.read()
                if not ok:
                    self.failed.emit("Camera frame could not be read.")
                    break

                frame_count += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)

                now = time.perf_counter()
                # Score at a controlled cadence to keep UI latency low.
                if now - last_score_at >= 0.25:
                    last_score_at = now
                    _, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    image_data = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
                    result = self.service.process_live_frame(
                        image_data,
                        threshold=self.threshold,
                        request_focus_screen=self.focus_screen,
                    )
                    elapsed = max(time.perf_counter() - started_at, 0.001)
                    result["camera_fps"] = frame_count / elapsed
                    self.result_ready.emit(result)
                QThread.msleep(1)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            cap.release()
            self.stopped.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False
