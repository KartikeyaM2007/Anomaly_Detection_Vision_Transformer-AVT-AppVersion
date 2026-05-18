from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

from desktop_app.live_intelligence import LiveIntelligence
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
        self.fast_live = LiveIntelligence()

    @Slot()
    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.failed.emit("Could not open camera.")
            self.stopped.emit()
            return

        last_heavy_score_at = 0.0
        frame_count = 0
        score_count = 0
        last_result: dict[str, Any] | None = None
        last_model_result: dict[str, Any] | None = None
        started_at = time.perf_counter()
        try:
            while self._running:
                ok, frame_bgr = cap.read()
                if not ok:
                    self.failed.emit("Camera frame could not be read.")
                    break

                frame_count += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                now = time.perf_counter()
                infer_started = time.perf_counter()
                result = self.fast_live.analyze(frame_rgb, self.threshold)
                instant_score = float((result.get("result") or {}).get("prob_anomaly", 0.0))
                runtime_ready = getattr(self.service, "_runtime", None) is not None
                should_confirm = (
                    runtime_ready
                    and now - last_heavy_score_at >= 1.6
                    and (instant_score >= min(0.18, self.threshold) or last_model_result is None)
                )
                if should_confirm:
                    last_heavy_score_at = now
                    model_result = self.service.process_live_array(
                        frame_rgb,
                        threshold=self.threshold,
                        request_focus_screen=self.focus_screen,
                    )
                    last_model_result = model_result
                result = self._combine_live_results(result, last_model_result)
                score_count += 1
                elapsed = max(time.perf_counter() - started_at, 0.001)
                result["latency_ms"] = (time.perf_counter() - infer_started) * 1000.0
                result["camera_fps"] = frame_count / elapsed
                result["processed_fps"] = score_count / elapsed
                last_result = result
                self.result_ready.emit(result)
                self.frame_ready.emit(self.fast_live.draw_overlay(frame_rgb, last_result))
                QThread.msleep(1)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            cap.release()
            self.stopped.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False

    def _combine_live_results(self, fast_result: dict[str, Any], model_payload: dict[str, Any] | None) -> dict[str, Any]:
        combined = dict(fast_result)
        runtime_ready = getattr(self.service, "_runtime", None) is not None
        combined["model_status"] = (model_payload or {}).get("status", "ready" if runtime_ready else "not loaded")
        combined["model_feature_count"] = (model_payload or {}).get("feature_count", 0)
        combined["events"] = list((model_payload or {}).get("events") or [])
        fast_prediction = dict(combined.get("result") or {})
        model_result = (model_payload or {}).get("result") or {}
        model_score = float(model_result.get("prob_anomaly", 0.0))
        fast_score = float(fast_prediction.get("prob_anomaly", 0.0))
        if model_score > 0:
            score = max(fast_score, model_score)
            fast_prediction["prob_anomaly"] = score
            fast_prediction["prob_normal"] = 1.0 - score
            fast_prediction["confidence"] = max(score, 1.0 - score)
            fast_prediction["prediction"] = "ANOMALY" if score >= self.threshold else "NORMAL"
            fast_prediction["basis"] = "fast+videomae"
        combined["result"] = fast_prediction
        combined["feature_count"] = combined.get("person_count", 0)
        return combined
