from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "artifacts" / "logs" / "desktop_smoke_test.log"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.ui.main_window import MainWindow
from vad_platform.detector import ViolenceDetectionService


def _make_smoke_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(40):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 5) % 255
        frame[:, :, 1] = 60
        frame[:, :, 2] = 120
        cv2.putText(frame, f"{i:02d}", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("desktop smoke start\n", encoding="utf-8")

    video_path = ROOT / "data" / "samples" / "ui_smoke_40f.mp4"
    if not video_path.exists():
        _make_smoke_video(video_path)

    QMessageBox.critical = lambda *args, **kwargs: None
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow(ROOT)
    window.threshold_input.setValue(0.06)
    _log("window created")

    service = ViolenceDetectionService(ROOT)
    result = service.analyze_video_path(
        video_path,
        threshold=0.06,
        progress=lambda message: window._append_log(message),
    )
    if "error" in result:
        print("UI_ANALYSIS_FAILED", result["error"], flush=True)
        return 1

    window._populate_result(result)
    window._set_workflow_step("completed", "Analysis completed")
    window.model_progress.setValue(100)
    window._append_terminal("analysis complete", "complete")

    terminal = window.terminal_output.toPlainText()
    print(
        "UI_ANALYSIS_DONE",
        result.get("operational", {}).get("prediction"),
        result.get("metrics", {}).get("frames"),
        window.model_progress.value(),
        window.analysis_state.text(),
        flush=True,
    )
    print("TERMINAL_HAS_RUNTIME", "[runtime] loading weights" in terminal, flush=True)
    print("TERMINAL_HAS_DONE", "[done] scoring completed" in terminal, flush=True)
    _log("done")
    app.quit()
    return 0


def _log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")




if __name__ == "__main__":
    raise SystemExit(main())
