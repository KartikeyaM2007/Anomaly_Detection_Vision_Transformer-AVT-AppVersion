from __future__ import annotations

import os
import sys
from pathlib import Path


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep Hugging Face/Transformers from attempting network access in packaged builds.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def _smoke_load_model() -> int:
    from vad_platform.config import default_config
    from vad_platform.detector import ViolenceDetectionService

    log_path = ROOT / "artifacts" / "logs" / "packaged_smoke_load_model.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("SMOKE_LOAD_MODEL_START\n", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    config = default_config(ROOT)
    service = ViolenceDetectionService(ROOT, config=config)
    messages: list[str] = []
    ready = service._ensure_runtime(progress=messages.append)
    health = service.health() if ready else {"ready": False, "error": service._runtime_error}
    for message in messages:
        emit(message)
    if not health.get("ready"):
        emit(f"SMOKE_LOAD_MODEL_FAILED {health.get('error')}")
        return 1
    emit(f"SMOKE_LOAD_MODEL_OK {health.get('device')}")
    return 0


def _smoke_live_model() -> int:
    import numpy as np

    from desktop_app.live_intelligence import LiveIntelligence
    from vad_platform.config import default_config
    from vad_platform.detector import ViolenceDetectionService

    log_path = ROOT / "artifacts" / "logs" / "packaged_smoke_live_model.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("SMOKE_LIVE_MODEL_START\n", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    service = ViolenceDetectionService(ROOT, default_config(ROOT))
    if not service._ensure_runtime(progress=emit):
        emit(f"SMOKE_LIVE_MODEL_FAILED {service._runtime_error}")
        return 1
    fast_live = LiveIntelligence()
    last = {}
    fast_last = {}
    try:
        for index in range(20):
            frame = np.zeros((160, 160, 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 9) % 255
            frame[:, :, 1] = 80
            frame[:, :, 2] = 120
            frame[40:120, 50 + (index % 8) : 90 + (index % 8), :] = 220
            fast_last = fast_live.analyze(frame, threshold=0.25)
            last = service.process_live_array(frame, threshold=0.25)
    except Exception as exc:
        emit(f"SMOKE_LIVE_MODEL_FAILED {exc}")
        return 1

    emit(f"SMOKE_LIVE_READY {last.get('ready')}")
    emit(f"SMOKE_LIVE_STATUS {last.get('status')}")
    emit(f"SMOKE_LIVE_FEATURES {last.get('feature_count')}")
    result = last.get("result") or {}
    emit(f"SMOKE_LIVE_PREDICTION {result.get('prediction')}")
    emit(f"SMOKE_LIVE_SCORE {float(result.get('prob_anomaly', 0.0)):.4f}")
    emit(f"SMOKE_FAST_PEOPLE {fast_last.get('person_count')}")
    emit(f"SMOKE_FAST_SCORE {float((fast_last.get('result') or {}).get('prob_anomaly', 0.0)):.4f}")
    if not last.get("ready") or last.get("status") != "scored":
        emit("SMOKE_LIVE_MODEL_FAILED not-scored")
        return 1
    if int(fast_last.get("person_count") or 0) < 1:
        emit("SMOKE_LIVE_MODEL_FAILED no-fast-people")
        return 1
    emit(f"SMOKE_LIVE_MODEL_OK {service._runtime.device_name if service._runtime else 'unknown'}")
    return 0


def main() -> int:
    if "--smoke-load-model" in sys.argv:
        return _smoke_load_model()
    if "--smoke-live-model" in sys.argv:
        return _smoke_live_model()

    from PySide6.QtWidgets import QApplication

    from desktop_app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AnomalyGuard")
    app.setOrganizationName("AVT")

    window = MainWindow(project_root=ROOT)
    window.resize(1440, 920)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
