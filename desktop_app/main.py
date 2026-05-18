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

from PySide6.QtWidgets import QApplication

from desktop_app.ui.main_window import MainWindow


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


def main() -> int:
    if "--smoke-load-model" in sys.argv:
        return _smoke_load_model()

    app = QApplication(sys.argv)
    app.setApplicationName("AnomalyGuard")
    app.setOrganizationName("AVT")

    window = MainWindow(project_root=ROOT)
    window.resize(1440, 920)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
