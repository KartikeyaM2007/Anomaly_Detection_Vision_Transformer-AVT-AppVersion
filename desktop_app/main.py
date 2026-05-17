from __future__ import annotations

import os
import sys
from pathlib import Path


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep Hugging Face/Transformers from attempting network access in packaged builds.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PySide6.QtWidgets import QApplication

from desktop_app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AnomalyGuard")
    app.setOrganizationName("AVT")

    window = MainWindow(project_root=ROOT)
    window.resize(1440, 920)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
