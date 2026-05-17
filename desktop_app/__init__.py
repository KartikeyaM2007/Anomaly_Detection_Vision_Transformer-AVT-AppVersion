"""PySide6 desktop client for the local anomaly detection platform."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
