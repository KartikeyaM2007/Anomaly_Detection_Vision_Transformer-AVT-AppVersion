from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts" / "models" / "videomae-base"
MODEL_NAME = "MCG-NJU/videomae-base"


def main() -> int:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from transformers.models.videomae.modeling_videomae import VideoMAEModel

    TARGET.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = VideoMAEModel.from_pretrained(MODEL_NAME, local_files_only=True)
    processor.save_pretrained(TARGET)
    model.save_pretrained(TARGET)
    print(f"Saved local VideoMAE assets to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
