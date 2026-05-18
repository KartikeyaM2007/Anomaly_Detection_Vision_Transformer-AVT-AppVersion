import base64
import io
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class PlatformSmokeTests(unittest.TestCase):
    def test_default_checkpoint_exists(self):
        from vad_platform.config import default_config

        config = default_config(ROOT)
        self.assertTrue(config.checkpoint_path.exists())
        self.assertEqual(config.checkpoint_path.name, "best_model.pt")

    def test_model_forward_shape(self):
        import torch

        from vad_platform.model import AnomalyTransformer

        model = AnomalyTransformer(
            feat_dim=8,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            dropout=0.0,
            max_frames=4,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 4, 8))
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_live_frame_requires_image(self):
        import app

        client = app.app.test_client()
        response = client.post("/api/live-frame", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing image", response.get_json()["error"])

    def test_live_frame_warmup_without_heavy_runtime(self):
        from vad_platform.config import RuntimeConfig
        from vad_platform.detector import ViolenceDetectionService

        service = ViolenceDetectionService(
            project_root=ROOT,
            config=RuntimeConfig(
                checkpoint_path=ROOT / "artifacts" / "checkpoints" / "best_model.pt",
                clip_len=2,
                frame_skip=1,
            ),
        )
        service._runtime = object()

        image = Image.new("RGB", (8, 8), (0, 0, 0))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        result = service.process_live_frame(image_data, threshold=0.25)
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "warming")
        self.assertEqual(result["needed_frames"], 1)

    def test_live_array_scores_without_base64_roundtrip(self):
        import numpy as np

        from vad_platform.config import RuntimeConfig
        from vad_platform.detector import ViolenceDetectionService

        class FakeRuntime:
            device_name = "fake-cpu"

            def extract_clip_feature(self, clip_frames):
                return np.ones(8, dtype=np.float32) * (len(clip_frames) / 2.0)

            def predict(self, feature_array, threshold, progress=None, label="window"):
                score = float(np.clip(np.mean(feature_array) * 0.8, 0.0, 1.0))
                return {
                    "prob_normal": 1.0 - score,
                    "prob_anomaly": score,
                    "prediction": "ANOMALY" if score >= threshold else "NORMAL",
                    "confidence": max(score, 1.0 - score),
                }

        service = ViolenceDetectionService(
            project_root=ROOT,
            config=RuntimeConfig(
                checkpoint_path=ROOT / "artifacts" / "checkpoints" / "best_model.pt",
                clip_len=2,
                frame_skip=1,
                live_clip_stride=1,
                live_segment_clips=1,
            ),
        )
        service._runtime = FakeRuntime()

        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        warmup = service.process_live_array(frame, threshold=0.25)
        scored = service.process_live_array(frame, threshold=0.25)

        self.assertFalse(warmup["ready"])
        self.assertTrue(scored["ready"])
        self.assertEqual(scored["status"], "scored")
        self.assertEqual(scored["result"]["prediction"], "ANOMALY")
        self.assertEqual(scored["feature_count"], 1)


if __name__ == "__main__":
    unittest.main()
