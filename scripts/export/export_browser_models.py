from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vad_platform.config import RuntimeConfig, default_config
from vad_platform.model import AnomalyTransformer


class VideoMAEFeatureWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        from transformers.models.videomae.modeling_videomae import VideoMAEModel

        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)


def load_anomaly_model(config: RuntimeConfig) -> AnomalyTransformer:
    model = AnomalyTransformer(
        feat_dim=config.feat_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_frames=config.max_frames,
    )
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_anomaly_classifier(config: RuntimeConfig, output_path: Path) -> None:
    model = load_anomaly_model(config)
    dummy = torch.zeros(1, config.max_frames, config.feat_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def export_videomae_feature_extractor(config: RuntimeConfig, output_path: Path) -> None:
    model = VideoMAEFeatureWrapper(config.videomae_name)
    dummy = torch.zeros(1, config.clip_len, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["pixel_values"],
        output_names=["features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def verify_anomaly_classifier(config: RuntimeConfig, output_path: Path) -> float:
    import onnx
    import onnxruntime as ort

    model = load_anomaly_model(config)
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(7)
    features = rng.normal(size=(1, config.max_frames, config.feat_dim)).astype(np.float32)
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(features)).numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(["logits"], {"features": features})
    return float(np.max(np.abs(torch_logits - onnx_logits)))


def write_manifest(config: RuntimeConfig, output_dir: Path, anomaly_path: Path, videomae_path: Path | None) -> None:
    manifest = {
        "version": 1,
        "clipLen": config.clip_len,
        "frameSkip": config.frame_skip,
        "clipStride": config.clip_stride,
        "featureDim": config.feat_dim,
        "maxFrames": config.max_frames,
        "segmentClips": config.segment_clips,
        "imageSize": 224,
        "imageMean": [0.485, 0.456, 0.406],
        "imageStd": [0.229, 0.224, 0.225],
        "models": {
            "anomalyClassifier": anomaly_path.name,
            "videomaeFeatureExtractor": videomae_path.name if videomae_path else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export browser-side ONNX models for AVT.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "web" / "static" / "models" / "browser",
    )
    parser.add_argument("--skip-videomae", action="store_true", help="Only export the anomaly classifier.")
    args = parser.parse_args()

    config = default_config(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_path = args.output_dir / "anomaly_transformer.onnx"
    videomae_path = args.output_dir / "videomae_feature_extractor.onnx"

    export_anomaly_classifier(config, anomaly_path)
    max_diff = verify_anomaly_classifier(config, anomaly_path)
    print(f"anomaly_classifier={anomaly_path}")
    print(f"anomaly_classifier_mb={anomaly_path.stat().st_size / (1024 * 1024):.2f}")
    print(f"anomaly_classifier_max_abs_diff={max_diff:.8f}")

    exported_videomae = None
    if not args.skip_videomae:
        export_videomae_feature_extractor(config, videomae_path)
        exported_videomae = videomae_path
        print(f"videomae_feature_extractor={videomae_path}")
        print(f"videomae_feature_extractor_mb={videomae_path.stat().st_size / (1024 * 1024):.2f}")

    write_manifest(config, args.output_dir, anomaly_path, exported_videomae)
    print(f"manifest={args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
