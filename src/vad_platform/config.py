from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint_path: Path
    threshold: float = 0.06
    live_threshold: float = 0.25
    live_clear_threshold: float = 0.12
    live_cooldown_seconds: float = 4.0
    clip_len: int = 16
    frame_skip: int = 2
    clip_stride: int = 8
    feat_dim: int = 768
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.30
    max_frames: int = 512
    segment_clips: int = 4
    live_history: int = 96
    live_clip_stride: int = 4
    live_segment_clips: int = 4
    use_amp: bool = False
    videomae_name: str = "MCG-NJU/videomae-base"


def default_config(project_root: Path) -> RuntimeConfig:
    checkpoint = project_root / "artifacts" / "checkpoints" / "best_model.pt"
    if not checkpoint.exists():
        checkpoint = project_root / "data" / "UCF-Crime_dataset" / "best_model.pt"
    return RuntimeConfig(checkpoint_path=checkpoint)
