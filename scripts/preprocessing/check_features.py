"""
check_features.py
Run this FIRST to verify your .npy files are compatible before training.
Usage: python check_features.py --embeddings path/to/embeddings.npy --labels path/to/labels.npy
"""

import numpy as np
import argparse
import sys

def check(embeddings_path, labels_path):
    print("=" * 50)
    print("Checking your .npy feature files...")
    print("=" * 50)

    try:
        emb = np.load(embeddings_path)
        print(f"[OK] Embeddings loaded: {embeddings_path}")
        print(f"     Shape : {emb.shape}")
        print(f"     Dtype : {emb.dtype}")
        print(f"     Min/Max: {emb.min():.4f} / {emb.max():.4f}")
    except Exception as e:
        print(f"[ERROR] Could not load embeddings: {e}")
        sys.exit(1)

    try:
        lbl = np.load(labels_path)
        print(f"\n[OK] Labels loaded: {labels_path}")
        print(f"     Shape  : {lbl.shape}")
        print(f"     Dtype  : {lbl.dtype}")
        print(f"     Unique : {np.unique(lbl)}")
        print(f"     Counts : { {v: int((lbl==v).sum()) for v in np.unique(lbl)} }")
    except Exception as e:
        print(f"[ERROR] Could not load labels: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("COMPATIBILITY CHECK")
    print("=" * 50)

    # Length match
    if emb.shape[0] != lbl.shape[0]:
        print(f"[FAIL] Length mismatch: {emb.shape[0]} embeddings vs {lbl.shape[0]} labels")
        sys.exit(1)
    else:
        print(f"[OK] Length match: {emb.shape[0]} samples")

    # Shape check
    if len(emb.shape) == 3:
        n, seq, feat = emb.shape
        print(f"[OK] 3D shape (videos={n}, frames={seq}, features={feat}) — ready to train!")
        input_size = feat
    elif len(emb.shape) == 2:
        n, feat = emb.shape
        print(f"[WARN] 2D shape (videos={n}, features={feat})")
        print(f"       Will be auto-expanded to (videos=1, frames=1, features={feat}) during training.")
        input_size = feat
    else:
        print(f"[FAIL] Unexpected shape: {emb.shape}")
        sys.exit(1)

    # Num classes
    num_classes = len(np.unique(lbl))
    print(f"[INFO] Detected {num_classes} classes -> use --num_classes {num_classes}")

    print("\n" + "=" * 50)
    print("SUGGESTED TRAINING COMMAND (SRU)")
    print("=" * 50)
    print(f"""
python sru_training.py \\
    --embeddings_path {embeddings_path} \\
    --labels_path {labels_path} \\
    --input_size {input_size} \\
    --num_classes {num_classes} \\
    --hidden_size 1024 \\
    --num_layers 2 \\
    --epochs 100 \\
    --batch_size 32 \\
    --learning_rate 0.001 \\
    --save_dir artifacts/models \\
    --show_plots
""")
    print("SUGGESTED TRAINING COMMAND (SRU++)")
    print("=" * 50)
    print(f"""
python srupp_training.py \\
    --embeddings_path {embeddings_path} \\
    --labels_path {labels_path} \\
    --input_size {input_size} \\
    --num_classes {num_classes} \\
    --hidden_size 1024 \\
    --proj_size 784 \\
    --num_layers 2 \\
    --epochs 100 \\
    --batch_size 32 \\
    --learning_rate 0.001 \\
    --save_dir artifacts/models \\
    --show_plots
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    args = parser.parse_args()
    check(args.embeddings, args.labels)
