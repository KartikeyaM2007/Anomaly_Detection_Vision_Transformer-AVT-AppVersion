"""
Consolidate VideoMAE features from per-category directories into embeddings.npy and labels.npy
Usage: python preprocess_videomae_features.py
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def consolidate_videomae_features(features_root, output_dir='artifacts/features'):
    """
    Load all VideoMAE .npy files from category subdirs and consolidate into:
      - embeddings.npy: (num_videos, num_frames, feature_dim) = (1940, 170, 768)
      - labels.npy: (num_videos,) with numeric class labels
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Map category folder names to numeric labels
    # Treat Normal/Normal1 as class 0 (normal), rest as anomaly (1) for binary
    # Or keep multiclass: each category gets unique label
    category_to_label = {}
    label_counter = 0
    
    all_embeddings = []
    all_labels = []
    
    print("=" * 70)
    print("CONSOLIDATING VideoMAE FEATURES")
    print("=" * 70)
    
    # Collect all categories and assign labels
    categories = sorted([d for d in os.listdir(features_root) 
                        if os.path.isdir(os.path.join(features_root, d))])
    
    for category in categories:
        cat_path = os.path.join(features_root, category)
        npy_files = sorted([f for f in os.listdir(cat_path) if f.endswith('.npy')])
        
        if not npy_files:
            print(f"[SKIP] {category}: no .npy files")
            continue
        
        category_to_label[category] = label_counter
        print(f"\n[{label_counter}] {category}: {len(npy_files)} videos")
        
        # Load all files in this category
        for npy_file in tqdm(npy_files, desc=f"Loading {category}", leave=False):
            file_path = os.path.join(cat_path, npy_file)
            try:
                features = np.load(file_path)  # shape: (170, 768)
                all_embeddings.append(features)
                all_labels.append(label_counter)
            except Exception as e:
                print(f"  ERROR loading {npy_file}: {e}")
                continue
        
        label_counter += 1
    
    # Convert lists to arrays
    embeddings = np.array(all_embeddings, dtype=np.float32)  # (num_videos, 170, 768)
    labels = np.array(all_labels, dtype=np.int64)  # (num_videos,)
    
    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)
    print(f"Total videos: {len(all_embeddings)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes: {len(category_to_label)}")
    print(f"Class mapping:\n{category_to_label}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Save consolidated files
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    labels_path = os.path.join(output_dir, 'labels.npy')
    
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    
    print(f"\n[SAVED] {embeddings_path}")
    print(f"[SAVED] {labels_path}")
    
    # Print suggested training command
    print("\n" + "=" * 70)
    print("SUGGESTED SRU TRAINING COMMAND")
    print("=" * 70)
    print(f"""
python sru_training.py \\
    --embeddings_path {embeddings_path} \\
    --labels_path {labels_path} \\
    --input_size 768 \\
    --num_classes {len(category_to_label)} \\
    --hidden_size 512 \\
    --num_layers 2 \\
    --epochs 100 \\
    --batch_size 16 \\
    --learning_rate 0.001 \\
    --save_dir artifacts/models \\
    --show_plots
""")
    
    print("\n" + "=" * 70)
    print("SUGGESTED SRU++ TRAINING COMMAND")
    print("=" * 70)
    print(f"""
python srupp_training.py \\
    --embeddings_path {embeddings_path} \\
    --labels_path {labels_path} \\
    --input_size 768 \\
    --num_classes {len(category_to_label)} \\
    --hidden_size 512 \\
    --proj_size 384 \\
    --num_layers 2 \\
    --epochs 100 \\
    --batch_size 16 \\
    --learning_rate 0.001 \\
    --save_dir artifacts/models \\
    --show_plots
""")
    
    return embeddings_path, labels_path, category_to_label


if __name__ == "__main__":
    features_root = r"data\UCF-Crime_dataset\VideoMAE_features"
    
    if not os.path.exists(features_root):
        print(f"ERROR: {features_root} not found!")
        exit(1)
    
    consolidate_videomae_features(features_root, output_dir=r'artifacts\features')
