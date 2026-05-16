# Edited Notebook Copies

These notebooks are compatibility copies. The original notebooks in `notebooks/`
were not modified.

## Files

- `files/Anomaly_Detection_MP4_Inference_VideoMAE.ipynb`
- `files/UCF_Crime_Anomaly_Detection_Training.ipynb`

## Compatibility Updates

- Uses project-root discovery so paths work from the organized repository.
- Uses `artifacts/checkpoints/best_model.pt` for inference.
- Uses `data/UCF-Crime_dataset/VideoMAE_features` for training features.
- Uses `torch.load(..., weights_only=False)` for PyTorch 2.6+ trusted local checkpoints.
- Replaces removed `VideoMAEFeatureExtractor` with `AutoImageProcessor`.
- Clears old cell outputs so the copies are clean to rerun.

The training copy saves to `artifacts/checkpoints/best_model_training_copy.pt`
so it does not overwrite the deployed app checkpoint by accident.
