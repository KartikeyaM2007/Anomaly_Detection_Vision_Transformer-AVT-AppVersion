# AnomalyGuard Desktop App

This desktop client runs the existing PyTorch inference service locally with a
PySide6 dashboard. It is designed for offline use after model assets are present
on the machine.

## Run in VS Code

```bash
pip install -r requirements-desktop.txt
python -m desktop_app.main
```

`requirements-desktop.txt` uses `PySide6-Essentials` instead of the full
`PySide6` package so first install is smaller. The app only needs Qt Widgets,
Qt Core, and Qt GUI modules.

## What the Desktop App Includes

- GPU/CPU runtime detection
- Auto / GPU / CPU runtime selection
- Manual model warm-up with a 1-100% load progress bar
- Upload video preview
- Local video analysis using `artifacts/checkpoints/best_model.pt`
- Analysis Workflow tracker
- PowerShell-style Analysis Terminal with live runtime logs
- Model/training Info dialog with notebook result screenshots
- Timeline graph
- Worm graph / smoothed score trend
- Detected frame samples
- FPS, confidence, latency, duration, frame count, anomaly coverage, event count,
  and peak score cards
- Camera tab for low-latency live scoring on a background `QThread`
- Screen Focus mode, live reset, and alert log

## Offline Model Requirement

The classifier checkpoint is already loaded from:

```text
artifacts/checkpoints/best_model.pt
```

The VideoMAE processor/model must also be available locally before the packaged
app is used without internet. During development, download/cache it once, then
package the cache or save it into a local folder and point the runtime config at
that folder.

Recommended final folder:

```text
artifacts/models/videomae-base/
```

When packaging, keep the app in `--onedir` mode so the large model files remain
beside the executable.

## Package Later

```bash
pyinstaller --noconfirm --onedir --windowed --name AnomalyGuard desktop_app/main.py
```
