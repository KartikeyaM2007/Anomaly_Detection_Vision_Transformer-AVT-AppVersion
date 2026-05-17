# Windows Installer Build

This project can be packaged as a shareable Windows desktop app.

Final outputs:

```text
dist/AVT/AVT.exe          # PyInstaller onedir app bundle
release/AVT-Setup.exe    # Installer when Inno Setup is installed
release/AVT-portable.zip # Fallback portable build when Inno Setup is missing
```

## Requirements

- Windows x64
- Python environment with the desktop dependencies installed
- `artifacts/checkpoints/best_model.pt`
- Local VideoMAE files under `artifacts/models/videomae-base`
- Optional: Inno Setup 6 for `AVT-Setup.exe`

Install desktop dependencies:

```powershell
pip install -r requirements-desktop.txt
```

If the VideoMAE model is already cached locally by Hugging Face, export it into
the project so the packaged app runs offline:

```powershell
python scripts\export_videomae_local.py
```

## Build

```powershell
.\scripts\build_windows_installer.ps1
```

The script:

1. Checks the classifier checkpoint.
2. Installs PyInstaller if it is missing.
3. Exports cached VideoMAE files if `artifacts/models/videomae-base` is missing.
4. Builds `dist/AVT/AVT.exe`.
5. Builds `release/AVT-Setup.exe` when Inno Setup is installed.
6. Creates `release/AVT-portable.zip` as a fallback when Inno Setup is missing.

## Sharing

Preferred:

```text
release/AVT-Setup.exe
```

Fallback:

```text
release/AVT-portable.zip
```

For the ZIP, users extract the folder and run `AVT.exe`.
