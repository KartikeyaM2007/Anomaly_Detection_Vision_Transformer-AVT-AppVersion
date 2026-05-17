# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd()
block_cipher = None

datas = [
    (str(ROOT / "src"), "src"),
    (str(ROOT / "desktop_app"), "desktop_app"),
    (str(ROOT / "artifacts" / "checkpoints"), "artifacts/checkpoints"),
]

local_videomae = ROOT / "artifacts" / "models" / "videomae-base"
if local_videomae.exists():
    datas.append((str(local_videomae), "artifacts/models/videomae-base"))

for package_name in ("transformers", "torchvision", "sklearn"):
    datas += collect_data_files(package_name)

hiddenimports = []
for package_name in (
    "desktop_app",
    "vad_platform",
    "transformers",
    "transformers.models.videomae",
    "transformers.models.auto",
    "torch",
    "torchvision",
    "cv2",
    "pyqtgraph",
    "PIL",
    "numpy",
):
    hiddenimports += collect_submodules(package_name)


a = Analysis(
    [str(ROOT / "desktop_app" / "main.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib.tests", "numpy.tests", "torch.testing"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AVT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AVT",
)
