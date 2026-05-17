param(
    [string]$Python = "C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Release = Join-Path $Root "release"
$Dist = Join-Path $Root "dist"
$Spec = Join-Path $Root "packaging\avt_desktop.spec"
$Checkpoint = Join-Path $Root "artifacts\checkpoints\best_model.pt"
$VideoMae = Join-Path $Root "artifacts\models\videomae-base"

Set-Location $Root
New-Item -ItemType Directory -Force -Path $Release | Out-Null

if (!(Test-Path $Python)) {
    throw "Python not found: $Python"
}

if (!(Test-Path $Checkpoint)) {
    throw "Missing checkpoint: $Checkpoint"
}

$PyInstallerCheck = & $Python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install pyinstaller
}

if (!(Test-Path $VideoMae)) {
    Write-Host "Local VideoMAE assets not found. Exporting from Hugging Face cache..."
    & $Python scripts\export_videomae_local.py
}

if (!(Test-Path $VideoMae)) {
    throw "Missing local VideoMAE assets: $VideoMae"
}

Write-Host "Building AVT desktop bundle with PyInstaller..."
& $Python -m PyInstaller --noconfirm --clean $Spec

$AppDir = Join-Path $Dist "AVT"
if (!(Test-Path (Join-Path $AppDir "AVT.exe"))) {
    throw "PyInstaller did not produce AVT.exe"
}

if (!$SkipInstaller) {
    $Iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue)
    if (!$Iscc) {
        $KnownIsccPaths = @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        )
        $IsccPath = $KnownIsccPaths | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
        if ($IsccPath) {
            $Iscc = [pscustomobject]@{ Source = $IsccPath }
        }
    }

    if ($Iscc) {
        Write-Host "Building AVT-Setup.exe with Inno Setup..."
        & $Iscc.Source (Join-Path $Root "packaging\installer.iss")
    } else {
        Write-Host "Inno Setup compiler not found. Creating ZIP fallback instead."
        $Zip = Join-Path $Release "AVT-portable.zip"
        if (Test-Path $Zip) {
            Remove-Item $Zip -Force
        }
        Compress-Archive -Path (Join-Path $AppDir "*") -DestinationPath $Zip -CompressionLevel Optimal
    }
}

Write-Host "Build complete."
Write-Host "Bundle: $AppDir"
Write-Host "Release folder: $Release"
