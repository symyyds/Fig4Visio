$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    python -m venv (Join-Path $Root ".venv")
}

Write-Host "Installing runtime and packaging dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirements.txt")
& $Python -m pip install pyinstaller

Write-Host "Building Fig4VisioGUI.exe..."
Push-Location $Root
try {
    & $Python -m PyInstaller `
        --clean `
        --noconfirm `
        --noconsole `
        --onefile `
        --name Fig4VisioGUI `
        --paths ".\scripts" `
        --add-data "scripts;scripts" `
        --add-data "templates;templates" `
        --add-data "references;references" `
        --add-data "requirements.txt;." `
        --collect-all "rapidocr_onnxruntime" `
        --collect-all "cv2" `
        --collect-all "onnxruntime" `
        --collect-all "numpy" `
        --hidden-import "pythoncom" `
        --hidden-import "pywintypes" `
        --hidden-import "win32timezone" `
        --hidden-import "win32com" `
        --hidden-import "win32com.client" `
        --hidden-import "win32com.client.gencache" `
        --hidden-import "cv2" `
        --hidden-import "rapidocr_onnxruntime" `
        ".\gui_app.py"
}
finally {
    Pop-Location
}

$ExePath = Join-Path $Root "dist\Fig4VisioGUI.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build failed: $ExePath was not created."
}

Write-Host ""
Write-Host "Build complete:"
Write-Host $ExePath
