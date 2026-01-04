# setup_env.ps1 — Setup virtualenv and install recommended packages for NeuralCloak
# Usage: Right-click -> Run with PowerShell, or run in PowerShell after creating/activating venv

param(
    [switch]$FullOptional  # install optional heavy deps (facenet-pytorch)
)

Write-Host "Setting up virtual environment and installing dependencies..."

# Activate venv if exists, else create
if (-Not (Test-Path -Path .\.venv)) {
    python -m venv .venv
}

Write-Host "Activating venv..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip, wheel, setuptools..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing core packages (opencv-python, pillow, numpy, scikit-image, requests)..."
pip install opencv-python Pillow numpy scikit-image requests

if ($FullOptional) {
    Write-Host "Installing optional CPU PyTorch + facenet-pytorch (may take several minutes)..."
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    pip install facenet-pytorch
    Write-Host "Optional packages installed."
} else {
    Write-Host "Skipping optional heavy packages. To install them, rerun with -FullOptional flag."
}

Write-Host "Done. You can now run: python test_disruption.py <original> <cloaked>"