<#
Demo runner for PowerShell.
Creates a venv if missing, installs dependencies into the venv, and runs Streamlit
using the venv's python executable so you do not need to manually activate.

Run in PowerShell as:
    .\demo_run.ps1
#>

# Allow running if execution policy is strict for this session
try {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -ErrorAction Stop
} catch {
    # ignore
}

if (-not (Test-Path -Path .venv)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

$venvPython = Join-Path -Path (Get-Location) -ChildPath ".venv\Scripts\python.exe"

Write-Host "Upgrading pip and installing requirements into venv..."
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt

Write-Host "Launching Streamlit (press Ctrl+C in this window to stop)..."
& $venvPython -m streamlit run app.py
