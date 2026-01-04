@echo off
REM Demo runner for Command Prompt / cmd.exe
REM Creates virtualenv if missing, installs deps and runs Streamlit using venv python

IF NOT EXIST .venv (
    echo Creating virtual environment .venv...
    python -m venv .venv
)

echo Upgrading pip and installing requirements into venv...
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python -m pip install -r requirements.txt

echo Launching Streamlit (press Ctrl+C to stop)...
.venv\Scripts\python -m streamlit run app.py
