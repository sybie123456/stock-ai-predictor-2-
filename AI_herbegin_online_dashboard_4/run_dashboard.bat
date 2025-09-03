@echo off
setlocal
REM Create and activate venv if missing
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate

REM Install light dashboard deps (no TF to avoid conflicts)
pip install --upgrade pip
pip install -r requirements_dash.txt

REM Run via python -m to avoid PATH issues
python -m streamlit run streamlit_app.py
