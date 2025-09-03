# Activate and run dashboard
$ErrorActionPreference = "Stop"
if (!(Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_dash.txt
python -m streamlit run streamlit_app.py
