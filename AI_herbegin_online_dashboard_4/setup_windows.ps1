# Creates venv and installs dashboard requirements
param(
  [string]$PythonExe = "python"
)
$ErrorActionPreference = "Stop"
if (!(Test-Path ".venv")) {
  & $PythonExe -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_dash.txt
Write-Host "Setup done. Start: python -m streamlit run streamlit_app.py"
