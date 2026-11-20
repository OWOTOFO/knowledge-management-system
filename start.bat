@echo off
rem Double-click to start (Windows). First run installs everything into .venv (a few minutes).
cd /d "%~dp0"
if not exist ".venv\.installed" (
  echo First run: installing, this takes a few minutes...
  py -3 -m venv .venv 2>nul || python -m venv .venv || goto :fail
  ".venv\Scripts\python" -m pip install -q --upgrade pip || goto :fail
  ".venv\Scripts\python" -m pip install -q -e . || goto :fail
  echo ok> ".venv\.installed"
)
start "" cmd /c "timeout /t 4 >nul & start http://localhost:8501"
".venv\Scripts\python" -m streamlit run src\km\app.py
goto :eof
:fail
echo.
echo Install failed - see the message above. Python 3.10 or newer is needed: https://www.python.org/downloads/
pause
