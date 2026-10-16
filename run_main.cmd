@echo off
setlocal
cd /d "%~dp0"
if defined IOV_PYTHON (
    set "IOV_SELECTED_PYTHON=%IOV_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "IOV_SELECTED_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "IOV_SELECTED_PYTHON=python"
)
"%IOV_SELECTED_PYTHON%" -B "%~dp0main.py" %*
set "IOV_EXIT_CODE=%ERRORLEVEL%"
if "%~1"=="" if not defined CI pause
exit /b %IOV_EXIT_CODE%
