@echo off
setlocal

cd /d "%~dp0"

set "SCRIPT=%~dp0metashape_360_gui.py"
set "PYTHONW="
set "PYTHON="

if exist "%~dp0env\Scripts\pythonw.exe" set "PYTHONW=%~dp0env\Scripts\pythonw.exe"
if not defined PYTHONW if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"

if exist "%~dp0env\Scripts\python.exe" set "PYTHON=%~dp0env\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%SCRIPT%" (
    echo metashape_360_gui.py was not found next to this launcher.
    pause
    exit /b 1
)

if defined PYTHONW (
    start "" "%PYTHONW%" "%SCRIPT%"
    exit /b 0
)

if defined PYTHON (
    start "" "%PYTHON%" "%SCRIPT%"
    exit /b 0
)

echo No virtual environment interpreter was found.
echo Expected one of these files:
echo   %~dp0env\Scripts\python.exe
echo   %~dp0.venv\Scripts\python.exe
echo.
echo Create the environment and install dependencies first.
pause
exit /b 1