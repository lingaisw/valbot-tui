@echo off
REM ValBot TUI Launcher for Windows
REM This script launches the Terminal User Interface version of ValBot

setlocal enabledelayedexpansion

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    pause
    exit /b 1
)

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at venv\
    echo Running with system Python...
)

REM Launch the TUI
echo Starting ValBot TUI...
python valbot_tui_launcher.py %*

REM Deactivate virtual environment if it was activated
if defined VIRTUAL_ENV (
    deactivate
)

endlocal
