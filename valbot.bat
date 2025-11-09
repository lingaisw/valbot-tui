@echo off
REM Run ValBot CLI using the virtual environment's Python interpreter
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found. Please run setup.bat first.
    echo.
    pause
    exit /b 1
)
venv\Scripts\python.exe app.py %*
if errorlevel 1 (
    echo.
    echo ERROR: ValBot encountered an error. See messages above.
    echo.
    pause
)
