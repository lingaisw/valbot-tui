#!/bin/bash
# ValBot TUI Launcher for Unix/Linux/macOS
# This script launches the Terminal User Interface version of ValBot

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found at venv/"
    echo "Running with system Python..."
fi

# Launch the TUI
echo "Starting ValBot TUI..."
python3 valbot_tui_launcher.py "$@"

# Deactivate virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
