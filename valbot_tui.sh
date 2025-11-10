#!/bin/bash
# ValBot TUI Launcher for Unix/Linux/macOS
# This script launches the Terminal User Interface version of ValBot

# Force UTF-8 encoding for proper emoji display
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

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

# Check for virtual environment in multiple locations
VENV_FOUND=0
VENV_PATH=""

# Check common venv locations
for VENV_DIR in "venv" "valbot-venv" ".venv"; do
    if [ -f "$SCRIPT_DIR/$VENV_DIR/bin/activate" ]; then
        VENV_PATH="$SCRIPT_DIR/$VENV_DIR"
        VENV_FOUND=1
        break
    fi
done

# Activate virtual environment if found
if [ $VENV_FOUND -eq 1 ]; then
    echo "Activating virtual environment at $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found in common locations (venv, valbot-venv, .venv)"
    echo "Running with system Python..."
fi

# Launch the TUI in a new xterm window
echo "Starting ValBot TUI in new terminal..."
xterm -e bash -c "
    # Re-activate virtual environment in the new terminal
    if [ $VENV_FOUND -eq 1 ]; then
        source '$VENV_PATH/bin/activate'
    fi
    
    # Launch the TUI
    python3 '$SCRIPT_DIR/valbot_tui_launcher.py' $@
" &

echo "ValBot TUI launched in new terminal window (PID: $!)"
