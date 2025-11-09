#!/usr/bin/env python3
"""
ValBot TUI Launcher
Entry point for the Terminal User Interface version of ValBot
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file first
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)

# Import platform setup
import platform_setup
platform_setup.setup_proxy_env()

from valbot_tui import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
