"""
ValBot TUI Demo Script
Run this to see a demo/test of the TUI interface
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗   ██╗ █████╗ ██╗     ██████╗  ██████╗ ████████╗       ║
║   ██║   ██║██╔══██╗██║     ██╔══██╗██╔═══██╗╚══██╔══╝       ║
║   ██║   ██║███████║██║     ██████╔╝██║   ██║   ██║          ║
║   ╚██╗ ██╔╝██╔══██║██║     ██╔══██╗██║   ██║   ██║          ║
║    ╚████╔╝ ██║  ██║███████╗██████╔╝╚██████╔╝   ██║          ║
║     ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝    ╚═╝          ║
║                                                               ║
║              Terminal User Interface (TUI)                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Welcome to ValBot TUI!

This is a modern, feature-rich terminal interface for ValBot.

Key Features:
  🎨 Beautiful terminal interface
  💬 Interactive chat with AI
  🖥️  Integrated terminal
  📁 File system browser
  ⌨️  Keyboard shortcuts

Getting Started:
  1. The chat panel is where you interact with AI
  2. Type your message and press Enter
  3. Use /help to see available commands
  4. Press Ctrl+T to toggle terminal panel
  5. Press Ctrl+F to toggle file explorer
  6. Press Ctrl+Q to quit

Slash Commands:
  /clear    - Clear conversation history
  /help     - Show help message
  /model    - Change AI model
  /terminal - Execute shell commands
  /file     - Load file content
  /quit     - Exit application

Keyboard Shortcuts:
  Ctrl+Q - Quit
  Ctrl+C - Clear chat
  Ctrl+T - Toggle terminal
  Ctrl+F - Toggle files
  Esc    - Cancel operation

Tips:
  • Resize your terminal for best experience (120x40 recommended)
  • Use a modern terminal emulator with color support
  • Try streaming responses for large outputs
  • Keep terminal panel closed when not needed

"""
    print(banner)
    input("\nPress Enter to launch ValBot TUI...")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    missing = []
    
    # Check for textual
    try:
        import textual
        print("✅ textual found")
    except ImportError:
        print("❌ textual not found")
        missing.append("textual")
    
    # Check for rich
    try:
        import rich
        print("✅ rich found")
    except ImportError:
        print("❌ rich not found")
        missing.append("rich")
    
    # Check for other core dependencies
    try:
        import openai
        print("✅ openai found")
    except ImportError:
        print("❌ openai not found")
        missing.append("openai")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\nTo install missing dependencies, run:")
        print("  pip install " + " ".join(missing))
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_config():
    """Check if configuration is set up."""
    print("\n🔍 Checking configuration...")
    
    # Check for .env file
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("ℹ️  No .env file found (optional)")
    
    # Check for config files
    if os.path.exists("user_config.json"):
        print("✅ user_config.json found")
    elif os.path.exists("default_config.json"):
        print("✅ default_config.json found")
    else:
        print("⚠️  No config file found")
        print("   The TUI will use default settings")
    
    return True


def main():
    """Main demo function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot launch TUI without required dependencies.")
        print("   Please install them and try again.")
        sys.exit(1)
    
    # Check config
    check_config()
    
    print("\n🚀 Launching ValBot TUI...\n")
    
    # Launch the TUI
    try:
        from valbot_tui import main as tui_main
        tui_main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching TUI: {e}")
        print("\nFor help, check README_TUI.md or QUICKSTART_TUI.md")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled. Goodbye!")
        sys.exit(0)
