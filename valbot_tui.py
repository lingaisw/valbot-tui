# -*- coding: utf-8 -*-
"""
ValBot TUI - A Modern Material Design Terminal Interface for ValBot
Inspired by Charm's Crush interface with beautiful gradients and sleek styling

FEATURE PARITY WITH CLI:
=======================
This TUI implementation includes all major features from the CLI version:

✅ System Prompt Integration:
   - Automatically loads system_prompt from config on startup
   - Ensures consistent AI behavior according to user preferences

✅ GPT-5 Reasoning Support:
   - Displays thinking process when display_reasoning is enabled
   - Configurable reasoning effort levels (low/medium/high)
   - Streams reasoning text in real-time via ResponseAudioDeltaEvent
   - Shows progress indicators during reasoning phase

✅ Full Command Support:
   - /clear, /new - Start new conversation
   - /quit - Exit application
   - /help - Show comprehensive help
   - /model - Interactive model picker with arrow keys
   - /agent - Interactive agent selection with descriptions
   - /context - Load files into conversation (supports globs)
   - /file - Display file contents with syntax highlighting
   - /terminal - Execute shell commands
   - /multi - Multi-line input via system editor
   - /prompts - Show custom prompts (via CommandManager)
   - /commands - Show all available commands (via CommandManager)
   - /settings - Settings information
   - /reload - Reinitialize chatbot
   - /update - Update information
   - /add_agent - Add agent information
   - /add_tool - Add tool information

✅ CommandManager Integration:
   - Uses actual CommandManager from bot_commands.py
   - Supports custom prompts with argument parsing
   - Supports custom commands from agent plugins
   - Automatic delegation to plugin manager

✅ Agent System:
   - Interactive agent picker with arrow navigation
   - Shows agent names and descriptions
   - Executes agent workflows with full context
   - Error handling and status reporting

✅ Context Management:
   - Load single files or glob patterns
   - Integrates with ContextManager
   - Visual feedback on loaded files
   - File content preview with syntax highlighting

✅ Response Streaming:
   - Real-time text streaming
   - Reasoning display for GPT-5 models
   - Proper event handling (ResponseAudioDeltaEvent, ResponseTextDeltaEvent)
   - Visual progress indicators

✅ Markdown Rendering:
   - Full markdown support with syntax highlighting
   - Code blocks with copy buttons
   - Tables, lists, quotes, emphasis
   - Collapsible code sections

✅ Material Design UI:
   - Modern dark theme
   - Gradient accents
   - Smooth animations
   - Responsive layout
   - Keyboard shortcuts
"""

import asyncio
import sys
import re
import glob
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from textual import on, work, events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, ScrollableContainer
from textual.reactive import reactive
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import (
    Header, Footer, Input, Static, Button, 
    DirectoryTree, Label, Markdown, RichLog, TabbedContent, TabPane, TextArea, OptionList
)
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from rich.panel import Panel
from rich.console import Group
from rich.syntax import Syntax

from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseAudioDeltaEvent,
    ResponseTextDeltaEvent
)

from chatbot import ChatBot
from config import ConfigManager
from context_management import ContextManager
from client_setup import initialize_agent_model, initialize_chat_client
from agent_plugins.plugin_manager import PluginManager
from agent_plugins import utilities
from terminal_manager import TerminalManager
from bot_commands import CommandManager
from rich.console import Console
from console_extensions import HistoryConsole
import sys
import os
import tempfile
import subprocess


# Platform-specific emoji/symbol mapping
print(sys.platform)
IS_LINUX = sys.platform.startswith('linux')

# Define emoji mappings based on platform
if IS_LINUX:
    # ASCII symbols for Linux
    EMOJI = {
        'checkmark': '✓',
        'clipboard': '[>_]',
        'info': 'ℹ️',
        'lightbulb': '◐',
        'robot': '⚞⚟',
        'user': '❰❱',
        'cross': '✕',
        'gear': '⚙️',
        'folder': '◨',
        'file': '◫',
        'keyboard': '⌨️',
    }
else:
    # Unicode emojis for Windows/Mac
    EMOJI = {
        'checkmark': '✅',
        'clipboard': '📋',
        'info': 'ℹ️',
        'lightbulb': '💡',
        'robot': '🤖',
        'user': '👤',
        'cross': '❌',
        'gear': '⚙️',
        'folder': '📁',
        'file': '📄',
        'keyboard': '⌨️',
    }


# Register custom ValBot dark theme
VALBOT_DARK_THEME = Theme(
    name="valbot-dark (default)",
    primary="#989af5",           # Indigo - primary brand color
    secondary="#4d4eba",         # Indigo darker - secondary/user color
    accent="#0696d4",            # Light blue - accent highlights
    foreground="#fafafa",        # Alabaster white - foreground & text 
    warning="#f59e0b",           # Amber - warnings
    error="#ef4444",             # Red - errors
    success="#10b981",           # Emerald - success states
    background="#0f172a",        # Slate-950 - main background
    surface="#1e293b",           # Slate-800 - surface/panel background
    panel="#353e4f",             # Slate-700 - elevated panels
    dark=True,
    # Additional variables for text colors
    variables={
        "text": "#fafafa",       # Slate-200 - primary text
        "text-muted": "#94a3b8",  # Slate-400 - muted text
        "text-bright": "#ffffff", # White - bright text
        "text-dim": "#a5b4fc",    # Indigo-300 - dimmed text
        "gray-light": "#374151",  # Gray-700 - light gray surfaces
        "blue-dark": "#1d4ed8",   # Blue-700 - dark blue
        "blue": "#2563eb",        # Blue-600 - standard blue
        "indigo-dark": "#4f46e5", # Indigo-600 - dark indigo
    }
)


class TUIConsoleWrapper:
    """Wrapper to make TUI's ChatPanel work like a Rich Console for CommandManager."""
    
    def __init__(self, chat_panel, app):
        self.chat_panel = chat_panel
        self.app = app
        self.history_console = HistoryConsole()
        self._live_display = None
        self._live_content = []
        self._current_response = []
        
    def print(self, *args, **kwargs):
        """Print to the chat panel."""
        import re
        from rich.text import Text
        
        # Handle Rich Markdown objects specially to preserve markdown formatting
        if args and hasattr(args[0], '__class__') and args[0].__class__.__name__ == 'Markdown':
            # Extract the markdown text from Rich Markdown object
            markdown_obj = args[0]
            if hasattr(markdown_obj, 'markup'):
                message = markdown_obj.markup
            elif hasattr(markdown_obj, '_text'):
                message = markdown_obj._text
            else:
                message = str(markdown_obj)
        elif args and isinstance(args[0], Text):
            # Handle Rich Text objects - convert to markup string
            rich_text = args[0]
            message = str(rich_text)
        else:
            # Convert to string
            message = " ".join(str(arg) for arg in args)
            
            # Handle styled text by preserving Rich markup
            # BUT: Don't wrap content with code blocks, as they need markdown rendering
            style = kwargs.get('style', '')
            has_code_blocks = '```' in message
            
            if style and not has_code_blocks:
                # Wrap content with Rich markup tags to preserve styling
                # (only if no code blocks present)
                message = f"[{style}]{message}[/]"
            # If message has code blocks, don't add markup - let Markdown handle it
        
        # Skip empty messages
        if not message.strip():
            return
        
        # Determine message role based on content
        # System notifications include: file detection, tool usage info, etc.
        role = "assistant"
        if any(indicator in message for indicator in [
            "Found local file:",
            "Using tools",
            "→ Using tools",
            "Running agent with tools"
        ]):
            role = "system"
        
        # Always add messages during agent execution
        # Use call_from_thread for thread-safe UI updates
        try:
            self.app.call_from_thread(self.chat_panel.add_message, role, message)
        except Exception as e:
            # Fallback: try direct call if we're already in main thread
            try:
                self.chat_panel.add_message(role, message)
            except Exception:
                pass  # Silently fail if neither works
        
    def add_command_completer(self, command_manager):
        """Stub for console command completion (not needed in TUI)."""
        pass
    
    def set_live(self, live):
        """Set the live display context (compatibility with Rich Console)."""
        self._live_display = live
    
    def clear_live(self):
        """Clear the live display context."""
        self._live_display = None
        self._live_content = []
    
    def input(self, prompt="", **kwargs):
        """Stub for input (not used in TUI mode - handled by TUIConsole instead)."""
        return ""
    
    def status(self, message, **kwargs):
        """Status message display."""
        class StatusContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return StatusContext()
    
    def rule(self, title="", **kwargs):
        """Display a horizontal rule (just show the title)."""
        if title:
            self.print(f"\n{'─' * 40}\n{title}\n{'─' * 40}\n")
    
    def is_terminal(self):
        """Indicate this is a terminal-like interface."""
        return True
    
    @property
    def width(self):
        """Return a reasonable default width."""
        return 80
    
    @property
    def height(self):
        """Return a reasonable default height."""
        return 24
    
    def ask(self, prompt, **kwargs):
        """Ask for user input - not supported in TUI streaming mode."""
        self.print(f"[Input required: {prompt}]")
        return ""
    
    def start_agent_stream(self):
        """Start streaming agent output."""
        self._streaming_agent = True
        self._current_response = []
        # Create a message for agent output
        # Check if header should be shown
        show_header = not self.chat_panel._assistant_header_shown
        self.chat_panel._assistant_header_shown = True
        self._current_msg = ChatMessage("assistant", f"{EMOJI['robot']} Agent is working...", show_header=show_header)
        self.chat_panel.mount(self._current_msg)
        self.chat_panel.scroll_end(animate=False)
        
    def end_agent_stream(self):
        """End streaming agent output."""
        self._streaming_agent = False
        if hasattr(self, '_current_msg') and self._current_msg:
            # Final update with complete response
            if self._current_response:
                final_response = "\n".join(self._current_response)
                self._current_msg.update_content(final_response)
            
            # Schedule copy button detection for agent output (same as direct AI responses)
            # Must use call_from_thread since this is called from agent thread
            if self._current_msg and self._current_msg.parent:
                msg_widget = self._current_msg
                self.app.call_from_thread(
                    self.chat_panel._add_copy_buttons_to_message,
                    msg_widget
                )
            
            self._current_msg = None
        self._current_response = []



class CodeBlockHeader(Container):
    """A simple header bar for code blocks with a copy button."""
    
    DEFAULT_CSS = """
    CodeBlockHeader {
        height: 1;
        background: $panel;
        padding: 0 1;
        margin: 0;
        layout: horizontal;
    }
    
    CodeBlockHeader .spacer {
        width: 1fr;
    }
    
    CodeBlockHeader #copy-btn {
        width: 8;
        height: 1;
        min-width: 8;
        text-style: bold;
        border: none;
        padding: 0;
        dock: right;
    }

    CodeBlockHeader #copy-btn:focus {
        background: $success;
    }
    """
    
    def __init__(self, code: str, **kwargs):
        super().__init__(**kwargs)
        self.code = code
    
    def compose(self) -> ComposeResult:
        """Compose the header with copy button on the right."""
        from textual.widgets import Button
        
        yield Static("", classes="spacer")
        yield Button("Copy", id="copy-btn", variant="primary")
    
    @on(Button.Pressed, "#copy-btn")
    async def copy_code(self):
        """Copy code to clipboard."""
        try:
            # Use pyperclip if available
            import pyperclip
            pyperclip.copy(self.code)
            btn = self.query_one("#copy-btn", Button)
            btn.label = EMOJI['checkmark']
            await asyncio.sleep(2)
            btn.label = "Copy"
        except ImportError:
            # Fallback: just show feedback
            btn = self.query_one("#copy-btn", Button)
            btn.label = "Copied"
            await asyncio.sleep(2)
            btn.label = "Copy"


class CopyButton(Static):
    """Overlay copy button that floats over code blocks."""
    
    DEFAULT_CSS = """
    CopyButton {
        width: 8;
        height: 1;
        offset: 0 0;
        layer: overlay;
        background: $surface;
        content-align: center middle;
        border: none;
    }
    
    CopyButton:hover {
        background: $surface-lighten-1;
    }
    
    CopyButton.success {
        background: $success;
    }
    """
    
    def __init__(self, code_content: str, code_index: int):
        super().__init__()
        self.code_content = code_content
        self.code_index = code_index
        self.label = "Copy"
        self.can_focus = True  # Make it clickable
        
    def render(self) -> str:
        return self.label
    
    def on_click(self) -> None:
        """Handle copy button click."""
        try:
            import pyperclip
            pyperclip.copy(self.code_content)
            # Show success state
            self.label = EMOJI['checkmark']
            self.add_class("success")
            self.refresh()
            # Reset after 1.5 seconds
            self.set_timer(1.5, self._reset_button)
        except Exception:
            pass
    
    def _reset_button(self):
        """Reset button to default state."""
        self.label = "Copy"
        self.remove_class("success")
        self.refresh()


class ChatMessage(Container):
    """A beautifully styled chat message widget with Material Design principles."""
    
    DEFAULT_CSS = """
    ChatMessage {
        height: auto;
        margin: 0 0 1 0;
        padding: 0;
    }
    
    ChatMessage.user-message {
        border: solid $accent-muted;
        border-left: thick $accent;
        padding: 1 2;
    }
    
    ChatMessage.assistant-message {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    ChatMessage.system-message {
        border-left: thick $foreground-muted;
        padding: 1 2;
    }
    
    ChatMessage.error-message {
        border-left: thick $error;
        padding: 1 2;
    }
    
    ChatMessage.welcome-message {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    .message-header {
        color: $text-muted;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .message-content {
        background: transparent;
    }
    
    /* RichLog styling to match Markdown */
    ChatMessage RichLog.message-content {
        background: transparent;
        border: none;
        padding: 0;
        scrollbar-size: 0 0;
        height: auto;
    }
    """
    
    can_focus = False  # Prevent focus/selection
    
    @staticmethod
    def _has_rich_color_markup(content: str) -> bool:
        """
        Check if content has Rich color markup tags.
        Returns True if content has color tags that would benefit from RichLog rendering.
        """
        # Pattern to detect Rich color tags (including combined style+color like [bold green])
        # Matches: [green], [bold green], [italic yellow], etc.
        color_pattern = r'\[(?:(?:bold|italic|dim|underline|strike|blink|reverse)\s+)?(?:red|green|blue|yellow|cyan|magenta|white|black|orange|purple|pink|deep_pink|grey\d+|#[0-9a-fA-F]{6}|rgb\([^\)]+\))'
        return re.search(color_pattern, content, re.IGNORECASE) is not None
    
    @staticmethod
    def _has_markdown_syntax(content: str) -> bool:
        """
        Check if content has markdown syntax (code blocks, headers, lists, etc.).
        Returns True if content needs markdown rendering.
        """
        # Check for common markdown patterns
        markdown_patterns = [
            r'```',           # Code blocks
            r'^#{1,6}\s',     # Headers
            r'^\s*[-*+]\s',   # Unordered lists
            r'^\s*\d+\.\s',   # Ordered lists
            r'\[.+?\]\(.+?\)', # Links
            r'^\|.+\|$',      # Tables
        ]
        for pattern in markdown_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    @staticmethod
    def _convert_rich_markup_to_markdown(content: str) -> str:
        """
        Convert Rich markup tags to markdown equivalents.
        Strips all Rich formatting and converts styles to markdown.
        
        Strategy:
        1. Convert style tags (bold, italic) to markdown
        2. Remove all color/style tags completely
        3. Clean up any remaining Rich markup
        """
        # First pass: Handle [bold ...] tags with any modifiers (colors, etc.)
        # Use a more aggressive pattern that captures the style word and ignores everything after
        # Match: [bold ...] content [/bold ...] OR [bold ...] content [/]
        
        # Bold: [bold X] -> **, [/bold X] or [/] -> **
        def replace_bold(match):
            return f"**{match.group(1)}**"
        content = re.sub(r'\[bold[^\]]*\](.*?)\[/bold[^\]]*\]', replace_bold, content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\[bold[^\]]*\](.*?)\[/\]', replace_bold, content, flags=re.IGNORECASE | re.DOTALL)
        
        # Italic: [italic X] -> *, [/italic X] or [/] -> *
        def replace_italic(match):
            return f"*{match.group(1)}*"
        content = re.sub(r'\[italic[^\]]*\](.*?)\[/italic[^\]]*\]', replace_italic, content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\[italic[^\]]*\](.*?)\[/\]', replace_italic, content, flags=re.IGNORECASE | re.DOTALL)
        
        # Dim: [dim X] -> _, [/dim X] or [/] -> _
        def replace_dim(match):
            return f"_{match.group(1)}_"
        content = re.sub(r'\[dim[^\]]*\](.*?)\[/dim[^\]]*\]', replace_dim, content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\[dim[^\]]*\](.*?)\[/\]', replace_dim, content, flags=re.IGNORECASE | re.DOTALL)
        
        # Second pass: Remove ALL remaining Rich markup tags
        # This includes colors, standalone styles, and any other Rich tags
        # Match any [tag] or [tag something] pattern
        content = re.sub(r'\[[^\]]+\]', '', content)
        
        return content
    
    @staticmethod
    def _process_backticks_only(content: str) -> str:
        """
        Process only backticks in user messages for inline code formatting.
        Escapes other markdown characters to prevent them from being interpreted,
        but preserves backticks so they render as inline code.
        """
        # Store backtick content temporarily with indices
        backtick_pattern = r'`([^`]+)`'
        backtick_matches = []
        
        # Find all backtick sections
        for match in re.finditer(backtick_pattern, content):
            backtick_matches.append({
                'start': match.start(),
                'end': match.end(),
                'full': match.group(0),
                'inner': match.group(1)
            })
        
        # Build the result by processing character by character
        result = []
        i = 0
        
        while i < len(content):
            # Check if we're at the start of a backtick section
            in_backtick = False
            for bt_match in backtick_matches:
                if i == bt_match['start']:
                    # Add the entire backtick section as-is
                    result.append(bt_match['full'])
                    i = bt_match['end']
                    in_backtick = True
                    break
            
            if not in_backtick:
                # Escape markdown special characters
                char = content[i]
                if char in '*_#[]()~>-+=|\\':
                    result.append('\\' + char)
                else:
                    result.append(char)
                i += 1
        
        return ''.join(result)
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None, show_header: bool = True, agent_name: Optional[str] = None):
        super().__init__()
        self.role = role
        self.timestamp = timestamp or datetime.now()
        self.show_header = show_header
        self.agent_name = agent_name  # Store the agent name if provided
        self._header_widget = None  # Store header widget reference
        self._content_widget = None  # Store content widget reference
        self._widget_type = None  # Track which widget type we're using ('markdown' or 'richlog')
        
        # Store original content before processing
        self.original_content = content
        
        # Determine which widget to use based on content
        # Priority: RichLog for pure Rich markup, Markdown for markdown syntax or mixed content
        if role != "user":
            has_rich_colors = self._has_rich_color_markup(content)
            has_markdown = self._has_markdown_syntax(content)
            
            # Decision logic:
            # - If has markdown syntax (code blocks, headers, etc.) -> Use Markdown (convert Rich to markdown)
            # - If has ONLY Rich colors with no markdown -> Use RichLog (preserve colors)
            # - Otherwise -> Use Markdown (safe default)
            if has_markdown or not has_rich_colors:
                # Use Markdown widget - convert Rich markup to markdown
                self._widget_type = 'markdown'
                content = self._convert_rich_markup_to_markdown(content)
            else:
                # Use RichLog widget - preserve Rich markup for color rendering
                self._widget_type = 'richlog'
                # Keep content as-is with Rich markup
        else:
            # User messages always use Markdown with backticks only
            self._widget_type = 'markdown'
            content = self._process_backticks_only(content)
        
        # Store the processed content
        self.content = content
        
        # Apply role-specific styling
        if role == "user":
            self.add_class("user-message")
        elif role == "assistant":
            self.add_class("assistant-message")
        elif role == "system":
            self.add_class("system-message")
        elif role == "error":
            self.add_class("error-message")
    
    def update_content(self, new_content: str):
        """Update the message content and refresh the display properly."""
        # Store original content
        self.original_content = new_content
        
        # Re-determine widget type based on new content
        if self.role != "user":
            has_rich_colors = self._has_rich_color_markup(new_content)
            has_markdown = self._has_markdown_syntax(new_content)
            
            if has_markdown or not has_rich_colors:
                new_widget_type = 'markdown'
                new_content = self._convert_rich_markup_to_markdown(new_content)
            else:
                new_widget_type = 'richlog'
                # Keep Rich markup for RichLog
        else:
            new_widget_type = 'markdown'
            new_content = self._process_backticks_only(new_content)
        
        self.content = new_content
        
        # If widget type changed, need to recompose
        if new_widget_type != self._widget_type:
            self._widget_type = new_widget_type
            self._content_widget = None
            self.refresh(recompose=True)
            return
        
        # Update the appropriate widget type
        try:
            if self._widget_type == 'markdown':
                content_widget = self.query_one(".message-content", Markdown)
                content_widget.update(new_content)
            elif self._widget_type == 'richlog':
                content_widget = self.query_one(".message-content", RichLog)
                content_widget.clear()
                content_widget.write(new_content, markup=True)
        except Exception:
            # If widget doesn't exist yet, do a full refresh
            self.refresh(recompose=True)
        
    def compose(self) -> ComposeResult:
        """Compose the message display with markdown."""
        role_icons = {
            "user": EMOJI['user'],
            "assistant": EMOJI['robot'],
            "system": EMOJI['info'],
            "error": EMOJI['cross']
        }
        
        role_labels = {
            "user": "You",
            "assistant": f"ValBot (Agent: {self.agent_name})" if self.agent_name else "ValBot",
            "system": "System",
            "error": "Error"
        }
        
        # Dynamically load theme colors from the app's theme
        theme = self.app.theme_variables
        role_colors = {
            "user": theme.get("accent", "#0696d4"),
            "assistant": theme.get("primary", "#989af5"),
            "system": theme.get("foreground-muted", "#fafafa"),
            "error": theme.get("error", "#ef4444")
        }
        
        icon = role_icons.get(self.role, "•")
        role_label = role_labels.get(self.role, self.role.title())
        role_color = role_colors.get(self.role, "#94a3b8")
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        # Create header with colored role label (only if show_header is True)
        if self.show_header:
            if self._header_widget is None:
                header = f"{icon} [{role_color}]{role_label}[/] · {time_str}"
                self._header_widget = Static(header, classes="message-header", markup=True)
            yield self._header_widget
        
        # Create content widget based on widget type determined in __init__
        # - 'markdown': Use Markdown widget for markdown syntax (code blocks, headers, etc.)
        # - 'richlog': Use RichLog widget to preserve Rich color markup
        if self._content_widget is None:
            if self._widget_type == 'richlog':
                # Use RichLog to preserve Rich markup with colors
                richlog = RichLog(highlight=True, markup=True, classes="message-content")
                richlog.write(self.content)
                self._content_widget = richlog
            else:
                # Use Markdown for markdown syntax or converted content
                self._content_widget = Markdown(self.content, classes="message-content")
        yield self._content_widget
    
    def add_copy_buttons_for_code_blocks(self):
        """Add overlay copy buttons positioned over code blocks (legacy method)."""
        # This method is now primarily used by the new post-streaming detection system
        # Find code blocks in the content
        code_block_pattern = r'```(\w*)\n(.*?)\n```'
        matches = list(re.finditer(code_block_pattern, self.content, re.DOTALL))
        
        if not matches:
            return
        
        # Remove any existing copy buttons first
        for existing_btn in list(self.query(CopyButton)):
            existing_btn.remove()
        
        # Get the markdown widget to calculate positions
        try:
            markdown_widget = self.query_one(".message-content", Markdown)
        except Exception:
            return
        
        # Add overlay copy button for each code block
        for i, match in enumerate(matches):
            code_content = match.group(2)
            # Create overlay button
            btn = CopyButton(code_content, i)
            # Position it - we'll update positions after mounting
            self.mount(btn)
            
        # Use the enhanced positioning method with a delay
        self.call_after_refresh(lambda: self.set_timer(0.1, self._position_copy_buttons_accurately))
    
    def _position_copy_buttons(self):
        """Position copy buttons at top-right of each code block."""
        try:
            buttons = list(self.query(CopyButton))
            if not buttons:
                return
            
            # Get markdown widget to reference
            markdown_widget = self.query_one(".message-content", Markdown)
            
            # Find positions of code blocks in the content
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            matches = list(re.finditer(code_block_pattern, self.content, re.DOTALL))
            
            if not matches:
                return
            
            # Get the width of the parent container to position from right
            parent_width = self.size.width if self.size.width > 0 else 80
            x_position = parent_width - 10  # 10 chars from right (button is 8 wide + margin)
            
            # Calculate cumulative line positions for each code block
            lines_before_content = 1  # Reduced to account for markdown padding
            
            for i, (btn, match) in enumerate(zip(buttons, matches)):
                # Get text from start to this code block start
                match_start_content = self.content[:match.start()]
                lines_to_match = match_start_content.count('\n')
                
                # Position button at the start of this code block
                # Lower by 2 rows for better visual alignment
                y_position = lines_before_content + lines_to_match + 2
                
                # Set the offset (x, y) from top-left of the message
                btn.styles.offset = (x_position, y_position)
                
        except Exception as e:
            # Fallback: position buttons in a simple vertical stack on the right
            try:
                parent_width = self.size.width if self.size.width > 0 else 80
                x_position = parent_width - 10
                for i, btn in enumerate(buttons):
                    y_offset = 1 + (i * 10)  # Rough spacing, starting closer to top
                    btn.styles.offset = (x_position, y_offset)
            except:
                pass
    
    def on_resize(self) -> None:
        """Reposition copy buttons when the widget is resized."""
        # Reposition buttons after a short delay to ensure layout is updated
        self.set_timer(0.2, self._position_copy_buttons_accurately)
    
    def _position_copy_buttons_accurately(self):
        """Enhanced copy button positioning with better accuracy after layout settling."""
        try:
            buttons = list(self.query(CopyButton))
            if not buttons:
                return
            
            # Get the markdown widget and ensure it's fully rendered
            try:
                markdown_widget = self.query_one(".message-content", Markdown)
                # Give markdown time to fully render if needed
                if not hasattr(markdown_widget, 'size') or markdown_widget.size.width == 0:
                    # Schedule retry if layout not ready
                    self.set_timer(0.1, self._position_copy_buttons_accurately)
                    return
            except Exception:
                return
            
            # Find code block positions in content
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            matches = list(re.finditer(code_block_pattern, self.content, re.DOTALL))
            
            if not matches or len(matches) != len(buttons):
                return
            
            # Calculate positioning based on actual widget dimensions
            container_width = self.size.width if self.size.width > 0 else 80
            x_position = max(container_width - 12, 2)  # Position from right edge with margin
            
            # More accurate line counting for positioning
            # Account for the markdown widget's internal structure
            lines_before_content = 1  # Reduced to account for markdown padding
            
            for i, (btn, match) in enumerate(zip(buttons, matches)):
                # Find the line where this code block starts
                match_start_content = self.content[:match.start()]
                lines_to_match = match_start_content.count('\n')
                
                # Position button at the start of the code block
                # Adjust for markdown rendering (code blocks typically have 1 line padding)
                # Lower by 2 rows for better visual alignment
                y_position = lines_before_content + lines_to_match + 2
                
                # Fine-tune positioning to align with the top border of the code block
                # This accounts for the ``` delimiter line
                y_position = max(y_position, 0)
                
                # Apply positioning with better alignment
                btn.styles.offset = (x_position, y_position)
                
        except Exception as e:
            # Fallback to original positioning method
            self._position_copy_buttons()
    
    def add_copy_buttons_if_needed(self):
        """Legacy method - now deprecated in favor of post-streaming detection."""
        # This method is kept for compatibility but should not be used during streaming
        # Copy button detection is now handled after streaming completes for better accuracy
        pass


class ChatTerminalOutput(Static):
    """A terminal output widget that displays inline in the chat with border."""
    
    DEFAULT_CSS = """
    ChatTerminalOutput {
        height: auto;
        margin: 0 0 1 0;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }
    
    #terminal-output-log {
        background: $surface;
        height: auto;
        scrollbar-size: 0 0;
    }
    """
    
    def __init__(self, command: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command = command
        self.border_title = f"{EMOJI['keyboard']} Terminal"
        self.terminal_manager = TerminalManager()
        
    def compose(self) -> ComposeResult:
        """Compose the terminal output display."""
        yield RichLog(highlight=True, markup=True, id="terminal-output-log")
        
    async def on_mount(self) -> None:
        """Execute command and display output when mounted."""
        log = self.query_one("#terminal-output-log", RichLog)
        log.write(Text(f"$ {self.command}", style="bold cyan"))
        
        async for output_type, line in self.terminal_manager.run_command(self.command):
            if output_type == "stdout":
                log.write(Text(line, style="white"))
            elif output_type == "stderr":
                log.write(Text(line, style="red"))
            elif output_type == "error":
                log.write(Text(f"Error: {line}", style="bold red"))
        
        log.write("")  # Empty line after command


class ChatPanel(VerticalScroll):
    """Modern Material Design panel for displaying chat messages."""
    
    DEFAULT_CSS = """
    ChatPanel {
        height: 1fr;
        background: $background;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    
    ChatPanel:focus {
        border: none;
    }
    
    ChatPanel > Static {
        width: 1fr;
    }
    
    ChatPanel > ChatMessage > Markdown {
        background: transparent;
    }
    """
    
    can_focus = False  # Prevent focus
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages: List[ChatMessage] = []
        self._assistant_header_shown = False  # Track if assistant header shown after last user message
        self.current_agent: Optional[str] = None  # Track the currently active agent
    
    def add_message(self, role: str, content: str, agent_name: Optional[str] = None):
        """Add a new message to the chat."""
        # Reset header flag when user sends a message
        if role == "user":
            self._assistant_header_shown = False
        
        # For assistant messages, only show header if not already shown
        show_header = True
        if role == "assistant":
            show_header = not self._assistant_header_shown
            self._assistant_header_shown = True
        
        # Use the provided agent_name, or fall back to the current_agent if not specified
        effective_agent_name = agent_name if agent_name is not None else self.current_agent
        
        msg = ChatMessage(role, content, show_header=show_header, agent_name=effective_agent_name)
        self.messages.append(msg)
        self.mount(msg)
        self.scroll_end(animate=True)
        
        # Schedule copy button detection for messages with code blocks
        # (especially important for agent output)
        if '```' in content:
            # Use delayed detection to ensure proper layout and positioning
            self.app.call_later(self._add_copy_buttons_to_message, msg)
    
    async def add_terminal_output(self, command: str):
        """Add a terminal output widget to the chat."""
        terminal_widget = ChatTerminalOutput(command)
        await self.mount(terminal_widget)
        self.scroll_end(animate=True)
    
    def _add_copy_buttons_to_message(self, message_widget):
        """
        Add copy buttons to a message with code blocks.
        Single-phase approach with adequate delay for layout settling.
        """
        if not message_widget or not message_widget.parent:
            return
        
        try:
            # Clear any existing copy buttons first
            for existing_btn in list(message_widget.query(CopyButton)):
                existing_btn.remove()
            
            # Detect code blocks in the final content
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            matches = list(re.finditer(code_block_pattern, message_widget.content, re.DOTALL))
            
            if matches:
                # Add copy buttons for each code block
                for i, match in enumerate(matches):
                    code_content = match.group(2)
                    btn = CopyButton(code_content, i)
                    message_widget.mount(btn)
                
                # Wait for layout to settle, then position accurately
                # Single delay is sufficient (0.5s total) instead of two phases (0.2s + 0.3s)
                self.set_timer(0.5, lambda: self._position_buttons_for_message(message_widget))
            
        except Exception:
            # Ignore errors in copy button detection
            pass
    
    def _position_buttons_for_message(self, message_widget):
        """Position copy buttons accurately after layout has settled."""
        if message_widget and message_widget.parent:
            try:
                # Use the improved positioning method
                message_widget._position_copy_buttons_accurately()
            except Exception:
                # Fallback to basic positioning
                try:
                    message_widget._position_copy_buttons()
                except Exception:
                    pass
        
    def clear_messages(self):
        """Clear all messages."""
        for msg in self.messages:
            msg.remove()
        self.messages.clear()
        # Reset the header flag and current agent
        self._assistant_header_shown = False
        self.current_agent = None
        # Also remove any ChatMessage widgets that were mounted directly
        # (e.g., streaming messages that weren't added to the messages list)
        for widget in self.query("ChatMessage"):
            widget.remove()
        # Remove any terminal output widgets
        for widget in self.query("ChatTerminalOutput"):
            widget.remove()

class CustomDirectoryTree(DirectoryTree):
    """Custom DirectoryTree with custom file icon."""
    
    ICON_FILE = EMOJI['file']


class FileExplorerPanel(Container):
    """Material Design file system navigation panel."""
    
    DEFAULT_CSS = """
    FileExplorerPanel {
        width: 35;
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
        display: none;
        layer: overlay;
    }
    
    FileExplorerPanel.visible {
        display: block;
    }
    
    #file-tree {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = f"{EMOJI['folder']} Files"
        
    def compose(self) -> ComposeResult:
        """Compose the file explorer."""
        yield CustomDirectoryTree("./", id="file-tree")


class StatusBar(Container):
    """Modern Material Design status bar with clickable model button."""
    
    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 3;
        background: $surface;
        padding: 0 2;
        layout: horizontal;
        align: right middle;
    }
    
    StatusBar Button {
        height: auto;
        min-width: 15;
        background: transparent !important;
        padding: 0 2;
        margin: 0 0;
        content-align: center middle;
        text-style: none;
        border: none !important;
    }
    """
    
    model_name = reactive("gpt-4o")
    
    def _truncate_model_name(self, model_name: str, max_length: int = 20) -> str:
        """Truncate model name if it exceeds max_length."""
        if len(model_name) <= max_length:
            return model_name
        return model_name[:max_length - 3] + "..."
    
    def compose(self) -> ComposeResult:
        """Compose the status bar with model button."""
        display_name = self._truncate_model_name(self.model_name)
        yield Button(f"{display_name} {EMOJI['gear']}", id="model-button")
    
    def watch_model_name(self, new_model: str) -> None:
        """Update button label when model changes."""
        try:
            button = self.query_one("#model-button", Button)
            display_name = self._truncate_model_name(new_model)
            button.label = f"{display_name} {EMOJI['gear']}"
        except Exception:
            pass
    
    @on(Button.Pressed, "#model-button")
    async def open_model_picker(self):
        """Open model picker when button is clicked."""
        # Get the main screen and call its action method
        main_screen = self.screen
        if hasattr(main_screen, 'action_change_model'):
            await main_screen.action_change_model()

class CommandInput(TextArea):
    """Modern Material Design multiline input widget with elegant styling."""
    
    DEFAULT_CSS = """
    CommandInput {
        background: rgba(30, 41, 59, 0);
        padding: 1 3;
        height: auto;
        min-height: 3;
        max-height: 10;
    }
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            placeholder=" Type your message or /command... (↓ for new line, Enter to submit)",
            highlight_cursor_line = False,
            tab_behavior = "indent",
            *args,
            **kwargs
        )
        self.show_line_numbers = False
        self._last_text = ""
    
    def on_mount(self) -> None:
        """Handle mount event."""
        super().on_mount()
        # Store reference in screen for autocomplete
        if hasattr(self.screen, '__class__') and 'MainScreen' in self.screen.__class__.__name__:
            self.screen._command_input = self
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Watch for text changes to trigger autocomplete."""
        new_text = self.text
        
        # Only trigger command autocomplete if text starts with /
        if new_text.startswith("/"):
            if new_text != self._last_text:
                self._last_text = new_text
                # Notify screen about text change
                if hasattr(self.screen, 'on_command_input_changed'):
                    self.screen.on_command_input_changed(new_text)
        else:
            # Not a command - update file path suggestions if already showing
            if hasattr(self.screen, '_file_autocomplete_context') and self.screen._file_autocomplete_context:
                # File autocomplete is active - update suggestions as user types
                if hasattr(self.screen, 'on_text_changed_check_paths'):
                    self.screen.on_text_changed_check_paths(new_text, self.cursor_location)
            
            # Hide command autocomplete if / was removed
            if self._last_text.startswith("/"):
                if hasattr(self.screen, 'hide_autocomplete'):
                    self.screen.hide_autocomplete()
            
            self._last_text = new_text
    
    def _on_key(self, event: events.Key) -> None:
        """Handle key events for multiline input."""
        # Check if autocomplete is visible
        autocomplete_visible = False
        autocomplete = None
        if hasattr(self.screen, '_autocomplete_overlay'):
            autocomplete = self.screen._autocomplete_overlay
            if autocomplete and autocomplete.styles.display != "none":
                autocomplete_visible = True
        
        if event.key == "enter":
            # Check if AI is processing - block Enter/submission
            if hasattr(self.screen, 'processing') and self.screen.processing:
                # Show toast notification
                if hasattr(self.screen, 'app'):
                    self.screen.app.notify(
                        "⚠️ AI is busy.\nPress `Esc` to cancel current response.",
                        severity="warning",
                        timeout=2
                    )
                event.prevent_default()
                event.stop()
                return
            
            if autocomplete_visible and autocomplete:
                # Trigger selection from autocomplete
                option_list = autocomplete.get_option_list()
                if option_list and option_list.highlighted is not None:
                    # Get the selected command and post the selection message
                    selected_idx = option_list.highlighted
                    if selected_idx < len(autocomplete._commands):
                        command = autocomplete._commands[selected_idx][0]
                        autocomplete.post_message(AutocompleteSelected(autocomplete, command))
                event.prevent_default()
                return
            
            # Regular Enter submits the message
            self.submit()
            event.prevent_default()
            
        elif event.key == "down":
            if autocomplete_visible and autocomplete:
                # Focus autocomplete for navigation
                option_list = autocomplete.get_option_list()
                if option_list:
                    option_list.focus()
                event.prevent_default()
                return
            
            # Explicitly handle for new lines
            self.insert("\n")
            return
            
        elif event.key == "up":
            if autocomplete_visible and autocomplete:
                # Focus autocomplete for navigation
                option_list = autocomplete.get_option_list()
                if option_list:
                    option_list.focus()
                    # Move to last item
                    if len(option_list._options) > 0:
                        option_list.highlighted = len(option_list._options) - 1
                event.prevent_default()
                return
                
        elif event.key == "tab":
            # Check if autocomplete is visible (command or file autocomplete)
            if autocomplete_visible and autocomplete:
                option_list = autocomplete.get_option_list()
                if option_list:
                    option_list.focus()
                    option_list.action_cursor_down()
                event.prevent_default()
                return
            
            # Try file path autocomplete only on Tab press
            if hasattr(self.screen, 'try_file_path_autocomplete'):
                # Get current word/path at cursor
                cursor_pos = self.cursor_location
                text = self.text
                
                # Don't autocomplete on whitespace
                if cursor_pos[1] > 0 and text:
                    char_before_cursor = text.split('\n')[cursor_pos[0]][max(0, cursor_pos[1] - 1):cursor_pos[1]]
                    if not char_before_cursor.isspace():
                        if self.screen.try_file_path_autocomplete(text, cursor_pos):
                            event.prevent_default()
                            return
            
            # Default: indent behavior (let TextArea handle it)
            # Don't prevent default - let the TextArea's tab_behavior work
                
        elif event.key == "escape":
            if autocomplete_visible:
                self.screen.hide_autocomplete()
                event.prevent_default()
                return
            
            # Propagate Esc to parent screen's action_cancel
            if hasattr(self.screen, 'action_cancel'):
                self.screen.action_cancel()
            event.prevent_default()
            event.stop()
    
    def submit(self):
        """Submit the current text content."""
        content = self.text
        # Always submit if content exists (even if blank) - let handle_input decide
        # Post a custom submission event
        self.post_message(TextAreaSubmitted(self, content))
        self.clear()

class TextAreaSubmitted(Message):
    """Custom message for TextArea submission."""
    def __init__(self, text_area, value):
        super().__init__()
        self.text_area = text_area
        self.value = value


class InlinePickerSelected(Message):
    """Custom message for inline picker selection."""
    def __init__(self, picker, value):
        super().__init__()
        self.picker = picker
        self.value = value


class InlinePicker(OptionList):
    """Inline picker that temporarily replaces the textarea for selections."""
    
    BINDINGS = [
        Binding("ctrl+a", "parent_agent_picker", "Agent", show=True, priority=True),
        Binding("ctrl+f", "parent_toggle_files", "Files", show=True, priority=True),
        Binding("ctrl+n", "parent_new_chat", "New Chat", show=True, priority=True),
        Binding("ctrl+m", "parent_change_model", "Model", show=True, priority=True),
        Binding("ctrl+s", "parent_save_session", "Save", show=True, priority=True),
        Binding("ctrl+l", "parent_load_session", "Load", show=True, priority=True),
        Binding("escape", "cancel_picker", "Cancel", show=True, priority=True),
        Binding("ctrl+q", "parent_quit", "Quit", show=True, priority=True),
    ]
    
    can_focus = True
    
    DEFAULT_CSS = """
    InlinePicker {
        height: auto;
        min-height: 10;
        max-height: 20;
        background: $surface;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
        scrollbar-size: 1 1;
        width: 100%;
    }
    
    InlinePicker .option-list--option {
        background: $panel;
        padding: 0 2;
        height: auto;
        min-height: 3;
        margin: 0 0 1 0;
        width: 100%;
        align-vertical: middle;
    }
    
    InlinePicker .option-list--option-highlighted {
        background: $primary-darken-3 !important;
        text-style: bold;
    }

    """
    
    def __init__(self, title: str, options: list, *args, **kwargs):
        """
        Initialize inline picker.
        
        Args:
            title: The title to show at the top
            options: List of (label, value) tuples or just strings
        """
        super().__init__(*args, **kwargs)
        self.title = title
        self._options_data = options
        # Create a mapping from label to value for tuple options
        self._value_map = {}
        # Set borders (using primary color from theme)
        self.border_title = f"{title}"
        self.border_subtitle = f"{len(options)} items"
        
    def on_mount(self) -> None:
        """Add options when mounted."""
        # Set border using the theme's primary color
        # Access the app's theme to get the actual color value
        try:
            theme = self.app.theme_variables
            primary_color = theme.get("primary", "#989af5")
            self.styles.border = ("solid", primary_color)
        except Exception:
            # Fallback to the valbot-dark primary color
            self.styles.border = ("solid", "#989af5")
        
        # Clear any existing options first
        self.clear_options()
        
        # Build list of options - use Rich Text objects that respect theme colors
        from rich.text import Text
        
        for idx, item in enumerate(self._options_data):
            if isinstance(item, tuple):
                label, value = item
                # Store the mapping from index to value
                self._value_map[idx] = value
                # Create a Rich Text object without hardcoded color - will use theme $text
                text = Text(str(label))
                self.add_option(text)
            else:
                # For simple strings, value is the same as label
                self._value_map[idx] = item
                # Create a Rich Text object without hardcoded color - will use theme $text
                text = Text(str(item))
                self.add_option(text)
        
        # Log for debugging
        self.log(f"InlinePicker mounted with {len(self._options_data)} options")
        self.log(f"Option count after adding: {len(self._options)}")
        
        # Force a refresh after adding all options
        self.refresh(layout=True)
        
        # Auto-highlight the first option after refresh completes
        if len(self._options_data) > 0:
            def highlight_first():
                self.highlighted = 0
                self.refresh()
            self.call_after_refresh(highlight_first)
    
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        # Get the actual value from our mapping using the option index
        selected_idx = event.option_index
        selected_value = self._value_map.get(selected_idx, event.option.id)
        self.post_message(InlinePickerSelected(self, selected_value))
    
    def action_cancel_picker(self) -> None:
        """Cancel the picker."""
        # Post a message with None value to indicate cancellation
        self.post_message(InlinePickerSelected(self, None))
        # Also try to call the parent screen's cancel action as fallback
        try:
            if hasattr(self.screen, 'action_cancel'):
                self.screen.action_cancel()
        except Exception:
            pass
    
    # Delegate parent actions to the main screen
    async def action_parent_agent_picker(self) -> None:
        """Delegate to parent screen's agent picker action."""
        if hasattr(self.screen, 'action_agent_picker'):
            await self.screen.action_agent_picker()
    
    async def action_parent_change_model(self) -> None:
        """Delegate to parent screen's change model action."""
        if hasattr(self.screen, 'action_change_model'):
            await self.screen.action_change_model()
    
    def action_parent_toggle_files(self) -> None:
        """Delegate to parent screen's toggle files action."""
        if hasattr(self.screen, 'action_toggle_files'):
            self.screen.action_toggle_files()
    
    def action_parent_new_chat(self) -> None:
        """Delegate to parent screen's new chat action."""
        if hasattr(self.screen, 'action_new_chat'):
            self.screen.action_new_chat()
    
    def action_parent_save_session(self) -> None:
        """Delegate to parent screen's save session action."""
        if hasattr(self.screen, 'action_save_session'):
            self.screen.action_save_session()
    
    def action_parent_load_session(self) -> None:
        """Delegate to parent screen's load session action."""
        if hasattr(self.screen, 'action_load_session'):
            self.screen.action_load_session()
    
    def action_parent_quit(self) -> None:
        """Delegate to parent app's quit action."""
        self.app.exit()


class AutocompleteSelected(Message):
    """Custom message for autocomplete selection."""
    def __init__(self, picker, command):
        super().__init__()
        self.picker = picker
        self.command = command


class AutocompleteOverlay(Container):
    """Floating autocomplete overlay for command suggestions."""
    
    DEFAULT_CSS = """
    AutocompleteOverlay {
        layer: overlay;
        width: 1fr;
        height: auto;
        max-height: 20;
        dock: bottom;
        offset-y: -8;
        background: $surface;
        padding: 0;
        display: none;
        overflow-y: auto;
        border: solid $primary;
    }
    
    AutocompleteOverlay > #autocomplete-list {
        width: 100%;
        height: auto;
        max-height: 18;
        background: transparent;
        border: none;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    
    AutocompleteOverlay #autocomplete-list > .option-list--option {
        background: $panel;
        padding: 0 2;
        height: auto;
        min-height: 1;
        margin: 0 0 1 0;
    }
    
    AutocompleteOverlay #autocomplete-list > .option-list--option-highlighted {
        background: $primary-darken-2 !important;
        text-style: bold;
    }
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize autocomplete overlay."""
        super().__init__(*args, **kwargs)
        self._commands = []
        self._option_list = None
        self.border_title = f"{EMOJI['lightbulb']} Suggestions"
    
    def compose(self) -> ComposeResult:
        """Compose the overlay contents."""
        option_list = OptionList(id="autocomplete-list")
        option_list.can_focus = True
        self._option_list = option_list
        yield option_list
    
    def on_key(self, event: events.Key) -> None:
        """Handle key events for the autocomplete overlay."""
        if event.key == "escape":
            self.styles.display = "none"
            # Clear file autocomplete context
            if hasattr(self.screen, '_file_autocomplete_context'):
                self.screen._file_autocomplete_context = None
            # Return focus to input
            if hasattr(self.screen, '_command_input'):
                self.screen._command_input.focus()
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            # If at the top, return to input
            if self._option_list and self._option_list.highlighted == 0:
                self.styles.display = "none"
                # Clear file autocomplete context
                if hasattr(self.screen, '_file_autocomplete_context'):
                    self.screen._file_autocomplete_context = None
                if hasattr(self.screen, '_command_input'):
                    self.screen._command_input.focus()
                event.prevent_default()
                event.stop()

    
    def update_suggestions(self, commands: list) -> None:
        """Update the list of command suggestions."""
        self._commands = commands
        
        if not self._option_list:
            return
        
        self._option_list.clear_options()
        
        if not commands:
            self.styles.display = "none"
            return
        
        from rich.text import Text
        
        for command, description in commands:
            # Format: /command - description
            text = Text()
            text.append(command)
            if description:
                text.append(f" - {description}", style="dim")
            self._option_list.add_option(text)
        
        # Auto-highlight the first option
        if len(commands) > 0:
            self._option_list.highlighted = 0
        
        self.styles.display = "block"
        self.refresh(layout=True)
    
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        selected_idx = event.option_index
        if 0 <= selected_idx < len(self._commands):
            command = self._commands[selected_idx][0]
            self.post_message(AutocompleteSelected(self, command))
        event.stop()
    
    def get_option_list(self) -> Optional[OptionList]:
        """Get the option list widget."""
        return self._option_list


class MainScreen(Screen):
    """Main application screen with modern Material Design layout."""
    
    BINDINGS = [
        Binding("ctrl+a", "agent_picker", "Agent", priority=True),
        Binding("ctrl+f", "toggle_files", "Files", priority=True),
        Binding("ctrl+n", "new_chat", "New Chat", priority=True),
        Binding("ctrl+m", "change_model", "Model", priority=True),
        Binding("ctrl+s", "save_session", "Save", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]
    
    CSS = """
    MainScreen {
        background: $background;
        layers: base overlay;
    }
    
    #main-container {
        layout: vertical;
        height: 100%;
        width: 100%;
        background: $background;
    }
    
    #chat-container {
        height: 1fr;
        width: 100%;
    }
    
    #file-panel {
        dock: right;
    }
    
    #input-container {
        height: auto;
        width: 100%;
        padding: 1 2;
        background: $surface;
        dock: bottom;
    }
    
    #command-input {
        width: 100%;
        border: none;
    }
    
    /* Markdown Styling - Let Textual handle natural styling */
    Markdown {
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    /* Scrollbar Styling */
    ScrollableContainer > .scrollbar {
        background: $surface;
        color: $primary;
    }
    
    ScrollableContainer > .scrollbar:hover {
        background: $panel;
        color: $accent;
    }
    
    /* Remove focus borders */
    *:focus {
        border: none !important;
    }
    
    Input:focus {
        border: none !important;
    }
    
    Static:focus {
        border: none !important;
    }
    
    Container:focus {
        border: none !important;
    }
    
    /*Footer */
    
    Footer {
        background: $surface;
    }
    
    Footer > .footer--key {
        background: $primary-darken-1;
    }
    """
    
    def __init__(self, config_manager: ConfigManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_manager = config_manager
        self.chatbot = None
        self.processing = False
        self.show_files = False
        self._active_picker_type = None  # Track which picker is currently open ('agent', 'model', etc.)
        self._cancel_streaming = False  # Flag to cancel ongoing streaming
        self._cancel_agent = False  # Flag to cancel ongoing agent execution
        self._file_autocomplete_context = None  # Track file path autocomplete context
        
    def compose(self) -> ComposeResult:
        """Compose the modern Material Design layout."""
        yield Header(show_clock=True)
        
        # File explorer overlay (appears on top when toggled)
        yield FileExplorerPanel(id="file-panel")
        
        with Container(id="main-container"):
            # Main chat area (scrollable)
            with Vertical(id="chat-container"):
                yield ChatPanel(id="chat-panel")
        
        # Input area at bottom (fixed)
        with Vertical(id="input-container"):
            yield CommandInput(id="command-input")
            yield StatusBar(id="status-bar")
        
        # Autocomplete overlay (on its own layer)
        yield AutocompleteOverlay(id="autocomplete-overlay")
            
        yield Footer()
        
    async def on_mount(self) -> None:
        """Initialize the application after mounting."""
        await self.initialize_chatbot()
        
        # Store reference to autocomplete overlay
        self._autocomplete_overlay = self.query_one("#autocomplete-overlay", AutocompleteOverlay)
        
        self.query_one("#command-input", CommandInput).focus()
        
        # Initialize CommandManager with TUI console wrapper
        if self.chatbot:
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            tui_console = TUIConsoleWrapper(chat_panel, self.app)
            self.chatbot.console = tui_console
            self.chatbot.command_manager = CommandManager(
                self.chatbot, 
                self.chatbot.plugin_manager,
                tui_console
            )
        
        # Display welcome message
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Display the welcome message."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        welcome = f"""# ░▒▓ Welcome to ValBot TUI ▓▒░

### {EMOJI['clipboard']} Available Commands

Type `/help` for complete list of commands, or try these:

- `/` - Show all commands
- `/prompts` - Show custom prompts  
- `/agent` - Run an agent workflow
- `/model` - Change AI model
- `/context` - Load files into context
- `/terminal <cmd>` - Run shell commands


### {EMOJI['info']} Tips
- Reference local files directly in chat
- Press `TAB` to autocomplete file paths
- Use key-bindings for quick access. E.g.: `ctrl + a` for Agent
- Change themes from palette


### {EMOJI['lightbulb']} Getting Started

Type your message and press **Enter** to chat with ValBot!

Try:
> Help me write a Python function to parse JSON

> Summarise *<file_path>*
"""
        # Use 'assistant' role but with plain styling (no border/background)
        msg = ChatMessage("assistant", welcome)
        msg.remove_class("assistant-message")  # Remove default styling
        msg.add_class("welcome-message")  # Add plain styling
        chat_panel.messages.append(msg)
        chat_panel.mount(msg)
        chat_panel.scroll_end(animate=True)
        
    async def initialize_chatbot(self):
        """Initialize the ChatBot instance."""
        try:
            agent_config = self.config_manager.get_setting('agent_model_config')
            agent_model = initialize_agent_model(
                endpoint=agent_config['default_endpoint'], 
                model_name=agent_config['default_model']
            )
            plugin_manager = PluginManager(self.config_manager, model=agent_model)
            
            self.chatbot = ChatBot(
                agent_model=agent_model,
                config_manager=self.config_manager,
                plugin_manager=plugin_manager
            )
            
            # Add system prompt to conversation history (like CLI does)
            system_prompt = self.config_manager.get_setting(
                "chat_model_config.system_prompt", 
                "You are a helpful assistant. Prioritize markdown format and code blocks when applicable."
            )
            self.chatbot.context_manager.conversation_history.append({
                "role": "system", 
                "content": system_prompt
            })
            
            # Update status bar
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.model_name = self.config_manager.get_setting("chat_model_config.default_model", "gpt-4o")
            
        except Exception as e:
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel.add_message("error", f"""## {EMOJI['cross']} Initialization Error

Failed to initialize chatbot:

```
{str(e)}
```

Please check your configuration and try again.
""")
    
    def get_available_commands(self) -> list:
        """Get list of all available commands with descriptions."""
        commands = []
        
        # Built-in TUI commands
        builtin_commands = [
            ("/help", "Show help information"),
            ("/quit", "Exit the application"),
            ("/clear", "Start a new conversation"),
            ("/new", "Start a new conversation"),
            ("/agent", "Select and run an agent workflow"),
            ("/model", "Change the AI model"),
            ("/context", "Load files into conversation context"),
            ("/file", "Display file contents"),
            ("/terminal", "Execute shell commands"),
            ("/multi", "Multi-line input via system editor"),
            ("/prompts", "Show custom prompts"),
            ("/commands", "Show all available commands"),
            ("/settings", "Show settings information"),
            ("/reload", "Reinitialize chatbot"),
            ("/update", "Update information"),
            ("/add_agent", "Add agent information"),
            ("/add_tool", "Add tool information"),
        ]
        commands.extend(builtin_commands)
        
        # Add commands from CommandManager if available
        if self.chatbot and hasattr(self.chatbot, 'command_manager'):
            cmd_mgr = self.chatbot.command_manager
            
            # Add registered commands
            if hasattr(cmd_mgr, 'command_registry'):
                for cmd_name, cmd_info in cmd_mgr.command_registry.items():
                    if not any(c[0] == cmd_name for c in commands):
                        commands.append((cmd_name, cmd_info.get('description', '')))
            
            # Add custom prompts
            if hasattr(cmd_mgr, 'prompts'):
                for prompt_name, prompt_obj in cmd_mgr.prompts.items():
                    if not any(c[0] == prompt_name for c in commands):
                        commands.append((prompt_name, prompt_obj.description))
        
        return commands
    
    def filter_commands(self, input_text: str) -> list:
        """Filter commands based on input text."""
        if not input_text.startswith("/"):
            return []
        
        # Get the command part (without /)
        query = input_text[1:].lower()
        
        # Get all available commands
        all_commands = self.get_available_commands()
        
        # If query is empty, show all commands
        if not query:
            return all_commands
        
        # Filter commands that start with or contain the query
        filtered = []
        for cmd, desc in all_commands:
            cmd_name = cmd[1:] if cmd.startswith("/") else cmd  # Remove leading /
            if cmd_name.lower().startswith(query):
                filtered.append((cmd, desc))
        
        # If no exact prefix matches, try contains matching
        if not filtered:
            for cmd, desc in all_commands:
                cmd_name = cmd[1:] if cmd.startswith("/") else cmd
                if query in cmd_name.lower():
                    filtered.append((cmd, desc))
        
        return filtered
    
    def on_command_input_changed(self, text: str) -> None:
        """Handle command input text changes to show/update autocomplete."""
        if not text.startswith("/"):
            self.hide_autocomplete()
            return
        
        # Clear file autocomplete context when showing commands
        self._file_autocomplete_context = None
        
        # Filter commands based on input
        filtered_commands = self.filter_commands(text)
        
        if not filtered_commands:
            self.hide_autocomplete()
            return
        
        # Show or update autocomplete overlay
        self.show_autocomplete(filtered_commands)
    
    def on_text_changed_check_paths(self, text: str, cursor_pos: tuple) -> None:
        """Check if current word is a partial path and show autocomplete."""
        if not text.strip():
            self.hide_autocomplete()
            return
        
        # Get the word at cursor
        word, start_col, end_col = self.get_word_at_cursor(text, cursor_pos)
        
        if not word or len(word) < 1:  # Require at least 1 character
            # Hide autocomplete if word is too short
            if hasattr(self, '_file_autocomplete_context') and self._file_autocomplete_context:
                self.hide_autocomplete()
            return
        
        # Get matching file paths
        matches = self.get_file_path_matches(word)
        
        if not matches:
            # Hide autocomplete if no matches
            if hasattr(self, '_file_autocomplete_context') and self._file_autocomplete_context:
                self.hide_autocomplete()
            return
        
        # Format matches for display
        formatted_matches = []
        for display_name, completion, is_dir in matches:
            desc = f"{EMOJI['folder']} Directory" if is_dir else f"{EMOJI['file']} File"
            formatted_matches.append((completion, desc))
        
        # Store the current word info for completion
        self._file_autocomplete_context = {
            'word': word,
            'start_col': start_col,
            'end_col': end_col,
            'cursor_row': cursor_pos[0],
            'matches': matches  # Store full match info including is_dir flag
        }
        
        # Show autocomplete with file paths
        self.show_autocomplete(formatted_matches)
    
    def show_autocomplete(self, commands: list) -> None:
        """Show autocomplete overlay with filtered commands."""
        if not hasattr(self, '_autocomplete_overlay') or self._autocomplete_overlay is None:
            return
        
        # Update suggestions
        self._autocomplete_overlay.update_suggestions(commands)
    
    def hide_autocomplete(self) -> None:
        """Hide autocomplete overlay."""
        if hasattr(self, '_autocomplete_overlay') and self._autocomplete_overlay:
            self._autocomplete_overlay.styles.display = "none"
            # Clear file autocomplete context when hiding
            self._file_autocomplete_context = None
            # Return focus to input
            command_input = self.query_one("#command-input", CommandInput)
            command_input.focus()
    
    def get_word_at_cursor(self, text: str, cursor_pos: tuple) -> tuple:
        """
        Get the word/path at cursor position.
        
        Returns:
            (word, start_col, end_col) tuple
        """
        row, col = cursor_pos
        lines = text.split('\n')
        
        if row >= len(lines):
            return ("", col, col)
        
        line = lines[row]
        if col > len(line):
            col = len(line)
        
        # Find start of word (go backwards until whitespace or start)
        start = col
        while start > 0 and not line[start - 1].isspace():
            start -= 1
        
        # Find end of word (go forwards until whitespace or end)
        end = col
        while end < len(line) and not line[end].isspace():
            end += 1
        
        word = line[start:end]
        return (word, start, end)
    
    def get_file_path_matches(self, partial_path: str, max_results: int = 20) -> list:
        """
        Get file/folder paths that match the partial path.
        
        Returns:
            List of (display_name, full_path, is_dir) tuples
        """
        import os
        from pathlib import Path
        
        if not partial_path:
            return []
        
        # Expand user home directory
        partial_path = os.path.expanduser(partial_path)
        
        # Handle both absolute and relative paths
        if os.path.isabs(partial_path):
            base_path = os.path.dirname(partial_path)
            prefix = os.path.basename(partial_path)
        else:
            # Relative path - search from current working directory
            base_path = os.getcwd()
            if os.path.dirname(partial_path):
                base_path = os.path.join(base_path, os.path.dirname(partial_path))
            prefix = os.path.basename(partial_path)
        
        matches = []
        
        try:
            # If base path doesn't exist, try parent
            if not os.path.exists(base_path):
                return []
            
            # List directory contents
            entries = os.listdir(base_path)
            
            for entry in entries:
                # Filter by prefix
                if prefix and not entry.lower().startswith(prefix.lower()):
                    continue
                
                full_path = os.path.join(base_path, entry)
                is_dir = os.path.isdir(full_path)
                
                # Create display name (add / for directories)
                display_name = entry + ("/" if is_dir else "")
                
                # Create the completion path
                if os.path.dirname(partial_path):
                    completion = os.path.join(os.path.dirname(partial_path), entry)
                else:
                    completion = entry
                
                matches.append((display_name, completion, is_dir))
                
                if len(matches) >= max_results:
                    break
            
            # Sort: directories first, then files, both alphabetically
            matches.sort(key=lambda x: (not x[2], x[0].lower()))
            
        except (PermissionError, OSError):
            # Can't read directory, return empty
            pass
        
        return matches
    
    def try_file_path_autocomplete(self, text: str, cursor_pos: tuple) -> bool:
        """
        Try to show file path autocomplete or trigger if already showing.
        
        Returns:
            True if autocomplete was shown/triggered, False otherwise
        """
        # Check if autocomplete is already showing file paths
        if hasattr(self, '_file_autocomplete_context') and self._file_autocomplete_context:
            if hasattr(self, '_autocomplete_overlay') and self._autocomplete_overlay.styles.display != "none":
                # Autocomplete is showing - focus it for navigation
                option_list = self._autocomplete_overlay.get_option_list()
                if option_list:
                    option_list.focus()
                return True
        
        # Get the word at cursor
        word, start_col, end_col = self.get_word_at_cursor(text, cursor_pos)
        
        if not word:
            return False
        
        # Get matching file paths
        matches = self.get_file_path_matches(word)
        
        if not matches:
            return False
        
        # If only one exact match and it's not a directory, complete it directly
        if len(matches) == 1 and not matches[0][2]:
            # Direct completion
            command_input = self.query_one("#command-input", CommandInput)
            lines = text.split('\n')
            row, col = cursor_pos
            
            if row < len(lines):
                line = lines[row]
                # Replace the word with the completion
                new_line = line[:start_col] + matches[0][1] + line[end_col:]
                lines[row] = new_line
                command_input.text = '\n'.join(lines)
                
                # Move cursor to end of completed word
                new_cursor_col = start_col + len(matches[0][1])
                command_input.move_cursor((row, new_cursor_col))
            
            return True
        
        # Multiple matches or single directory - trigger the real-time check to show autocomplete
        self.on_text_changed_check_paths(text, cursor_pos)
        
        return True
    
    @on(AutocompleteSelected)
    def handle_autocomplete_selected(self, event: AutocompleteSelected) -> None:
        """Handle autocomplete selection."""
        import os
        selection = event.command
        command_input = self.query_one("#command-input", CommandInput)
        
        # Check if this is a file path completion
        if hasattr(self, '_file_autocomplete_context') and self._file_autocomplete_context:
            context = self._file_autocomplete_context
            text = command_input.text
            lines = text.split('\n')
            row = context['cursor_row']
            
            # Find the selected match to check if it's a directory
            is_directory = False
            if 'matches' in context:
                for display_name, completion, is_dir in context['matches']:
                    if completion == selection:
                        is_directory = is_dir
                        break
            
            # Add appropriate separator for directories
            if is_directory:
                # Use OS-appropriate separator (/ for Unix-like, \ for Windows)
                separator = os.sep
                selection = selection + separator
            
            if row < len(lines):
                line = lines[row]
                # Replace the word with the selected path
                new_line = line[:context['start_col']] + selection + line[context['end_col']:]
                lines[row] = new_line
                command_input.text = '\n'.join(lines)
                
                # Move cursor to end of completed path
                new_cursor_col = context['start_col'] + len(selection)
                command_input.move_cursor((row, new_cursor_col))
            
            # Always close the autocomplete and clear context after selection
            # User must press Tab again to trigger autocomplete
            self.hide_autocomplete()
            self._file_autocomplete_context = None
            
        else:
            # Command completion - replace entire input with command + space
            command_input.text = selection + " "
            command_input.move_cursor_relative(columns=len(selection) + 1)
            
            # Hide autocomplete
            self.hide_autocomplete()
        
        # Return focus to input
        command_input.focus()
        command_input.focus()
    
    def _update_footer_bindings(self, picker_open: bool):
        """Update footer bindings based on whether picker is open.
        
        Args:
            picker_open: True if picker is open, False otherwise
        """
        # The bindings are already in the correct order:
        # agent, files, new chat, model, save, load, cancel, quit
        # When picker is open, we just need to ensure the footer refreshes
        # Textual will automatically show the bindings from BINDINGS
        try:
            footer = self.app.query_one(Footer)
            footer.refresh()
        except Exception:
            pass
    
    async def show_inline_picker(self, title: str, options: list, picker_type: str = None):
        """Show an inline picker that replaces the textarea.
        
        Args:
            title: The title for the picker
            options: List of options to display
            picker_type: Type identifier for the picker ('agent', 'model', etc.)
        """
        # Check if same picker type is already open - if so, close it instead
        if picker_type and self._active_picker_type == picker_type:
            self.hide_inline_picker()
            return
        
        # Debug: Log what we're about to show
        self.log(f"show_inline_picker called with title='{title}', options={options[:3] if len(options) > 3 else options}...")
        
        # First, hide any existing inline picker (but don't clear callback yet)
        try:
            existing_picker = self.query_one("#inline-picker", InlinePicker)
            await existing_picker.remove()
        except Exception:
            pass
        
        input_container = self.query_one("#input-container", Vertical)
        command_input = self.query_one("#command-input", CommandInput)
        
        # Hide the command input
        command_input.display = False
        
        # Create and mount the picker
        picker = InlinePicker(title, options, id="inline-picker")
        await input_container.mount(picker, before=0)  # Mount at the top of the container
        
        # Track which picker is now open
        self._active_picker_type = picker_type
        
        # Update footer bindings to show the desired order when picker is open
        self._update_footer_bindings(picker_open=True)
        
        # Debug: Check the picker state after mounting
        self.log(f"Picker mounted. Option count: {len(picker._options)}")
        for idx, opt in enumerate(picker._options[:3]):
            self.log(f"  Option {idx}: prompt='{opt.prompt}'")
        
        picker.focus()
    
    def hide_inline_picker(self):
        """Hide the inline picker and restore the textarea."""
        try:
            picker = self.query_one("#inline-picker", InlinePicker)
            picker.remove()
        except Exception:
            pass
        
        # Clear the active picker type
        self._active_picker_type = None
        
        # Restore original footer bindings
        self._update_footer_bindings(picker_open=False)
        
        # Show the command input again
        command_input = self.query_one("#command-input", CommandInput)
        command_input.display = True
        command_input.focus()
    
    @on(InlinePickerSelected)
    async def handle_picker_selection(self, event: InlinePickerSelected) -> None:
        """Handle selection from inline picker."""
        selected_value = event.value
        
        # Check if AI is processing - block selection
        if self.processing and selected_value is not None:
            self.app.notify(
                "⚠️ AI is busy.\nPress `Esc` to cancel current response.",
                severity="warning",
                timeout=2
            )
            # Don't hide the picker - let user try again or cancel
            return
        
        # Hide the picker first
        self.hide_inline_picker()
        
        # Check what type of picker this was
        if hasattr(self, '_picker_callback'):
            callback = self._picker_callback
            self._picker_callback = None
            
            # Execute the callback only if not cancelled
            if callback and selected_value is not None:
                # Check if callback is async or sync
                import inspect
                if inspect.iscoroutinefunction(callback):
                    await callback(selected_value)
                else:
                    # Call sync function directly
                    callback(selected_value)
    
    def action_cancel(self):
        """Cancel current operation (e.g., close inline picker, cancel command/agent prompts)."""
        # Cancel any ongoing streaming
        if self._cancel_streaming == False:  # Only set flag and notify if not already cancelled
            self._cancel_streaming = True
            # Check if we're actually streaming
            if self.processing:
                self.app.notify(f"{EMOJI['cross']} Response cancelled.", severity="warning", timeout=2)
        
        # Cancel any ongoing agent execution
        if hasattr(self, '_agent_input_state') and self._agent_input_state:
            # Check if agent is actively running (not just waiting for input)
            if self._agent_input_state.get('waiting_for_input', False):
                # Agent is waiting for input - abort it
                self._agent_input_state['result'] = None
                self._agent_input_state['waiting_for_input'] = False
                self._agent_input_state['cancelled'] = True  # Mark as cancelled
                self._agent_input_state['ready'].set()
                # Clear current agent indicator
                chat_panel = self.query_one("#chat-panel", ChatPanel)
                chat_panel.current_agent = None
                # Don't show message here - the agent thread will show it
                return
        
        # Check if we're waiting for command input
        if hasattr(self, '_command_input_state') and self._command_input_state.get('waiting_for_input', False):
            self._command_input_state['result'] = None
            self._command_input_state['waiting_for_input'] = False
            self._command_input_state['ready'].set()
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel.add_message("system", f"{EMOJI['cross']} Command cancelled.")
            return
        
        # Check if inline picker is visible
        try:
            picker = self.query_one("#inline-picker", InlinePicker)
            self.hide_inline_picker()
            # Clear any pending callback
            if hasattr(self, '_picker_callback'):
                self._picker_callback = None
        except Exception:
            pass
    
    @on(TextAreaSubmitted)
    async def handle_input(self, event: TextAreaSubmitted) -> None:
        """Handle user input submission."""
        message = event.value
        
        # Check if we're waiting for command input (like /add_agent prompts)
        if hasattr(self, '_command_input_state') and self._command_input_state.get('waiting_for_input', False):
            # Display user's response
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            display_msg = message if message else ""
            chat_panel.add_message("user", display_msg)
            
            # Provide the response to the command
            self._command_input_state['result'] = message
            self._command_input_state['waiting_for_input'] = False
            self._command_input_state['ready'].set()
            return
        
        # Check if we're waiting for agent input (allow blank entries for agents)
        if hasattr(self, '_agent_input_state') and self._agent_input_state.get('waiting_for_input', False):
            # Display user's response (show empty message as well)
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            display_msg = message if message else ""
            chat_panel.add_message("user", display_msg)
            
            # Provide the response to the agent (can be empty string)
            self._agent_input_state['result'] = message
            self._agent_input_state['waiting_for_input'] = False  # Reset the flag
            self._agent_input_state['ready'].set()
            return
        
        # For normal chat/commands, require non-empty input
        if not message.strip():
            return
        
        # Process command or message
        if message.strip().startswith("/"):
            await self.handle_command(message.strip())
        else:
            # For multiline messages, we need to format them properly for Markdown
            # Convert single newlines to double newlines for proper Markdown line breaks
            formatted_message = message.replace('\n', '\n\n')
            
            # Display user message immediately
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel.add_message("user", formatted_message)
            
            # Send the original message to the chatbot (without double newlines) in background thread
            import threading
            thread = threading.Thread(
                target=self._send_chat_message_in_thread,
                args=(message,),
                daemon=True
            )
            thread.start()
    
    async def handle_command(self, command: str):
        """Handle slash commands using CommandManager."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        
        # Display the user's command input in the chat with backtick formatting
        chat_panel.add_message("user", f"`{command}`")
        
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Check for /add_agent and /add_tool commands - handle with threading directly
        if cmd == "/add_agent":
            # Start in a separate thread using Python's threading module directly
            import threading
            thread = threading.Thread(
                target=self._run_add_agent_in_thread,
                args=(args,),
                daemon=True
            )
            thread.start()
            return
        
        elif cmd == "/add_tool":
            # Start in a separate thread using Python's threading module directly
            import threading
            thread = threading.Thread(
                target=self._run_add_tool_in_thread,
                args=(args,),
                daemon=True
            )
            thread.start()
            return
        
        # Handle TUI-specific commands first (these need special UI handling)
        if cmd == "/quit":
            self.app.exit()
            return
        
        elif cmd == "/agent":
            # Override CLI's agent command with TUI-specific inline picker
            if self.chatbot and hasattr(self.chatbot, 'plugin_manager'):
                agents = self.chatbot.plugin_manager.plugin_info
                if agents:
                    async def on_agent_selected(selected_agent_name):
                        if selected_agent_name:
                            # Find the agent details
                            for agent_name, agent_desc in agents:
                                if agent_name == selected_agent_name:
                                    # Run agent in background thread using threading directly
                                    import threading
                                    thread = threading.Thread(
                                        target=self._run_agent_workflow_in_thread,
                                        args=(agent_name, agent_desc),
                                        daemon=True
                                    )
                                    thread.start()
                                    break
                    
                    # Format agents for display: show name and description
                    agent_options = [(f"{name}: {desc}", name) for name, desc in agents]
                    
                    # Show the inline picker first
                    await self.show_inline_picker(f"{EMOJI['robot']} Select Agent Workflow", agent_options)
                    
                    # Store callback for when selection is made (after picker is shown)
                    self._picker_callback = on_agent_selected
                else:
                    chat_panel.add_message("system", "No agents available. Use `/add_agent` to add one.")
            else:
                chat_panel.add_message("error", "Plugin manager not initialized.")
            return
        
        elif cmd == "/model":
            # Override CLI's model command with TUI-specific picker
            await self.action_change_model()
            return
        
        elif cmd == "/clear" or cmd == "/new":
            # Override CLI's clear command to completely restart from scratch
            chat_panel.clear_messages()
            # Clear any streaming message reference
            if hasattr(self, '_streaming_msg'):
                self._streaming_msg = None
            if hasattr(self, '_last_code_block_count'):
                del self._last_code_block_count
            
            # Completely reinitialize the chatbot (like quitting and relaunching)
            if self.chatbot:
                # Clear conversation history
                self.chatbot.clear_conversation()
                
                # Reinitialize system prompt (like during startup)
                system_prompt = self.config_manager.get_setting(
                    "chat_model_config.system_prompt", 
                    "You are a helpful assistant. Prioritize markdown format and code blocks when applicable."
                )
                self.chatbot.context_manager.conversation_history.append({
                    "role": "system", 
                    "content": system_prompt
                })
                
                # Reset conversation ID to start fresh backend session
                self.chatbot._conversation_id = None
            
            chat_panel.add_message("system", f"{EMOJI['checkmark']} Started new chat.")
            return
            
        elif cmd == "/help":
            chat_panel.add_message("system", """## 📖 ValBot Help

**Available Commands:**
- `/agent` - Select and run an agent workflow
- `/model` - Change the AI model
- `/context <files>` - Load files into conversation context  
- `/clear` or `/new` - Start a new conversation
- `/prompts` - Show available custom prompts
- `/commands` - Show all available commands
- `/settings` - Show settings (CLI mode only)
- `/quit` - Exit the application

**Usage Tips:**
- Type messages normally for chat
- Use `/` prefix for commands
- Agent workflows provide specialized functionality
- Context files are remembered throughout conversation

For more detailed help, try the CLI mode with `--help` flag.
""")
            return
            
        elif cmd == "/settings":
            chat_panel.add_message("system", """## {EMOJI['gear']} Settings

Settings management is available in CLI mode only.
To modify settings, exit TUI and run:

```bash
python app.py
/settings
```

**Current Model:** """ + str(self.config_manager.get_setting("chat_model_config.default_model", "gpt-4o")) + """

Use `/model` to change the current model.
""")
            return
            
        # Try to use CommandManager for standard commands
        if hasattr(self.chatbot, 'command_manager'):
            # Check if command exists in CommandManager
            if cmd in self.chatbot.command_manager.command_registry or cmd in self.chatbot.command_manager.prompts:
                try:
                    # Let CommandManager handle the command
                    self.chatbot.command_manager.handle_command(command)
                    return
                except Exception as e:
                    chat_panel.add_message("error", f"""## {EMOJI['cross']} Command Error

Error executing command `{cmd}`:

```
{str(e)}
```
""")
                    return
        
        # Handle additional TUI-specific commands not in CommandManager
        if cmd == "/help":
            help_text = """# 📚 ValBot TUI Help

## Available Commands

### Chat Management
- `/new` or `/clear` - Start a new chat session
- `/quit` - Exit the application
- `/prompts` - Show available custom prompts
- `/commands` - Show all available commands

### Model & AI Configuration
- `/model` - Change AI model (interactive picker with ↑/↓ keys)
  - Available models: gpt-4o, gpt-5, gpt-4.1, gpt-oss:20b
  
### Context & File Management
- `/context <file_or_pattern>` - Load file(s) into conversation context
  - Example: `/context file.py` - Load single file
  - Example: `/context *.py` - Load all Python files
  - Example: `/context src/**/*.js` - Load all JS files in src/
- `/file <path>` - Display file content with syntax highlighting
  - Example: `/file ./config.py`

### Agent System (Advanced Workflows)
- `/agent` - Run an agent flow (interactive selection with ↑/↓ keys)
  - Agents can perform complex multi-step tasks
  - Select from available agents like File Edit, Terminal, Spec Expert, etc.
- `/add_agent <url_or_path>` - Add a new agent from git repo or local path
- `/add_tool <url_or_path>` - Add a new tool from git repo or local path

### System Integration
- `/terminal <command>` - Execute a shell command
  - Example: `/terminal ls -la`
  - Example: `/terminal python --version`
  - Example: `/terminal git status`
- `/multi` - Multi-line input using system editor
  - Opens your default editor (set via $EDITOR env variable)
  - Default: notepad (Windows) or vim (Linux/Mac)

### Configuration & Updates
- `/settings` - View settings information
- `/reload` - Reinitialize chatbot with current configuration
- `/update` - Check for updates (see instructions)

## {EMOJI['keyboard']} Keyboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+A** | Agent | Select and run an agent workflow |
| **Ctrl+F** | Files | Toggle file explorer panel |
| **Ctrl+N** | New Chat | Clear conversation and start fresh |
| **Ctrl+M** | Model | Open model picker dialog |
| **Ctrl+S** | Save | Save session (coming soon) |
| **Ctrl+L** | Load | Load session (coming soon) |
| **Escape** | Cancel | Cancel current operation/close dialogs |
| **Ctrl+Q** | Quit | Exit the application |

## 💬 Chat Features

- **Markdown Support**: Full markdown rendering with headers, lists, tables, quotes
- **Code Highlighting**: Syntax-highlighted code blocks with copy buttons
  - Supports: Python, JavaScript, Java, C++, Go, Rust, and more
- **Streaming**: Real-time response streaming as AI generates text
- **Reasoning Display**: GPT-5 shows its thinking process (if enabled)
  - Configure in `user_config.json`: `"display_reasoning": true`
  - Set effort level: `"reasoning_effort": "low|medium|high"`
- **Context Awareness**: Maintains full conversation history
- **Agents**: Run complex agentic workflows for tasks like:
  - File editing
  - Terminal operations
  - Spec analysis
  - Project scaffolding
  - And more!

## 🎨 UI Features

- **Material Design**: Modern, beautiful dark theme interface
- **Gradient Accents**: Colorful borders and highlights
- **Message Types**: Color-coded by role (user/assistant/system/error)
  - 🔵 Blue border = Assistant response
  - 🟡 Yellow border = System message
  - 🔴 Red border = Error message
- **Responsive Layout**: Adapts to your terminal size
- **Smooth Scrolling**: Elegant animations and transitions
- **Interactive Dialogs**: Beautiful modal dialogs for selection tasks

## 📚 Custom Prompts

ValBot supports custom prompts defined in your config file. These are shortcuts for common tasks:

- Use `/prompts` to see all available custom prompts
- Custom prompts can include placeholders and execute shell commands
- Example: `/my-prompt "argument1" "argument2"`

**Define custom prompts in `user_config.json`:**
```json
{
  "custom_prompts": [
    {
      "prompt_cmd": "review",
      "description": "Review code in a file",
      "prompt": "Please review this code: {file}",
      "args": ["file"]
    }
  ]
}
```

## {EMOJI['robot']} Agent Plugins

Agents are powerful workflows that can perform complex tasks:

**Built-in Agents:**
- **Spec Expert** - Q&A with specification documents
- **File Edit** - Edit files with AI assistance
- **File Creator** - Create new files with AI assistance
- **Terminal** - Run terminal commands intelligently
- **Repo Index** - Summarize and index repository context
- **Project** - Make project-wide changes and add features

**Using Agents:**
1. Type `/agent` or press Ctrl+M
2. Use ↑/↓ arrow keys to select
3. Press Enter to run
4. Follow the agent's prompts

## 🔧 Configuration Tips

Edit `user_config.json` to customize your experience:

```json
{
  "chat_model_config": {
    "default_model": "gpt-5",
    "You are a helpful assistant. Prioritize markdown format and code blocks when applicable.",
    "display_reasoning": false,
    "reasoning_effort": "low"
  },
  "agent_model_config": {
    "default_model": "gpt-4.1"
  },
  "general": {
    "display_commands_on_startup": true
  }
}
```

## 🆘 Troubleshooting

**Q: Commands not working?**
- Make sure command starts with `/`
- Check spelling carefully
- Use `/commands` to see all available commands

**Q: Can't see GPT-5 reasoning?**
- Set `"display_reasoning": true` in config
- Only works with gpt-5 models
- Restart TUI after config changes

**Q: File not loading with /context?**
- Check path is correct (use absolute paths if needed)
- Ensure you have read permissions
- Try with `/file` first to verify path

**Q: Agent not found?**
- Use `/agent` to see available agents
- Some agents may need to be installed
- Check config for agent definitions

**Q: How do I exit?**
- Press **Ctrl+Q** or type `/quit`

## 📖 More Resources

- **Full Documentation**: See `README_TUI.md`
- **Feature List**: See `TUI_FEATURES_IMPLEMENTED.md`
- **Quick Reference**: See `TUI_QUICK_REFERENCE_CARD.md`

Need more help? Just ask ValBot directly!

---
**ValBot TUI** - Your AI-Powered Assistant
"""
            chat_panel.add_message("system", help_text)
                
        elif cmd == "/terminal":
            if args:
                await chat_panel.add_terminal_output(args)
            else:
                chat_panel.add_message("system", """## {EMOJI['keyboard']} Terminal Command

**Usage**: `/terminal <command>`

**Examples**:
- `/terminal ls -la`
- `/terminal python --version`
- `/terminal git status`
""")
                
        elif cmd == "/file":
            if args:
                try:
                    file_path = Path(args).expanduser()
                    if file_path.exists() and file_path.is_file():
                        # Check file size before reading
                        file_size = file_path.stat().st_size
                        max_size_bytes = 10 * 1024 * 1024  # 10 MB limit for display
                        
                        if file_size > max_size_bytes:
                            chat_panel.add_message("error", 
                                f"{EMOJI['cross']} File too large to display: {file_size / (1024*1024):.2f} MB (limit: 10 MB)\n"
                                f"Use `/context` to load it into conversation context instead.")
                        else:
                            # Read in chunks for better memory handling
                            content_chunks = []
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                while True:
                                    chunk = f.read(8192)  # Read in 8KB chunks
                                    if not chunk:
                                        break
                                    content_chunks.append(chunk)
                            content = ''.join(content_chunks)
                            
                            # Determine file type for syntax highlighting
                            file_ext = file_path.suffix.lower()
                            lang_map = {
                                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                                '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
                                '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php',
                                '.html': 'html', '.css': 'css', '.json': 'json', '.xml': 'xml',
                                '.yaml': 'yaml', '.yml': 'yaml', '.sh': 'bash', '.md': 'markdown'
                            }
                            lang = lang_map.get(file_ext, 'text')
                            
                            formatted_content = f"""## 📄 File: `{file_path}`

```{lang}
{content}
```

**Lines**: {len(content.splitlines())} | **Size**: {file_size:,} bytes ({file_size / 1024:.2f} KB)
"""
                            chat_panel.add_message("system", formatted_content)
                    else:
                        chat_panel.add_message("error", f"{EMOJI['cross']} File not found: `{args}`")
                except MemoryError:
                    chat_panel.add_message("error", f"{EMOJI['cross']} Memory error: File too large to read")
                except Exception as e:
                    chat_panel.add_message("error", f"{EMOJI['cross']} Error reading file: {str(e)}")
            else:
                chat_panel.add_message("system", "**Usage**: `/file <path>`\n\nExample: `/file ./config.py`")
        
        elif cmd == "/context":
            # Load file(s) into conversation context
            if args and self.chatbot:
                try:
                    # Use context manager to load context
                    import glob
                    files = []
                    # Support glob patterns
                    for pattern in args.split():
                        expanded = glob.glob(pattern)
                        if expanded:
                            files.extend(expanded)
                        elif os.path.exists(pattern):
                            files.append(pattern)
                    
                    if files:
                        self.chatbot.context_manager.load_context(files)
                        file_list = "\n".join(f"- `{f}`" for f in files)
                        chat_panel.add_message("system", f"""## 📂 Context Loaded

Loaded {len(files)} file(s) into conversation context:

{file_list}

You can now ask questions about these files!
""")
                    else:
                        chat_panel.add_message("error", f"{EMOJI['cross']} No files found matching: `{args}`")
                except Exception as e:
                    chat_panel.add_message("error", f"{EMOJI['cross']} Error loading context: {str(e)}")
            else:
                chat_panel.add_message("system", """## 📂 Load Context

**Usage**: `/context <file_or_pattern>`

**Examples**:
- `/context main.py` - Load single file
- `/context *.py` - Load all Python files
- `/context src/**/*.js` - Load all JS files in src/
""")
        
        elif cmd == "/multi":
            # Multi-line input using system editor
            chat_panel.add_message("system", """## 📝 Multi-line Input

Opening your default editor for multi-line input...

**Note**: Set your `EDITOR` environment variable to choose your preferred editor.
Default: vim (Linux/Mac) or notepad (Windows)
""")
            try:
                editor = os.environ.get('EDITOR', 'notepad' if sys.platform == 'win32' else 'vim')
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.md') as temp_file:
                    temp_file_name = temp_file.name
                
                # Open editor
                subprocess.run([editor, temp_file_name], check=True)
                
                # Read content
                with open(temp_file_name, 'r') as f:
                    content = f.read().strip()
                
                os.remove(temp_file_name)
                
                if content:
                    chat_panel.add_message("user", content)
                    # Send in background thread
                    import threading
                    thread = threading.Thread(
                        target=self._send_chat_message_in_thread,
                        args=(content,),
                        daemon=True
                    )
                    thread.start()
                else:
                    chat_panel.add_message("system", "No content entered.")
                    
            except Exception as e:
                chat_panel.add_message("error", f"{EMOJI['cross']} Error with multi-line input: {str(e)}")
        
        elif cmd == "/settings":
            chat_panel.add_message("system", """## {EMOJI['gear']} Settings

The settings TUI is not yet available in this version.

You can manually edit your configuration file at:
`user_config.json`

**Available Settings**:
- `chat_model_config.default_model` - Default AI model
- `chat_model_config.system_prompt` - System instructions
- `chat_model_config.display_reasoning` - Show reasoning (GPT-5)
- `chat_model_config.reasoning_effort` - Reasoning effort level
- `agent_model_config.default_model` - Agent model
- `general.ascii_banner_size` - Banner size
- `general.display_commands_on_startup` - Show commands on start
""")
        
        elif cmd == "/reload":
            chat_panel.add_message("system", """## 🔄 Reload Configuration

Reinitializing chatbot with current settings...
""")
            try:
                await self.initialize_chatbot()
                chat_panel.add_message("system", f"{EMOJI['checkmark']} ChatBot reinitialized with current configuration!")
            except Exception as e:
                chat_panel.add_message("error", f"{EMOJI['cross']} Error reinitializing: {str(e)}")
        
        elif cmd == "/update":
            chat_panel.add_message("system", """## 📦 Check for Updates

The update feature is not yet fully integrated in TUI mode.

To update ValBot:
1. Exit the TUI (Ctrl+Q)
2. Run: `git pull` in the ValBot directory
3. Run: `pip install -r requirements.txt`
4. Restart ValBot TUI

Or use the CLI version with: `python app.py` and run `/update`
""")
                
        else:
            chat_panel.add_message("error", f"{EMOJI['cross']} Unknown command: `{cmd}`\n\nType `/help` for available commands.")
    
    def _run_add_agent_in_thread(self, args: str):
        """Run /add_agent in a separate thread with monkey patches (not @work decorator)."""
        import asyncio
        import sys
        import threading
        import traceback
        import re
        import importlib
        from rich.prompt import Prompt, Confirm
        from rich.console import Console
        from prompt_toolkit import PromptSession
        import rich.prompt
        import rich.console
        from extension_manager import agent_adder
        
        # Set up event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if sys.version_info >= (3, 10):
            try:
                asyncio.get_event_loop_policy()
            except RuntimeError:
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        console_wrapper = self.chatbot.command_manager.console
        
        # Save original classes/functions
        original_prompt_ask = Prompt.ask
        original_confirm_ask = Confirm.ask
        original_prompt_session_prompt = PromptSession.prompt
        original_console_class = rich.console.Console
        
        # Create shared state for input handling
        command_input_state = {
            'waiting_for_input': False,
            'prompt_text': None,
            'result': None,
            'ready': threading.Event()
        }
        
        self._command_input_state = command_input_state
        
        # Monkey-patch Prompt.ask
        def tui_prompt_ask(prompt="", **kwargs):
            """Replacement for Prompt.ask that uses chat-based input."""
            clean_prompt = re.sub(r'\[/?[^\]]+\]', '', str(prompt))
            
            choices = kwargs.get('choices', None)
            default = kwargs.get('default', None)
            
            if choices:
                choices_text = ", ".join(f"`{c}`" for c in choices)
                full_prompt = f"**{clean_prompt}**\n\nChoices: {choices_text}"
                if default:
                    full_prompt += f"\n\n_Default: `{default}`_"
            else:
                full_prompt = f"**{clean_prompt}**"
                if default:
                    full_prompt += f"\n\n_Default: `{default}`_"
            
            self.app.call_from_thread(
                chat_panel.add_message,
                "assistant",
                full_prompt
            )
            
            command_input_state['waiting_for_input'] = True
            command_input_state['prompt_text'] = clean_prompt
            command_input_state['result'] = None
            command_input_state['ready'].clear()
            
            command_input_state['ready'].wait()
            
            result = command_input_state['result']
            command_input_state['waiting_for_input'] = False
            
            if not result and default:
                return default
            return result if result else ""
        
        # Monkey-patch Confirm.ask
        def tui_confirm_ask(prompt="", **kwargs):
            """Replacement for Confirm.ask that uses chat-based input."""
            clean_prompt = re.sub(r'\[/?[^\]]+\]', '', str(prompt))
            default = kwargs.get('default', True)
            
            full_prompt = f"**{clean_prompt}**\n\nChoices: `y`, `n`"
            if default is not None:
                full_prompt += f"\n\n_Default: `{'y' if default else 'n'}`_"
            
            self.app.call_from_thread(
                chat_panel.add_message,
                "assistant",
                full_prompt
            )
            
            command_input_state['waiting_for_input'] = True
            command_input_state['prompt_text'] = clean_prompt
            command_input_state['result'] = None
            command_input_state['ready'].clear()
            
            command_input_state['ready'].wait()
            
            result = command_input_state['result']
            command_input_state['waiting_for_input'] = False
            
            if not result and default is not None:
                return default
            
            return result.lower() in ['y', 'yes', 'true', '1']
        
        # Create TUI Console wrapper
        _captured_console_wrapper = console_wrapper
        
        class TUIConsole:
            """Console replacement for TUI mode."""
            def __init__(self, *args, **kwargs):
                self._wrapper = _captured_console_wrapper
            
            def print(self, *args, **kwargs):
                return self._wrapper.print(*args, **kwargs)
            
            def rule(self, *args, **kwargs):
                return self._wrapper.rule(*args, **kwargs)
            
            def status(self, *args, **kwargs):
                return self._wrapper.status(*args, **kwargs)
            
            def input(self, prompt="", **kwargs):
                return self._wrapper.input(prompt, **kwargs)
            
            def log(self, *args, **kwargs):
                return self._wrapper.print(*args, **kwargs)
            
            def path_prompt(self, prompt="", **kwargs):
                """Handle path_prompt used by AgentAdder."""
                return tui_prompt_ask(prompt, **kwargs)
            
            @property
            def width(self):
                return self._wrapper.width
            
            @property
            def height(self):
                return self._wrapper.height
            
            def is_terminal(self):
                return self._wrapper.is_terminal()
        
        # Apply monkey patches
        rich.prompt.Prompt.ask = classmethod(lambda cls, *args, **kwargs: tui_prompt_ask(*args, **kwargs))
        rich.prompt.Confirm.ask = classmethod(lambda cls, *args, **kwargs: tui_confirm_ask(*args, **kwargs))
        Prompt.ask = classmethod(lambda cls, *args, **kwargs: tui_prompt_ask(*args, **kwargs))
        Confirm.ask = classmethod(lambda cls, *args, **kwargs: tui_confirm_ask(*args, **kwargs))
        rich.console.Console = TUIConsole
        
        # Reload extension_manager modules
        importlib.reload(agent_adder)
        
        # Add starting message
        self.app.call_from_thread(
            chat_panel.add_message,
            "system",
            "🔄 Starting `/add_agent` command..."
        )
        
        try:
            # Use the reloaded module
            AgentAdder = agent_adder.AgentAdder
            tui_console_instance = TUIConsole()
            adder = AgentAdder(tui_console_instance, self.chatbot.config_manager)
            adder.run(args if args else None)
            
            self.app.call_from_thread(
                chat_panel.add_message,
                "system",
                f"{EMOJI['checkmark']} Agent added successfully! Run `/reload` to load the new agent."
            )
            
        except Exception as e:
            error_details = f"{EMOJI['cross']} Error adding agent: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            self.app.call_from_thread(
                chat_panel.add_message,
                "error",
                error_details
            )
        
        finally:
            # Clean up
            if hasattr(self, '_command_input_state'):
                self._command_input_state['waiting_for_input'] = False
                delattr(self, '_command_input_state')
            
            # Restore originals
            rich.prompt.Prompt.ask = original_prompt_ask
            rich.prompt.Confirm.ask = original_confirm_ask
            Prompt.ask = original_prompt_ask
            Confirm.ask = original_confirm_ask
            rich.console.Console = original_console_class
            
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass
    
    def _run_add_tool_in_thread(self, args: str):
        """Run /add_tool in a separate thread with monkey patches (not @work decorator)."""
        import asyncio
        import sys
        import threading
        import traceback
        import re
        import importlib
        from rich.prompt import Prompt, Confirm
        from rich.console import Console
        from prompt_toolkit import PromptSession
        import rich.prompt
        import rich.console
        from extension_manager import tool_adder
        
        # Set up event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if sys.version_info >= (3, 10):
            try:
                asyncio.get_event_loop_policy()
            except RuntimeError:
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        console_wrapper = self.chatbot.command_manager.console
        
        # Save original classes/functions
        original_prompt_ask = Prompt.ask
        original_confirm_ask = Confirm.ask
        original_prompt_session_prompt = PromptSession.prompt
        original_console_class = rich.console.Console
        
        # Create shared state for input handling
        command_input_state = {
            'waiting_for_input': False,
            'prompt_text': None,
            'result': None,
            'ready': threading.Event()
        }
        
        self._command_input_state = command_input_state
        
        # Monkey-patch Prompt.ask
        def tui_prompt_ask(prompt="", **kwargs):
            """Replacement for Prompt.ask that uses chat-based input."""
            clean_prompt = re.sub(r'\[/?[^\]]+\]', '', str(prompt))
            
            choices = kwargs.get('choices', None)
            default = kwargs.get('default', None)
            
            if choices:
                choices_text = ", ".join(f"`{c}`" for c in choices)
                full_prompt = f"**{clean_prompt}**\n\nChoices: {choices_text}"
                if default:
                    full_prompt += f"\n\n_Default: `{default}`_"
            else:
                full_prompt = f"**{clean_prompt}**"
                if default:
                    full_prompt += f"\n\n_Default: `{default}`_"
            
            self.app.call_from_thread(
                chat_panel.add_message,
                "assistant",
                full_prompt
            )
            
            command_input_state['waiting_for_input'] = True
            command_input_state['prompt_text'] = clean_prompt
            command_input_state['result'] = None
            command_input_state['ready'].clear()
            
            command_input_state['ready'].wait()
            
            result = command_input_state['result']
            command_input_state['waiting_for_input'] = False
            
            if not result and default:
                return default
            return result if result else ""
        
        # Monkey-patch Confirm.ask
        def tui_confirm_ask(prompt="", **kwargs):
            """Replacement for Confirm.ask that uses chat-based input."""
            clean_prompt = re.sub(r'\[/?[^\]]+\]', '', str(prompt))
            default = kwargs.get('default', True)
            
            full_prompt = f"**{clean_prompt}**\n\nChoices: `y`, `n`"
            if default is not None:
                full_prompt += f"\n\n_Default: `{'y' if default else 'n'}`_"
            
            self.app.call_from_thread(
                chat_panel.add_message,
                "assistant",
                full_prompt
            )
            
            command_input_state['waiting_for_input'] = True
            command_input_state['prompt_text'] = clean_prompt
            command_input_state['result'] = None
            command_input_state['ready'].clear()
            
            command_input_state['ready'].wait()
            
            result = command_input_state['result']
            command_input_state['waiting_for_input'] = False
            
            if not result and default is not None:
                return default
            
            return result.lower() in ['y', 'yes', 'true', '1']
        
        # Create TUI Console wrapper
        _captured_console_wrapper = console_wrapper
        
        class TUIConsole:
            """Console replacement for TUI mode."""
            def __init__(self, *args, **kwargs):
                self._wrapper = _captured_console_wrapper
            
            def print(self, *args, **kwargs):
                return self._wrapper.print(*args, **kwargs)
            
            def rule(self, *args, **kwargs):
                return self._wrapper.rule(*args, **kwargs)
            
            def status(self, *args, **kwargs):
                return self._wrapper.status(*args, **kwargs)
            
            def input(self, prompt="", **kwargs):
                return self._wrapper.input(prompt, **kwargs)
            
            def log(self, *args, **kwargs):
                return self._wrapper.print(*args, **kwargs)
            
            def path_prompt(self, prompt="", **kwargs):
                """Handle path_prompt used by ToolAdder."""
                return tui_prompt_ask(prompt, **kwargs)
            
            @property
            def width(self):
                return self._wrapper.width
            
            @property
            def height(self):
                return self._wrapper.height
            
            def is_terminal(self):
                return self._wrapper.is_terminal()
        
        # Apply monkey patches
        rich.prompt.Prompt.ask = classmethod(lambda cls, *args, **kwargs: tui_prompt_ask(*args, **kwargs))
        rich.prompt.Confirm.ask = classmethod(lambda cls, *args, **kwargs: tui_confirm_ask(*args, **kwargs))
        Prompt.ask = classmethod(lambda cls, *args, **kwargs: tui_prompt_ask(*args, **kwargs))
        Confirm.ask = classmethod(lambda cls, *args, **kwargs: tui_confirm_ask(*args, **kwargs))
        rich.console.Console = TUIConsole
        
        # Reload extension_manager modules
        importlib.reload(tool_adder)
        
        # Add starting message
        self.app.call_from_thread(
            chat_panel.add_message,
            "system",
            "🔄 Starting `/add_tool` command..."
        )
        
        try:
            # Use the reloaded module
            ToolAdder = tool_adder.ToolAdder
            tui_console_instance = TUIConsole()
            adder = ToolAdder(tui_console_instance, self.chatbot.config_manager)
            adder.run(args if args else None)
            
            self.app.call_from_thread(
                chat_panel.add_message,
                "system",
                f"{EMOJI['checkmark']} Tool added successfully! Run `/reload` to load the new tool."
            )
            
        except Exception as e:
            error_details = f"{EMOJI['cross']} Error adding tool: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            self.app.call_from_thread(
                chat_panel.add_message,
                "error",
                error_details
            )
        
        finally:
            # Clean up
            if hasattr(self, '_command_input_state'):
                self._command_input_state['waiting_for_input'] = False
                delattr(self, '_command_input_state')
            
            # Restore originals
            rich.prompt.Prompt.ask = original_prompt_ask
            rich.prompt.Confirm.ask = original_confirm_ask
            Prompt.ask = original_prompt_ask
            Confirm.ask = original_confirm_ask
            rich.console.Console = original_console_class
            
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass
    
    def _run_agent_workflow_in_thread(self, agent_name: str, agent_desc: str, agent_init_args: dict = None):
        """Run an agent workflow in a separate thread to avoid asyncio conflicts."""
        import asyncio
        import sys
        import threading
        from rich.prompt import Prompt
        from rich.console import Console
        from prompt_toolkit import PromptSession
        
        # For worker threads, we need to ensure asyncio works properly
        # Set a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Store original policy if needed
        if sys.version_info >= (3, 10):
            # In Python 3.10+, asyncio.run() can have issues in threads
            # We need to ensure the event loop policy is set correctly
            try:
                asyncio.get_event_loop_policy()
            except RuntimeError:
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        console_wrapper = self.chatbot.command_manager.console
        
        # Save original classes/functions for restoration
        original_prompt_ask = Prompt.ask
        original_prompt_session_prompt = PromptSession.prompt
        import rich.console
        original_console_class = rich.console.Console
        
        # Create a shared state for agent input handling
        agent_input_state = {
            'waiting_for_input': False,
            'prompt_text': None,
            'result': None,
            'ready': threading.Event(),
            'last_menu_choices': None,  # Track menu choices from printed tables
            'cancelled': False  # Flag to cancel agent execution
        }
        
        # Store reference in MainScreen for access in handle_input
        self._agent_input_state = agent_input_state
        
        # Monkey-patch Prompt.ask to use chat-based input
        
        def tui_prompt_ask(prompt="", **kwargs):
            """Replacement for Prompt.ask that uses chat-based input."""
            import re
            
            # Check if agent was cancelled before prompting
            if agent_input_state.get('cancelled', False):
                raise KeyboardInterrupt("Agent cancelled by user")
            
            # Clean up rich markup from prompt
            clean_prompt = re.sub(r'\[/?[^\]]+\]', '', str(prompt))
            
            # Check if there are choices
            choices = kwargs.get('choices', None)
            if choices:
                choices_text = ", ".join(f"`{c}`" for c in choices)
                full_prompt = f"**{clean_prompt}**\n\nChoices: {choices_text}"
            else:
                full_prompt = f"**{clean_prompt}**"
            
            # Show prompt in chat
            self.app.call_from_thread(
                chat_panel.add_message,
                "assistant",
                full_prompt
            )
            
            # Set state to waiting
            agent_input_state['waiting_for_input'] = True
            agent_input_state['prompt_text'] = clean_prompt
            agent_input_state['result'] = None
            agent_input_state['ready'].clear()
            
            # Wait for user input
            agent_input_state['ready'].wait()
            
            # Check if cancelled while waiting
            if agent_input_state.get('cancelled', False):
                agent_input_state['waiting_for_input'] = False
                raise KeyboardInterrupt("Agent cancelled by user")
            
            # Get result and reset state
            result = agent_input_state['result']
            agent_input_state['waiting_for_input'] = False
            
            return result if result else ""
        
        def tui_prompt_session_prompt(self, *args, **kwargs):
            """Replacement for PromptSession.prompt that uses chat-based input."""
            prompt_text = args[0] if args else "Enter valuesss"
            return tui_prompt_ask(prompt_text, **kwargs)
        
        # Create a TUI Console wrapper class that captures the console_wrapper in closure
        _captured_console_wrapper = console_wrapper
        main_app = self.app  # Capture the ValbotTUI app instance (self is MainScreen, self.app is ValbotTUI)
        _original_console_class = original_console_class  # Capture original Console before patching
        main_screen = self  # Capture MainScreen instance to access _agent_input_state dynamically
        
        class TUIConsole:
            """A Console replacement that forwards to our TUI console wrapper."""
            def __init__(self, *args, **kwargs):
                # Capture the console wrapper for this instance
                self._wrapper = _captured_console_wrapper
                self._main_screen = main_screen  # Store MainScreen to access current _agent_input_state
                self._chat_panel = chat_panel  # Capture chat panel
                self._main_app = main_app  # Capture main ValbotTUI instance (outer 'self')
                self._original_console = _original_console_class  # Store original Console class
            
            @property
            def _agent_input_state(self):
                """Get the current agent_input_state from MainScreen dynamically."""
                return self._main_screen._agent_input_state if hasattr(self._main_screen, '_agent_input_state') else None
            
            def print(self, *args, **kwargs):
                # Check if printing a Rich Table with menu choices
                from rich.table import Table
                if args and isinstance(args[0], Table):
                    table = args[0]
                    
                    # Try to extract table data by rendering it
                    choices = []
                    try:
                        # Use the ORIGINAL Rich Console (not our patched one) to render the table
                        from io import StringIO
                        
                        # Create a temporary console using the original Console class
                        string_buffer = StringIO()
                        temp_console = self._original_console(file=string_buffer, force_terminal=False, width=80, legacy_windows=False)
                        temp_console.print(table)
                        rendered_output = string_buffer.getvalue()
                        
                        # Parse the rendered table to extract rows
                        # Look for lines that match the pattern: "│ key │ description │"
                        import re
                        lines = rendered_output.split('\n')
                        for line in lines:
                            # Match table rows (lines with │ separators)
                            if '│' in line and not any(char in line for char in ['─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼']):
                                # Split by │ and extract cells
                                cells = [cell.strip() for cell in line.split('│') if cell.strip()]
                                if len(cells) >= 2:
                                    # First cell is key, second is description
                                    key = cells[0].strip()
                                    description = cells[1].strip()
                                    if key and description:
                                        choices.append((description, key))
                        
                        if choices and self._agent_input_state:
                            self._agent_input_state['last_menu_choices'] = choices
                        
                        # Don't print the rendered table - it will be shown as InlinePicker
                        return
                    except Exception as e:
                        # Fallback - just pass through
                        pass
                
                # For non-table content, pass through to wrapper
                return self._wrapper.print(*args, **kwargs)
            
            def rule(self, *args, **kwargs):
                return self._wrapper.rule(*args, **kwargs)
            
            def status(self, *args, **kwargs):
                return self._wrapper.status(*args, **kwargs)
            
            def input(self, prompt="", **kwargs):
                """Handle input requests from agents - use InlinePicker for menus, text input otherwise."""
                import re
                import threading
                # Clean up rich markup from prompt
                clean_prompt = re.sub(r'\[/?[^\]]+\]', '', str(prompt))
                
                # Check if we have menu choices from a recently printed table
                menu_choices = self._agent_input_state.get('last_menu_choices') if self._agent_input_state else None
                
                if menu_choices:
                    # Show InlinePicker for menu selection
                    if self._agent_input_state:
                        self._agent_input_state['last_menu_choices'] = None  # Clear after using
                    
                    # Use a container to store the result
                    result_container = {'value': None, 'ready': threading.Event()}
                    
                    # Define the callback that will be triggered when user selects
                    def on_picker_selection(selected_value):
                        result_container['value'] = selected_value
                        result_container['ready'].set()
                    
                    # Call the show_inline_picker method from the main thread
                    # This uses the proper async flow that MainScreen already has
                    import asyncio
                    
                    async def show_picker_wrapper():
                        main_screen = self._main_app.screen
                        # Set up the callback before showing picker
                        main_screen._picker_callback = on_picker_selection
                        # Show the picker
                        await main_screen.show_inline_picker(
                            title=clean_prompt,
                            options=menu_choices,
                            picker_type="agent_menu"
                        )
                    
                    # Schedule the async function on the main event loop
                    self._main_app.call_from_thread(
                        lambda: asyncio.create_task(show_picker_wrapper())
                    )
                    
                    # Wait for user to make a selection
                    result_container['ready'].wait()
                    return result_container['value'] if result_container['value'] else ""
                else:
                    # No menu - use text input
                    if not self._agent_input_state:
                        return ""  # Safety check - no state available
                    
                    # Show prompt in chat
                    self._main_app.call_from_thread(
                        self._chat_panel.add_message,
                        "assistant",
                        f"**{clean_prompt}**"
                    )
                    
                    # Set state to waiting
                    self._agent_input_state['waiting_for_input'] = True
                    self._agent_input_state['prompt_text'] = clean_prompt
                    self._agent_input_state['result'] = None
                    self._agent_input_state['ready'].clear()
                    
                    # Wait for user input
                    self._agent_input_state['ready'].wait()
                    
                    # Get result and reset state
                    result = self._agent_input_state['result']
                    self._agent_input_state['waiting_for_input'] = False
                    return result if result is not None else ""
            
            def log(self, *args, **kwargs):
                # Treat log same as print
                return self._wrapper.print(*args, **kwargs)
            
            @property
            def width(self):
                return self._wrapper.width
            
            @property
            def height(self):
                return self._wrapper.height
            
            def is_terminal(self):
                return self._wrapper.is_terminal()
            
            def set_live(self, live_instance):
                """Stub for Rich Live compatibility - we'll intercept Live context manager instead."""
                return True
            
            def clear_live(self):
                """Stub for Rich Live compatibility."""
                pass
            
            def show_cursor(self, show=True):
                """Stub for Rich cursor control - not needed in TUI."""
                pass
            
            def push_render_hook(self, hook):
                """Stub for Rich render hooks - not needed in TUI."""
                pass
            
            def pop_render_hook(self):
                """Stub for Rich render hooks - not needed in TUI."""
                pass
            
            def __enter__(self):
                """Context manager support for Rich Live."""
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                """Context manager support for Rich Live."""
                return False
        
        # Monkey patch Rich's Live class to intercept interactive menus
        from rich.live import Live as OriginalLive
        
        class TUILive:
            """Intercepts Rich Live displays and converts them to TUI InlinePicker."""
            def __init__(self, renderable=None, *, console=None, **kwargs):
                self.renderable = renderable
                self.console = console
                self.kwargs = kwargs
                self._entered = False
            
            def __enter__(self):
                self._entered = True
                # Don't actually start Live display in TUI
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self._entered = False
                return False
            
            def start(self, refresh=True):
                pass
            
            def stop(self):
                pass
            
            def update(self, renderable):
                pass
        
        # Apply monkey patches
        Prompt.ask = staticmethod(tui_prompt_ask)
        PromptSession.prompt = tui_prompt_session_prompt
        
        # Replace Console class entirely with our wrapper
        import rich.console
        import rich.live
        original_console_class = rich.console.Console
        rich.console.Console = TUIConsole
        
        # Replace Live class with our interceptor
        rich.live.Live = TUILive
        
        # Monkey patch readchar to force fallback to console.input()
        try:
            import readchar
            original_readkey = readchar.readkey
            def tui_readkey():
                """Raise exception to force agent to use console.input() fallback."""
                raise RuntimeError("readchar not available in TUI - using console.input() fallback")
            readchar.readkey = tui_readkey
        except ImportError:
            pass  # readchar not installed, that's fine
        
        # Patch AgentRunResult to add .data property as alias for .output (for backward compatibility)
        try:
            from pydantic_ai import AgentRunResult
            if not hasattr(AgentRunResult, 'data'):
                # Add data property that returns output
                AgentRunResult.data = property(lambda self: self.output)
        except (ImportError, AttributeError):
            pass  # If AgentRunResult doesn't exist or is different, skip
        
        # Also patch Console and Live in any modules that may have already imported them
        import sys
        for module_name, module in list(sys.modules.items()):
            if module:
                # Patch Console
                if hasattr(module, 'Console'):
                    try:
                        if module.Console is original_console_class:
                            module.Console = TUIConsole
                    except (AttributeError, TypeError):
                        pass
                # Patch Live
                if hasattr(module, 'Live'):
                    try:
                        module.Live = TUILive
                    except (AttributeError, TypeError):
                        pass
        
        # Add starting message
        self.app.call_from_thread(
            chat_panel.add_message, 
            "system", 
            f"{EMOJI['robot']} Running agent: **{agent_name}**\n\n[dim]{agent_desc}[/dim]"
        )
        
        # Set the current agent in the chat panel
        self.app.call_from_thread(setattr, chat_panel, 'current_agent', agent_name)
        
        # CRITICAL: Monkey-patch the SPF2PDL agent's get_initializer_args method
        # to completely bypass its interactive menu system
        original_spf2pdl_get_init = None
        try:
            from agent_plugins.agents import pdl_agent
            original_spf2pdl_get_init = pdl_agent.MyCustomPlugin.get_initializer_args
            
            def tui_get_initializer_args_replacement(agent_self):
                """TUI-compatible version that returns mode from initializer_args without prompting."""
                # Simply return what was already set via kwargs in __init__
                mode = agent_self.initializer_args.get('mode', '2')
                result = {'mode': mode}
                
                # If mode is 1 and input_path not provided, we need it
                if mode == '1' and not agent_self.initializer_args.get('input_path'):
                    result['input_path'] = agent_self.initializer_args.get('input_path', '')
                
                return result
            
            # Apply the monkey patch
            pdl_agent.MyCustomPlugin.get_initializer_args = tui_get_initializer_args_replacement
            self._original_get_initializer_args = original_spf2pdl_get_init
        except (ImportError, AttributeError) as e:
            pass
        
        try:
            # Run the agent with proper context (pass init args if provided)
            # All console.print() calls from the agent will now appear as assistant messages
            kwargs = agent_init_args or {}
            
            # Monkey-patch PluginManager.run_plugin to replace agent console after creation
            original_run_plugin = self.chatbot.plugin_manager.run_plugin
            tui_console_instance = TUIConsole()  # Create our TUI console
            
            def patched_run_plugin(plugin_name, context, **run_kwargs):
                """Patched run_plugin that replaces agent's console with TUIConsole."""
                def normalize(name):
                    return name.replace('_', '').replace(' ', '').lower()

                normalized_input = normalize(plugin_name)
                matched_plugin_class = None
                for key in self.chatbot.plugin_manager.plugins:
                    if normalize(key) == normalized_input:
                        matched_plugin_class = self.chatbot.plugin_manager.plugins[key]
                        break
                if matched_plugin_class:
                    plugin_instance = matched_plugin_class(self.chatbot.plugin_manager.model, **run_kwargs)
                    
                    # CRITICAL: Replace the agent's console BEFORE running it
                    if hasattr(plugin_instance, 'console'):
                        plugin_instance.console = tui_console_instance
                    
                    plugin_instance.run_agent_flow_with_init_args(context, **run_kwargs)
                else:
                    print(f"Plugin {plugin_name} not found.")
            
            # Apply the patch
            self.chatbot.plugin_manager.run_plugin = patched_run_plugin
            
            # Now run the plugin
            self.chatbot.plugin_manager.run_plugin(
                agent_name, 
                self.chatbot.context_manager.conversation_history,
                **kwargs
            )
            
            # Restore original
            self.chatbot.plugin_manager.run_plugin = original_run_plugin
            
            # Check if cancelled after completion
            if agent_input_state.get('cancelled', False):
                # Add cancellation message
                self.app.call_from_thread(
                    chat_panel.add_message, 
                    "system", 
                    f"{EMOJI['cross']} Agent **{agent_name}** was cancelled."
                )
            else:
                # Add success message
                self.app.call_from_thread(
                    chat_panel.add_message, 
                    "assistant", 
                    f"✅ Agent **{agent_name}** completed successfully!"
                )
        except KeyboardInterrupt:
            # Agent was cancelled by user
            self.app.call_from_thread(
                chat_panel.add_message, 
                "system", 
                f"{EMOJI['cross']} Agent: **{agent_name}** was cancelled by user."
            )
        except Exception as e:
            import traceback
            
            # Add error message with traceback for debugging
            error_details = f"{EMOJI['cross']} Agent error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            self.app.call_from_thread(
                chat_panel.add_message, 
                "error", 
                error_details
            )
        finally:
            # Restore SPF2PDL agent's get_initializer_args if it was patched
            try:
                from agent_plugins.agents import pdl_agent
                if hasattr(self, '_original_get_initializer_args'):
                    pdl_agent.MyCustomPlugin.get_initializer_args = self._original_get_initializer_args
                    delattr(self, '_original_get_initializer_args')
            except (ImportError, AttributeError):
                pass
            
            # Clear the current agent in the chat panel
            self.app.call_from_thread(setattr, chat_panel, 'current_agent', None)
            
            # Clean up agent input state (but don't delete it - just reset)
            if hasattr(self, '_agent_input_state'):
                self._agent_input_state['waiting_for_input'] = False
                self._agent_input_state['last_menu_choices'] = None
                self._agent_input_state['cancelled'] = False  # Reset cancellation flag
                # Keep _agent_input_state around for next agent run
            
            # Restore original prompt functions and Console class
            Prompt.ask = original_prompt_ask
            PromptSession.prompt = original_prompt_session_prompt
            import rich.console
            rich.console.Console = original_console_class
            
            # Restore Console in modules that were patched
            import sys
            for module_name, module in list(sys.modules.items()):
                if module and hasattr(module, 'Console'):
                    try:
                        if module.Console is TUIConsole:
                            module.Console = original_console_class
                    except (AttributeError, TypeError):
                        pass
            
            # Clean up the event loop
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass
    
    def _send_chat_message_in_thread(self, message: str):
        """Send a message to the chatbot (runs in background thread)."""
        # Reset cancellation flag at the start of a new message
        self._cancel_streaming = False
        
        if not self.chatbot:
            self.app.call_from_thread(self._add_error_message, f"""## {EMOJI['cross']} ChatBot Not Initialized

ChatBot is not ready. Please check your configuration.

**Troubleshooting**:
1. Verify your API keys are set
2. Check your model configuration
3. Ensure network connectivity
""")
            return
            
        # Set processing state in UI thread
        self.app.call_from_thread(self._set_processing_state, True)
        
        # Add loading indicator in chat
        self.app.call_from_thread(self._add_loading_message)
        
        try:
            # Check if the message should use tools
            if self.chatbot.agent_model and self.chatbot._should_use_tools(message):
                # Use tool-enabled agent (synchronous version to avoid event loop conflicts)
                self._send_with_tools(message)
            else:
                # Use standard streaming chat (synchronous version)
                self._send_standard_message_sync(message)
                
        except Exception as e:
            # Remove loading indicator on error
            self.app.call_from_thread(self._remove_loading_message)
            
            error_message = f"""## {EMOJI['cross']} Error

An error occurred while processing your message:

```
{str(e)}
```
"""
            self.app.call_from_thread(self._add_error_message, error_message)
        
        finally:
            # Reset processing state
            self.app.call_from_thread(self._set_processing_state, False)
    
    # Helper methods for thread-safe UI updates
    def _set_processing_state(self, processing: bool):
        """Set processing state (called from worker thread)."""
        self.processing = processing
        # Status bar now only shows model, no processing state
    
    def _add_loading_message(self):
        """Add loading message (called from worker thread)."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        # Check if header should be shown
        show_header = not chat_panel._assistant_header_shown
        chat_panel._assistant_header_shown = True
        loading_msg = ChatMessage("assistant", "⠋ Thinking...", show_header=show_header)
        chat_panel.mount(loading_msg)
        chat_panel.scroll_end(animate=True)
        # Store reference for removal later
        self._loading_msg = loading_msg
        # Start animation
        self._start_loading_animation()
    
    def _remove_loading_message(self):
        """Remove loading message (called from worker thread)."""
        if hasattr(self, '_loading_msg') and self._loading_msg and self._loading_msg.parent:
            self._loading_msg.remove()
            self._loading_msg = None
            # Reset the header flag so the next message (streaming response) will show the header
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel._assistant_header_shown = False
        # Stop animation
        self._stop_loading_animation()
    
    def _start_loading_animation(self):
        """Start the loading spinner animation."""
        self._loading_animation_active = True
        self._loading_spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._loading_frame_index = 0
        # Schedule the animation update - store the timer handle so we can cancel it
        self._loading_timer = self.app.set_interval(0.1, self._update_loading_animation)
    
    def _stop_loading_animation(self):
        """Stop the loading spinner animation."""
        self._loading_animation_active = False
        # Cancel the timer if it exists
        if hasattr(self, '_loading_timer') and self._loading_timer:
            self._loading_timer.stop()
            self._loading_timer = None
    
    def _update_loading_animation(self):
        """Update the loading spinner frame."""
        if not self._loading_animation_active or not hasattr(self, '_loading_msg') or not self._loading_msg:
            return
        
        if self._loading_msg and self._loading_msg.parent:
            self._loading_frame_index = (self._loading_frame_index + 1) % len(self._loading_spinner_frames)
            spinner = self._loading_spinner_frames[self._loading_frame_index]
            # Update the content of the message widget (it's a Markdown widget)
            try:
                content_widget = self._loading_msg.query_one(".message-content", Markdown)
                content_widget.update(f"{spinner} Just a sec...")
            except Exception:
                # Fallback: update the message content directly
                self._loading_msg.content = f"{spinner} Just a sec..."
                self._loading_msg.refresh()
    
    def _add_error_message(self, message: str):
        """Add error message (called from worker thread)."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        chat_panel.add_message("error", message)
    
    def _add_system_message(self, message: str):
        """Add system message (called from worker thread)."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        chat_panel.add_message("system", message)
    
    def _add_assistant_message(self, message: str):
        """Add assistant message (called from worker thread)."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        chat_panel.add_message("assistant", message)
        # Schedule copy button detection after UI settles (for non-streaming messages)
        try:
            if chat_panel.messages:
                last_msg = chat_panel.messages[-1]
                # Use the same delayed detection system as streaming
                self.app.call_later(self._schedule_copy_button_detection, last_msg)
        except Exception:
            pass  # Ignore errors
    
    def _update_loading_message(self, content: str):
        """Update loading message content (called from worker thread)."""
        if hasattr(self, '_loading_msg') and self._loading_msg and self._loading_msg.parent:
            self._loading_msg.content = content
            self._loading_msg.refresh()
    
    def _send_with_tools(self, message: str):
        """Send a message using the tool-enabled pydantic-ai agent."""
        # Import ChatDeps from chatbot module
        from chatbot import ChatDeps
        import asyncio
        
        # Check if tool agent is available
        if not hasattr(self.chatbot, 'tool_agent') or self.chatbot.tool_agent is None:
            self.app.call_from_thread(self._add_system_message, "⚠️ Tool agent not available. Using standard chat.")
            self._send_standard_message_sync(message)
            return
        
        # Add to conversation history
        self.chatbot.context_manager.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Create deps with conversation history
        deps = ChatDeps(
            conversation_history=self.chatbot.context_manager.conversation_history
        )
        
        try:
            # Show status
            self.app.call_from_thread(self._update_loading_message, "🔧 Using tools to process your request...")
            
            # Run the agent with tools in a new event loop for this thread
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.chatbot.tool_agent.run(message, deps=deps))
            finally:
                loop.close()
            
            # Remove loading indicator
            self.app.call_from_thread(self._remove_loading_message)
            
            # Display the response - use .output instead of .data
            response_text = str(result.output) if hasattr(result, 'output') else str(result)
            self.app.call_from_thread(self._add_assistant_message, response_text)
            
            # Add to conversation history
            self.chatbot.context_manager.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Remove loading indicator
            self.app.call_from_thread(self._remove_loading_message)
            
            error_msg = f"""## ⚠️ Tool Error

Error using tools: {str(e)}

Falling back to standard chat...

<details>
<summary>Error Details</summary>

```
{error_details}
```
</details>
"""
            self.app.call_from_thread(self._add_system_message, error_msg)
            
            # Remove the user message we added
            if self.chatbot.context_manager.conversation_history and \
               self.chatbot.context_manager.conversation_history[-1]["role"] == "user":
                self.chatbot.context_manager.conversation_history.pop()
            
            # Fall back to standard message
            self._send_standard_message_sync(message)
    
    def _send_standard_message_sync(self, message: str):
        """Send a message using standard OpenAI streaming (synchronous for worker thread)."""
        # Add to conversation history (if not already added)
        if not self.chatbot.context_manager.conversation_history or \
           self.chatbot.context_manager.conversation_history[-1]["content"] != message:
            self.chatbot.context_manager.conversation_history.append({
                "role": "user",
                "content": message
            })
        
        # Get response from AI with reasoning support
        model_name = self.chatbot.modelname
        messages = self.chatbot.context_manager.conversation_history
        effort_level = self.config_manager.get_setting("chat_model_config.reasoning_effort", "low")
        
        # Validate effort_level
        valid_effort_levels = ["low", "medium", "high"]
        if effort_level not in valid_effort_levels:
            effort_level = "low"
        
        # Prepare kwargs for API call
        kwargs = {
            "model": model_name,
            "input": messages,
            "stream": True,
        }
        
        # Add reasoning for gpt-5 models
        if model_name.startswith("gpt-5"):
            kwargs["reasoning"] = {"summary": "auto", "effort": effort_level}
        
        try:
            stream = self.chatbot.client.responses.create(**kwargs)
            
            # Collect streaming response with reasoning support
            response_text = ""
            reasoning_text = ""
            display_reasoning = self.config_manager.get_setting("chat_model_config.display_reasoning", False)
            
            # Track state
            reasoning_msg_id = None
            first_response_received = False
            
            for response in stream:
                # Check for cancellation
                if self._cancel_streaming:
                    # Break out of streaming loop
                    break
                    
                if isinstance(response, ResponseAudioDeltaEvent):
                    # This is reasoning content
                    if response.delta:
                        reasoning_text += response.delta
                        if display_reasoning:
                            # Update or create reasoning message
                            reasoning_content = f"**🧠 Thinking... (effort: {effort_level})**\n\n{reasoning_text}"
                            if reasoning_msg_id is None:
                                # Create new reasoning message
                                self.app.call_from_thread(self._add_reasoning_message, reasoning_content)
                                reasoning_msg_id = True
                            else:
                                # Update existing reasoning message
                                self.app.call_from_thread(self._update_reasoning_message, reasoning_content)
                        
                elif isinstance(response, ResponseTextDeltaEvent):
                    # This is the actual response text
                    if response.delta:
                        # Remove loading indicator on first response
                        if not first_response_received:
                            self.app.call_from_thread(self._remove_loading_message)
                            first_response_received = True
                        
                        # Remove reasoning message if it exists
                        if reasoning_msg_id is not None:
                            self.app.call_from_thread(self._remove_reasoning_message)
                            reasoning_msg_id = None
                        
                        response_text += response.delta
                        # Stream the response text as it comes in
                        self.app.call_from_thread(self._update_streaming_response, response_text)
            
            # Handle cancellation cleanup
            if self._cancel_streaming:
                # Remove any loading or reasoning messages
                if not first_response_received:
                    self.app.call_from_thread(self._remove_loading_message)
                if reasoning_msg_id is not None:
                    self.app.call_from_thread(self._remove_reasoning_message)
                # Keep the partial response but finalize it (don't remove it)
                if response_text:
                    self.app.call_from_thread(self._finalize_streaming_response)
                    # Still add partial response to conversation history
                    self.chatbot.context_manager.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                # Always show cancellation message (even if no response yet)
                self.app.call_from_thread(self._add_system_message, f"{EMOJI['cross']} *Response cancelled by user.*")
                return
                        
        except Exception as e:
            # Remove loading indicator on error
            self.app.call_from_thread(self._remove_loading_message)
            
            response_text = f"""## {EMOJI['cross']} API Communication Error

```
{str(e)}
```

**Possible causes**:
- Invalid API key
- Network connectivity issues
- Model not available
- Rate limit exceeded
"""
            self.app.call_from_thread(self._add_error_message, response_text)
            return
        
        # Finalize the streamed response
        if response_text:
            self.app.call_from_thread(self._finalize_streaming_response)
            self.chatbot.context_manager.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
    
    def _add_reasoning_message(self, content: str):
        """Add reasoning message (called from worker thread)."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        reasoning_msg = ChatMessage("system", content)
        chat_panel.mount(reasoning_msg)
        chat_panel.scroll_end(animate=False)
        self._reasoning_msg = reasoning_msg
    
    def _update_reasoning_message(self, content: str):
        """Update reasoning message (called from worker thread)."""
        if hasattr(self, '_reasoning_msg') and self._reasoning_msg and self._reasoning_msg.parent:
            self._reasoning_msg.content = content
            self._reasoning_msg.refresh()
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel.scroll_end(animate=False)
    
    def _remove_reasoning_message(self):
        """Remove reasoning message (called from worker thread)."""
        if hasattr(self, '_reasoning_msg') and self._reasoning_msg and self._reasoning_msg.parent:
            self._reasoning_msg.remove()
            self._reasoning_msg = None
    
    def _remove_streaming_message(self):
        """Remove streaming message when cancelled (called from worker thread)."""
        if hasattr(self, '_streaming_msg') and self._streaming_msg and self._streaming_msg.parent:
            self._streaming_msg.remove()
            self._streaming_msg = None
            # Reset the header flag
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel._assistant_header_shown = False
    
    def _update_streaming_response(self, content: str):
        """Update streaming response message (called from worker thread)."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        
        # Create or update the streaming message
        if not hasattr(self, '_streaming_msg') or self._streaming_msg is None:
            # Check if header should be shown
            show_header = not chat_panel._assistant_header_shown
            chat_panel._assistant_header_shown = True
            self._streaming_msg = ChatMessage("assistant", content, show_header=show_header)
            chat_panel.mount(self._streaming_msg)
        else:
            # Update the content attribute
            self._streaming_msg.content = content
            # Find and update the Markdown widget(s) inside the message
            try:
                markdown_widgets = self._streaming_msg.query(".message-content")
                for md_widget in markdown_widgets:
                    if isinstance(md_widget, Markdown):
                        md_widget.update(content)
            except Exception:
                # Fallback: remove and recreate the message
                show_header = not chat_panel._assistant_header_shown
                chat_panel._assistant_header_shown = True
                self._streaming_msg.remove()
                self._streaming_msg = ChatMessage("assistant", content, show_header=show_header)
                chat_panel.mount(self._streaming_msg)
        
        chat_panel.scroll_end(animate=False)
    
    def _finalize_streaming_response(self):
        """Finalize the streaming response (called from worker thread)."""
        # Schedule copy button detection after streaming completes and UI settles
        if hasattr(self, '_streaming_msg') and self._streaming_msg:
            # Store reference to the message for later processing
            final_msg = self._streaming_msg
            # Use the simplified single-phase approach
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            self.app.call_later(chat_panel._add_copy_buttons_to_message, final_msg)
        
        # Clear the streaming message reference
        if hasattr(self, '_streaming_msg'):
            self._streaming_msg = None

    
    async def action_new_chat(self):
        """Start a new chat - completely restart everything from scratch."""
        # Cancel any ongoing streaming
        self._cancel_streaming = True
        
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        
        # Cancel any active agent
        agent_was_active = False
        if hasattr(self, '_agent_input_state') and self._agent_input_state:
            # Check if agent is running (either waiting for input or actively executing)
            if self._agent_input_state.get('waiting_for_input', False) or hasattr(chat_panel, 'current_agent') and chat_panel.current_agent:
                agent_was_active = True
                # Signal the agent to stop by setting the cancelled flag
                self._agent_input_state['cancelled'] = True
                self._agent_input_state['result'] = None
                self._agent_input_state['waiting_for_input'] = False
                self._agent_input_state['ready'].set()  # Unblock the waiting thread if it's waiting
                
                # Give the agent thread a moment to process the cancellation
                await asyncio.sleep(0.2)
                
                # Clean up the agent input state
                if hasattr(self, '_agent_input_state'):
                    delattr(self, '_agent_input_state')
        
        # Clear current agent in chat panel
        if hasattr(chat_panel, 'current_agent') and chat_panel.current_agent:
            agent_was_active = True
            chat_panel.current_agent = None
        
        # Clear messages (this should now catch any lingering agent messages)
        chat_panel.clear_messages()
        
        # Clear any streaming message reference
        if hasattr(self, '_streaming_msg'):
            self._streaming_msg = None
        
        # Completely reinitialize the chatbot (like quitting and relaunching)
        if self.chatbot:
            # Clear conversation history
            self.chatbot.clear_conversation()
            
            # Reinitialize system prompt (like during startup)
            system_prompt = self.config_manager.get_setting(
                "chat_model_config.system_prompt", 
                "You are a helpful assistant. Prioritize markdown format and code blocks when applicable."
            )
            self.chatbot.context_manager.conversation_history.append({
                "role": "system", 
                "content": system_prompt
            })
            
            # Reset conversation ID to start fresh backend session
            self.chatbot._conversation_id = None
        
        # Show welcome message again
        self.show_welcome_message()
        
        # Notify user if an agent was cancelled via toast
        if agent_was_active:
            self.app.notify(f"{EMOJI['cross']} Active agent cancelled.", severity="warning", timeout=3)
    
    async def action_agent_picker(self):
        """Open the inline agent picker."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        
        if self.chatbot and hasattr(self.chatbot, 'plugin_manager'):
            agents = self.chatbot.plugin_manager.plugin_info
            if agents:
                async def on_agent_selected(selected_agent_name):
                    if selected_agent_name:
                        # Find the agent details
                        for agent_name, agent_desc in agents:
                            if agent_name == selected_agent_name:
                                # Run agent in background thread using threading directly
                                import threading
                                thread = threading.Thread(
                                    target=self._run_agent_workflow_in_thread,
                                    args=(agent_name, agent_desc),
                                    daemon=True
                                )
                                thread.start()
                                break
                # Format agents for display: show name and description
                agent_options = [(f"{name}: {desc}", name) for name, desc in agents]
                # Show the inline picker first (with 'agent' type for tracking)
                await self.show_inline_picker(f"{EMOJI['robot']} Select Agent Workflow", agent_options, picker_type="agent")
                # Store callback for when selection is made (after picker is shown)
                self._picker_callback = on_agent_selected
            else:
                chat_panel.add_message("system", "No agents available. Use `/add_agent` to add one.")
        else:
            chat_panel.add_message("error", "Plugin manager not initialized.")
    
    def action_toggle_files(self):
        """Toggle the file explorer panel visibility."""
        file_panel = self.query_one("#file-panel", FileExplorerPanel)
        if self.show_files:
            file_panel.remove_class("visible")
            self.show_files = False
        else:
            file_panel.add_class("visible")
            self.show_files = True
            # Focus the file tree when opening the panel
            file_tree = self.query_one("#file-tree", DirectoryTree)
            file_tree.focus()
    
    async def action_change_model(self):
        """Prompt to change the AI model with inline picker."""
        async def handle_model_selection(selected_model):
            if selected_model:
                chat_panel = self.query_one("#chat-panel", ChatPanel)
                chat_panel.add_message("system", f"✅ Model changed to: **{selected_model}**\n\nUpdating chatbot...")
                
                # Update status bar and chatbot's model
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.model_name = selected_model
                
                # Update the chatbot's model directly
                if self.chatbot:
                    self.chatbot.modelname = selected_model
                    chat_panel.add_message("system", f"✅ Now using model: **{selected_model}**")
                else:
                    chat_panel.add_message("error", f"{EMOJI['cross']} ChatBot not initialized.")
        
        # Show inline picker for model selection
        status_bar = self.query_one("#status-bar", StatusBar)
        current_model = status_bar.model_name
        model_names = ["gpt-4o", "gpt-5", "gpt-4.1", "gpt-oss:20b"]
        
        # Format model names EXACTLY like agent picker does - with newline and description
        model_options = [(f"{name}", name) for name in model_names]
        
        # Show the inline picker first (with 'model' type for tracking)
        await self.show_inline_picker(f"{EMOJI['robot']} Select AI Model", model_options, picker_type="model")
        
        # Store callback for when selection is made (after picker is shown)
        self._picker_callback = handle_model_selection
    
    def action_save_session(self):
        """Save the current chat session to a log file."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        
        try:
            # Generate filename with timestamp
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            log_filename = f"chatlog_{timestamp}.log"
            log_filepath = Path.cwd() / log_filename
            
            # Collect all messages from the chat panel
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write(f"ValBot Chat Log\n")
                f.write(f"Saved: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                for msg in chat_panel.messages:
                    # Get role and format it
                    role = msg.role.upper()
                    agent_info = f" ({msg.agent_name})" if hasattr(msg, 'agent_name') and msg.agent_name else ""
                    
                    f.write(f"[{role}{agent_info}]\n")
                    f.write(msg.content)
                    f.write("\n\n" + "-" * 80 + "\n\n")
            
            # Show success message
            chat_panel.add_message("system", f"✅ Chat saved to `{log_filepath}`")
            
        except Exception as e:
            chat_panel.add_message("error", f"{EMOJI['cross']} Failed to save chat: {str(e)}")
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle click on a file in the directory tree.
        
        Single-click behavior:
        - Click on file: Adds path to chat and closes file explorer
        """
        # Only process if it's actually a file (not a directory)
        if not event.path.is_file():
            return
        
        # Add file to chat immediately
        file_path = event.path
        try:
            # Convert to relative path from current directory
            import os
            relative_path = os.path.relpath(file_path, os.getcwd())
            
            # Get the command input widget
            command_input = self.query_one("#command-input", CommandInput)
            
            # Insert the path at the cursor position or append if empty
            current_text = command_input.text
            if current_text:
                # Add space before the path only if current text does not end with whitespace
                if not current_text[-1].isspace():
                    command_input.insert(f" {relative_path}")
                else:
                    command_input.insert(relative_path)
            else:
                command_input.insert(relative_path)
            
            # Close the file explorer
            self.action_toggle_files()
            
            # Focus the command input
            command_input.focus()
            
        except Exception as e:
            # If there's an error, just use the absolute path
            command_input = self.query_one("#command-input", CommandInput)
            command_input.insert(str(file_path))
            self.action_toggle_files()
            command_input.focus()


class ValbotTUI(App):
    """ValBot Terminal User Interface with Material Design."""
    
    TITLE = f"{EMOJI['robot']} ValBot TUI - AI Assistant"
    
    # Material Design inspired CSS
    CSS = """
    App {
        background: $background;
    }
    
    Screen {
        background: $background;
    }
    
    /* Enhanced Header */
    Header {
        dock: top;
        height: 1;
        text-style: bold;
        text-align: center;
        align: center middle;
    }
    
    Header .header--title {
        text-style: bold;
        text-align: center;
    }
    
    Header .header--clock {
        color: $primary-lighten-2;
    }
    
    /* Enhanced Footer */
    Footer {
        dock: bottom;
        height: 1;
        background: $surface;
    }
    
    Footer > .footer--highlight {
        background: $accent;
    }
    
    Footer > .footer--key {
        background: $primary-darken-1;
    }
    
    Footer > .footer--description {
        color: $text-muted;
    }
    
    /* Smooth animations */
    * {
        transition: background 200ms, color 200ms;
    }
    
    /* Remove all focus borders */
    *:focus {
        border: none !important;
    }
    
    /* Loading spinner */
    LoadingIndicator {
        background: $surface;
        color: $accent;
    }
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config_manager = ConfigManager(config_path)
        utilities.set_config_manager(self.config_manager)
        # Register and use the custom valbot-dark theme
        self.register_theme(VALBOT_DARK_THEME)
        self.theme = "valbot-dark (default)"
        
    def on_mount(self) -> None:
        """Set up the application after mounting."""
        self.push_screen(MainScreen(self.config_manager))


def main():
    """Main entry point for the TUI application."""
    import argparse
    import locale
    
    # Force UTF-8 encoding for UNIX systems to properly display emojis
    try:
        # Set locale to UTF-8 if available
        locale.setlocale(locale.LC_ALL, '')
        
        # For Python 3.7+, ensure UTF-8 mode is enabled
        if sys.version_info >= (3, 7):
            # Set environment variables before any I/O operations
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # Reconfigure stdout/stderr for UTF-8
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (locale.Error, Exception) as e:
        # Fallback if locale setting fails
        import os
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        pass
    
    parser = argparse.ArgumentParser(description="ValBot TUI - Terminal User Interface")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    args = parser.parse_args()
    
    try:
        app = ValbotTUI(config_path=args.config)
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
