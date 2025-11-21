import os
import glob
import subprocess
import tempfile
import sys
import threading
from time import monotonic
from rich.spinner import Spinner
from typing import List, Union, Optional

import readchar
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.prompt import Prompt, Confirm

from bot_commands import Command, PromptCommand, CommandManager
import config as configuration
from tui.settings_app import SettingsApp
from context_management import ContextManager
from client_setup import *
from console_extensions import HistoryConsole
from help_bot import HelpBotApp
from agent_plugins.plugin_manager import PluginManager
from extension_manager import AgentAdder, ToolAdder
from valbot_updater import ValbotUpdater
from openai.types.responses import *

# Import pydantic-ai for tool integration
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dataclasses import dataclass

# Import file and terminal tools from common_tools
from agent_plugins.common_tools.file_tools import (
    read_file as file_read_tool,
    read_partial_file,
    create_file,
    write_file,
    edit_string_in_file,
    list_files_in_directory,
    find_files_with_name
)
from agent_plugins.common_tools.terminal_tools import (
    list_files,
    find_files,
    grep_in_files,
    show_file_tree,
    run_shell_command
)

@dataclass
class ChatDeps:
    """Dependencies for the chat agent with tools."""
    conversation_history: List[dict]

class PromptDisplay:
    def __init__(self, sync_enabled_getter=lambda: True, conversation_id_getter=lambda: None, display_conversation_id=True):
        self.sync_enabled_getter = sync_enabled_getter
        self.conversation_id_getter = conversation_id_getter
        self.display_conversation_id = display_conversation_id
        self.default_prompt_text = "[bold green]You[/bold green]"

    def display(self, prompt: Optional[str] = None) -> str:
        if not prompt:
            prompt = self.default_prompt_text
        if self.sync_enabled_getter():
            conv_id = self.conversation_id_getter()
            conv_id_display = f"({conv_id[:8]})" if conv_id is not None and self.display_conversation_id else ""
            sync_text = f"[green1]↑↓{conv_id_display}[/green1] "
        else:
            sync_text = ""
        return f"{sync_text}{prompt}"

class ChatBot:
    def __init__(self, agent_model=None, config_manager=None, plugin_manager=None):
        self.console = HistoryConsole()
        self.config_manager = config_manager
        self.modelname = self.config_manager.get_setting("chat_model_config.default_model", "gpt-4o")
        self.client = initialize_chat_client(
            endpoint=self.config_manager.get_setting("chat_model_config.default_endpoint"),
            model_name=self.modelname
        )
        self.context_manager = ContextManager(self.console)
        self.rag_knowledge_base = None  # Will be initialized when database is created/loaded
        if config_manager is None:
            self.console.print("[bold red]Error loading configuration. Exiting.[/bold red]")
            sys.exit(1)
        self.plugin_manager = plugin_manager
        self.plugin_manager.load_plugins()
        self.command_manager = CommandManager(self, self.plugin_manager, self.console)
        self.console.add_command_completer(self.command_manager)
        self.cloud_sync_enabled = self.config_manager.get_setting('cloud_settings.conversation_sync_enabled', False)
        self._conversation_id: Optional[str] = None
        display_conversation_id = self.config_manager.get_setting('cloud_settings.display_conversation_id', False)
        self.prompt_display = PromptDisplay(lambda: self.cloud_sync_enabled, lambda: self._conversation_id, display_conversation_id)
        
        # Store agent_model and create pydantic-ai agent with file/terminal tools
        self.agent_model = agent_model
        self.tool_agent = None  # Initialize to None
        if self.agent_model:
            try:
                self._setup_tool_agent()
                self.console.print("[dim green]✓ Tool agent initialized with file and terminal tools[/dim green]")
            except Exception as e:
                self.console.print(f"[yellow]⚠ Could not initialize tool agent: {e}[/yellow]")
                self.agent_model = None  # Disable tools if setup fails

    def _setup_tool_agent(self):
        """Setup pydantic-ai agent with file and terminal tools."""
        system_prompt = (
            'You are a helpful AI assistant with access to file system and terminal tools. '
            'You can help users:\n'
            '- List and search for files in directories\n'
            '- Read file contents (full or partial)\n'
            '- Create, edit, and write files\n'
            '- Search text within files using grep\n'
            '- Display directory tree structures\n'
            '- Execute shell commands\n\n'
            'When users ask about files or want to execute commands, use the appropriate tools. '
            'Always provide clear, helpful responses. Prioritize markdown format and code blocks when applicable.'
        )
        
        self.tool_agent = Agent[ChatDeps](
            model=self.agent_model,
            system_prompt=system_prompt,
            deps_type=ChatDeps,
            retries=3,
            instrument=True,
        )
        
        # Register all file and terminal tools
        self.tool_agent.tool(list_files)
        self.tool_agent.tool(find_files)
        self.tool_agent.tool(grep_in_files)
        self.tool_agent.tool(show_file_tree)
        self.tool_agent.tool(run_shell_command)
        self.tool_agent.tool(file_read_tool)
        self.tool_agent.tool(read_partial_file)
        self.tool_agent.tool(create_file)
        self.tool_agent.tool(write_file)
        self.tool_agent.tool(edit_string_in_file)
        self.tool_agent.tool(list_files_in_directory)
        self.tool_agent.tool(find_files_with_name)


    def display_banner(self):
        ascii_banner_small = r"""
__   __    _ ___      _      ___ _    ___
\ \ / /_ _| | _ ) ___| |_   / __| |  |_ _|
 \ V / _` | | _ \/ _ \  _| | (__| |__ | |
  \_/\__,_|_|___/\___/\__|  \___|____|___|
        """
        ascii_banner_large = r"""
$$\    $$\          $$\ $$$$$$$\             $$\            $$$$$$\  $$\       $$$$$$\
$$ |   $$ |         $$ |$$  __$$\            $$ |          $$  __$$\ $$ |      \_$$  _|
$$ |   $$ |$$$$$$\  $$ |$$ |  $$ | $$$$$$\ $$$$$$\         $$ /  \__|$$ |        $$ |
\$$\  $$  |\____$$\ $$ |$$$$$$$\ |$$  __$$\\_$$  _|        $$ |      $$ |        $$ |
 \$$\$$  / $$$$$$$ |$$ |$$  __$$\ $$ /  $$ | $$ |          $$ |      $$ |        $$ |
  \$$$  / $$  __$$ |$$ |$$ |  $$ |$$ |  $$ | $$ |$$\       $$ |  $$\ $$ |        $$ |
   \$  /  \$$$$$$$ |$$ |$$$$$$$  |\$$$$$$  | \$$$$  |      \$$$$$$  |$$$$$$$$\ $$$$$$\
    \_/    \_______|\__|\_______/  \______/   \____/        \______/ \________|\______|
        """
        ascii_banner_size = self.config_manager.get_setting('general.ascii_banner_size', 'small')
        ascii_banner = {
            'small': ascii_banner_small,
            'large': ascii_banner_large
        }.get(ascii_banner_size, "")

        if ascii_banner:
            self.console.print(Panel(ascii_banner, title="[bold blue]Validation BOT[/bold blue]",
                                    border_style="bright_blue", expand=False))
        updater = ValbotUpdater(self.console)
        self.console.print(f"[bold yellow]Version: {updater.get_version()}[/bold yellow]")
        self.console.print(f"[bold blue]Welcome to ValBot-CLI. Chat away![/bold blue]")
        self.console.print(f"Chatting with: {self.modelname} (/model to change models)\n")
        self.context_manager.conversation_history.append({"role": "system", "content": self.config_manager.get_setting("chat_model_config.system_prompt", "You are a helpful assistant. Prioritize markdown format and code blocks when applicable.")})

    def _should_use_tools(self, message: str) -> tuple[bool, list[str]]:
        """Determine if a message should use the tool-enabled agent.
        
        Returns:
            tuple: (should_use_tools: bool, detected_files: list[str])
        """
        message_lower = message.lower()
        
        # Check for valid file paths in the message
        import re
        from pathlib import Path
        
        # Look for potential file paths (with extensions or absolute/relative paths)
        # Match patterns like: ./file.txt, ../dir/file.py, C:\path\file.js, /path/to/file, file.md, agent_plugins/init.py
        file_path_patterns = [
            r'[./\\][\w/\\.-]+\.\w+',      # Relative paths starting with . or \ with extension
            r'[A-Za-z]:\\[\w\\.-]+',        # Windows absolute paths
            r'/[\w/.-]+',                   # Unix-like absolute paths
            r'\b[\w-]+(?:[/\\][\w-]+)+\.\w+', # Subdirectory paths (word/word/file.ext)
            r'(?<![/\\])(?<!\w)\b[\w-]+\.\w{2,5}\b(?![/\\])'  # Simple filenames not part of a path
        ]
        
        potential_paths = []
        for pattern in file_path_patterns:
            matches = re.findall(pattern, message)
            potential_paths.extend(matches)
        
        # Check if any of these paths actually exist
        valid_files = []
        for path_str in potential_paths:
            try:
                path = Path(path_str)
                if path.exists() and path.is_file():
                    valid_files.append(path_str)
            except:
                continue
        
        # Print all found files in a single message
        valid_path_found = len(valid_files) > 0
        if valid_path_found:
            files_str = ", ".join([f"[bold]{f}[/bold]" for f in valid_files])
            file_word = "file" if len(valid_files) == 1 else "files"
            self.console.print(f"Found local {file_word}: {files_str}")
            return True, valid_files
        
        # Keywords that suggest directory/file listing operations
        directory_keywords = ['list files', 'show files', 'ls ', 'dir ', 'file tree', 
                             'directory structure', 'show tree', 'list all', 'show all']
        if any(keyword in message_lower for keyword in directory_keywords):
            self.console.print(f"[dim]→ Using tools (directory operation)[/dim]")
            return True, []
        
        return False, []
    
    async def send_message_with_tools(self, message: str):
        """Send a message using the tool-enabled pydantic-ai agent."""
        if not self.agent_model or not self.tool_agent:
            self.console.print("[bold yellow]Agent model not available. Using standard chat.[/bold yellow]")
            self.send_message(message)
            return
            
        self.console.print(Rule(f"[bold blue]Bot with Tools [/bold blue]", style="bright_blue"))
        
        try:
            # Add user message to conversation history first
            self.context_manager.conversation_history.append({"role": "user", "content": message})
            
            # Create deps with conversation history
            deps = ChatDeps(conversation_history=self.context_manager.conversation_history)
            
            # Run the agent with the user's message
            self.console.print("[dim]Running agent with tools...[/dim]")
            result = await self.tool_agent.run(message, deps=deps)
            
            # Display the response - use .output instead of .data
            response_text = str(result.output) if hasattr(result, 'output') else str(result)
            self.console.print(Markdown(response_text))
            
            # Add assistant response to conversation history
            self.context_manager.conversation_history.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            import traceback
            self.console.print(f"[bold red]Error using tools: {e}[/bold red]")
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.console.print("[bold yellow]Falling back to standard chat...[/bold yellow]")
            # Remove the user message we added, since we'll add it again in send_message
            if self.context_manager.conversation_history and \
               self.context_manager.conversation_history[-1]["role"] == "user":
                self.context_manager.conversation_history.pop()
            self.send_message(message)
        
        self.console.print(Rule(style="bright_blue"))

    def send_message(self, message):
        self.context_manager.conversation_history.append({"role": "user", "content": message})
        messages = self.context_manager.conversation_history
        effort_level = self.config_manager.get_setting("chat_model_config.reasoning_effort", "low")

        # Validate effort_level
        valid_effort_levels = ["low", "medium", "high"]
        if effort_level not in valid_effort_levels:
            self.console.print(f"[bold red]Warning: Invalid reasoning_effort value '{effort_level}'. Must be one of: {', '.join(valid_effort_levels)}[/bold red]")
            self.console.print(f"[bold yellow]Falling back to default value 'low'[/bold yellow]")
            effort_level = "low"

        try:
            start = monotonic()
            kwargs = {
                "model": self.modelname,
                "input": messages,
                "stream": True,
            }
            if self.modelname.startswith("gpt-5"):
                kwargs["reasoning"] = {"summary": "auto", "effort": effort_level}

            stream = self.client.responses.create(**kwargs)
        except Exception as e:
            self.console.print(f"[bold red]Error communicating with the API: {e}[/bold red]")
            if "Token has expired and is not valid anymore" in str(e):
                self.console.print("[bold yellow] Attempting to reinitialize the client with new credentials...[/bold yellow]")
                # reobtain the client with new credentials, and retry the message
                self.client = initialize_chat_client(
                    endpoint=self.config_manager.get_setting("chat_model_config.default_endpoint"),
                    model_name=self.modelname
                )
                self.send_message(message)
            return

        response_text = self.gather_and_display_response(stream, effort_level)
        self.context_manager.conversation_history.append({"role": "assistant", "content": response_text})

    def gather_and_display_response(self, stream, effort_level="medium"):
        response_text = ""
        reasoning_text = ""
        display_reasoning = self.config_manager.get_setting("chat_model_config.display_reasoning", False)

        self.console.print(Rule(f"[bold blue]Bot [/bold blue]({self.modelname})", style="bright_blue"))
        with Live('', refresh_per_second=6, console=self.console) as live:
            for response in stream:
                match response:
                    case ResponseCreatedEvent():
                        pass
                    case ResponseInProgressEvent():
                        pass
                    case ResponseAudioDeltaEvent():
                        if response.delta:
                            reasoning_text += response.delta
                            if display_reasoning:
                                content = Markdown(reasoning_text)
                            else:
                                content = Text.assemble(
                                    (f"Thinking... (effort: {effort_level})\n\n", "bold cyan"),
                                    ("(Use /settings to enable reasoning display and change effort level)", "dim bold yellow")
                                )
                                content.justify = "center"

                            live.update(Panel(
                                content,
                                title=f"[bold cyan]Thinking... (effort: {effort_level})[/bold cyan]" if display_reasoning else None,
                                border_style="cyan",
                                padding=(1, 2) if display_reasoning else (0, 2)
                            ))
                    case ResponseTextDeltaEvent():
                        if response.delta:
                            if reasoning_text: live.update("")
                            response_text += response.delta
                            live.update(Markdown(response_text))
                    case _:
                        pass
        self.console.print(Rule(style="bright_blue"))
        return response_text

    @CommandManager.register_command('/prompts')
    def display_prompts(self):
        """Display available prompts with their descriptions and arguments."""
        self.command_manager.display_prompts()

    @CommandManager.register_command('/commands')
    def display_slash_commands(self):
        """Display available slash commands with their descriptions."""
        self.command_manager.display_slash_commands()

    @CommandManager.register_command('/clear')
    def clear_conversation(self):
        """Clear the conversation history."""
        self.context_manager.clear_conversation()
        self._conversation_id = None   # reset backend session

    @CommandManager.register_command('/quit')
    def exit_chat(self):
        """Exit the chat."""
        self.console.print("\n[bold red]Exiting...[/bold red]")
        sys.exit(0)

    @CommandManager.register_command('/context')
    def load_more_context(self):
        """Load more context from a file or list of files."""
        self.context_manager.load_more_context()

    @CommandManager.register_command('/help')
    def load_valbot_context_and_display_help_prompt(self):
        """Chat with the help bot."""
        HelpBotApp(self.config_manager.get_setting("chat_model_config.default_endpoint"), self.modelname).run()

    @CommandManager.register_command('/settings')
    def display_settings(self):
        """Display and modify settings."""
        SettingsApp(self.config_manager).run()

    @CommandManager.register_command('/new', "Start a new conversation")
    def new_conversation(self):
        """Start a new conversation."""
        self.context_manager.clear_conversation()
        self.console.print("[bold blue]New conversation started.[/bold blue]")

    @CommandManager.register_command('/model')
    def change_chat_model(self):
        """Model picker."""
        model_names = ["gpt-4o", "gpt-5", "gpt-4.1", "gpt-oss:20b"]  # TODO: fetch dynamically from the client when supported

        try:
            selected_index = model_names.index(self.modelname)
        except ValueError:
            selected_index = 0

        def render():
            lines = [
                "Select a model (use ↑ ↓ and Enter, or press a number 1-{}):".format(len(model_names))
            ]
            for i, name in enumerate(model_names, start=1):
                prefix = "> " if (i - 1) == selected_index else "  "
                line = f"{prefix}{i}) {name}"
                if (i - 1) == selected_index:
                    lines.append(f"[bold green]{line}[/bold green]")
                else:
                    lines.append(line)
            return Text.from_markup("\n".join(lines))

        with Live(render(), refresh_per_second=10, console=self.console) as live:
            while True:
                key = readchar.readkey()
                if key == readchar.key.UP:
                    selected_index = (selected_index - 1) % len(model_names)
                elif key == readchar.key.DOWN:
                    selected_index = (selected_index + 1) % len(model_names)
                elif key == readchar.key.ENTER:
                    break
                elif key.isdigit():
                    # interpret single digit as selection (1-based)
                    num = int(key)
                    if 1 <= num <= len(model_names):
                        selected_index = num - 1
                        break
                live.update(render())

        self.modelname = model_names[selected_index]
        self.console.print(f"[bold green]Model changed to {self.modelname}[/bold green]")

        # Reinitialize client to use correct client type for the new model
        self.client = initialize_chat_client(
            endpoint=self.config_manager.get_setting("chat_model_config.default_endpoint"),
            model_name=self.modelname
        )

    @CommandManager.register_command('/add_agent')
    def add_agent(self, path_or_url: Optional[str] = None):
        """Add a new agent from a git repo or local path."""
        AgentAdder(self.console, self.config_manager).run(path_or_url)
        self.reload_app()

    @CommandManager.register_command('/add_tool')
    def add_tool(self, path_or_url: Optional[str] = None):
        """Add a new tool from a git repo or local path."""
        ToolAdder(self.console, self.config_manager).run(path_or_url)
        self.reload_app()

    @CommandManager.register_command('/reload')
    def reload_app(self):
        """Restart the application to reload configuration."""
        if Confirm.ask("[bold yellow]Restart now to reload configuration?[/bold yellow]", default=True):
            self.console.print("[yellow]Restarting...[/yellow]")
            python = sys.executable
            argv = [python] + sys.argv
            try:
                # Replace the current process (preserves args and env)
                os.execv(python, argv)
            except Exception as e:
                # Fallback: spawn a new process and exit the current one
                self.console.print(f"[red]Direct restart failed ({e}). Spawning a new process...[/red]")
                subprocess.Popen(argv)
                sys.exit(0)

    @CommandManager.register_command('/update')
    def update(self):
        # Ask the user if they want to check for updates on the plugins or the main app
        """Check for updates to the main app and installed plugins."""
        choice = Prompt.ask("Check for updates to [bold]valbot[/bold] or [bold]plugins[/bold]?", choices=["valbot", "plugins", "both", "none"], default="none")
        if choice in ("valbot", "both"):
            self.valbot_update()
        if choice in ("plugins", "both"):
            self.plugin_manager.plugin_update_flow()
            self.plugin_manager.plugin_cleanup_flow(dry_run=False)

    def valbot_update(self):
        """Check for updates to the main application."""
        updater = ValbotUpdater(self.console, self.reload_app, self.exit_chat)
        updater.update()


