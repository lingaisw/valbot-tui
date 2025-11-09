import os
import asyncio
from agent_plugins.common_agent_imports import *
from agent_plugins.agent_plugin import AgentPlugin
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from agent_plugins.common_tools.terminal_tools import list_files, find_files, grep_in_files, show_file_tree, run_shell_command
from agent_plugins.common_tools.ask_human import ask_human, ask_human_for_more_instructions
from agent_plugins.common_tools.file_tools import read_file, FileContent

@dataclass
class TerminalAgentDeps:
    user_request: str

class TerminalAgentPlugin(AgentPlugin):
    REQUIRED_ARGS = {
        'request': (AgentPlugin.simple_prompt, 'Describe what you want the terminal agent to do'),
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.terminal_agent = self.create_terminal_agent(model)
        self.apply_tool_decorators()
        self.console = Console()

    def create_terminal_agent(self, model):
        return Agent[TerminalAgentDeps](
            model=model,
            system_prompt=(
                'You are a terminal agent. You can navigate files, search, and run shell commands in the given root_dir. '\
                'You have access to the following tools:'\
                '  - list_files: list files in a directory, optionally using a pattern and recursive mode.'\
                '  - find_files: search for a filename recursively in root_dir.'\
                '  - grep_in_files: search within files in a directory for a specific text.'\
                '  - show_file_tree: display a visual directory tree (max_depth: 3 by default).'\
                '  - run_shell_command: run custom shell commands with output.'\
                '  - ask_human: ask the user a question and get their response.'\
                '  - ask_human_for_more_instructions: ask the user for more instructions if needed.'\
                '  - read_file: read the content of a file.'\
                'Given the user_request, compose an appropriate plan, run one or more tools, and summarize the results.'\
                'Always ask by using your `ask_human` and/or `ask_human_for_more_instructions` tools before running any writing or manipulating commands to get confirmation from the user'\
                'You can only ask using your tools, not by responding directly.'\
                'If a directory appears to have hundreds or thousands of files, always prompt for a narrower pattern.'\
                'Do not assume anything about the type of files until you see what files types are present.'\
            ),
            deps_type=TerminalAgentDeps,
            retries=2,
            instrument=True,
        )

    def apply_tool_decorators(self):
        self.terminal_agent.tool(list_files)
        self.terminal_agent.tool(find_files)
        self.terminal_agent.tool(grep_in_files)
        self.terminal_agent.tool(show_file_tree)
        self.terminal_agent.tool(run_shell_command)
        self.terminal_agent.tool(ask_human)
        self.terminal_agent.tool(ask_human_for_more_instructions)
        self.terminal_agent.tool(read_file)

    def run_agent_flow(self, context, **kwargs):
        self.console.print("[bold green]Entering terminal agent mode.[/bold green]")

        root_dir = os.getcwd()
        user_request = self.initializer_args['request']
        deps = TerminalAgentDeps(user_request=user_request)
        message_history = []
        while True:
            self.console.print(f"[grey74]Processing your request...[/grey74]")
            result = asyncio.run(self.terminal_agent.run(
                f"User request: {user_request}\nRoot directory: {root_dir}",
                deps=deps,
                model_settings={'max_completion_tokens': 6000},
                message_history=message_history
            ))
            self.console.print(Markdown(result.data))
            message_history.extend(result.new_messages())

            self.console.print("[bold yellow]Type your next request to continue the session. Leave empty and press Enter to exit.[/bold yellow]")
            try:
                next_request = self.console.input("[bold green]Request:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("[bold green]Exiting terminal agent mode.[/bold green]")
                break
            if next_request.strip() == "":
                self.console.print("[bold green]Exiting terminal agent mode.[/bold green]")
                break

            user_request = next_request.strip()
            deps = TerminalAgentDeps(user_request=user_request)
