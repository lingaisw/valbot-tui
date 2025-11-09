from abc import ABC, abstractmethod
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from rich.prompt import Prompt
from rich.console import Console

class AgentPlugin(ABC):
    # Define required arguments and their prompt methods
    REQUIRED_ARGS = {}

    def __init__(self, model, **kwargs):
        self.model = model
        self.initializer_args = kwargs
        self.console = Console(force_terminal=True)

    @abstractmethod
    def run_agent_flow(self, context, **kwargs):
        """Run the agent with the given context."""
        pass

    def run_agent_flow_with_init_args(self, context, **kwargs):
        self.initializer_args = self.initialize_args(**kwargs)
        self.run_agent_flow(context, **kwargs)

    def initialize_args(self, **kwargs):
        """Initialize arguments using defaults, passed values, or prompting."""
        args = {}
        for arg_name, (prompt_method, prompt_text, *extra_args) in self.REQUIRED_ARGS.items():
            arg_value = kwargs.get(arg_name) or self.initializer_args.get(arg_name)
            if not arg_value:
                arg_value = prompt_method(self, prompt_text, *extra_args)
            args[arg_name] = arg_value
        return args

    def simple_prompt(self, prompt_text):
        """Simple console prompt for getting a string."""
        return Prompt.ask(f"[bold green]{prompt_text}[/bold green]").strip()

    def path_prompt(self, prompt_text):
        """Prompt for a path using PromptSession with PathCompleter.
        
        In TUI mode, falls back to simple_prompt since PathCompleter doesn't work
        with the TUI's chat-based input system.
        """
        # Detect if we're in TUI mode by checking if Console class has been monkey-patched
        # The TUI replaces Console with a wrapper class at runtime
        try:
            import rich.console
            # Check if Console class name indicates it's been patched
            # The TUI wrapper is named 'TUIConsole'
            console_class_name = rich.console.Console.__name__
            if console_class_name == 'TUIConsole' or console_class_name != 'Console':
                # TUI mode detected - use simple_prompt instead
                return self.simple_prompt(prompt_text)
        except (ImportError, AttributeError):
            pass  # If check fails, proceed with normal path_prompt
        
        # Normal CLI mode - use path completion
        self.console.print(f"[bold green]{prompt_text}[/bold green]")
        session = PromptSession()
        return session.prompt(completer=PathCompleter()).strip()

    def choice_prompt(self, prompt_text, choices, default=None):
        """Prompt for a choice from a list of options."""
        return Prompt.ask(f"[bold green]{prompt_text}[/bold green]", choices=choices, default=default).strip()
