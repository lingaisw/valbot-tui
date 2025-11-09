from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter, Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from getpass import getpass

####################### Console with History Support ################################
class HistoryConsole(Console):
    def __init__(self, *args, **kwargs):
        self.history = InMemoryHistory()
        self.prompt_session = PromptSession(history=self.history, reserve_space_for_menu=3)
        return super().__init__(*args, **kwargs)

    def input(
        self, prompt="", markup=True, emoji=True, password=False, stream=None
    ) -> str:
        if prompt:
            self.print(prompt, markup=markup, emoji=emoji, end="\n")
        if password:
            result = getpass("", stream=stream)
        else:
            if stream:
                result = stream.readline()
            else:
                result = self.prompt_session.prompt("")
        return result

    def path_prompt(self, prompt="", single_path_only=False, markup=True, emoji=True, password=False, stream=None):
        """Prompt for a file path using a temporary session with PathCompleter."""
        completer = PathCompleter(expanduser=True, only_directories=False)
        session = PromptSession(completer=completer)
        bindings = KeyBindings()

        @bindings.add('enter')
        def _(event):
            event.current_buffer.insert_text(' ')
            event.current_buffer.validate_and_handle()

        input_text = ""
        while True:
            part = session.prompt(prompt, key_bindings=bindings)
            if single_path_only:
                return part.strip()
            self.print("[white]Hit[/white] [bold green]Enter[/bold green] [white]again to finish input or continue adding paths.[/white]")
            if not part.strip():
                break
            input_text += part + ' '
        return input_text.strip()

    def add_command_completer(self, command_manager):
        """Add the command completer to the console."""
        self.command_completer = CommandCompleter(command_manager)
        self.prompt_session.completer = self.command_completer

class CommandCompleter(Completer):
    def __init__(self, command_manager):
        self.command_manager = command_manager

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith('/'):
            # Yield completions for registered slash commands
            for command_name in self.command_manager.command_registry.keys():
                if command_name.startswith(text):
                    yield Completion(command_name, start_position=-len(text))
            # Yield completions for prompt commands
            for prompt_name in self.command_manager.prompts.keys():
                if prompt_name.startswith(text):
                    yield Completion(prompt_name, start_position=-len(text))