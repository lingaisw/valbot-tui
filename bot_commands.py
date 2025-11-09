import re
import subprocess
from typing import Callable, Dict, List
import shlex

class Command:
    def __init__(self, method: Callable, description: str):
        self.method = method
        self.description = description

class PromptCommand:
    def __init__(self, custom_prompt):
        """Initialize the PromptCommand with custom prompt details."""
        self.prompt_cmd = custom_prompt['prompt_cmd']
        self.description = custom_prompt.get('description', "No description available.")
        self.prompt_template = custom_prompt['prompt']
        self.arg_names = custom_prompt.get('args', [])

    def execute_command(self, command):
        """Execute a shell command and return its output."""
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.stdout.decode('utf-8').strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command '{command}' failed with error: {e.stderr.decode('utf-8').strip()}")

    def format_prompt(self, rest_of_line):
        """Format the prompt by replacing placeholders with actual values."""
        try:
            arg_values = self.parse_arguments(rest_of_line)
            formatted_prompt = self.replace_placeholders(arg_values)
            return formatted_prompt
        except Exception as e:
            raise RuntimeError(f"Failed to format prompt: {str(e)}")

    def parse_arguments(self, rest_of_line):
        """Parse arguments and execute commands if needed."""
        if len(self.arg_names) == 1:
            return {self.arg_names[0]: self.process_argument(rest_of_line.strip(), self.arg_names[0])}

        parsed_args = re.findall(r'"(.*?)"', rest_of_line)
        if len(parsed_args) != len(self.arg_names):
            expected_args = ', '.join(self.arg_names)
            example_format = ' '.join([f'"{arg}"' for arg in self.arg_names])
            raise ValueError(
                f"Expected {len(self.arg_names)} arguments ({expected_args}), "
                f"but got {len(parsed_args)}. Please use quotes to separate arguments, e.g., {example_format}."
            )

        return {arg_name: self.process_argument(arg_value, arg_name) for arg_name, arg_value in zip(self.arg_names, parsed_args)}

    def process_argument(self, arg_value, arg_name):
        """Process an argument, executing it as a command if indicated."""
        if f"{{cmd:{arg_name}}}" in self.prompt_template:
            return self.execute_command(arg_value)
        return arg_value

    def replace_placeholders(self, arg_values):
        """Replace placeholders in the prompt template with actual values."""
        formatted_prompt = self.prompt_template
        for arg_name, arg_value in arg_values.items():
            formatted_prompt = formatted_prompt.replace(f"{{cmd:{arg_name}}}", arg_value)
            formatted_prompt = formatted_prompt.replace(f"{{{arg_name}}}", arg_value)
        return formatted_prompt

class CommandManager:
    command_registry = {}

    @staticmethod
    def register_command(command_name, description=None, bind_to_chatbot=True):
        def decorator(func):
            cmd_description = description or func.__doc__ or "No description available."
            CommandManager.command_registry[command_name] = {
                'function': func,
                'description': cmd_description,
                'bind_to_chatbot': bind_to_chatbot
            }
            return func
        return decorator

    def __init__(self, chatbot, plugin_manager, console):
        self.chatbot = chatbot
        self.plugin_manager = plugin_manager
        self.console = console
        self.setup_custom_commands()
        self.setup_prompts()

    def handle_command(self, command):
        # Parse the command using shlex so quoted args are handled properly
        try:
            parts = shlex.split(command)
        except ValueError as e:
            self.console.print(f"[bold red]Failed to parse command: {e}[/bold red]")
            return
        if not parts:
            return

        name = parts[0]
        args = parts[1:]

        if name in self.command_registry:
            command_info = self.command_registry[name]
            func = command_info['function']
            if command_info['bind_to_chatbot']:
                bound_func = func.__get__(self.chatbot, type(self.chatbot))
            else:
                bound_func = func

            # Try to pass args; fall back to no-arg call for legacy commands
            try:
                bound_func(*args)
            except TypeError:
                bound_func()
        elif name in self.prompts:
            # Keep prompt parsing behavior unchanged
            rest_of_line = command[len(name):].lstrip()
            try:
                prompt = self.prompts[name].format_prompt(rest_of_line)
                self.chatbot.send_message(prompt)
            except Exception as e:
                self.console.print(f"[bold red]Error executing prompt: {str(e)}[/bold red]")
        else:
            self.console.print(f"[bold red]Unknown command: {command}[/bold red]")

    def execute_prompt(self, command):
        """Execute a prompt command with additional arguments."""
        command_parts = command.split()
        prompt_name = command_parts[0]
        rest_of_line = ' '.join(command_parts[1:])
        try:
            prompt = self.prompts[prompt_name].format_prompt(rest_of_line)
            self.chatbot.send_message(prompt)
        except Exception as e:
            self.console.print(f"[bold red]Error executing prompt: {str(e)}[/bold red]")

    def setup_custom_commands(self):
        """Setup user-defined custom commands from the plugin manager."""
        custom_command_info = self.plugin_manager.get_custom_command_info()
        for custom_command in custom_command_info:
            command_name = custom_command['command']
            description = custom_command['description']
            # Register a function to delegate execution to the PluginManager
            self.register_command(command_name, description=description, bind_to_chatbot=False)(
                lambda command_name=command_name: self.plugin_manager.run_custom_command(command_name, self.chatbot.context_manager.conversation_history)
            )

    def setup_prompts(self):
        """Setup user-defined prompts from the plugin manager."""
        self.prompts = {}
        custom_prompt_info = self.plugin_manager.get_custom_prompt_info()
        for custom_prompt in custom_prompt_info:
            prompt = '/' + custom_prompt['prompt_cmd'] if not custom_prompt['prompt_cmd'].startswith('/') else custom_prompt['prompt_cmd']
            self.prompts[prompt] = PromptCommand(custom_prompt)

    def display_slash_commands(self):
        """Display available slash commands with their descriptions."""
        self.console.print("[bold green]Available commands:[/bold green]")
        for command_name, command_info in self.command_registry.items():
            self.console.print(f" - {command_name}: {command_info['description']}")

    def display_prompts(self):
        """Display available prompts with their descriptions."""
        self.console.print("[bold green]Available prompts:[/bold green]")
        for prompt_name, prompt in self.prompts.items():
            self.console.print(f" - {prompt_name}: {prompt.description}")