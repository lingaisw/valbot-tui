from agent_plugins.common_agent_imports import *
from agent_plugins.common_tools.ask_human import ask_human, ask_human_for_more_instructions
from agent_plugins.common_tools.file_tools import *
from agent_plugins.agent_plugin import AgentPlugin

@dataclass
class Deps:
    file_path: str
    edit_request: str
    current_file_content: Optional[str] = None

class FileEditPlugin(AgentPlugin):
    REQUIRED_ARGS = {
        'file_path': (AgentPlugin.path_prompt, 'Enter the file you want to modify'),
        'edit_request': (AgentPlugin.simple_prompt, 'Enter your request')
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.console = Console()
        system_prompt = (
            'You are an agent responsible for editing files based on user requests. '
            'Your workflow is as follows:\n'
            '1. Parse and understand the user\'s edit request.\n'
            '2. Plan the changes by analyzing both the intent and the source code.\n'
            '3. Generate a unified diff showing the required modifications. Display this diff to the user.\n'
            '4. Prompt the user to confirm these changes with `ask_human`.\n'
            '   a. If confirmed, use either the write_file or edit_string_in_file tools to apply the required changes and save the file.\n'
            '   b. If not confirmed, ask for further instructions or new edits from the user.\n'
            '5. Repeat as needed until edits are confirmed or the session is ended.\n'
            'Important:\n'
            '- Do not apply file changes before user confirmation.\n'
            '- Propose a meaningful, minimal diff. If no changes are needed, inform the user and ask for further clarification.\n'
            '- All interactions with the user MUST use the `ask_human` or `ask_human_for_more_instructions` tools.\n'
            '- Use each tool in a safe and proper sequence. DO NOT skip steps or tools.\n'
        )
        self.file_edit_agent = Agent[Deps](
            model=model,
            system_prompt=system_prompt,
            deps_type=Deps,
            retries=5,
            instrument=True,
        )
        self.apply_tool_decorators()

    def apply_tool_decorators(self):
        self.file_edit_agent.tool(self.display_diff_of_changes_to_the_user)
        self.file_edit_agent.tool(ask_human)
        self.file_edit_agent.tool(ask_human_for_more_instructions)
        self.file_edit_agent.tool(write_file)
        self.file_edit_agent.tool(edit_string_in_file)

    async def display_diff_of_changes_to_the_user(self, ctx: RunContext[Deps], content: str) -> None:
        """
        Display the changes made to the file in a diff format to the user using code blocks.
        Shows only the actual diff, highlights additions and deletions for clarity.
        """
        if not content.strip():
            return "No changes to display to the user"
        for line in content.splitlines():
            if line.startswith('@@'):
                self.console.print(line, style="bold magenta")
            elif line.startswith('+') and not line.startswith('+++'):
                self.console.print(line, style="green")
            elif line.startswith('-') and not line.startswith('---'):
                self.console.print(line, style="red")

    def _read_file_content(self, file_path: str) -> str:
        while True:
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                self.console.print(f"[bold red]File not found: {file_path}[/bold red]")
                file_path = self.path_prompt('Enter the file you want to modify')

    def run_agent_flow(self, context, **kwargs):
        file_path = self.initializer_args.get('file_path')
        edit_request = self.initializer_args.get('edit_request')
        file_content = self._read_file_content(file_path)
        deps = Deps(file_path=file_path, edit_request=edit_request, current_file_content=file_content)

        def make_agent_input(context, dep):
            prior_context = f"\nGiven the following context of the prior conversation:\n{context}" if context else ""
            return (
                f"{prior_context}\n\nNow for this file: {dep.file_path}\n"
                f"Apply these edits as requested by the user: {dep.edit_request}.\n"
                f"Current content of file:\n{dep.current_file_content}"
            )

        agent_input = make_agent_input(context, deps)
        message_history = []

        self.agent_edit_loop(agent_input, message_history, deps, make_agent_input)

    def agent_edit_loop(self, agent_input, message_history, deps, make_agent_input):
        while True:
            result = asyncio.run(self.file_edit_agent.run(agent_input, deps=deps, message_history=message_history))
            self.console.print(Markdown(result.data))
            new_edit_request = Prompt.ask("Edit request (leave blank to exit)")
            if not new_edit_request.strip():
                self.console.print("[bold green]Exiting file edit agent.[/bold green]")
                break
            deps.edit_request = new_edit_request
            deps.current_file_content = self._read_file_content(deps.file_path)
            message_history.extend(result.new_messages())
            agent_input = make_agent_input(None, deps)
