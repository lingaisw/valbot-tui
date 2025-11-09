from agent_plugins.common_agent_imports import *
from agent_plugins.common_tools.file_tools import *
from agent_plugins.agent_plugin import AgentPlugin

@dataclass
class NewFileDeps:
    file_path: str
    edit_request: str

class NewFileEditPlugin(AgentPlugin):
    REQUIRED_ARGS = {
        'file_path': (AgentPlugin.path_prompt, 'Enter the file you want to create'),
        'edit_request': (AgentPlugin.simple_prompt, 'Enter your request')
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.console = Console()
        self.file_edit_agent = Agent[NewFileDeps](
            model=model,
            system_prompt=(
                'You are an agent responsible for creating and editing files based on user requests. '
                'Your task is to do the following using your tools:'
                '1. Create a new file with the specified path and initial content.'
                '2. if needed, read the file, and continue to write the file to achieve the desired file content.'
            ),
            deps_type=NewFileDeps,
            retries=4,
            instrument=True,
        )
        self.file_edit_agent.tool(create_file)
        self.file_edit_agent.tool(read_file)
        self.file_edit_agent.tool(write_file)

    def run_agent_flow(self, context):
        """Run the file edit agent."""
        file_path = self.initializer_args.get('file_path')
        edit_request = self.initializer_args.get('edit_request')
        context = f"\nGiven the following context of the previous converstaion: {context}" if context else ""
        deps = NewFileDeps(file_path=file_path, edit_request=edit_request)
        asyncio.run(self.file_edit_agent.run(f"for this file: {file_path}, do this as requested by the user: {edit_request}. {context}", deps=deps))
        try:
            with open(file_path, 'r') as f:
                final_content = f.read()
                self.console.print(f"[bold green]Content of #{file_path}[/bold green]:\n{final_content}")
        except Exception as e:
            self.console.print(f"[bold red]Error reading final content of {file_path}: {e}[/bold red]")
