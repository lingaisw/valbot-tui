from agent_plugins.common_agent_imports import *
from agent_plugins.agent_plugin import AgentPlugin
import os
import platform
import subprocess
from dataclasses import dataclass
from agent_plugins.common_tools.file_tools import read_partial_file

@dataclass
class SpecDeps:
    spec_path: str
    user_request: str

@dataclass
class GrepResult:
    file_name: str
    matches: dict[int, str]  # line_number -> line_content

class SpecExpertAgentPlugin(AgentPlugin):
    # Define required arguments and their prompt methods
    REQUIRED_ARGS = {
        'spec_path': (AgentPlugin.path_prompt, 'Enter the specification path')
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.spec_expert_agent = self.create_spec_expert_agent(model)
        self.apply_tool_decorators()
        self.read_files = set()

    def create_spec_expert_agent(self, model):
        return Agent[SpecDeps](
            model=model,
            system_prompt=(
                'You are an agent responsible for providing expert advice based on design specifications.'
                'Your task is to do the following:'
                '1. Analyze the user\'s request and figure out what they are looking for.'
                '2. if you already know the answer based on files you\'ve already analyzed, provide it to the user.'
                '3. if you don\'t know the answer, you must come up with a searching strategy to find the answer in a directory of specification files.'
                '4. Come up with multiple grep commands to search for keywords to figure out the right files to read. Use the run_grep_command tool to run the command.'
                '5. Read the specified file or files.'
                '6. If needed, run the grep command again to find more specific information and load in those files if needed.'
                '7. Answer the user\'s request based on the content of the specification.'
                'IMPORTANT: Try to limit grep strings to single unique words, or patterns of unique words using wildcards, to find relevant information. And keep using the grep command until you find the right files.'
                'Strive to give succinct answers.'
                'Provide your final answer in an organized Markdown format'
            ),
            deps_type=SpecDeps,
            retries=4,
            instrument=True,
        )

    def apply_tool_decorators(self):
        self.spec_expert_agent.tool(self.run_grep_command)
        self.spec_expert_agent.tool(read_partial_file)

    async def run_grep_command(self, ctx: RunContext[SpecDeps], spec_path: str, string_to_grep: str) -> list[GrepResult]:
        """Runs a grep command for a given string to search for keywords in the specification files.
        Returns a list of GrepResult dataclasses with file names and line number->content mappings."""
        console = Console()
        if platform.system() == "Windows":
            grep_command = f'findstr /S /I /N /C:"{string_to_grep}" "{spec_path}\\*"'
        else:
            grep_command = f"grep -rin '{string_to_grep}' {spec_path}"
        #console.print(f"[bold green]Running grep command on {spec_path} for: {string_to_grep}[/bold green]")
        file_results = {}  # file_name -> {line_number -> line_content}
        try:
            result = subprocess.run(grep_command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_name = parts[0]
                            line_number = int(parts[1])
                            content = parts[2].strip()

                            if file_name not in file_results:
                                file_results[file_name] = {}
                            file_results[file_name][line_number] = content
        except Exception as e:
            return [GrepResult(file_name="ERROR", matches={0: f"Error running grep command: {grep_command}\n Error: {str(e)}"})]

        return [GrepResult(file_name=fname, matches=matches) for fname, matches in file_results.items()]

    def run_agent_flow(self, context, **kwargs):
        """Switch to agent mode for specification expert."""
        self.console.print("[bold green]Entering agent mode for specification expert.[/bold green]")

        user_request = Prompt.ask("[bold green]Enter the request[/bold green]").strip()

        deps = SpecDeps(spec_path=self.initializer_args['spec_path'], user_request=user_request)
        self.console.print("[bold green]Processing your request...[/bold green]")

        async def run_agent():
            nonlocal deps  # Declare deps as nonlocal to avoid scope issues
            message_history = []
            while True:
                result = await self.spec_expert_agent.run(
                    f"for this spec area: {deps.spec_path}, answer the user's question: {deps.user_request}",
                    deps=deps,
                    message_history=message_history
                )
                message_history.extend(result.new_messages())
                self.console.print(Markdown(result.data))

                # Ask the user for the next question (leave blank to exit)
                next_question = Prompt.ask("[bold green]Next question (leave blank to exit)[/bold green]").strip()
                if not next_question:
                    self.console.print("[bold green]Exiting spec expert agent.[/bold green]")
                    break
                # Update deps with the next question
                deps.user_request = next_question

        asyncio.run(run_agent())