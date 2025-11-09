import os
from agent_plugins.common_agent_imports import *
from agent_plugins.agent_plugin import AgentPlugin
from typing import List, Dict, Any, TypedDict
from rich.prompt import Prompt
from agent_plugins.common_tools.ask_human import ask_human
from agent_plugins.common_tools.file_tools import read_file, list_files_in_directory, write_file, create_file

@dataclass
class ProjectIdentifyDeps:
    repo_index_path: str
    user_request: str

@dataclass
class ProjectImplementDeps:
    files_to_open: List[str]
    rationale: str
    user_request: str

class IdentifierOutput(TypedDict, total=False):
    rationale: str
    files_to_open: List[str]


class ProjectAgentPlugin(AgentPlugin):
    REQUIRED_ARGS = {
        'repo_index': (AgentPlugin.path_prompt, 'Enter the path to .REPO_INDEX.md file'),
        'user_request': (AgentPlugin.simple_prompt, 'Enter the user request for the project')
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.console = Console()
        self.identifier_agent = self.create_identifier_agent(model)
        self.implementor_agent = self.create_implementor_agent(model)
        self.apply_tool_decorators()
        if os.path.exists('.REPO_INDEX.md') and 'repo_index' not in kwargs:
            use_found_repo_index = Prompt.ask("[bold yellow]Found existing .REPO_INDEX.md file. Do you want to use it? (yes/no)[/bold yellow]", choices=["yes", "no"], default="yes")
            if use_found_repo_index.lower() == "yes":
                self.initializer_args['repo_index'] = '.REPO_INDEX.md'

    def create_identifier_agent(self, model):
        return Agent[ProjectIdentifyDeps](
            model=model,
            system_prompt=(
                'You are a project analysis agent. You are given a REPO_INDEX file (in markdown or structured form) that summarizes the files and structure of a repository, '
                'and a user request to perform a change/addition/fix/feature etc. in the project.\n'
                'Your ONLY job is to:\n'
                '- Carefully read and interpret the user request and the repo index summary.\n'
                '- Identify exactly which files, modules, or sections of the codebase (including suggesting related examples that could be referenced) should be reviewed or opened in order to proceed with the request.\n'
                '- Do NOT solve the user request, just return:\n'
                '    1. A concise rationale for which files must be read and why,\n'
                '    2. A list of all file paths (relative to repo root) that should be opened/read for implementation/planning, including anything that might be relevant (examples, utilities, configs).\n'
                'DO NOT attempt to implement or propose code changes; only provide the required file list and rationale.'
                'You should always provide the full file paths when listing out the files to open so make sure you include the prepend the Directory of the files in your list'
            ),
            deps_type=ProjectIdentifyDeps,
            retries=3,
            instrument=True,
            result_type=IdentifierOutput,
        )

    def create_implementor_agent(self, model):
        return Agent[ProjectImplementDeps](
            model=model,
            system_prompt=(
                'You are an implementation planning agent. Given:\n'
                '- The user request describing the intended change/addition/etc.,\n'
                '- The rationale for why the files are chosen,\n'
                '- A list of file paths provided,\n'
                'You must OPEN/READ these files YOURSELF to get their content for planning. Use the read_file tool as needed.\n'
                'As you are reading the files, you may decide you need more information or context (i.e more examples, utilities, libraries, etc.) to implement the user request.'
                'In this case you should use the obtain_more_project_info tool to request additional information regarding certain libraries, modules, other clases, examples, etc.'
                'Your task is to:\n'
                '1. Read and understand all provided files and the request.\n'
                '2. Propose a plan outlining how you would implement the user request.\n'
                '    - The plan must list: steps, impacted files, what changes/additions are required, and potential issues/decision points.\n'
                '    - When applicable, show code snippets or examples to illustrate your plan.\n'
                '3. Only produce the implementation plan, do not implement yet.'
            ),
            deps_type=ProjectImplementDeps,
            retries=3,
            instrument=True,
        )

    def apply_tool_decorators(self):
        self.implementor_agent.tool(read_file)
        self.implementor_agent.tool(list_files_in_directory)
        self.implementor_agent.tool(self.obtain_more_project_info)
        self.implementor_agent.tool(write_file)
        self.implementor_agent.tool(create_file)


    async def obtain_more_project_info(self, ctx: RunContext, more_info_request: str) -> IdentifierOutput:
        """
        This tool allows the implementor agent to request more information about the project.
        Requests should be specific and actionable, and directly related to understanding functionality of the project/repo.
        Requests should not be related to getting more information about the user request or additional requirements.
        For example, "Show me the CheckUtils Module" or "What are the examples of how to use the DataProcessor class?"
        """
        self.console.print(f"[bold cyan]Gathering more information: {more_info_request}[/bold cyan]")
        identify_deps = ProjectIdentifyDeps(repo_index_path=self.repo_index_path, user_request=more_info_request)
        response = await self.identifier_agent.run(
            f"You have an additional request for more information regarding:\n{more_info_request}",
            deps=identify_deps,
            model_settings={'max_completion_tokens': 10000},
            message_history=self.identifier_agent_history
        )
        self.console.print(f"[bold green]Received additional project information:[/bold green]\n{response.data}")
        return response.data


    def run_agent_flow(self, context, **kwargs):
        """
        Main flow for using project agent: (1) Identify relevant files (2) Provide the files and request to implementation planning.
        """
        self.repo_index_path = self.initializer_args['repo_index']
        user_request = self.initializer_args['user_request']

        # Change CWD to the repo_index_path directory so all file operations are relative to the index file
        abs_repo_index = os.path.abspath(self.repo_index_path)
        os.chdir(os.path.dirname(abs_repo_index))

        # Ask if the user wants to provide a specification file
        provide_spec = Prompt.ask("[bold green]Do you want to provide a specification path? (yes/no)[/bold green]", choices=["yes", "no"], default="no")
        spec_content = None
        spec_path = None
        additional_spec_info = None
        if provide_spec.lower() == "yes":
            spec_path = self.path_prompt('Enter the path to the specification file')
            try:
                with open(spec_path, 'r') as sf:
                    spec_content = sf.read()
                self.console.print(f"[bold green]Read specification file: {spec_path}[/bold green]")
            except Exception as e:
                self.console.print(f"[bold red]Failed to read specification file: {e}[/bold red]")
            # Ask user if there's any additional info to relay with the spec
            wants_to_add_info = Prompt.ask("[bold cyan]Do you want to provide any additional information to relay to the implementor agent along with the spec? (yes/no)[/bold cyan]", choices=["yes", "no"], default="no")
            if wants_to_add_info.lower() == "yes":
                additional_spec_info = Prompt.ask("[bold cyan]Please enter the additional information or directions for how to consume the spec.[/bold cyan]")

        # Read the REPO_INDEX file content up front, pass it to identifier agent.
        with open(os.path.basename(self.repo_index_path), 'r') as f:
            repo_index_content = f.read()

        identify_deps = ProjectIdentifyDeps(repo_index_path=self.repo_index_path, user_request=user_request)

        async def main_flow():
            # Step 1: Identifier agent
            self.console.print("[bold green]Analyzing repo index and user request to identify required files...[bold green]")
            identifier_result = await self.identifier_agent.run(
                f"Request: {user_request}\nREPO_INDEX file content follows.\n{repo_index_content}\n",
                deps=identify_deps,
                model_settings={'max_completion_tokens': 10000}
            )
            self.console.print("\n[bold yellow]Identifier agent output:[/bold yellow]")
            self.console.print(identifier_result.data)
            if not identifier_result.data.get('files_to_open'):
                self.console.print("[bold red]No files identified for implementation. Exiting.[/bold red]")
                return
            file_paths = identifier_result.data['files_to_open']
            rationale = identifier_result.data.get('rationale', 'No rationale provided.')
            self.identifier_agent_history = identifier_result.all_messages()

            # Step 2: Implementation planning agent
            implement_deps = ProjectImplementDeps(files_to_open=file_paths, rationale=rationale, user_request=user_request)
            self.console.print("[bold green]\nProviding relevant file paths and rationale to the implementor agent for planning...[bold green]")

            implementor_prompt = (
                f"User request: {user_request}\nRelevant file paths and rationale received from another agent: \n{identifier_result.data}.\n"
            )
            if spec_content is not None:
                implementor_prompt += f"\nSPECIFICATION:\n{spec_content}\n"
                if additional_spec_info:
                    implementor_prompt += f"\nADDITIONAL INFORMATION for consuming the specification:\n{additional_spec_info}\n"

            # call the implementor loop
            await self.implementor_loop(implementor_prompt, implement_deps)

        asyncio.run(main_flow())

    async def implementor_loop(self, initial_implementor_prompt, implement_deps):
        implementor_prompt = initial_implementor_prompt
        implementor_agent_history = []

        async def show_plan_and_get_decision(prompt, history):
            implementor_result = await self.implementor_agent.run(
                prompt,
                deps=implement_deps,
                message_history=history
            )
            self.console.print("\n[bold yellow]IMPLEMENTATION PLAN:[/bold yellow]")
            self.console.print(Markdown(implementor_result.data))
            decision = Prompt.ask(
                "[bold green]Do you want to proceed with the implementation based on this plan? (yes/no/change)[/bold green]",
                choices=["yes", "no", "change"],
                default="no"
            ).lower()
            return implementor_result, decision

        while True:
            implementor_result, decision = await show_plan_and_get_decision(implementor_prompt, implementor_agent_history)
            if decision == "yes":
                self.console.print("[bold green]Proceeding with implementation...[bold green]")
                implementor_result = await self.implementor_agent.run(
                    "The user has confirmed the implementation plan. Now proceed with implementing the changes based on the plan provided.\n",
                    message_history=implementor_result.all_messages(),
                    deps=implement_deps,
                    model_settings={'max_completion_tokens': 16000}
                )
                self.console.print("[bold green]Implementation completed successfully![bold green]")
                self.console.print(Markdown(implementor_result.data))
                break
            elif decision == "no":
                self.console.print("[bold red]Implementation cancelled by user.[/bold red]")
                break
            elif decision == "change":
                new_request = self.simple_prompt("Enter your new request or changes desired:")
                implementor_prompt = f"The user requested changes to the plan, this is what they said:\n{new_request}\n"
                implementor_agent_history = implementor_result.all_messages()
