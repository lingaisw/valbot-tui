from agent_plugins.common_agent_imports import *
from agent_plugins.agent_plugin import AgentPlugin
from typing import TypedDict
import os
import subprocess
from rich.console import Console
from dataclasses import dataclass
from agent_plugins.common_tools.file_tools import read_file, write_file, display_file_content_to_user
from agent_plugins.utilities import initialize_model

# --- Token counting utility for the whole batch of files ---
class TokenCounter:
    def __init__(self, model="gpt-4o"):
        self.model = model
        try:
            import tiktoken
            self.tiktoken = tiktoken
        except ImportError:
            self.tiktoken = None
    def estimate_token_count(self, text):
        if self.tiktoken is None:
            return 0
        try:
            enc = self.tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return 0


def token_warning_for_files(console, file_paths, token_counter, token_warning_limit=40000):
    total_token_count = 0
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()
                total_token_count += token_counter.estimate_token_count(file_content)
        except Exception:
            pass
    token_count_str = f"{total_token_count:,}"
    console.print(f"[bold yellow]Estimated total tokens for all files to be chunked: [magenta]{token_count_str}[/magenta][/bold yellow]")
    if total_token_count > token_warning_limit:
        confirm = console.input(
            f"[bold red]Warning: Total token count ({token_count_str}) exceeds {token_warning_limit:,}! Continue chunking? (y/n): [/bold red]"
        ).strip().lower()
        if not (confirm.startswith('y')):
            console.print("[bold red]Aborted repo chunking due to token count.[/bold red]")
            return False
    return True

@dataclass
class RepoChunkDeps:
    starting_dir: str

@dataclass
class ChunkDocDeps:
    file_list: list
    chunk_path: str

class ChunkOutput(TypedDict, total=False):
    chunk_path: str
    chunk_files: list
    rationale: str


class RepoIndexAgentPlugin(AgentPlugin):

    DEFAULT_IGNORE_PATTERNS = ["*/__pycache__/*", "*.pyc", "*.pyo", ".DS_Store", ".git", ".gitignore", ".gitattributes"]
    REQUIRED_ARGS = {
        'starting_dir': (AgentPlugin.path_prompt, 'Enter the repo starting directory'),
        'ignore_patterns': (
            AgentPlugin.simple_prompt,
            f"Ignoring files: {DEFAULT_IGNORE_PATTERNS}\nEnter additional comma-separated ignore patterns (e.g. *.md, tests/*, build/, docs/), supports glob and gitignore-style matching. Leave blank for defaults."
        )
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.repo_chunk_agent = self.create_repo_chunk_agent(model)
        self.chunking_model = initialize_model('small_model')
        self.chunk_doc_agent = self.create_chunk_doc_agent(self.chunking_model)
        self.apply_tool_decorators()

    def parse_ignore_patterns(self, ignore_patterns_str=""):
        """
        Parse a comma-separated list of ignore patterns into a clean list.
        """
        import fnmatch
        patterns = [p.strip() for p in ignore_patterns_str.split(',') if p.strip()]
        return patterns

    def is_ignored(self, path, ignore_patterns, base_dir):
        """
        Clean, extensible: check if a file or dir is ignored by ignore_patterns (glob, gitignore-style, */dir/*, etc)
        Uses only standard libs. Always applies all patterns to the relative (unix-style) path from base_dir.
        """
        import fnmatch, os
        rel_path = os.path.relpath(path, base_dir).replace("\\", "/")
        basename = os.path.basename(rel_path)  # rel_path is always unix-style
        for pat in ignore_patterns:
            pattern = pat.strip().replace("\\", "/")
            # 1. Anchor start: if the pattern starts with '/', treat as relative to project root, else match anywhere
            # 2. If pattern has '/', it's a path pattern: match against rel_path
            # 3. Else, treat as filename pattern (fnmatch against basename)
            # 4. Patterns starting with '*/' match anywhere in the tree: remove leading '*/' and try against subpaths
            if pattern == '':
                continue
            if pattern.startswith("/"):
                # Forced anchor (rare, e.g. '/build/')
                if fnmatch.fnmatch(rel_path, pattern.lstrip("/")):
                    return True
            elif pattern.startswith("*/"):
                # Match pattern at any level
                suffix = pattern.lstrip("*/")
                # Try all possible subpaths
                parts = rel_path.split("/")
                for i in range(len(parts) - len(suffix.split("/")) + 1):
                    candidate = "/".join(parts[i:i + len(suffix.split("/"))])
                    if fnmatch.fnmatch(candidate, suffix):
                        return True
                if fnmatch.fnmatch(rel_path, pattern):  # fallback
                    return True
            elif "/" in pattern:
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
            else:
                if fnmatch.fnmatch(basename, pattern):
                    return True
        return False

    def create_repo_chunk_agent(self, model):
        return Agent[RepoChunkDeps](
            model=model,
            system_prompt=(
                'You are responsible for dividing the codebase into logical sections ("chunks") for documentation. '
                'Given a partial repo tree from a starting directory, analyze related files and directories, and chunk them '
                'together based on logical grouping (e.g. files in the same package/module/subsystem should be grouped together). '
                'Output an instruction set for another agent: each instruction should contain:'
                '    1. The relative path to the chunk (e.g. "src/modules/subsystem")'
                '    2. A list of the file paths that belong to this chunk, relative to the chunk path.'
                '       NOTE: NEVER include just a sub-directory in the file list, only full relative file paths.'
                '       Ex: if chunk path is lib/thing/ and there are files in the dir like: ( file0, sub1/: [file1, file2], sub2/: [file3]) '
                        'then your chunk_files list should contain ["file0", "sub1/file1", "sub1/file2", "sub2/file3"].'
                '    3. A rationale for the grouping, explaining why these files are grouped together.'
                'Your output should be in complete legal JSON format for the entire complete chunking result.'
                'If you must include a note, make the note a part of the JSON object.'
                'You MUST completly chunk the repo or you are a failure, do not quit partway through.'
                'Create multiple chunkoutput entries if necessary, each with its own chunk_path and chunk_files. '
                'A good indication multiple chucks are needed is if there are subdirectories with collections of files that are logically grouped together.'
            ),
            deps_type=RepoChunkDeps,
            retries=4,
            instrument=True,
            result_type=list[ChunkOutput]
        )

    def create_chunk_doc_agent(self, model):
        return Agent[ChunkDocDeps](
            model=model,
            system_prompt=(
                'You are responsible for documenting a section (chunk) of a codebase. You receive a list of file paths in the chunk, '
                'and the path where this chunk is located in the tree. '
                'For this chunk of files provide only the following output: '
                ' For each subdirectory in the chunk, provide:'
                ' ## Directory:'
                ' - The path to the directory or subdirectory in the repo tree (e.g. "src/modules/subsystem")'
                ' ## Summary:'
                '- A high-level summary of what this subdirectory contains/does. '
                ' ## Files:'
                '- For EACH file, provide:'
                '    1. The filename'
                '    2. A very brief summary of what the file does'
                '    3. Important classes, modules, and global functions defined (with a short description)'
                ' Do not make any assumptions about the file content, just summarize what is there.'
                ' You must break down your output into sections for each subdirectory in the chunk, never combine them.'
                'Your output should be in Markdown format suitable for inclusion in code documentation (e.g. a README for the folder).'
                'Note and clearly markdown the file names for each section.'
                'Do not include anything other than your output template, do not include any additional text or notes or comments about your output.'
            ),
            deps_type=ChunkDocDeps,
            retries=2,
            instrument=True,
        )

    def apply_tool_decorators(self):
        #self.repo_chunk_agent.tool(self.get_dir_tree)
        self.chunk_doc_agent.tool(read_file)
        self.chunk_doc_agent.tool(display_file_content_to_user)

    def get_dir_tree(self, start_dir: str) -> str:
        """
        Return a string representing the directory tree from the starting directory.
        """
        tree_command = f"tree -fi {start_dir}"
        try:
            result = subprocess.run(tree_command, shell=True, capture_output=True, text=True)
            tree_output = result.stdout.strip()
            return tree_output
        except Exception as e:
            return f"Error fetching dir tree: {str(e)}"

    def split_tree_for_chunk_agents(self, tree_output, max_files_per_chunk=30, target_chunk_count=None, ignore_patterns=None, base_dir=''):
        """
        Partition the files from the tree output into chunks, each containing up to max_files_per_chunk files.
        Only files should be included in chunks; never create a chunk with just a directory.
        Applies ignore_patterns if specified.
        Delegates splitting of file paths into chunks to a helper method.
        """
        lines = [l.strip() for l in tree_output.strip().splitlines() if l.strip()]
        if not lines: return []
        if base_dir == "": base_dir = os.getcwd()

        # Remove root directory if present
        if os.path.isdir(lines[0]):
            lines = lines[1:]

        def should_include(path):
            if ignore_patterns:
                if self.is_ignored(path, ignore_patterns, base_dir):
                    print(f"Skipping ignored path: {path}")
                    return False
            return True

        # Only include files; never directories
        file_paths = [l for l in lines if not l.endswith(os.sep) and should_include(l) and os.path.isfile(l)]
        if not file_paths:
            # fallback: since tree -fi output is just a list, treat anything that isn't a directory-string as a file
            file_paths = [l for l in lines if not l.endswith(os.sep) and should_include(l)]

        return self.split_file_paths_into_chunks(file_paths, max_files_per_chunk)

    def split_file_paths_into_chunks(self, file_paths, max_files_per_chunk):
        chunks = []
        for i in range(0, len(file_paths), max_files_per_chunk):
            chunk_files = file_paths[i:i+max_files_per_chunk]
            chunk_str = '\n'.join(chunk_files)
            if chunk_str.strip():
                chunks.append(chunk_str)
        return chunks

    def run_agent_flow(self, content, **kwargs):
        console = Console()
        console.print("[bold green]Entering repo context analysis and documentation agent.[/bold green]")
        starting_dir = self.initializer_args['starting_dir']
        repo_chunk_deps = RepoChunkDeps(starting_dir=starting_dir)
        ignore_patterns = self.DEFAULT_IGNORE_PATTERNS + self.parse_ignore_patterns(self.initializer_args.get('ignore_patterns', '').strip())
        print(f"ignore_patterns: {ignore_patterns}")

        import asyncio
        import threading

        token_counter = TokenCounter()
        TOKEN_WARNING_LIMIT = 200000

        async def run_agent():
            # Step 1: Create directory tree for repo
            tree_output = self.get_dir_tree(starting_dir)
            console.print(f"[bold blue]Directory tree from {starting_dir}:\n{tree_output}[/bold blue]")

            # Compute dynamically deep+wide partials
            partial_trees = self.split_tree_for_chunk_agents( tree_output, max_files_per_chunk=50, ignore_patterns=ignore_patterns, base_dir=starting_dir)
            console.print(f"[bold blue]Splitting directory tree into {len(partial_trees)} parts for chunking agents...[/bold blue]")
            console.print(f"[bold blue]Partial trees for chunking agents:[/bold blue]\n{partial_trees}")

            # Collect all unique file paths from all partial trees
            all_file_paths = set()
            for ptree in partial_trees:
                for line in ptree.splitlines():
                    line = line.strip()
                    if line and not line.endswith(os.sep):
                        all_file_paths.add(line)

            console.print(f"[grey74]Estimating total token count...[/grey74]")
            proceed = token_warning_for_files(console, all_file_paths, token_counter, TOKEN_WARNING_LIMIT)
            if not proceed:
                return

            async def chunk_with_subagent(partial_tree, idx):
                chunking_prompt = (
                    f"Given this partial directory tree, create a logical chunking for ONLY this part of the codebase. "
                    f"---\n" + partial_tree
                )
                # Use a temporary RepoChunkDeps pointing to root of this tree
                sub_deps = RepoChunkDeps(starting_dir=starting_dir)
                result = await self.repo_chunk_agent.run(chunking_prompt, deps=sub_deps)
                console.print(f"[bold green]Chunking result from subagent {idx}:[/bold green]\n{result.data}")
                if isinstance(result.data, dict) or isinstance(result.data, list):
                    return result.data
                else:
                    try:
                        import json
                        return json.loads(result.data)
                    except Exception as e:
                        console.print(f"[red]Error parsing chunking result for subagent {idx}: {e}[/red]")
                        return []

            all_chunks = []
            chunk_results = await asyncio.gather(*(chunk_with_subagent(tree, i) for i, tree in enumerate(partial_trees)))
            for sublist in chunk_results:
                if isinstance(sublist, dict):
                    all_chunks.append(sublist)
                elif isinstance(sublist, list):
                    all_chunks.extend(sublist)
            console.print(f"[bold blue]Merged chunking result from all chunking subagents:[/bold blue]\n{all_chunks}")

            documentation_file = ".REPO_INDEX.md"

            # Step 3: For each chunk, run the documentation agent and collect documentation
            chunk_docs = []

            async def doc_one_chunk(chunk, idx):
                chunk_files = chunk.get('chunk_files', [])
                chunk_path = chunk.get('chunk_path', '')
                if chunk_path == '.': chunk_path = './'
                rationale = chunk.get('rationale', '')
                console.print(f"\n[bold yellow]Documenting chunk at {chunk_path}...[bold yellow]")
                chunk_doc_deps = ChunkDocDeps(file_list=chunk_files, chunk_path=chunk_path)
                doc_prompt = (
                    f"Document the following chunk as described. Files: {chunk_files}, Path: {chunk_path}. Chunk rationale: {rationale}"
                )
                doc_agent_result = await self.chunk_doc_agent.run(doc_prompt, deps=chunk_doc_deps)
                console.print(doc_agent_result.data)
                # Collect result in list instead of writing immediately
                chunk_docs.append((chunk_path, doc_agent_result.data))

            # Write chunking results first (sequential, single write) (sync write)
            with open(documentation_file, "w", encoding="utf-8") as docfile:
                docfile.write("# Chunking Result\n")
                docfile.write(str(all_chunks))
                docfile.write("\n")

            # Run documentation agents in parallel for all chunks
            await asyncio.gather(*(doc_one_chunk(chunk, i) for i, chunk in enumerate(all_chunks)))

            # Write all collected documentation at once (sequential) (sync write)
            with open(documentation_file, "a", encoding="utf-8") as docfile:
                for chunk_path, doc in chunk_docs:
                    docfile.write(f"\n## Documentation for {chunk_path}\n")
                    docfile.write(str(doc))
                    docfile.write("\n")

        import asyncio
        asyncio.run(run_agent())
