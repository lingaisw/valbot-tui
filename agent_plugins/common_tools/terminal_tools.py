import os
import fnmatch
import subprocess
from typing import Generator, Optional, Union, Any
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from pydantic_ai import RunContext
import asyncio

async def list_files(
    ctx: RunContext,
    directory: str = '.',
    pattern: str = '*',
    recursive: bool = False,
) -> list[str]:
    """
    List files in the given directory matching a pattern.
    Optionally search recursively.
    Args:
        directory (str): Directory to search in.
        pattern (str): Pattern to match files, e.g. '*.py'.
        recursive (bool): If True, search subdirs as well.
    Returns:
        list[str]: List of matched file paths (relative to directory).
    """
    if not recursive:
        return fnmatch.filter(os.listdir(directory), pattern)
    matches = []
    for dirpath, _, filenames in os.walk(directory):
        for name in fnmatch.filter(filenames, pattern):
            matches.append(os.path.relpath(os.path.join(dirpath, name), directory))
    return matches

async def find_files(
    ctx: RunContext,
    root_dir: str,
    file_name: str
) -> list[str]:
    """
    Recursively finds files with a given name starting from root_dir.
    Args:
        root_dir (str): Root directory to search from.
        file_name (str): Filename to search for (exact match).
    Returns:
        list[str]: Paths of found files.
    """
    console = Console()
    console.print(Markdown(f"- Searching for files named `{file_name}` in `{root_dir}`"), markup=False)
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name == file_name:
                matches.append(os.path.join(dirpath, name))
    return matches

async def grep_in_files(
    ctx: RunContext,
    directory: str,
    search_text: str,
    file_pattern: str = '*',
    case_sensitive: bool = True
) -> list[tuple[str, int, str]]:
    """
    Search for text inside files under a directory (optionally filtered by pattern).
    Args:
        directory (str): Directory to search under.
        search_text (str): Text to search for.
        file_pattern (str): Only search files matching this pattern.
        case_sensitive (bool): Whether the search is case-sensitive.
    Returns:
        list of tuples: (file_path, line_number, line_content) for each match.
    """
    console = Console()
    console.print(Markdown(f"- Searching for `{search_text}` in files matching `{file_pattern}` under `{directory}`"), markup=False)
    results = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, file_pattern):
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for idx, line in enumerate(f, start=1):
                        hay = line if case_sensitive else line.lower()
                        needle = search_text if case_sensitive else search_text.lower()
                        if needle in hay:
                            results.append((file_path, idx, line.strip()))
            except Exception:
                pass
    return results

async def show_file_tree(
    ctx: RunContext,
    start_path: str = '.',
    max_depth: int = 3
) -> list[str]:
    """
    Returns a formatted tree of the directory structure (up to max_depth) as a list of strings.
    Args:
        start_path (str): Directory to start at.
        max_depth (int): Maximum depth to show.
    Returns:
        list[str]: Lines representing the directory tree
    """
    console = Console()
    console.print(Markdown(f"- Getting file tree for `{start_path}`"), markup=False)
    lines = []
    def walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        entries = list(path.iterdir())
        for idx, entry in enumerate(sorted(entries, key=lambda e: (not e.is_dir(), e.name))):
            is_last = idx == len(entries) - 1
            branch = '┗━ ' if is_last else '┣━ '
            lines.append(f"{prefix}{branch}{entry.name}")
            if entry.is_dir():
                walk(entry, prefix + ('   ' if is_last else '┃  '), depth + 1)
    walk(Path(start_path), '', 1)
    return lines


async def run_shell_command(
    ctx: RunContext,
    command: str,
    cwd: Optional[str] = None,
    capture_output: bool = True,
    timeout: Optional[int] = 60
) -> dict[str, Any]:
    """
    Run a shell command and return the result.
    Args:
        command (str): The shell command to execute.
        cwd (str, optional): Current working dir to run in.
        capture_output (bool): Whether to capture stdout/stderr.
        timeout (int, optional): Seconds to wait before killing proc.
    Returns:
        dict: {"stdout": ..., "stderr": ..., "returncode": ...}
    """
    try:
        console = Console()
        console.print(Markdown(f"- Running: `{command}`"), markup=False)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE if capture_output else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"stdout": "", "stderr": "Timeout expired", "returncode": -1}
        return {
            "stdout": (stdout.decode() if stdout else "") if capture_output else None,
            "stderr": (stderr.decode() if stderr else "") if capture_output else None,
            "returncode": proc.returncode,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}
