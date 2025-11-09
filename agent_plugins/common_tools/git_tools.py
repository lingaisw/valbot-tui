from pydantic_ai import RunContext
from pydantic import BaseModel
from rich.console import Console
from typing import Optional
from .terminal_tools import run_shell_command

# ------- Output Models -------

class GitCommandOutput(BaseModel):
    stdout: str
    stderr: str
    returncode: int

# ------- Git Tools -------

async def git_status(ctx: RunContext, repo_path: str = ".") -> GitCommandOutput:
    """
    Get the status of the current git repository.
    Args:
        repo_path (str): Path to the git repository.
    Returns:
        GitCommandOutput: Output of the git status command.
    """
    console = Console()
    result = await run_shell_command(ctx, "git status", cwd=repo_path)
    console.print(result["stdout"])
    return GitCommandOutput(**result)

async def git_add(ctx: RunContext, pathspec: str = ".", repo_path: str = ".") -> GitCommandOutput:
    """
    Stage changes for commit in the git repository.
    Args:
        pathspec (str): File(s) or pattern to add. Default is all.
        repo_path (str): Path to the git repository.
    Returns:
        GitCommandOutput: Output of the git add command.
    """
    console = Console()
    result = await run_shell_command(ctx, f"git add {pathspec}", cwd=repo_path)
    msg = result["stdout"] or result["stderr"]
    console.print(msg)
    return GitCommandOutput(**result)

class GitCommitInput(BaseModel):
    message: str
    repo_path: str = "."

async def git_commit(ctx: RunContext, input: GitCommitInput) -> GitCommandOutput:
    """
    Commit staged changes in the git repository.
    Args:
        input (GitCommitInput): Contains commit message and repo path.
    Returns:
        GitCommandOutput: Output of the git commit command.
    """
    console = Console()
    command = f"git commit -m {input.message!r}"
    result = await run_shell_command(ctx, command, cwd=input.repo_path)
    msg = result["stdout"] or result["stderr"]
    console.print(msg)
    return GitCommandOutput(**result)

async def git_diff(ctx: RunContext, options: Optional[str] = None, repo_path: str = ".") -> GitCommandOutput:
    """
    Show changes in the current git repository.
    Args:
        options (str): Extra options for git diff (optional).
        repo_path (str): Path to the git repository.
    Returns:
        GitCommandOutput: Output of the git diff command.
    """
    console = Console()
    cmd = "git diff"
    if options:
        cmd += f" {options}"
    result = await run_shell_command(ctx, cmd, cwd=repo_path)
    msg = result["stdout"] or result["stderr"]
    console.print(msg)
    return GitCommandOutput(**result)

async def git_log(ctx: RunContext, options: Optional[str] = None, repo_path: str = ".") -> GitCommandOutput:
    """
    Show commit logs of the git repository.
    Args:
        options (str): Extra options for git log (optional).
        repo_path (str): Path to the git repository.
    Returns:
        GitCommandOutput: Output of the git log command.
    """
    console = Console()
    cmd = "git log"
    if options:
        cmd += f" {options}"
    result = await run_shell_command(ctx, cmd, cwd=repo_path)
    msg = result["stdout"] or result["stderr"]
    console.print(msg)
    return GitCommandOutput(**result)

async def git_branch(ctx: RunContext, options: Optional[str] = None, repo_path: str = ".") -> GitCommandOutput:
    """
    Show local branches of the git repository.
    Args:
        options (str): Extra options for git branch (optional).
        repo_path (str): Path to the git repository.
    Returns:
        GitCommandOutput: Output of the git branch command.
    """
    console = Console()
    cmd = "git branch"
    if options:
        cmd += f" {options}"
    result = await run_shell_command(ctx, cmd, cwd=repo_path)
    msg = result["stdout"] or result["stderr"]
    console.print(msg)
    return GitCommandOutput(**result)

class GitCheckoutInput(BaseModel):
    branch: str
    repo_path: str = "."

async def git_checkout(ctx: RunContext, input: GitCheckoutInput) -> GitCommandOutput:
    """
    Checkout a branch in the git repository.
    Args:
        input (GitCheckoutInput): branch name and repo path.
    Returns:
        GitCommandOutput: Output of the git checkout command.
    """
    console = Console()
    command = f"git checkout {input.branch}"
    result = await run_shell_command(ctx, command, cwd=input.repo_path)
    msg = result["stdout"] or result["stderr"]
    console.print(msg)
    return GitCommandOutput(**result)

# Optionally, add more commands such as pull, push, remote, etc.
