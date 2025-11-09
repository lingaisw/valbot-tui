from pydantic_ai import RunContext
from pydantic import BaseModel, Field
from typing import Optional, Protocol
import requests
from .github_auth import TokenProvider

####################################### GITHUB API TOOLS ################################################

# ------- Status Printer Protocol -------

class StatusPrinter(Protocol):
    """Protocol for objects that can print status messages to the user."""
    def print_status(self, message: str) -> None:
        """Prints a status message to the user."""
        ...

class GitHubToolsDeps(TokenProvider, StatusPrinter, Protocol):
    """Combined protocol for dependencies needed by GitHub tools."""
    pass

# ------- Output Models -------

class GitHubFileContentResult(BaseModel):
    """Contains the full content of a file retrieved from GitHub."""
    repository: str = Field(description="Repository in format 'owner/repo'")
    file_path: str = Field(description="Path to the file within the repository")
    content: str = Field(description="Full content of the file")
    error: Optional[str] = Field(default=None, description="Error message if request failed")

# ------- GitHub API Tools -------

async def get_github_file_content(ctx: RunContext[GitHubToolsDeps], repository: str, file_path: str) -> GitHubFileContentResult:
    """
    Retrieves the full content of any file from a GitHub repository using the GitHub API.
    Works with any file type (code, markdown, configuration files, etc.) from the default branch.

    Args:
        repository: Repository in format 'owner/repo' (e.g., 'intel-innersource/applications.ai.valbot-cli')
        file_path: Path to the file within the repository (e.g., 'src/main.py' or 'docs/README.md')

    Returns:
        GitHubFileContentResult with the full file content or error message
    """
    if hasattr(ctx.deps, 'print_status'):
        ctx.deps.print_status(f"Fetching file: {file_path}")
    token = ctx.deps.get_github_token()
    # Validate repository format
    if '/' not in repository:
        return GitHubFileContentResult(repository=repository, file_path=file_path, content="",
                                     error=f"Invalid repository format. Expected 'owner/repo', got: {repository}")
    # Build GitHub API URL to fetch file content from default branch
    api_url = f"https://api.github.com/repos/{repository}/contents/{file_path}"
    headers = {"Accept": "application/vnd.github.v3.raw", "Authorization": f"Bearer {token}"}
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        return GitHubFileContentResult(repository=repository, file_path=file_path, content=response.text, error=None)
    except requests.exceptions.RequestException as e:
        return GitHubFileContentResult(repository=repository, file_path=file_path, content="",
                                     error=f"Failed to fetch file: {str(e)}")
