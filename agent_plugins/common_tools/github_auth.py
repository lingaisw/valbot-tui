"""
Shared GitHub authentication utilities for accessing GitHub APIs.
Provides token retrieval and common protocol definitions.
"""
from typing import Protocol
import subprocess

####################################### GITHUB AUTHENTICATION ################################################

# ------- Token Provider Protocol -------

class TokenProvider(Protocol):
    """Protocol for objects that can provide a GitHub token."""
    def get_github_token(self) -> str:
        """Returns the GitHub personal access token."""
        ...

# ------- Token Retrieval -------

def get_github_token_from_shell(command: str = "dt github print-token") -> str:
    """
    Retrieves GitHub token by executing a shell command.
    Default uses 'dt github print-token' but can be customized.

    Args:
        command: Shell command to execute to retrieve the token

    Returns:
        The GitHub token as a string

    Raises:
        RuntimeError: If token retrieval fails
    """
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Failed to get token: {result.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error retrieving GitHub token: {e}")
