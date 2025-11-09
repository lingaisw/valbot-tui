"""
Terminal Manager for executing shell commands and streaming output
"""

import asyncio
import os
import sys
import subprocess
from typing import AsyncIterator, Optional, Tuple
from pathlib import Path


class TerminalManager:
    """Manages terminal command execution with real-time output streaming."""
    
    def __init__(self, default_cwd: Optional[str] = None):
        """
        Initialize the terminal manager.
        
        Args:
            default_cwd: Default working directory for commands
        """
        self.default_cwd = default_cwd or os.getcwd()
        self.current_cwd = self.default_cwd
        self.shell = self._get_shell()
        
    def _get_shell(self) -> bool:
        """Determine if we should use shell mode based on platform."""
        return sys.platform == 'win32'
    
    async def run_command(
        self, 
        command: str, 
        cwd: Optional[str] = None,
        timeout: Optional[float] = 60.0
    ) -> AsyncIterator[Tuple[str, str]]:
        """
        Run a shell command and yield output line by line.
        
        Args:
            command: The command to execute
            cwd: Working directory (defaults to current_cwd)
            timeout: Maximum execution time in seconds
            
        Yields:
            Tuples of (output_type, line) where output_type is 'stdout', 'stderr', or 'error'
        """
        work_dir = cwd or self.current_cwd
        
        # Handle cd command specially
        if command.strip().startswith('cd '):
            new_dir = command.strip()[3:].strip()
            try:
                new_path = Path(new_dir).expanduser().resolve()
                if new_path.exists() and new_path.is_dir():
                    self.current_cwd = str(new_path)
                    yield ("stdout", f"Changed directory to: {self.current_cwd}")
                else:
                    yield ("error", f"Directory not found: {new_dir}")
            except Exception as e:
                yield ("error", f"Failed to change directory: {str(e)}")
            return
        
        try:
            # Create subprocess
            if sys.platform == 'win32':
                # Windows: use shell for built-in commands
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=work_dir,
                    shell=True
                )
            else:
                # Unix: use shell for proper command interpretation
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=work_dir
                )
            
            # Read output streams concurrently using asyncio.Queue
            output_queue = asyncio.Queue()
            
            async def consume_stream(stream, stream_type):
                """Consume a stream and put lines in the queue."""
                if stream is None:
                    return
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    try:
                        decoded = line.decode('utf-8', errors='replace').rstrip()
                        if decoded:
                            await output_queue.put((stream_type, decoded))
                    except Exception as e:
                        await output_queue.put(("error", f"Error decoding output: {str(e)}"))
            
            # Create tasks for both streams
            stdout_task = asyncio.create_task(consume_stream(process.stdout, "stdout"))
            stderr_task = asyncio.create_task(consume_stream(process.stderr, "stderr"))
            
            # Yield output as it comes from the queue
            async def monitor_tasks():
                """Wait for both tasks to complete."""
                await asyncio.gather(stdout_task, stderr_task)
                await output_queue.put(None)  # Sentinel to signal completion
            
            monitor_task = asyncio.create_task(monitor_tasks())
            
            while True:
                item = await output_queue.get()
                if item is None:  # Sentinel value indicating completion
                    break
                yield item
            
            # Wait for process to complete with timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                yield ("error", f"Command timed out after {timeout} seconds")
                return
            
            # Check return code
            if process.returncode != 0:
                yield ("stderr", f"Command exited with code: {process.returncode}")
                
        except FileNotFoundError:
            yield ("error", f"Command not found: {command.split()[0]}")
        except PermissionError:
            yield ("error", f"Permission denied: {command}")
        except Exception as e:
            yield ("error", f"Failed to execute command: {str(e)}")
    
    def get_cwd(self) -> str:
        """Get the current working directory."""
        return self.current_cwd
    
    def set_cwd(self, path: str):
        """Set the current working directory."""
        path_obj = Path(path).expanduser().resolve()
        if path_obj.exists() and path_obj.is_dir():
            self.current_cwd = str(path_obj)
            return True
        return False
    
    async def run_command_simple(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = 60.0
    ) -> dict:
        """
        Run a command and return all output at once.
        
        Args:
            command: The command to execute
            cwd: Working directory
            timeout: Maximum execution time
            
        Returns:
            Dict with 'stdout', 'stderr', and 'returncode' keys
        """
        work_dir = cwd or self.current_cwd
        
        try:
            if sys.platform == 'win32':
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=work_dir,
                    shell=True
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=work_dir
                )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'returncode': process.returncode
            }
            
        except asyncio.TimeoutError:
            process.kill()
            return {
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'returncode': -1
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': f'Error: {str(e)}',
                'returncode': -1
            }
    
    def get_env(self) -> dict:
        """Get current environment variables."""
        return os.environ.copy()
    
    def set_env(self, key: str, value: str):
        """Set an environment variable."""
        os.environ[key] = value
    
    async def list_dir(self, path: Optional[str] = None) -> list:
        """List contents of a directory."""
        target = path or self.current_cwd
        try:
            path_obj = Path(target).expanduser().resolve()
            if not path_obj.exists():
                return []
            if not path_obj.is_dir():
                return []
            return [str(item.name) for item in sorted(path_obj.iterdir())]
        except Exception:
            return []
