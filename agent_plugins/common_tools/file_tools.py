from pydantic_ai import RunContext
from pydantic import BaseModel
from rich.console import Console
from rich.prompt import Prompt
from typing import Optional, Union
import os
from pathlib import Path
import re

class FilePath(BaseModel):
    file_path: str

class FileContent(BaseModel):
    content: str

class WriteResult(BaseModel):
    message: str

class WriteInput(BaseModel):
    file_path: str
    content: str

async def create_file(
    ctx: RunContext,
    file_path: str,
    initial_content: Optional[str] = None
) -> str:
    """
    Create a file with the specified path and optional initial content.
    If the file already exists, ask the user if they want to overwrite it.
    """
    console = Console()
    if os.path.exists(file_path):
        overwrite = Prompt.ask(f"The file '{file_path}' already exists. Do you want to overwrite it? (yes/no)", choices=["yes", "no"])
        if overwrite.lower() == "no":
            return f"File '{file_path}' was not overwritten."

    with open(file_path, 'w') as file:
        if initial_content:
            file.write(initial_content)
        console.print(f"File '{file_path}' created successfully.")
    return f"File '{file_path}' created successfully."

async def read_file(ctx: RunContext, file_path: str, max_size_mb: int = 100) -> FileContent:
    """Read the specified file with size limit checking."""
    console = Console()
    try:
        # Check file size before reading
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            console.print(f"[yellow]Warning: File is large ({file_size / (1024*1024):.2f} MB). Reading first {max_size_mb} MB...[/yellow]")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read(max_size_bytes)
                console.print(f"File read successfully (partial): {file_path}")
            return FileContent(content=content + f"\n\n[... File truncated at {max_size_mb} MB ...]")
        
        # Read file in chunks for better memory handling
        content_chunks = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            while True:
                chunk = file.read(8192)  # Read in 8KB chunks
                if not chunk:
                    break
                content_chunks.append(chunk)
        
        content = ''.join(content_chunks)
        console.print(f"File read successfully: {file_path}")
        return FileContent(content=content)
        
    except FileNotFoundError:
        console.print(f"File not found: {file_path}")
        return FileContent(content=f"File not found: {file_path}")
    except IsADirectoryError:
        console.print(f"Expected a file but found a directory: {file_path}")
        return FileContent(content=f"Expected a file but found a directory: {file_path}")
    except UnicodeDecodeError:
        console.print(f"Error decoding file: {file_path}. It may not be a text file.")
        return FileContent(content=f"Error decoding file: {file_path}. It may not be a text file.")
    except MemoryError:
        console.print(f"Error: File too large to read into memory: {file_path}")
        return FileContent(content=f"Error: File too large to read into memory: {file_path}")

async def read_partial_file(ctx: RunContext, file_path: str, starting_line_num: int, ending_line_num: Union[int, None] = None) -> FileContent:
    """Read a specific range of lines from the specified file. If ending_line_num is None, read to the end of the file."""
    console = Console()
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if ending_line_num is None or ending_line_num > len(lines):
                ending_line_num = len(lines)
            if starting_line_num < 1 or starting_line_num > len(lines) or starting_line_num > ending_line_num:
                console.print(f"Invalid line range: {starting_line_num} to {ending_line_num}, file has {len(lines)} lines.")
                return FileContent(content=f"Invalid line range: {starting_line_num} to {ending_line_num}, file has {len(lines)} lines.")
            content = ''.join(lines[starting_line_num - 1:ending_line_num])
            console.print(f"File read successfully: {file_path} (lines {starting_line_num} to {ending_line_num})")
        return FileContent(content=content)
    except FileNotFoundError:
        console.print(f"File not found: {file_path}")
        return FileContent(content=f"File not found: {file_path}")
    except IsADirectoryError:
        console.print(f"Expected a file but found a directory: {file_path}")
        return FileContent(content=f"Expected a file but found a directory: {file_path}")
    except UnicodeDecodeError:
        console.print(f"Error decoding file: {file_path}. It may not be a text file.")
        return FileContent(content=f"Error decoding file: {file_path}. It may not be a text file.")


async def write_file(ctx: RunContext, input: WriteInput) -> WriteResult:
    """Write the modified content back to the file."""
    console = Console()
    try:
        with open(input.file_path, 'w') as file:
            file.write(input.content)
            console.print(f"[bold green]Changes made to {input.file_path} successfully.[/bold green]")
        return WriteResult(message=f"File '{input.file_path}' has been edited successfully.")
    except Exception as e:
        console.print(f"Error writing to file: {e}")
        return WriteResult(message=f"Error writing to file: {e}")

async def edit_string_in_file(ctx: RunContext, file_path: str, search_string_regex: str, replace_exact_string: str) -> WriteResult:
    """Edit a specific string in the file, writing the new content to the file.
    Use a regular expression to find the string and replace it with the specified string.
    This should be prefered over write_file for simple edits for a file, such as replacing a string with another string."""
    console = Console()
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        # Replace the string using regex
        new_content = re.sub(search_string_regex, replace_exact_string, content)
        with open(file_path, 'w') as file:
            file.write(new_content)
        console.print(f"[bold green]Changes made to {file_path} successfully.[/bold green]")
        return WriteResult(message=f"File '{file_path}' has been edited successfully.")
    except FileNotFoundError:
        console.print(f"File not found: {file_path}")
        return WriteResult(message=f"File not found: {file_path}")
    except Exception as e:
        console.print(f"Error editing file: {e}")
        return WriteResult(message=f"Error editing file: {e}")


async def display_file_content_to_user(ctx: RunContext, file_path: str) -> None:
    """Display the content of the specified file to the user's screen."""
    console = Console()
    console.print(f"[bold green]Displaying content of file: {file_path}[/bold green]")
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            console.print(f"Content of '{file_path}':\n{content}")
    except FileNotFoundError:
        console.print(f"File not found: {file_path}")

async def list_files_in_directory(ctx: RunContext, directory_path: str) -> list:
    """List all files in the specified directory."""
    console = Console()
    try:
        files = os.listdir(directory_path)
        console.print(f"Files in '{directory_path}': {files}")
        return files
    except FileNotFoundError:
        console.print(f"Directory not found: {directory_path}")
        return []

async def find_files_with_name(ctx: RunContext, directory_path: str, file_name: str) -> Union[list[str], str]:
    """Find files with the specified name in the directory and its subdirectories. Returns a list of file paths matching the name."""
    try:
        path = Path(directory_path)
        files = list(path.rglob(file_name))
        if not files:
            return f"No files found with name {file_name} in dir {directory_path}"
        return [str(file) for file in files]
    except FileNotFoundError:
        return f"Directory not found {directory_path}"
