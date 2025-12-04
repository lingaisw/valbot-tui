import glob
import os
import gzip
import PyPDF2
import shlex
try:
    from chonkie import TokenChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False


def open_file_smart(file_path, mode='r', **kwargs):
    """
    Open regular or .gz compressed files transparently.
    
    Args:
        file_path: Path to file (can be .gz compressed)
        mode: File open mode ('r', 'rb', etc.)
        **kwargs: Additional arguments passed to open/gzip.open
        
    Returns:
        File handle for reading
    """
    if file_path.endswith('.gz'):
        # For text mode, ensure we use text mode with gzip
        if 'b' not in mode:
            return gzip.open(file_path, mode + 't', **kwargs)
        return gzip.open(file_path, mode, **kwargs)
    return open(file_path, mode, **kwargs)


# --- Token Counting Utility ---
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

class ContextManager:
    def __init__(self, console, token_counter=None):
        self.console = console
        self.conversation_history = []
        self.token_counter = token_counter or TokenCounter()

    def _validate_file_type(self, file_path, allowed_extensions=None):
        """Validate file type and provide helpful error messages for unsupported files.
        
        Args:
            file_path: Path to the file to validate
            allowed_extensions: List of allowed file extensions. If None, allow common text/document types.
            
        Returns:
            True if file is valid, False otherwise
        """
        # Default allowed extensions for context loading
        if allowed_extensions is None:
            allowed_extensions = [
                '.txt', '.md', '.py', '.js', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go',
                '.rs', '.rb', '.php', '.json', '.xml', '.yaml', '.yml', '.csv', '.log',
                '.sh', '.bash', '.sql', '.html', '.css', '.scss', '.ts', '.jsx', '.tsx',
                '.pdf',  # PDFs are handled specially
                '.gz'    # Compressed files (.txt.gz, .log.gz, etc.)
            ]
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # For .gz files, check the underlying file type
        if file_ext == '.gz':
            # Check if there's a file extension before .gz (e.g., .txt.gz)
            base_name = os.path.splitext(file_path[:-3])[1].lower()  # Remove .gz and check
            if base_name and base_name not in allowed_extensions:
                self.console.print(f"[bold red]Error:[/bold red] Compressed file '{os.path.basename(file_path)}' contains unsupported type '{base_name}'.")
                return False
            # .gz is allowed, will be decompressed transparently
            return True
        
        # If file has no extension and path is valid, treat as normal text file
        if not file_ext and os.path.isfile(file_path):
            return True
        
        # Check if file extension is in allowed list
        if file_ext not in allowed_extensions:
            self.console.print(f"[bold red]Error:[/bold red] File '{os.path.basename(file_path)}' has unsupported file type '{file_ext}'.")
            
            # Provide specific hints for common unsupported types
            if file_ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
                self.console.print(f"[yellow]Excel files are not supported for context loading.[/yellow]")
            elif file_ext in ['.exe', '.dll', '.bin', '.so', '.dylib']:
                self.console.print(f"[yellow]Binary executable files cannot be loaded.[/yellow]")
            elif file_ext in ['.zip', '.rar', '.7z', '.tar']:
                self.console.print(f"[yellow]Archive files (except .gz) are not supported.[/yellow]")
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
                self.console.print(f"[yellow]Image files cannot be loaded as text.[/yellow]")
            elif file_ext in ['.mp3', '.mp4', '.avi', '.mov', '.wav']:
                self.console.print(f"[yellow]Media files are not supported.[/yellow]")
            elif file_ext in ['.doc', '.ppt', '.pptx']:
                self.console.print(f"[yellow]Microsoft Office documents (.doc, .ppt, .pptx) are not supported.[/yellow]")
                if file_ext == '.doc':
                    self.console.print(f"[yellow]Note: .docx files are supported, but older .doc format is not.[/yellow]")
            
            # Show allowed formats
            common_formats = ['.txt', '.md', '.py', '.js', '.json', '.xml', '.csv', '.log', '.pdf']
            allowed_str = ", ".join(common_formats)
            self.console.print(f"[yellow]Common supported formats:[/yellow] {allowed_str}")
            return False
        
        return True

    def expand_paths(self, paths):
        """Expand wildcards in a list of paths."""
        expanded_paths = []
        for path in paths:
            expanded_paths.extend(glob.glob(os.path.expanduser(path)))
        return expanded_paths

    def token_summary_and_prompt(self, token_warning_limit, expanded_context, file_texts, silent):
        """Helper to display token counts, prompt user if needed, and return whether to proceed."""
        token_counts = []
        total_tokens = 0
        for file, context_data in file_texts:
            tokens = self.token_counter.estimate_token_count(context_data)
            token_counts.append((file, tokens))
            total_tokens += tokens

        if not silent:
            self.console.print("[bold yellow]Estimated token counts for files to be loaded into context:[/bold yellow]")
            for file, count in token_counts:
                self.console.print(f"[cyan]{file}[/cyan]: [magenta]{count}[/magenta] tokens")
            self.console.print(f"[bold]Total estimated tokens:[/bold] [magenta]{total_tokens}[/magenta]")

            if total_tokens > token_warning_limit:
                confirm = self.console.input(
                    f"[bold red]Warning: The context to load is large ({total_tokens} tokens). This may slow responses and reduce output quality.\nContinue loading? (y/n): [/bold red]"
                ).strip().lower()
                if not (confirm.startswith('y')):
                    self.console.print("[bold red]Aborted context loading due to token count.[/bold red]")
                    return False

            self.console.print(f"[green]Loading context from: {expanded_context}[/green]")
        return True

    def load_context(self, context, token_warning_limit=80000, silent=False):
        """Load context from a file or list of files, showing estimated token counts before loading (unless silent)."""
        if isinstance(context, str):
            context = [context]
        expanded_context = self.expand_paths(context)

        file_texts = []
        for file in expanded_context:
            try:
                context_data = self.read_context_file(file)
                file_texts.append((file, context_data))
            except Exception:
                file_texts.append((file, 'ERR'))

        proceed = self.token_summary_and_prompt(token_warning_limit, expanded_context, file_texts, silent)
        if not proceed:
            return

        for idx, (file, context_data) in enumerate(file_texts):
            try:
                self.append_context(file, context_data)
                if not silent: self.console.print(f"[bold green]Context loaded from: [/bold green][cyan]{file}[/cyan]")
            except FileNotFoundError:
                self.console.print(f"[bold red]File not found: {file}[/bold red]")
            except Exception as e:
                self.console.print(f"[bold red]Error loading file {file}: {e}[/bold red]")

    def load_more_context(self):
        """Load more context from a file or list of files."""
        self.console.print("[bold green]Enter file(s) to load[/bold green] (space-separated list, wildcards supported '*'):")
        context = self.console.path_prompt(prompt='Path: ').strip()
        if context:
            self.console.print(f"[bold cyan]You entered:[/bold cyan] {context}")
            files_to_load = self.expand_paths(shlex.split(context))
            self.load_context(files_to_load)

    def read_context_file(self, file, max_size_mb=50):
        """Read content from a file, handling both text and PDF formats with size limits.
        Uses chonkie for intelligent document chunking when available.
        Supports .gz compressed files transparently."""
        # Validate file type first
        if not self._validate_file_type(file):
            raise ValueError(f"Unsupported file type: {file}")
        
        # Check if this is a compressed PDF
        if file.endswith('.pdf.gz'):
            # Decompress PDF first, then pass to PDF reader
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with gzip.open(file, 'rb') as gz_file:
                    tmp_file.write(gz_file.read())
            try:
                result = self.read_context_pdf(tmp_path)
                return result
            finally:
                os.unlink(tmp_path)
        elif file.endswith('.pdf'):
            return self.read_context_pdf(file)
        else:
            try:
                # Check file size before reading
                file_size = os.path.getsize(file)
                max_size_bytes = max_size_mb * 1024 * 1024
                
                if file_size > max_size_bytes:
                    self.console.print(f"[yellow]Warning: File {file} is large ({file_size / (1024*1024):.2f} MB). Reading first {max_size_mb} MB...[/yellow]")
                    with open_file_smart(file, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(max_size_bytes)
                    
                    # Use chonkie for better chunking if available
                    if CHONKIE_AVAILABLE:
                        try:
                            chunker = TokenChunker(chunk_size=4000, chunk_overlap=200)
                            chunks = chunker.chunk(content)
                            # Return chunked content with separators for better readability
                            return "\n\n--- CHUNK BOUNDARY ---\n\n".join([chunk.text for chunk in chunks])
                        except Exception as e:
                            self.console.print(f"[yellow]Chonkie chunking failed, using default: {e}[/yellow]")
                            return content + f"\n\n[... File truncated at {max_size_mb} MB ...]"
                    else:
                        return content + f"\n\n[... File truncated at {max_size_mb} MB ...]"
                
                # Read full file
                with open_file_smart(file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Use chonkie for semantic chunking if the file is reasonably large
                if CHONKIE_AVAILABLE and len(content) > 10000:  # Only chunk files > 10KB
                    try:
                        chunker = TokenChunker(chunk_size=4000, chunk_overlap=200)
                        chunks = chunker.chunk(content)
                        # For smaller files, just return as-is; for larger ones, show chunk structure
                        if len(chunks) > 1:
                            self.console.print(f"[cyan]Chunked file into {len(chunks)} semantic chunks using chonkie[/cyan]")
                        return "\n\n--- CHUNK BOUNDARY ---\n\n".join([chunk.text for chunk in chunks])
                    except Exception as e:
                        self.console.print(f"[yellow]Chonkie chunking failed, using raw content: {e}[/yellow]")
                        return content
                
                return content
                
            except MemoryError:
                self.console.print(f"[red]Error: File {file} too large to read into memory[/red]")
                return f"[Error: File too large to read into memory]"

    def read_context_pdf(self, file):
        """Extract text from a PDF file."""
        with open(file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return ''.join(page.extract_text() for page in reader.pages)

    def append_context(self, file, context_data):
        """Append context data to the conversation history."""
        before_context = f"\nThe following is the context for the conversation it came from a file called {file}:\n"
        after_file_context = f"\nThis marks the end of the context from the file {file}.\n"
        self.conversation_history.extend([
            {"role": "system", "content": before_context},
            {"role": "system", "content": context_data},
            {"role": "system", "content": after_file_context}
        ])

    def clear_conversation(self):
        """Clear the conversation history."""
        self.console.print("[bold yellow]Conversation history cleared.[/bold yellow]")
        self.conversation_history = []
    
    def remove_file_from_context(self, file_path: str):
        """
        Remove a specific file's context from the conversation history.
        
        Args:
            file_path: Path to the file to remove from context
        """
        # Find and remove the context blocks for this file
        # Context blocks are marked with system messages containing the file name
        indices_to_remove = []
        
        # Normalize the file path for comparison (resolve to absolute path)
        normalized_file_path = os.path.abspath(os.path.normpath(file_path))
        
        i = 0
        while i < len(self.conversation_history):
            msg = self.conversation_history[i]
            
            # Look for the "beginning of context" marker
            if (msg.get("role") == "system" and 
                "following is the context for the conversation it came from a file called" in msg.get("content", "")):
                
                # Extract the file path from the message
                # Message format: "\nThe following is the context for the conversation it came from a file called {file}:\n"
                content = msg.get("content", "")
                try:
                    # Extract the file path from the message
                    file_marker = "it came from a file called "
                    if file_marker in content:
                        start_idx = content.index(file_marker) + len(file_marker)
                        # Find the end (either newline or colon)
                        end_idx = content.find(":", start_idx)
                        if end_idx == -1:
                            end_idx = content.find("\n", start_idx)
                        if end_idx == -1:
                            end_idx = len(content)
                        
                        stored_file_path = content[start_idx:end_idx].strip()
                        
                        # Normalize the stored path for comparison
                        normalized_stored_path = os.path.abspath(os.path.normpath(stored_file_path))
                        
                        # Compare normalized paths
                        if normalized_file_path == normalized_stored_path:
                            # Mark this message for removal
                            indices_to_remove.append(i)
                            
                            # Also mark the next two messages (content + end marker)
                            if i + 1 < len(self.conversation_history):
                                indices_to_remove.append(i + 1)
                            if i + 2 < len(self.conversation_history):
                                indices_to_remove.append(i + 2)
                            
                            # Skip ahead past these messages
                            i += 3
                            continue
                except Exception:
                    # If parsing fails, skip this message
                    pass
            
            i += 1
        
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            if 0 <= idx < len(self.conversation_history):
                self.conversation_history.pop(idx)
        
        if indices_to_remove:
            self.console.print(f"[bold green]Removed file from context:[/bold green] [cyan]{file_path}[/cyan]")
        else:
            self.console.print(f"[bold yellow]File not found in context:[/bold yellow] [cyan]{file_path}[/cyan]")

