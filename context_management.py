import glob
import os
import PyPDF2
import shlex


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
        """Read content from a file, handling both text and PDF formats with size limits."""
        if file.endswith('.pdf'):
            return self.read_context_pdf(file)
        else:
            try:
                # Check file size before reading
                file_size = os.path.getsize(file)
                max_size_bytes = max_size_mb * 1024 * 1024
                
                if file_size > max_size_bytes:
                    self.console.print(f"[yellow]Warning: File {file} is large ({file_size / (1024*1024):.2f} MB). Reading first {max_size_mb} MB...[/yellow]")
                    with open(file, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(max_size_bytes)
                    return content + f"\n\n[... File truncated at {max_size_mb} MB ...]"
                
                # Read in chunks for better memory handling
                content_chunks = []
                with open(file, 'r', encoding='utf-8', errors='replace') as f:
                    while True:
                        chunk = f.read(8192)  # Read in 8KB chunks
                        if not chunk:
                            break
                        content_chunks.append(chunk)
                return ''.join(content_chunks)
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
