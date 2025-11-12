import os
import sys
import subprocess
import json
from rich.rule import Rule
from rich.prompt import Prompt, Confirm

class ValbotUpdater:
    def __init__(self, console, reload_callback=None, exit_callback=None):
        self.console = console
        self.reload_callback = reload_callback
        self.exit_callback = exit_callback

    def update(self):
        is_frozen = getattr(sys, 'frozen', False)
        if is_frozen:
            app_directory =  self.find_git_repo_for_exe()
            if not app_directory:
                self.console.print("[bold red]Could not locate git repository.[/bold red]")
                return
        else:
            app_directory = os.path.dirname(os.path.abspath(__file__))

        try:
             self.perform_update(app_directory, is_frozen)
        except subprocess.CalledProcessError as e:
            error_msg = e.output.decode('utf-8') if e.output else str(e)
            self.console.print(f"[bold red]Error:[/bold red] {error_msg}")
        except Exception as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")

    def perform_update(self, app_directory, is_frozen):
        subprocess.check_output(["git", "fetch", "--tags", "--force"], cwd=app_directory, stderr=subprocess.STDOUT)
        local_version =  self.get_version().split('-', 1)[0] #strip off the -<extra sha>
        
        # Try to get remote version from tags first, fall back to commit comparison
        try:
            remote_version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0", "origin/HEAD"], cwd=app_directory, stderr=subprocess.STDOUT).strip().decode('utf-8')
            use_tags = True
        except subprocess.CalledProcessError:
            # No tags available, use commit-based comparison
            self.console.print("[yellow]No version tags found, comparing commits instead...[/yellow]")
            local_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=app_directory).strip().decode('utf-8')
            remote_commit = subprocess.check_output(["git", "rev-parse", "origin/HEAD"], cwd=app_directory).strip().decode('utf-8')
            
            if local_commit == remote_commit:
                self.console.print(f"[bold green]ValBot is up to date (commit: {local_commit[:7]}).[/bold green]")
                return
            
            self.console.print(f"[bold yellow]Updates available![/bold yellow]")
            self.console.print(f"Current commit: {local_commit[:7]} → Latest commit: {remote_commit[:7]}")
            
            # Check if requirements changed between commits
            requirements_changed = self.check_requirements_changed_by_commit(app_directory, local_commit, remote_commit)
            if requirements_changed:
                self.console.print("[bold yellow]Dependencies have changed - you'll need to update python package requirements after updating.[/bold yellow]")
            if not Confirm.ask("Update now?", default=True): return
            self.apply_update_by_commit(app_directory, remote_commit, requirements_changed, is_frozen)
            return
        
        # Tag-based update path
        if local_version == remote_version:
            self.console.print(f"[bold green]ValBot is up to date ({local_version}).[/bold green]")
            return
        self.console.print(f"[bold yellow]New version available![/bold yellow]")
        self.console.print(f"Current: {local_version} → Latest: {remote_version}")
        requirements_changed =  self.check_requirements_changed(app_directory, remote_version)
        if requirements_changed:
            self.console.print("[bold yellow]Dependencies have changed - you'll need to update python package requirements after updating.[/bold yellow]")
        if not Confirm.ask("Update now?", default=True): return
        self.apply_update(app_directory, remote_version, requirements_changed, is_frozen)

    def apply_update(self, app_directory, remote_version, requirements_changed, is_frozen):
        self.console.print("[yellow]Updating...[/yellow]")
        try:
            subprocess.check_output(["git", "checkout", remote_version], cwd=app_directory, stderr=subprocess.STDOUT)
            self.console.print("[bold green]ValBot repository updated successfully![/bold green]")
        except subprocess.CalledProcessError as e:
            error_msg = e.output.decode('utf-8') if e.output else str(e)
            self.console.print(f"[bold red]Update failed:[/bold red]\n{error_msg}")
            return
        
        # Fix file permissions on Linux
        if sys.platform.startswith('linux'):
            self.fix_linux_permissions(app_directory)
        
        if requirements_changed:  self.display_requirements_reinstall_instructions(app_directory, is_frozen)
        if is_frozen:
             self.display_exe_rebuild_instructions(app_directory)
        else:
            if self.reload_callback:
                self.reload_callback()

    def apply_update_by_commit(self, app_directory, remote_commit, requirements_changed, is_frozen):
        """Apply update using commit hash instead of tag."""
        self.console.print("[yellow]Updating...[/yellow]")
        # First, ensure we're on a branch that can be updated
        current_branch = "main"  # Default
        try:
            current_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=app_directory, stderr=subprocess.DEVNULL).strip().decode('utf-8')
            if current_branch == "HEAD":
                # Detached HEAD state, checkout main/master first
                self.console.print("[yellow]Detached HEAD detected, checking out main branch...[/yellow]")
                try:
                    subprocess.check_output(["git", "checkout", "main"], cwd=app_directory, stderr=subprocess.STDOUT)
                    current_branch = "main"
                except subprocess.CalledProcessError:
                    try:
                        subprocess.check_output(["git", "checkout", "master"], cwd=app_directory, stderr=subprocess.STDOUT)
                        current_branch = "master"
                    except subprocess.CalledProcessError as e:
                        error_msg = e.output.decode('utf-8') if e.output else str(e)
                        self.console.print(f"[bold red]Failed to checkout branch:[/bold red] {error_msg}")
                        return
        except subprocess.CalledProcessError as e:
            error_msg = e.output.decode('utf-8') if e.output else str(e)
            self.console.print(f"[bold red]Failed to get current branch:[/bold red] {error_msg}")
            # Try to continue with default branch
        
        # Check for local changes before pulling
        try:
            status_output = subprocess.check_output(["git", "status", "--porcelain"], cwd=app_directory).decode('utf-8').strip()
            if status_output:
                # Parse modified files and filter out ignored files/folders
                modified_files = []
                ignored_patterns = ['.env', '__pycache__', 'valbot-venv']
                
                for line in status_output.split('\n'):
                    if line.strip():
                        # Get filename (last part after spaces)
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            filename = parts[-1]
                            # Skip files matching ignored patterns
                            if not any(ignored in filename for ignored in ignored_patterns):
                                modified_files.append(filename)
                
                if modified_files:
                    self.console.print("[bold yellow]⚠️  Local Changes Detected[/bold yellow]\n")
                    self.console.print("The following files have been modified locally:\n")
                    for f in modified_files:
                        self.console.print(f"  • {f}")
                    self.console.print("")
                    
                    # Ask user how to handle conflicts
                    self.console.print("[yellow]Choose how to handle these changes:[/yellow]")
                    self.console.print("1. [bold]Overwrite All[/bold] - Discard all local changes and use repository version")
                    self.console.print("2. [bold]Keep All[/bold] - Keep all local changes and skip update for those files")
                    self.console.print("3. [bold]Resolve Each File[/bold] - Choose action for each file individually")
                    self.console.print("4. [bold]Cancel[/bold] - Cancel the update\n")
                    
                    choice = Prompt.ask("Your choice", choices=["1", "2", "3", "4"], default="4")
                    
                    if choice == "1":
                        # Stash changes and pull
                        self.console.print("[yellow]Discarding local changes...[/yellow]")
                        try:
                            subprocess.check_output(["git", "reset", "--hard", "HEAD"], cwd=app_directory, stderr=subprocess.STDOUT)
                            self.console.print("[green]Local changes discarded.[/green]")
                        except subprocess.CalledProcessError as e:
                            error_msg = e.output.decode('utf-8') if e.output else str(e)
                            self.console.print(f"[bold red]Failed to discard changes:[/bold red] {error_msg}")
                            return
                    
                    elif choice == "2":
                        # Stash and try to reapply
                        self.console.print("[yellow]Saving local changes...[/yellow]")
                        try:
                            subprocess.check_output(["git", "stash"], cwd=app_directory, stderr=subprocess.STDOUT)
                            self.console.print("[green]Local changes saved.[/green]")
                            
                            # Note: We'll try to pop stash after pull
                            stashed = True
                        except subprocess.CalledProcessError as e:
                            error_msg = e.output.decode('utf-8') if e.output else str(e)
                            self.console.print(f"[bold red]Failed to stash changes:[/bold red] {error_msg}")
                            return
                    
                    elif choice == "3":
                        # Resolve each file individually
                        self.console.print("\n[bold yellow]Resolving files individually...[/bold yellow]\n")
                        files_to_discard = []
                        files_to_keep = []
                        
                        for file_path in modified_files:
                            self.console.print(f"[bold cyan]File:[/bold cyan] {file_path}")
                            self.console.print("  1. [bold]Overwrite[/bold] - Use repository version")
                            self.console.print("  2. [bold]Keep[/bold] - Keep your local changes")
                            
                            file_choice = Prompt.ask("  Your choice", choices=["1", "2"], default="2")
                            
                            if file_choice == "1":
                                files_to_discard.append(file_path)
                            else:
                                files_to_keep.append(file_path)
                            self.console.print("")
                        
                        # Apply choices
                        if files_to_discard:
                            self.console.print(f"[yellow]Discarding changes for {len(files_to_discard)} file(s)...[/yellow]")
                            try:
                                for file_path in files_to_discard:
                                    # Checkout from origin to get the repository version, not local HEAD
                                    subprocess.check_output(["git", "checkout", f"origin/{current_branch}", "--", file_path], cwd=app_directory, stderr=subprocess.STDOUT)
                                self.console.print("[green]Selected files reverted to repository version.[/green]")
                            except subprocess.CalledProcessError as e:
                                error_msg = e.output.decode('utf-8') if e.output else str(e)
                                self.console.print(f"[bold red]Failed to discard changes:[/bold red] {error_msg}")
                                return
                        
                        if files_to_keep:
                            self.console.print(f"[yellow]Saving local changes for {len(files_to_keep)} file(s)...[/yellow]")
                            try:
                                # Stash only the files we want to keep
                                for file_path in files_to_keep:
                                    subprocess.check_output(["git", "add", file_path], cwd=app_directory, stderr=subprocess.STDOUT)
                                subprocess.check_output(["git", "stash", "push", "-m", "ValBot update - kept files"], cwd=app_directory, stderr=subprocess.STDOUT)
                                self.console.print("[green]Local changes saved.[/green]")
                                stashed = True
                            except subprocess.CalledProcessError as e:
                                error_msg = e.output.decode('utf-8') if e.output else str(e)
                                self.console.print(f"[bold red]Failed to stash changes:[/bold red] {error_msg}")
                                return
                    
                    else:  # choice == "4"
                        self.console.print("[yellow]Update cancelled.[/yellow]")
                        return
        except subprocess.CalledProcessError:
            pass  # Continue if status check fails
        
        # Pull the latest changes with explicit origin and branch
        stashed = False
        try:
            pull_output = subprocess.check_output(
                ["git", "pull", "origin", current_branch], 
                cwd=app_directory, 
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            self.console.print("[bold green]ValBot repository updated successfully![/bold green]")
            if pull_output.strip() and not "Already up to date" in pull_output:
                self.console.print(f"{pull_output}")
        except subprocess.CalledProcessError as e:
            error_msg = e.output.decode('utf-8') if e.output else str(e)
            
            # Check if it's a merge conflict
            if "would be overwritten" in error_msg or "Your local changes" in error_msg:
                self.console.print(f"[bold red]Merge conflict detected:[/bold red]\n{error_msg}\n")
                self.console.print("[yellow]This shouldn't happen if changes were handled properly.[/yellow]")
                self.console.print("[yellow]Please report this issue.[/yellow]")
            else:
                self.console.print(f"[bold red]Update failed:[/bold red]\n{error_msg}")
            return
        
        # If we stashed changes, try to reapply them
        if stashed:
            self.console.print("\n[yellow]Attempting to restore your local changes...[/yellow]")
            try:
                stash_output = subprocess.check_output(["git", "stash", "pop"], cwd=app_directory, stderr=subprocess.STDOUT).decode('utf-8')
                self.console.print("[green]Local changes restored successfully![/green]")
                if "CONFLICT" in stash_output:
                    self.console.print("[bold yellow]⚠️  Some files have conflicts that need manual resolution:[/bold yellow]")
                    self.console.print(stash_output)
                    self.console.print("\n[yellow]Run 'git status' to see conflicted files.[/yellow]")
            except subprocess.CalledProcessError as e:
                error_msg = e.output.decode('utf-8') if e.output else str(e)
                if "CONFLICT" in error_msg:
                    self.console.print("[bold yellow]⚠️  Conflicts occurred while restoring changes:[/bold yellow]")
                    self.console.print(error_msg)
                    self.console.print("\n[yellow]Your changes are still in the stash.[/yellow]")
                    self.console.print("[yellow]Run 'git stash list' to see stashed changes.[/yellow]")
                else:
                    self.console.print(f"[bold red]Failed to restore changes:[/bold red] {error_msg}")
        
        # Fix file permissions on Linux
        if sys.platform.startswith('linux'):
            self.fix_linux_permissions(app_directory)
        
        if requirements_changed:  self.display_requirements_reinstall_instructions(app_directory, is_frozen)
        if is_frozen:
             self.display_exe_rebuild_instructions(app_directory)
        else:
            if self.reload_callback:
                self.reload_callback()

    def check_requirements_changed_by_commit(self, app_directory, local_commit, remote_commit):
        """Check if requirements.txt changed between two commits."""
        try:
            current_requirements = subprocess.check_output(["git", "show", f"{local_commit}:requirements.txt"], cwd=app_directory, stderr=subprocess.DEVNULL).decode('utf-8')
            new_requirements = subprocess.check_output(["git", "show", f"{remote_commit}:requirements.txt"], cwd=app_directory, stderr=subprocess.DEVNULL).decode('utf-8')
            return current_requirements != new_requirements
        except subprocess.CalledProcessError:
            return False

    def fix_linux_permissions(self, app_directory):
        """Fix file permissions for Linux shell scripts."""
        try:
            scripts = ['ec_linux_setup.sh', 'valbot_tui.sh']
            for script in scripts:
                script_path = os.path.join(app_directory, script)
                if os.path.exists(script_path):
                    subprocess.check_output(['chmod', '+x', script_path], cwd=app_directory, stderr=subprocess.DEVNULL)
            self.console.print("[dim]Fixed permissions for Linux scripts[/dim]")
        except Exception as e:
            # Non-critical error, just log it
            self.console.print(f"[dim yellow]Note: Could not fix script permissions: {e}[/dim yellow]")

    def find_git_repo_for_exe(self):
        """Find the git repository when running from a PyInstaller exe."""
        # First, try to get the repo path from build_info.json
        build_info = self.get_build_info()
        repo_path = build_info.get("repo_path")

        # Validate the stored repo path
        if repo_path and os.path.exists(os.path.join(repo_path, '.git')):
            self.console.print(f"[green]Using repository path from build: {repo_path}[/green]")
            return repo_path

        # If stored path is invalid or missing, prompt the user
        if repo_path:
            self.console.print(f"[yellow]Stored repository path is invalid: {repo_path}[/yellow]")
        else:
            self.console.print("[yellow]Running from executable - please provide the repository location.[/yellow]")

        while True:
            repo_path = Prompt.ask("Enter the path to your valbot-cli repository")
            if os.path.exists(os.path.join(repo_path, '.git')):
                return repo_path
            self.console.print("[bold red]Invalid repository path - no .git directory found.[/bold red]")
            if not Confirm.ask("Try again?", default=True):
                return None

    def display_exe_rebuild_instructions(self, repo_path):
        self.console.print(Rule(style="bright_blue", title="[bold blue]Rebuild Instructions for Windows Executable[/bold blue]"))
        self.console.print("\n[bold yellow]You are running valbot from a pre-built executable - rebuild required of exe:[/bold yellow]")
        self.console.print(f"1. Navigate to the valbot repository at: [bold]{repo_path}[/bold]")
        self.console.print("2. Either double-click the setup.bat file, or run the setup.bat from the command prompt.")
        self.exit_callback()

    def check_requirements_changed(self, app_directory, new_version):
        try:
            current_requirements = subprocess.check_output(["git", "show", "HEAD:requirements.txt"], cwd=app_directory, stderr=subprocess.DEVNULL).decode('utf-8')
            new_requirements = subprocess.check_output(["git", "show", f"{new_version}:requirements.txt"], cwd=app_directory, stderr=subprocess.DEVNULL).decode('utf-8')
            return current_requirements != new_requirements
        except subprocess.CalledProcessError:
            return False

    def display_requirements_reinstall_instructions(self, app_directory,is_frozen):
        self.console.print("\n[bold yellow]Python Dependencies changed! Please update your packages:[/bold yellow]")
        if is_frozen or sys.platform == "win32":
            self.console.print("Run setup.bat to rebuild the exe with new dependencies.")
        else:
            requirements_path = os.path.join(app_directory, "requirements.txt")
            answer = Prompt.ask("Would you like to automatically reinstall dependencies now? (y/n)", choices=["y", "n"], default="y")
            if answer.lower() == "y":
                use_proxy = Confirm.ask("Are you behind the corporate vpn/proxy?", default=True)
                proxy_args = ["--proxy", "http://proxy-dmz.intel.com:912"] if use_proxy else []
                subprocess.call([sys.executable, "-m", "pip", "install", "-r", requirements_path] + proxy_args)
            else:
                self.console.print(f"You can manually run:\n[bold]pip install -r {requirements_path} --proxy http://proxy-dmz.intel.com:912[/bold]")



    def get_build_info(self):
        """
        Get build information (version and repo_path).
        Returns a dict with 'version' and 'repo_path' keys.
        """
        if getattr(sys, 'frozen', False):  # Check if running from PyInstaller exe
            if hasattr(sys, '_MEIPASS'):
                build_info_file = os.path.join(sys._MEIPASS, 'build_info.json')
                if os.path.exists(build_info_file):
                    try:
                        with open(build_info_file, 'r') as f:
                            return json.load(f)
                    except (json.JSONDecodeError, IOError):
                        pass
            return {"version": "unknown (exe)", "repo_path": None}
        # Normal git-based version detection (for non-exe runs)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        version = "unknown"
        try:
            # Try git describe with tags first
            version = subprocess.check_output(["git", "describe", "--tags"], cwd=app_dir, stderr=subprocess.DEVNULL).strip().decode('utf-8')
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fall back to short commit hash if no tags
                version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=app_dir, stderr=subprocess.DEVNULL).strip().decode('utf-8')
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        return {"version": version, "repo_path": app_dir}

    def get_version(self):
        return self.get_build_info()["version"]

