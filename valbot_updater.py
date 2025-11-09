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
        remote_version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0", "origin/HEAD"], cwd=app_directory).strip().decode('utf-8')
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
        if subprocess.call(["git", "checkout", remote_version], cwd=app_directory) != 0:
            self.console.print("[bold red]Update failed.[/bold red]")
            return
        self.console.print("[bold green]ValBot repository updated successfully![/bold green]")
        if requirements_changed:  self.display_requirements_reinstall_instructions(app_directory, is_frozen)
        if is_frozen:
             self.display_exe_rebuild_instructions(app_directory)
        else:
            self.reload_callback()

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
            version = subprocess.check_output(["git", "describe", "--tags"], cwd=app_dir).strip().decode('utf-8')
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=app_dir).strip().decode('utf-8')
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        return {"version": version, "repo_path": app_dir}

    def get_version(self):
        return self.get_build_info()["version"]

