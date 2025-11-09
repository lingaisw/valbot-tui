import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional
from rich.prompt import Prompt
from extension_manager.registry_parser import RegistryParser


class ToolAdder:
    """
    Interactive flow to add tool_extensions entries to ~/.valbot_config.json.
    Works similarly to AgentAdder but for tool files.
    """

    def __init__(self, console, config_manager):
        self.console = console
        self.cfg = config_manager

    def run(self, source_input: Optional[str] = None):
        if source_input is None:
            source_input = Prompt.ask(
                "[bold green]Enter a path or URL[/bold green] (a .py file for local, or a git repo dir/.git/GitHub URL)",
                console=self.console,
            ).strip()
        if not source_input:
            self.console.print("[bold red]A path or URL is required.[/bold red]")
            return

        expanded = os.path.expanduser(os.path.expandvars(source_input))
        entries = []

        if os.path.isfile(expanded) and expanded.lower().endswith(".py"):
            entry = self._build_local_entry(raw_location=expanded)
            entries = [entry] if entry else []
        else:
            entries = self._build_github_entry(repo_source=expanded) or []

        if not entries:
            return

        for entry in entries:
            # Handle duplicates before persisting
            existing_idx = self._find_existing_by_name(entry.get("name"))
            if existing_idx is not None:
                choice = Prompt.ask(
                    f"[yellow]A tool named '{entry['name']}' already exists. Overwrite?[/yellow]",
                    choices=["y", "n"],
                    default="n",
                    console=self.console,
                )
                if choice == "n":
                    new_name_default = f"{entry['name']} (2)"
                    new_name = (
                        Prompt.ask(
                            "[bold green]Enter a new name[/bold green]",
                            default=new_name_default,
                            console=self.console,
                        ).strip()
                        or new_name_default
                    )
                    entry["name"] = new_name

            saved, action = self._persist_entry(entry, overwrite=True)
            if saved:
                path_display = getattr(self.cfg, "home_config_path", os.path.expanduser("~/.valbot_config.json"))
                self.console.print(
                    f"[bold green]Tool '{entry['name']}' {action} in {path_display}[/bold green]"
                )
                self.console.print("[dim]Note: restart or reload to use it in this session.[/dim]")

    def _build_local_entry(self, raw_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        name = Prompt.ask("[bold green]Tool name[/bold green]", console=self.console).strip()
        description = (
            Prompt.ask("[bold green]Description[/bold green]", default="", console=self.console).strip()
        )
        if not raw_location:
            raw_location = self.console.path_prompt("Path to the .py file: ", single_path_only=True).strip()

        if not name or not raw_location:
            self.console.print("[bold red]Name and location are required.[/bold red]")
            return None

        abs_loc = os.path.abspath(os.path.expanduser(os.path.expandvars(raw_location)))

        if not (os.path.isfile(abs_loc) and abs_loc.lower().endswith(".py")):
            self.console.print("[bold red]Provided path must be an existing .py file.[/bold red]")
            return None

        return {"name": name, "description": description, "location": abs_loc}

    def _build_github_entry(self, repo_source: Optional[str] = None) -> Optional[list]:
        if not repo_source:
            repo_source = Prompt.ask(
                "[bold green]Git repo URL or local repo directory[/bold green]",
                console=self.console,
            ).strip()
            if not repo_source:
                self.console.print("[bold red]Repo source is required.[/bold red]")
                return None

        repo_source = os.path.expanduser(os.path.expandvars(repo_source))
        registry = None
        repo_url_for_entry = None

        # Local directory or .git path
        if os.path.isdir(repo_source) or (os.path.isfile(repo_source) and os.path.basename(repo_source) == ".git"):
            repo_dir = repo_source
            if os.path.basename(repo_source) == ".git":
                repo_dir = os.path.dirname(repo_source)
            reg_path = os.path.join(repo_dir, "registry.json")
            if not os.path.isfile(reg_path):
                self.console.print("[bold red]registry.json not found in the provided repository directory.[/bold red]")
                return None
            try:
                with open(reg_path, "r") as f:
                    registry = json.load(f)
                repo_url_for_entry = repo_dir
            except Exception as e:
                self.console.print(f"[bold red]Failed to read registry.json:[/bold red] {e}")
                return None
        else:
            # Treat as URL
            is_url_like = (
                repo_source.startswith("http://")
                or repo_source.startswith("https://")
                or repo_source.startswith("git@")
                or repo_source.endswith(".git")
            )
            if not is_url_like:
                self.console.print("[bold red]Not a valid git source. Provide a GitHub URL or a git repository directory/.git.[/bold red]")
                return None
            try:
                with TemporaryDirectory() as tmpdir:
                    subprocess.run(["git", "clone", "--depth", "1", repo_source, tmpdir], check=True)
                    reg_path = os.path.join(tmpdir, "registry.json")
                    if not os.path.isfile(reg_path):
                        self.console.print("[bold red]registry.json not found in repo root.[/bold red]")
                        return None
                    with open(reg_path, "r") as f:
                        registry = json.load(f)
                    repo_url_for_entry = repo_source
            except subprocess.CalledProcessError as e:
                self.console.print(f"[bold red]Failed to clone repo:[/bold red] {e}")
                return None
            except Exception as e:
                self.console.print(f"[bold red]Failed to read registry.json:[/bold red] {e}")
                return None

        tools = self._parse_registry(registry)
        if not tools:
            self.console.print("[bold red]No valid tools found in registry.json.[/bold red]")
            return None

        selected_list = self._select_tools(tools)
        if not selected_list:
            return None

        entries = []
        for selected in selected_list:
            entry = {
                "name": selected["name"],
                "description": selected.get("description", ""),
                "repo": repo_url_for_entry if repo_url_for_entry is not None else "",
                "path": selected["path"],
            }
            if selected.get("ref"):
                entry["ref"] = selected["ref"]
            entries.append(entry)
        return entries

    def _select_tools(self, tools: list) -> Optional[list]:
        if len(tools) == 1:
            return [tools[0]]

        self.console.print("\n[bold]Select tool(s) to add:[/bold]")
        for i, t in enumerate(tools, start=1):
            desc = t.get("description", "")
            self.console.print(f"  {i}) [bold]{t['name']}[/bold] - {desc}")

        while True:
            choice = Prompt.ask(
                "[bold green]Enter number(s)[/bold green] (e.g., 1,3 or 'all')",
                console=self.console,
            ).strip()
            lc = choice.lower()
            if lc in ("all", "a"):
                return tools
            parts = [p.strip() for p in choice.split(",") if p.strip()]
            if not parts:
                self.console.print("[yellow]Invalid selection. Try again.[/yellow]")
                continue
            indices = []
            valid = True
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    if 1 <= idx <= len(tools):
                        if idx not in indices:
                            indices.append(idx)
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid and indices:
                return [tools[i - 1] for i in indices]
            self.console.print("[yellow]Invalid selection. Try again.[/yellow]")

    def _parse_registry(self, data: Any) -> list:
        """Parse registry.json for tools section using common RegistryParser."""
        return RegistryParser.parse_tools(data)

    def _find_existing_by_name(self, name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        existing_list = self.cfg.home_config.get("tool_extensions") or []
        for i, entry in enumerate(existing_list):
            if isinstance(entry, dict) and entry.get("name") == name:
                return i
        return None

    def _persist_entry(self, entry: Dict[str, Any], overwrite: bool = True):
        """Persist the entry into tool_extensions via ConfigManager."""
        if hasattr(self.cfg, "append_to_list"):
            before_idx = self._find_existing_by_name(entry["name"])
            self.cfg.append_to_list(
                "tool_extensions",
                entry,
                unique_key="name",
                overwrite=overwrite,
                persist=True,
            )
            after_idx = self._find_existing_by_name(entry["name"])
            action = "updated" if before_idx is not None else "added"
            return True, action

        # Fallback
        tool_list = self.cfg.home_config.get("tool_extensions")
        if not isinstance(tool_list, list):
            tool_list = []
            self.cfg.home_config["tool_extensions"] = tool_list

        idx = self._find_existing_by_name(entry["name"])
        if idx is not None and overwrite:
            tool_list[idx] = entry
            action = "updated"
        else:
            tool_list.append(entry)
            action = "added"

        try:
            self.cfg.save_user_config()
            return True, action
        except Exception as e:
            self.console.print(f"[bold red]Failed to save config:[/bold red] {e}")
            return False, None
