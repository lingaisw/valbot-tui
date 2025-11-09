import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from rich.prompt import Prompt
from extension_manager.tool_dependency_handler import ToolDependencyHandler
from extension_manager.registry_parser import RegistryParser


class AgentAdder:
    """
    Interactive flow to add agent_extensions entries to ~/.valbot_config.json.
    Source is inferred from a single input:
    - If a .py file path is provided: treated as local (stores absolute path under 'location').
    - If a GitHub URL, git URL, or a directory (optionally pointing to a .git folder/file): treated as git-based,
      reads registry.json from the repo (by cloning for URLs or reading directly for local directories) and lets you pick.

    This class only updates configuration; it does not install or load plugins.
    """

    def __init__(self, console, config_manager):
        self.console = console
        self.cfg = config_manager  # ConfigManager instance
        self.tool_handler = ToolDependencyHandler(console, config_manager)
        # Ensure a stable default install directory when running under PyInstaller

    def _is_pyinstaller(self) -> bool:
        """
        Detect if running from a PyInstaller-built executable.
        """
        return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")

    def _is_in_meipass(self, path: str) -> bool:
        """
        Return True if the given path resolves inside PyInstaller's temporary _MEIPASS directory
        (or otherwise contains a transient _MEI path). Such paths are not persistent across runs.
        """
        if not self._is_pyinstaller():
            return False
        try:
            meipass = os.path.abspath(getattr(sys, "_MEIPASS"))
        except Exception:
            meipass = None
        abspath = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if meipass and abspath.startswith(meipass):
            return True
        # Defensive: also guard against random Temp/_MEIxxxx segments on Windows
        return "_MEI" in abspath

    def _normalize_dir(self, path: str) -> str:
        """Expand env vars and user, and convert to an absolute normalized directory path."""
        return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    def _get_default_install_dir(self) -> str:
        """
        Determine the default install directory for agents.
        - When running under PyInstaller, avoid the temporary _MEIPASS unpack dir.
        - On Windows (PyInstaller): use a directory next to the executable, e.g., <exe_dir>/agent_plugins/my_agents.
        - On Unix: use XDG_DATA_HOME or ~/.local/share/valbot/agent_plugins/my_agents.
        - Otherwise (non-frozen), keep the project-relative 'agent_plugins/my_agents'.
        """
        if self._is_pyinstaller():
            if os.name == "nt":
                # Mimic project-relative default when frozen: place under the executable's directory
                base = os.path.dirname(sys.executable)
                return os.path.join(base, "agent_plugins", "my_agents")
            else:
                base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
                return os.path.join(base, "valbot", "agent_plugins", "my_agents")
        # Default for non-frozen Python environments relative to this file:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "agent_plugins", "my_agents"))

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
                    f"[yellow]An agent named '{entry['name']}' already exists. Overwrite?[/yellow]",
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
                    f"[bold green]Agent '{entry['name']}' {action} in {path_display}[/bold green]"
                )
                self.console.print("[dim]Note: restart or reload plugins to use it in this session.[/dim]")

    # ----- Interactive prompts -----

    def _ask_source_type(self) -> str:
        return Prompt.ask(
            "[bold green]Add agent from[/bold green]",
            choices=["local", "github"],
            default="local",
            console=self.console,
        )

    def _build_local_entry(self, raw_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        name = Prompt.ask("[bold green]Agent name[/bold green]", console=self.console).strip()
        description = (
            Prompt.ask("[bold green]Description[/bold green]", default="", console=self.console).strip()
        )
        if not raw_location:
            raw_location = self.console.path_prompt("Path to the .py file: ", single_path_only=True).strip()

        if not name or not raw_location:
            self.console.print("[bold red]Name and location are required.[/bold red]")
            return None

        abs_loc = os.path.abspath(os.path.expanduser(os.path.expandvars(raw_location)))
        # Disallow selecting files from the transient PyInstaller _MEIPASS directory
        if self._is_in_meipass(abs_loc):
            self.console.print(
                "[bold red]The selected .py file is inside the application's temporary directory (PyInstaller _MEIPASS).[/bold red]\n"
                "[bold red]Please copy it to a persistent user directory (e.g., your Documents or a folder under LocalAppData) and try again.[/bold red]"
            )
            return None
        if not (os.path.isfile(abs_loc) and abs_loc.lower().endswith(".py")):
            self.console.print("[bold red]Provided path must be an existing .py file.[/bold red]")
            return None

        return {"name": name, "description": description, "location": abs_loc}

    def _build_github_entry(self, repo_source: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        # Determine source: prompt if not provided
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
            # Treat as URL (http(s), git@, or anything ending with .git)
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

        agents = self._parse_registry(registry)
        if not agents:
            self.console.print("[bold red]No valid agents found in registry.json.[/bold red]")
            return None

        selected_list = self._select_agents(agents)
        if not selected_list:
            return None

        entries: List[Dict[str, Any]] = []
        for selected in selected_list:
            entry: Dict[str, Any] = {
                "name": selected["name"],
                "description": selected.get("description", ""),
                "repo": repo_url_for_entry if repo_url_for_entry is not None else "",
                "path": selected["path"],
            }
            if selected.get("ref"):
                entry["ref"] = selected["ref"]

            # Optional custom install path per entry
            default_install_dir = self._normalize_dir(self._get_default_install_dir())
            default_hint = default_install_dir
            # Determine the project-relative default path (without PyInstaller special handling)
            project_relative_default = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent_plugins", "my_agents"))

            use_custom = Prompt.ask(
                f"[bold green]Use a custom install directory for '{entry['name']}'?[/bold green] (default: {default_hint})",
                choices=["y", "n"],
                default="n",
                console=self.console,
            )

            # Only set install_path if it differs from the project-relative default
            install_path_to_use = default_install_dir
            if use_custom == "y":
                install_dir = self.console.path_prompt(
                    f"Install directory for '{entry['name']}' (leave empty to cancel and use default: {default_hint}): ", single_path_only=True
                ).strip()
                if install_dir:
                    resolved_dir = self._normalize_dir(install_dir)
                    if self._is_in_meipass(resolved_dir):
                        self.console.print(
                            "[yellow]Chosen install directory is inside the temporary PyInstaller folder (_MEIPASS). Using the persistent default instead.[/yellow]"
                        )
                        install_path_to_use = default_install_dir
                    else:
                        install_path_to_use = resolved_dir

            # Only add install_path key if it's not the project-relative default
            if install_path_to_use != project_relative_default:
                entry["install_path"] = install_path_to_use

            # Ensure the directory exists (best-effort)
            dir_to_create = install_path_to_use
            try:
                os.makedirs(dir_to_create, exist_ok=True)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Could not create install directory '{dir_to_create}': {e}. Falling back to default.[/yellow]"
                )
                try:
                    os.makedirs(default_install_dir, exist_ok=True)
                    # Only set install_path if fallback differs from project-relative default
                    if default_install_dir != project_relative_default:
                        entry["install_path"] = default_install_dir
                except Exception:
                    # If even default can't be created, keep as-is and proceed; loader can handle/raise later
                    pass
            entries.append(entry)

        # Check for required tools in selected agents
        all_required_tools = []
        common_install_path = None

        for selected in selected_list:
            if "required_tools" in selected and isinstance(selected["required_tools"], list):
                all_required_tools.extend(selected["required_tools"])

        # Use the install_path from the first agent (all from same repo)
        if entries and "install_path" in entries[0]:
            common_install_path = entries[0]["install_path"]

        if all_required_tools:
            # Remove duplicates
            unique_required_tools = list(set(all_required_tools))
            # Delegate tool handling to ToolDependencyHandler
            self.tool_handler.prompt_and_install_tools(
                unique_required_tools,
                registry,
                repo_url_for_entry,
                common_install_path
            )

        return entries

    def _select_agents(self, agents: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if len(agents) == 1:
            return [agents[0]]

        self.console.print("\n[bold]Select agent(s) to add:[/bold]")
        for i, a in enumerate(agents, start=1):
            desc = a.get("description", "")
            self.console.print(f"  {i}) [bold]{a['name']}[/bold] - {desc}")

        while True:
            choice = Prompt.ask(
                "[bold green]Enter number(s)[/bold green] (e.g., 1,3 or 'all')",
                console=self.console,
            ).strip()
            lc = choice.lower()
            if lc in ("all", "a"):
                return agents
            parts = [p.strip() for p in choice.split(",") if p.strip()]
            if not parts:
                self.console.print("[yellow]Invalid selection. Try again.[/yellow]")
                continue
            indices: List[int] = []
            valid = True
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    if 1 <= idx <= len(agents):
                        if idx not in indices:
                            indices.append(idx)
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid and indices:
                return [agents[i - 1] for i in indices]
            self.console.print("[yellow]Invalid selection. Try again.[/yellow]")

    # ----- Registry parsing -----

    def _parse_registry(self, data: Any) -> List[Dict[str, Any]]:
        """Parse registry for agents with required_tools using common RegistryParser."""
        return RegistryParser.parse_agents(data)

    # ----- Config persistence -----

    def _find_existing_by_name(self, name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        # Prefer home_config view for duplicates in the user’s file
        existing_list = self.cfg.home_config.get("agent_extensions") or []
        for i, entry in enumerate(existing_list):
            if isinstance(entry, dict) and entry.get("name") == name:
                return i
        return None

    def _persist_entry(self, entry: Dict[str, Any], overwrite: bool = True):
        """
        Persist the entry into agent_extensions via ConfigManager if available.
        Falls back to manual upsert + save_user_config.
        Returns (saved: bool, action: 'added'|'updated'|None).
        """
        # Preferred path: use append_to_list if the ConfigManager provides it
        if hasattr(self.cfg, "append_to_list"):
            before_idx = self._find_existing_by_name(entry["name"])
            self.cfg.append_to_list(
                "agent_extensions",
                entry,
                unique_key="name",
                overwrite=overwrite,
                persist=True,
            )
            after_idx = self._find_existing_by_name(entry["name"])  # re-check
            action = "updated" if before_idx is not None else "added"
            return True, action

        # Fallback: manual upsert into home_config and save
        agent_list = self.cfg.home_config.get("agent_extensions")
        if not isinstance(agent_list, list):
            agent_list = []
            self.cfg.home_config["agent_extensions"] = agent_list

        idx = self._find_existing_by_name(entry["name"])
        if idx is not None and overwrite:
            agent_list[idx] = entry
            action = "updated"
        else:
            agent_list.append(entry)
            action = "added"

        try:
            # Persist using ConfigManager's save method if present
            if hasattr(self.cfg, "save_home_config"):
                self.cfg.save_home_config()
            else:
                self.cfg.save_user_config()
            # Refresh merged_config so current session sees the change
            if hasattr(self.cfg, "merge_configs"):
                self.cfg.merged_config = self.cfg.merge_configs()
            return True, action
        except Exception as e:
            self.console.print(f"[bold red]Failed to save config:[/bold red] {e}")
            return False, None
