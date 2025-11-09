"""
Tool Dependency Handler - Centralized tool management for agents and tool extensions.

This module handles:
- Checking tool availability (common_tools, tool_extensions)
- Categorizing required tools
- Parsing registry.json for tools
- Installing missing tools from registries
"""

import os
from typing import Any, Dict, List, Optional, Set
from rich.prompt import Confirm
from extension_manager.registry_parser import RegistryParser


class ToolDependencyHandler:
    """
    Handles tool dependency resolution and installation.
    Separated from AgentAdder/ToolAdder to follow Single Responsibility Principle.
    """

    def __init__(self, console, config_manager):
        self.console = console
        self.cfg = config_manager

    # ----- Tool availability checks -----

    def get_common_tool_names(self) -> Set[str]:
        """Get set of all available common tools by scanning the common_tools directory."""
        # Go up one level from extension_manager to reach agent_plugins
        common_tools_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "agent_plugins", "common_tools"
        )
        if not os.path.isdir(common_tools_dir):
            return set()

        tool_names = set()
        for filename in os.listdir(common_tools_dir):
            if filename.endswith("_tools.py"):
                # Extract tool name without .py extension
                tool_names.add(filename[:-3])
        return tool_names

    def get_extension_tool_names(self) -> Set[str]:
        """Get set of all tool names in merged config (tool_extensions + default_tools)."""
        existing_tools = []
        if self.cfg.merged_config.get("tool_extensions"):
            existing_tools.extend(self.cfg.merged_config.get("tool_extensions", []))
        if self.cfg.merged_config.get("default_tools"):
            existing_tools.extend(self.cfg.merged_config.get("default_tools", []))

        return {
            entry["name"] for entry in existing_tools
            if isinstance(entry, dict) and entry.get("name")
        }

    def categorize_tools(self, required_tools: List[str]) -> Dict[str, List[str]]:
        """
        Categorize required tools into:
        - 'common': available as common_tools
        - 'extensions': already configured as tool_extensions
        - 'missing': not available anywhere

        Returns dict with these three keys.
        """
        if not required_tools:
            return {"common": [], "extensions": [], "missing": []}

        common_tools = self.get_common_tool_names()
        extension_tools = self.get_extension_tool_names()

        categorized = {
            "common": [],
            "extensions": [],
            "missing": []
        }

        for tool in required_tools:
            if tool in common_tools:
                categorized["common"].append(tool)
            elif tool in extension_tools:
                categorized["extensions"].append(tool)
            else:
                categorized["missing"].append(tool)

        return categorized

    def get_missing_tools(self, required_tools: List[str]) -> List[str]:
        """
        Check which required tools are not available (neither in common_tools nor tool_extensions).
        Returns only truly missing tools.
        """
        if not required_tools:
            return []
        categorized = self.categorize_tools(required_tools)
        return categorized["missing"]

    # ----- Registry parsing -----

    def parse_tools_from_registry(self, registry: Any) -> List[Dict[str, Any]]:
        """
        Parse registry.json for tools section using common RegistryParser.
        """
        return RegistryParser.parse_tools(registry)

    def find_tools_in_registry(self, registry: Any, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Find specific tools in the registry by name."""
        if not tool_names:
            return []

        all_tools = self.parse_tools_from_registry(registry)
        tool_names_set = set(tool_names)
        return [tool for tool in all_tools if tool["name"] in tool_names_set]

    # ----- Tool installation flow -----

    def build_tool_entry(self, tool: Dict[str, Any], repo_url: str, install_path: Optional[str] = None) -> Dict[str, Any]:
        """Build a tool entry for user_config.json."""
        entry = {
            "name": tool["name"],
            "repo": repo_url,
            "path": tool["path"],
        }
        if tool.get("description"):
            entry["description"] = tool["description"]
        if tool.get("ref"):
            entry["ref"] = tool["ref"]
        if install_path:
            entry["install_path"] = install_path
        return entry

    def prompt_and_install_tools(
        self,
        required_tools: List[str],
        registry: Any,
        repo_url: str,
        install_path: Optional[str] = None,
        silent: bool = False
    ) -> Dict[str, Any]:

        if not required_tools:
            return {"installed": [], "already_available": [], "not_found": []}

        result = {
            "installed": [],
            "already_available": [],
            "not_found": []
        }

        # Show what tools are required
        if not silent:
            self.console.print(f"\n[yellow]This agent requires the following tools: {', '.join(required_tools)}[/yellow]")

        # Categorize tools
        categorized = self.categorize_tools(required_tools)

        # Show already available tools
        if categorized["common"]:
            if not silent:
                self.console.print(f"[green]✓ Already available (common_tools): {', '.join(categorized['common'])}[/green]")
            result["already_available"].extend(categorized["common"])

        if categorized["extensions"]:
            if not silent:
                self.console.print(f"[green]✓ Already available (tool_extensions): {', '.join(categorized['extensions'])}[/green]")
            result["already_available"].extend(categorized["extensions"])

        # Handle truly missing tools
        truly_missing = categorized["missing"]
        if not truly_missing:
            if not silent:
                self.console.print("[green]All required tools are already available![/green]")
            return result

        # Find missing tools in registry
        found_tools = self.find_tools_in_registry(registry, truly_missing)
        not_found = [t for t in truly_missing if t not in [ft["name"] for ft in found_tools]]

        # Show found tools
        if found_tools and not silent:
            self.console.print(f"\n[cyan]Found {len(found_tools)} tool(s) in registry:[/cyan]")
            for tool in found_tools:
                self.console.print(f"  • [bold]{tool['name']}[/bold] - {tool.get('description', '')}")

        # Show tools not found
        if not_found:
            if not silent:
                self.console.print(f"\n[yellow]⚠ Not found in registry: {', '.join(not_found)}[/yellow]")
                self.console.print("[dim]These tools will need to be added manually or may cause issues if the agent uses them.[/dim]")
            result["not_found"] = not_found

        if not found_tools:
            if not silent:
                self.console.print("[yellow]No tools from the registry can be auto-installed.[/yellow]")
                self.console.print("[dim]Add missing tools manually with /add_tool[/dim]")
            return result

        # Prompt to install (unless silent mode)
        should_install = silent
        if not silent:
            should_install = Confirm.ask(
                f"\n[bold]Add the {len(found_tools)} found tool(s) now?[/bold]",
                default=True,
                console=self.console
            )

        if not should_install:
            return result

        # Install tools using ToolAdder
        from extension_manager.tool_adder import ToolAdder
        tool_adder = ToolAdder(self.console, self.cfg)

        for tool in found_tools:
            entry = self.build_tool_entry(tool, repo_url, install_path)
            saved, action = tool_adder._persist_entry(entry, overwrite=True)

            if saved:
                result["installed"].append(tool["name"])
                if not silent:
                    self.console.print(f"[green]✓ {action.capitalize()} tool '{tool['name']}'[/green]")
            else:
                if not silent:
                    self.console.print(f"[red]✗ Failed to add tool '{tool['name']}'[/red]")

        return result
