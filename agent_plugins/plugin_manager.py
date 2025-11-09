import importlib.util
import os
import sys
import traceback
import subprocess
import hashlib
from agent_plugins.agent_plugin import AgentPlugin
from agent_plugins.plugin_updater import PluginUpdater

def _default_install_root(app_directory):
    """Default install root: <valbot_repo>/agent_plugins/my_agents/"""
    return os.path.join(app_directory, "agent_plugins", "my_agents")

def _resolve_install_root(app_directory, install_path):
    """Resolve install_path; make relative paths relative to <valbot_repo>."""
    if not install_path:
        return _default_install_root(app_directory)
    return install_path if os.path.isabs(install_path) else os.path.join(app_directory, install_path)

def _repo_target_dir(install_root, repo_url, ref=None):
    """Stable dir name for a repo/ref: <basename>-<short-hash>."""
    base = os.path.basename(repo_url.rstrip("/")).removesuffix(".git")
    key = f"{repo_url}@{ref or ''}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(install_root, f"{base}-{h}")

def _ensure_repo_clone(repo_url, ref, install_root):
    """Clone repo into install_root if not present; checkout ref if provided."""
    os.makedirs(install_root, exist_ok=True)
    repo_dir = _repo_target_dir(install_root, repo_url, ref)
    if not os.path.exists(repo_dir):
        print(f"Cloning plugin repo {repo_url} into {repo_dir}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        if ref:
            subprocess.run(["git", "fetch", "--tags"], cwd=repo_dir, check=True)
            subprocess.run(["git", "checkout", ref], cwd=repo_dir, check=True)
    return repo_dir

def _resolve_plugin_location(app_directory, plugin_info):
    """
    Resolve the plugin .py file location.
    - If 'repo' and 'path' are provided, clone (once) into install_path or default location.
    - Else, use 'location' (absolute or relative to <valbot_repo>).
    Returns: absolute path to plugin .py, or None if cannot resolve.
    """
    # Remote fields
    repo_url = plugin_info.get("repo")
    subpath = plugin_info.get("path")
    ref = plugin_info.get("ref")
    install_path = plugin_info.get("install_path")

    # Local field
    plugin_location = plugin_info.get("location")

    if repo_url and subpath:
        try:
            install_root = _resolve_install_root(app_directory, install_path)
            repo_dir = _ensure_repo_clone(repo_url, ref, install_root)
            plugin_location = os.path.join(repo_dir, subpath)
        except Exception as e:
            print(f"Error preparing repo for plugin {plugin_info.get('name')}: {e}")
            traceback.print_exc()
            return None
    elif plugin_location:
        if not os.path.isabs(plugin_location):
            plugin_location = os.path.join(app_directory, plugin_location)
    else:
        print(f"No location or repo/path provided for plugin {plugin_info.get('name')}. Skipping.")
        return None

    if not plugin_location.endswith(".py"):
        print(f"Plugin {plugin_info.get('name')} does not point to a .py file: {plugin_location}")
        return None

    if not os.path.isfile(plugin_location):
        print(f"Plugin file not found for {plugin_info.get('name')}: {plugin_location}")
        return None

    return plugin_location

def _load_python_plugin(plugin_name, plugin_description, plugin_location, plugins_dict, plugin_info_list):
    """Import a plugin module from a file and register AgentPlugin subclasses."""
    module_name = os.path.basename(plugin_location)[:-3]
    try:
        spec = importlib.util.spec_from_file_location(module_name, plugin_location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in dir(module):
            plugin_class = getattr(module, attr)
            if isinstance(plugin_class, type) and issubclass(plugin_class, AgentPlugin) and plugin_class is not AgentPlugin:
                plugins_dict[plugin_name] = plugin_class
                plugin_info_list.append((plugin_name, plugin_description))
    except Exception as e:
        print(f"Error loading plugin {plugin_name}: {e}")
        traceback.print_exc()

class PluginManager:
    def __init__(self, config_manager, model=None):
        self.config_manager = config_manager
        self.config = config_manager.merged_config
        self.plugins = {}  # Store plugin *classes*
        self.plugin_info = []  # Store plugin info including name and description
        self.custom_commands = config_manager.get_setting("custom_agent_commands", [])
        self.custom_prompts = config_manager.get_setting("custom_prompts", [])
        self.model = model  # Store the model to pass to plugins

    def load_plugins(self):
        self.load_tools()  # Load tools first
        self.load_agents()  # Then load agents that depend on tools

    def load_agents(self):
        # <valbot_repo> directory (agent_plugins is under this)
        app_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Combine agent_extensions and default_agents
        plugin_sources = []
        if self.config.get("agent_extensions"):
            plugin_sources.extend(self.config.get("agent_extensions", []))
        if self.config.get("default_agents"):
            plugin_sources.extend(self.config.get("default_agents", []))

        for pinfo in plugin_sources:
            plugin_name = pinfo.get("name")
            plugin_description = pinfo.get("description")

            plugin_location = _resolve_plugin_location(app_directory, pinfo)
            if not plugin_location:
                continue

            _load_python_plugin(plugin_name, plugin_description, plugin_location, self.plugins, self.plugin_info)

    def load_tools(self):
        """
        Load tool modules from config and register them in the tool registry.
        Tools can then be imported like:
            from agent_plugins.tool_extensions import docs_search_tools
        Or:
            from agent_plugins.tool_extensions import import_tool
            docs_tools = import_tool('docs_search_tools')
        """
        from agent_plugins.tool_extensions import register_tool

        app_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Combine tool_extensions and default_tools
        tool_sources = []
        if self.config.get("tool_extensions"):
            tool_sources.extend(self.config.get("tool_extensions", []))
        if self.config.get("default_tools"):
            tool_sources.extend(self.config.get("default_tools", []))

        for tinfo in tool_sources:
            tool_name = tinfo.get("name")
            tool_location = _resolve_plugin_location(app_directory, tinfo)
            if not tool_location:
                continue
            register_tool(tool_name, tool_location)

    def get_custom_command_info(self):
        """Retrieve custom command information including descriptions."""
        return [
            {
                "command": custom_command['command'],
                "description": custom_command.get('description', "No description available.")
            }
            for custom_command in self.custom_commands
        ]

    def get_custom_prompt_info(self):
        """Retrieve custom prompt information including descriptions."""
        return self.custom_prompts


    def run_plugin(self, plugin_name, context, **kwargs):
        def normalize(name):
            return name.replace('_', '').replace(' ', '').lower()

        normalized_input = normalize(plugin_name)
        matched_plugin_class = None
        for key in self.plugins:
            if normalize(key) == normalized_input:
                matched_plugin_class = self.plugins[key]
                break
        if matched_plugin_class:
            plugin_instance = matched_plugin_class(self.model, **kwargs)  # Create a new instance each time
            plugin_instance.run_agent_flow_with_init_args(context, **kwargs)
        else:
            print(f"Plugin {plugin_name} not found.")

    def run_custom_command(self, command, context):
        for custom_command in self.custom_commands:
            if custom_command['command'] == command:
                agent_name = custom_command['uses_agent']
                agent_args = custom_command.get('agent_args', {})
                self.run_plugin(agent_name, context, **agent_args)
                return
        print(f"Custom command {command} not found.")

    def get_agent_plugin_sources(self):
        plugin_sources = []
        if self.config.get("agent_extensions"):
            plugin_sources.extend(self.config.get("agent_extensions", []))
        if self.config.get("default_agents"):
            plugin_sources.extend(self.config.get("default_agents", []))
        return plugin_sources

    def plugin_update_flow(self):
        app_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        updator = PluginUpdater(app_directory, self.get_agent_plugin_sources(), self.config_manager)
        updates_available = updator.check_for_updates()
        if not updates_available:
            print("All plugins are up to date.")
            return
        plugins_to_update = updator.ask_which_to_update(updates_available)
        if plugins_to_update:
            updator.update_selected_plugins(plugins_to_update)
            self.load_plugins()  # Reload plugins after update
            print("Plugin update process completed.")
        else:
            print("No plugins selected for update.")

    def plugin_cleanup_flow(self, dry_run=True):
        """Clean up orphaned plugin repositories."""
        app_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        updator = PluginUpdater(app_directory, self.get_agent_plugin_sources(), self.config_manager)
        updator.cleanup_orphaned_repos(dry_run=dry_run)