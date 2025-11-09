import os
import subprocess
import shutil

class PluginUpdater:
    """Handles checking for and managing plugin updates."""

    def __init__(self, app_directory, plugin_configs, config_manager=None):
        self.app_directory = app_directory
        self.plugin_configs = plugin_configs
        self.config_manager = config_manager

    @staticmethod
    def _is_branch_ref(ref):
        """Check if ref looks like a branch name (not a commit hash or tag)."""
        if not ref:
            return True
        return not (len(ref) >= 7 and all(c in '0123456789abcdef' for c in ref))

    @staticmethod
    def _run_git_command(args, cwd=None, capture=False):
        """Execute git command and return output if capture=True."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=capture,
                text=True,
                check=True
            )
            return result.stdout.strip() if capture else True
        except Exception as e:
            print(f"Git command error: {e}")
            return None if capture else False

    def _get_remote_commit(self, repo_url, ref=None):
        """Get the latest commit hash from remote for given ref/branch."""
        output = self._run_git_command(["ls-remote", repo_url, ref or "HEAD"], capture=True)
        return output.split()[0] if output else None

    def _get_local_commit(self, repo_dir):
        """Get the current commit hash of local repo."""
        return self._run_git_command(["rev-parse", "HEAD"], cwd=repo_dir, capture=True)

    def _update_repo(self, repo_dir, ref=None):
        """Update a local repo to latest commit on ref."""
        print(f"Updating repository in {repo_dir}...")
        if not self._run_git_command(["fetch", "--all"], cwd=repo_dir):
            return False

        # If no ref specified, checkout the remote's default branch
        checkout_target = ref if ref else "origin/HEAD"

        if not self._run_git_command(["checkout", checkout_target], cwd=repo_dir):
            return False
        new_commit = self._get_local_commit(repo_dir)
        print(f"Updated to commit {new_commit[:12]}")
        return True

    def _get_repo_info(self, pinfo):
        """Extract repo directory and metadata from plugin info."""
        from agent_plugins.plugin_manager import _resolve_install_root, _repo_target_dir
        repo_url = pinfo.get("repo")
        if not repo_url:
            return None
        ref = pinfo.get("ref")
        install_root = _resolve_install_root(self.app_directory, pinfo.get("install_path"))
        repo_dir = _repo_target_dir(install_root, repo_url, ref)
        return {
            'name': pinfo.get("name"),
            'repo_url': repo_url,
            'ref': ref,
            'repo_dir': repo_dir
        }

    def check_for_updates(self):
        """Check which plugin repos have updates available."""
        updates_available = []
        for pinfo in self.plugin_configs:
            repo_info = self._get_repo_info(pinfo)
            if not repo_info or not self._is_branch_ref(repo_info['ref']):
                if repo_info:
                    print(f"Skipping {repo_info['name']}: pinned to specific ref '{repo_info['ref']}'")
                continue

            if not os.path.exists(repo_info['repo_dir']):
                continue

            local_commit = self._get_local_commit(repo_info['repo_dir'])
            remote_commit = self._get_remote_commit(repo_info['repo_url'], repo_info['ref'])

            if local_commit and remote_commit:
                if local_commit != remote_commit:
                    updates_available.append({
                        'name': repo_info['name'],
                        'repo': repo_info['repo_url'],
                        'ref': repo_info['ref'] or 'default',
                        'local_commit': local_commit[:12],
                        'remote_commit': remote_commit[:12],
                        'repo_dir': repo_info['repo_dir']
                    })
                    print(f"Update available for {repo_info['name']}: {local_commit[:12]} -> {remote_commit[:12]}")
                else:
                    print(f"{repo_info['name']} is up to date ({local_commit[:12]})")
        return updates_available

    @staticmethod
    def ask_which_to_update(updates_available):
        """Prompt user to select which plugins to update."""
        if not updates_available:
            print("No plugin updates available.")
            return []

        print("\nThe following plugins have updates available:")
        for idx, update in enumerate(updates_available, start=1):
            print(f"\t{idx}. {update['name']} ({update['local_commit']} -> {update['remote_commit']})")

        selection = input("Enter numbers (comma-separated), or 'all': ").strip()
        if selection.lower() == 'all':
            return updates_available

        selected_indices = {int(p.strip()) - 1 for p in selection.split(',')
                          if p.strip().isdigit() and 1 <= int(p.strip()) <= len(updates_available)}
        return [updates_available[i] for i in selected_indices]

    def _find_affected_plugins(self, selected_repo_groups):
        """Find plugins affected by repo updates but not selected."""
        selected_names = {p['name'] for plugins in selected_repo_groups.values() for p in plugins}
        affected_by_repo = {}

        for pinfo in self.plugin_configs:
            plugin_name = pinfo.get("name")
            if plugin_name in selected_names:
                continue

            repo_info = self._get_repo_info(pinfo)
            if not repo_info or not self._is_branch_ref(repo_info['ref']):
                continue

            if repo_info['repo_dir'] in selected_repo_groups:
                affected_by_repo.setdefault(repo_info['repo_dir'], []).append({
                    'name': plugin_name,
                    'config': pinfo,
                    'repo_dir': repo_info['repo_dir']
                })
        return affected_by_repo

    def _pin_plugin_to_current_ref(self, plugin_name, repo_dir):
        """Pin a plugin to its current commit hash in the user config."""
        if not self.config_manager:
            print(f"Cannot pin {plugin_name}: no config manager available")
            return False

        current_commit = self._get_local_commit(repo_dir)
        if not current_commit:
            print(f"Cannot pin {plugin_name}: unable to get current commit")
            return False

        for setting_name in ['agent_extensions', 'default_agents']:
            items = self.config_manager.get_setting(setting_name, [])
            for item in items:
                if item.get('name') == plugin_name:
                    item['ref'] = current_commit
                    self.config_manager.update_setting(setting_name, items, persist=True)
                    print(f"Pinned {plugin_name} to commit {current_commit[:12]}")
                    return True

        print(f"Cannot pin {plugin_name}: not found in config")
        return False

    def update_selected_plugins(self, selected_updates):
        """Update the selected plugins, handling shared repos."""
        if not selected_updates:
            return []

        repo_groups = {}
        for update in selected_updates:
            repo_groups.setdefault(update['repo_dir'], []).append(update)

        multi_plugin_repos = {repo: plugins for repo, plugins in repo_groups.items() if len(plugins) > 1}
        if multi_plugin_repos:
            print("\nNote: The following repositories contain multiple plugins:")
            for repo_dir, plugins in multi_plugin_repos.items():
                print(f"  {os.path.basename(repo_dir)}: {', '.join(p['name'] for p in plugins)}")
            print("All plugins in these repos will be updated together.\n")

        affected_by_repo = self._find_affected_plugins(repo_groups)
        repos_to_skip = set()

        if affected_by_repo:
            print("Warning: The following plugins share the same repository:")
            for repo_dir, affected_plugins in affected_by_repo.items():
                print(f"  {os.path.basename(repo_dir)}: {', '.join(p['name'] for p in affected_plugins)}")

            if input("\nWould you like to update these as well? (y/n): ").strip().lower() != 'y':
                if input("Would you like to pin the affected plugins to their current commit? (y/n): ").strip().lower() == 'y':
                    for repo_dir, affected_plugins in affected_by_repo.items():
                        for plugin_info in affected_plugins:
                            self._pin_plugin_to_current_ref(plugin_info['name'], repo_dir)
                    # Don't skip repos that have selected plugins - only pin the unselected ones
                    print("Note: Affected plugins have been pinned to their current commits.")
                else:
                    print("Update cancelled.")
                    return []

        updated_plugins = []
        for repo_dir, plugins in repo_groups.items():
            ref = plugins[0]['ref']
            print(f"\nUpdating repository: {os.path.basename(repo_dir)}")
            if self._update_repo(repo_dir, ref if ref != 'default' else None):
                updated_plugins.extend([p['name'] for p in plugins])
            else:
                print(f"Failed to update plugins: {', '.join(p['name'] for p in plugins)}")

        if updated_plugins:
            print(f"\nSuccessfully updated {len(updated_plugins)} plugin(s):")
            for name in updated_plugins:
                print(f"  ✓ {name}")

        return updated_plugins

    def cleanup_orphaned_repos(self, dry_run=True):
        """Clean up old plugin repository directories that are no longer in use."""
        from agent_plugins.plugin_manager import _resolve_install_root, _repo_target_dir

        active_repo_dirs = {
            _repo_target_dir(_resolve_install_root(self.app_directory, pinfo.get("install_path")),
                           pinfo.get("repo"), pinfo.get("ref"))
            for pinfo in self.plugin_configs if pinfo.get("repo")
        }

        install_roots = {
            _resolve_install_root(self.app_directory, pinfo.get("install_path"))
            for pinfo in self.plugin_configs if pinfo.get("repo")
        }

        dirs_to_remove = []
        for install_root in install_roots:
            if not os.path.exists(install_root):
                continue
            for dirname in os.listdir(install_root):
                full_path = os.path.join(install_root, dirname)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, ".git")):
                    if full_path not in active_repo_dirs:
                        dirs_to_remove.append(full_path)

        if not dirs_to_remove:
            print("No orphaned plugin repositories found.")
            return []

        print(f"\nFound {len(dirs_to_remove)} orphaned plugin repository(s):")
        for repo_dir in dirs_to_remove:
            size = self._get_dir_size(repo_dir)
            print(f"  - {repo_dir} ({self._format_size(size)})")

        if dry_run:
            print("\nDry run - no directories were deleted.")
            print("Run with dry_run=False to actually remove these directories.")
            return dirs_to_remove

        if input("\nDelete these directories? (y/n): ").strip().lower() != 'y':
            print("Cleanup cancelled.")
            return []

        removed = []
        for repo_dir in dirs_to_remove:
            try:
                print(f"Removing {repo_dir}...")
                shutil.rmtree(repo_dir)
                removed.append(repo_dir)
                print("  ✓ Removed")
            except Exception as e:
                print(f"  ✗ Error removing {repo_dir}: {e}")

        if removed:
            total_size = sum(self._get_dir_size(d) for d in removed if os.path.exists(d))
            print(f"\nSuccessfully removed {len(removed)} directory(s), freed {self._format_size(total_size)}")

        return removed

    @staticmethod
    def _get_dir_size(path):
        """Get total size of directory in bytes."""
        total = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
        except Exception:
            pass
        return total

    @staticmethod
    def _format_size(bytes_size):
        """Format bytes as human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"