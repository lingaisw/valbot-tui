import os
import json
import logging
import tempfile
import shutil

# ---------------- Configuration Manager ----------------

class ConfigManager:
    def __init__(self, config_file=None):
        self.config_dir = os.path.dirname(os.path.abspath(__file__))
        self.user_config_path = os.path.join(self.config_dir, 'user_config.json')
        self.default_config_path = os.path.join(self.config_dir, 'default_config.json')
        self.home_config_path = os.path.join(os.path.expanduser('~'), '.valbot_config.json')
        self.command_line_config_path = config_file

        self.default_config = self.read_json(self.default_config_path)
        self.user_config = self.read_json(self.user_config_path)
        self.command_line_config = self.read_json(self.command_line_config_path) if config_file else {}
        self.home_config = self.read_json(self.home_config_path)

        self.merged_config = self.merge_configs()

    def read_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"Config {file_path} not found or invalid. Using empty object.")
            return {}

    def deep_merge_dicts(self, base, override):
        """Recursively merge two dictionaries. Values from override take precedence."""
        result = dict(base)
        for k, v in override.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = self.deep_merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    def merge_configs(self):
        # Merge priority: default <- user <- command_line <- home
        merged = self.deep_merge_dicts(self.default_config, self.user_config)
        merged = self.deep_merge_dicts(merged, self.command_line_config)
        merged = self.deep_merge_dicts(merged, self.home_config)
        return merged

    def save_user_config(self):
        """
        Atomically write home_config to ~/.valbot_config.json.
        Only persists settings in home_config (not the full merged config).
        Filters out keys with value None to avoid writing JSON null for uninitialized values.
        Also refreshes merged_config after saving.
        """
        data = {k: v for k, v in self.home_config.items() if v is not None}
        dir_name = os.path.dirname(self.home_config_path) or "."
        os.makedirs(dir_name, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".valbot_config.", suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            # Best-effort backup
            if os.path.exists(self.home_config_path):
                try:
                    shutil.copy2(self.home_config_path, self.home_config_path + ".bak")
                except Exception:
                    pass
            os.replace(tmp_path, self.home_config_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        # Refresh merged_config after saving
        self.merged_config = self.merge_configs()

    def update_setting(self, key, value, persist=False):
        """
        Update a setting in merged_config and, if persist=True, in home_config using dotted paths.
        """
        self._set_nested(self.merged_config, key, value)
        if persist:
            self._set_nested(self.home_config, key, value)
        
            self.save_user_config()

    def get_setting(self, key, default=None):
        keys = key.split('.')
        value = self.merged_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    # ---------- Helpers for nested paths and list upserts ----------

    def _set_nested(self, root: dict, dotted: str, value):
        parts = dotted.split('.')
        cur = root
        for i, p in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if is_last:
                cur[p] = value
            else:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]

    def _ensure_nested_container(self, root: dict, dotted: str, container_type):
        """
        Ensure that the path exists and ends with a container of container_type (dict or list).
        Returns the container reference.
        """
        parts = dotted.split('.')
        cur = root
        for i, p in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if is_last:
                if p not in cur or not isinstance(cur[p], container_type):
                    cur[p] = container_type()
                return cur[p]
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]

    def append_to_list(self, key, item, unique_key=None, overwrite=True, persist=False):
        """
        Upsert an item into a list at dotted path `key` (e.g., 'agent_extensions').
        Operates on home_config when persist=True; otherwise only merged_config.
        """
        # Update merged_config
        m_list = self._ensure_nested_container(self.merged_config, key, list)
        if unique_key and isinstance(item, dict):
            idx = next((i for i, x in enumerate(m_list) if isinstance(x, dict) and x.get(unique_key) == item.get(unique_key)), None)
            if idx is not None:
                if overwrite:
                    m_list[idx] = item
            else:
                m_list.append(item)
        else:
            m_list.append(item)

        if persist:
            h_list = self._ensure_nested_container(self.home_config, key, list)
            if unique_key and isinstance(item, dict):
                idx = next((i for i, x in enumerate(h_list) if isinstance(x, dict) and x.get(unique_key) == item.get(unique_key)), None)
                if idx is not None:
                    if overwrite:
                        h_list[idx] = item
                else:
                    h_list.append(item)
            else:
                h_list.append(item)
            self.save_user_config()

    def reload(self):
        """Re-read all config files and recompute merged_config."""
        self.default_config = self.read_json(self.default_config_path)
        self.user_config = self.read_json(self.user_config_path)
        self.home_config = self.read_json(self.home_config_path)
        self.command_line_config = self.read_json(self.command_line_config_path) if self.command_line_config_path else {}
        self.merged_config = self.merge_configs()
