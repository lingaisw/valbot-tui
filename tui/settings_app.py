from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer, Tree, Input, Static
from textual.containers import VerticalScroll, Container, Center
from textual.message import Message
from textual import on

from rich.style import Style
from rich.text import Text

# Import ConfigManager from config.py
from config import ConfigManager

class SettingsApp(App):
    TITLE = "⚙️  Settings Manager"
    CSS = """
    Screen {
        align: center middle;
    }
    #expand_collapse_controls {
        layout: horizontal;
        width: auto;
        height: auto;  /* Set height to auto to minimize space usage */
        margin: 1 0;
    }
    Button {
        width: auto;
        margin: 0 1;
        background: #21262d;
        color: #FFD700;
        border: round #444e5c;
        transition: background 0.2s, color 0.2s;
    }
    Button:hover {
        background: #FFD700;
        color: #21262d;
        border: round #FFD700;
        text-style: bold;
    }
    #save_button:hover {
        background: #9f1c09;
        color: #fff;
        border: round #FFD700;
    }
    #expand_all_button:hover, #collapse_all_button:hover {
        background: #1752b8;
        color: #fff;
        border: round #FFD700;
        text-style: bold;
    }
    #settings_tree {
        height: 1fr;  /* Make the tree fill the remaining vertical space */
        margin: 1 0;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "close_modal", "Cancel Edit"),
        ("e", "edit", "Edit Value"),
        ("space", "toggle_expand", "Expand/collapse tree node"),
        ("A", "expand_all", "Expand All"),
        ("C", "collapse_all", "Collapse All"),
    ]

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self.settings_tree = Tree("Settings", id="settings_tree")
        self.edit_input = Input(placeholder="Enter new value", id="edit_input")
        self.save_button = Button("Save", id="save_button")
        self.expand_all_button = Button("Expand All", id="expand_all_button")
        self.collapse_all_button = Button("Collapse All", id="collapse_all_button")
        self.modal: Container = None
        self.edit_path = []

    def _walk_tree(self, node, fn):
        # Recursively apply fn to each node in the tree
        fn(node)
        for child in getattr(node, 'children', []):
            self._walk_tree(child, fn)

    def compose(self) -> ComposeResult:
        yield Header()
        helper = Static(
            Text("↑/↓ Move   [<Space>] Expand/Collapse   [A] Expand All   [C] Collapse All   [e] Edit   [q] Quit", style="bold cyan"),
            classes="helper-bar"
        )
        yield helper

        settings_info = Static(
            Text("Changes applied after restart.", style="yellow"),
            classes="settings-info-message"
        )
        yield settings_info

        with VerticalScroll():
            with Container(id="expand_collapse_controls"):
                yield self.expand_all_button
                yield self.collapse_all_button
            yield self.settings_tree  # Ensure the tree fills the remaining space
        with Center():
            with Container(id="edit_modal", classes="modal") as self.modal:
                yield Static(
                    Text("Edit Setting", style="bold green"),
                    classes="modal-title"
                )
                yield self.edit_input
                yield self.save_button
        yield Footer()

    async def on_mount(self) -> None:
        self.populate_tree(self.settings_tree.root, self.config_manager.merged_config)
        self.settings_tree.root.expand()
        self.settings_tree.focus()
        self.modal.display = False
        self.settings_tree.styles.background = "#181A1B"
        self.settings_tree.styles.color = "#C9D1D9"
        self.settings_tree.styles.border = ("panel", "#30363D")

    def populate_tree(self, node, config, path=None):
        path = path or []
        for key, value in config.items():
            full_path = path + [key]
            if isinstance(value, dict):
                # Folder nodes: soft blue
                child = node.add(Text(f"{key}", style="bold #78A9FF"))
                child.data = full_path
                child.is_list = False
                self.populate_tree(child, value, full_path)
            elif isinstance(value, list):
                # List object containers: orange for the container node
                list_node = node.add(Text(f"{key} (list)", style="bold #FFA657"))  # changed color/style here
                list_node.data = full_path
                list_node.is_list = True
                for idx, item in enumerate(value):
                    item_path = full_path + [idx]
                    if isinstance(item, dict):
                        # Index: purple
                        item_node = list_node.add(Text(f"[{idx}]", style="#D2A8FF"))
                        item_node.data = item_path
                        item_node.is_list = False
                        self.populate_tree(item_node, item, item_path)
                    else:
                        # Index: blue, value: yellow
                        leaf = list_node.add_leaf(
                            Text(f"[{idx}]", style="#58A6FF") + Text(f": {item}", style="bold #FFD700")
                        )
                        leaf.data = item_path
                        leaf.is_list = False
            else:
                # Key: bright green, colon: light gray, value: bright yellow -- easier to read on black
                leaf = node.add_leaf(
                    Text(f"{key}", style="bold #7CFC00") + Text(": ", style="#C9D1D9") + Text(f"{value}", style="bold #FFD700")
                )
                leaf.data = full_path
                leaf.is_list = False

    def _get_value(self, path):
        cfg = self.config_manager.merged_config
        for p in path:
            cfg = cfg[p]
        return cfg

    def _set_value(self, path, value):
        cfg = self.config_manager.home_config
        for key in path[:-1]:
            if isinstance(key, int):
                cfg = cfg[key]
            else:
                cfg = cfg.setdefault(key, {})
        cfg[path[-1]] = value
        self.config_manager.save_user_config()
        self.config_manager.home_config = self.config_manager.read_json(self.config_manager.home_config_path)
        self.config_manager.merged_config = self.config_manager.merge_configs()

    def _format_label(self, path, value):
        key = str(path[-1])
        return Text(f"{key}", style="bold #7CFC00") + Text(": ", style="#C9D1D9") + Text(f"{value}", style="bold #FFD700")

    def _parse_value(self, val_str):
        lower = str(val_str).lower().strip()
        if lower in ["true", "yes"]:
            return True
        if lower in ["false", "no"]:
            return False
        if lower in ["none", "null", "undefined", ""]:
            return ""
        try:
            return int(val_str)
        except (ValueError, TypeError):
            pass
        try:
            return float(val_str)
        except (ValueError, TypeError):
            pass
        return val_str

    async def action_edit(self) -> None:
        node = self.settings_tree.cursor_node
        if node and node.data and not node.children and not getattr(node, "is_list", False):
            self.edit_path = node.data
            value = self._get_value(self.edit_path)
            self.edit_input.value = str(value)
            self.edit_input.styles.background = "#353575"
            self.edit_input.styles.color = "#FFD700"
            self.modal.display = True
            self.edit_input.focus()
        else:
            return

    async def action_toggle_expand(self) -> None:
        node = self.settings_tree.cursor_node
        if node:
            node.toggle()

    @on(Input.Submitted)
    async def save_edited_value(self, event: Input.Submitted) -> None:
        if not self.edit_path:
            return
        new_val = self._parse_value(event.value)
        self._set_value(self.edit_path, new_val)

        # Clear all children of the root node
        self.settings_tree.root.remove_children()

        # Repopulate the tree with updated configuration
        self.populate_tree(self.settings_tree.root, self.config_manager.merged_config)

        self.modal.display = False
        self.edit_path = []

    @on(Button.Pressed, "#save_button")
    async def on_save_clicked(self, _) -> None:
        # Simulate an Input.Submitted event to reuse save_edited_value logic
        class DummyEvent:
            def __init__(self, value):
                self.value = value
        await self.save_edited_value(DummyEvent(self.edit_input.value))

    async def action_close_modal(self) -> None:
        self.modal.display = False
        self.edit_path = []

    async def action_quit(self) -> None:
        class ThankYouMessage(Message):
            def __init__(self) -> None:
                super().__init__()
                self.static = Static(Text("Thank you for using Settings Manager!", style="bold green"))
        self.post_message(ThankYouMessage())
        self.exit()

    async def action_expand_all(self) -> None:
        def expand_node(node):
            if hasattr(node, "expand"):
                node.expand()
        self._walk_tree(self.settings_tree.root, expand_node)

    async def action_collapse_all(self) -> None:
        def collapse_node(node):
            if hasattr(node, "collapse"):
                node.collapse()
        self._walk_tree(self.settings_tree.root, collapse_node)

    @on(Button.Pressed, "#expand_all_button")
    async def on_expand_all_clicked(self, _):
        await self.action_expand_all()

    @on(Button.Pressed, "#collapse_all_button")
    async def on_collapse_all_clicked(self, _):
        await self.action_collapse_all()


# ---------------- Run If Called Directly ----------------

if __name__ == "__main__":
    config_manager = ConfigManager()
    app = SettingsApp(config_manager)
    app.run()
