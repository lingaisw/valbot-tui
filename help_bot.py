from __future__ import annotations
import asyncio
import os
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Markdown, Static

from client_setup import initialize_chat_client
from context_management import ContextManager

HELP_CONTEXT_FILES = [
    "app.py",
    "chatbot.py",
    "bot_commands.py",
    "context_management.py",
    "default_config.json",
    "config.py",
    "README.md",
]

HELP_SYSTEM_PROMPT = (
    "You are **ValBot-CLI Help** - a friendly assistant that explains ValBot "
    "commands, agent flows and configuration.  Answer concisely in Markdown."
)

# ─────────────────────────────────────────────────────────────
#  Main Textual app
# ─────────────────────────────────────────────────────────────
class HelpBotApp(App):  # type: ignore[misc]
    """Interactive help chat shown by ValBot-CLI's **/help** command."""

    TITLE = "ValBot-CLI-Help"

    CSS = """
        Screen          { background: #101318;  color: #D7DCE2; }

        #view           { height: 1fr; width: 100%; }
        #status         { dock: bottom; height: 1; padding-left: 1; color: #42A5F5; }
        Input           { dock: bottom; height: 4; }
    """

    BINDINGS = [
        ("escape", "quit", "Quit"),
        ("q", "quit", ""),
    ]

    def __init__(self, endpoint: str = "valbot-proxy", model: str = "gpt-4o") -> None:
        super().__init__()
        self._client = initialize_chat_client(endpoint=endpoint, model_name=model)
        self.model = model
        self.context_manager = ContextManager(Console())
        self.load_help_context()  # Load help context files into the conversation history
        self.partial: str = ""  # live assistant reply in‑progress

    def load_help_context(self) -> None:
        valbot_root_dir = os.path.dirname(os.path.abspath(__file__))
        self.context_manager.conversation_history.append({"role": "system", "content": HELP_SYSTEM_PROMPT})
        self.context_manager.conversation_history.append({"role": "system", "content": "Below is reference material for ValBot-CLI:\n"})
        self.context_manager.load_context(
            [os.path.join(valbot_root_dir, file) for file in HELP_CONTEXT_FILES],
            silent=True,
        )

    # ───────────────────── compose UI ────────────────────────
    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=False)

        self.view = Markdown(id="view")
        self.status = Static("", id="status")
        self.input = Input(placeholder="Ask me anything…")

        yield self.view
        yield self.status
        yield self.input
        yield Footer()

    # ────────────────────── lifecycle ────────────────────────
    async def on_mount(self) -> None:
        self.input.focus()
        self._render()

    async def on_input_submitted(self, evt: Input.Submitted) -> None:  # noqa: N802
        query = evt.value.strip()
        if not query:
            return
        self.context_manager.conversation_history.append({"role": "user", "content": query})
        evt.input.value = ""  # clear line
        self._render()

        # Kick off background streaming so the UI stays snappy
        self.partial = ""
        self.status.update("⏳ Thinking…")
        asyncio.create_task(self._stream_reply())

    async def action_quit(self) -> None:  # noqa: D401
        self.exit()

    # ──────────────────── rendering helpers ──────────────────

    def format_message(self, msg):
        role_prefix = "**YOU**: " if msg["role"] == "user" else "**HELP**: "
        return f"{role_prefix}{msg['content']}"

    def _render(self) -> None:
        """Re-render the *entire* conversation (cheap for short logs)."""
        non_system_messages = (msg for msg in self.context_manager.conversation_history if msg["role"] != "system")
        body = "\n\n".join(map(self.format_message, non_system_messages)) or (
            "# Welcome to **ValBot-CLI Help**\n"
            "\n"
            "_Ask me anything about ValBot-CLI!_\n"
            "\n"
            "## Examples:\n"
            "- `\"How do I use the /agent command?\"`\n"
            "- `\"How can I make my own custom configuration?\"`\n"
            "- `\"Show me how to set up a new agent.\"`\n"
            "\n"
        )
        self.view.update(body)
        try:  # Textual ≥0.51
            self.view.scroll_end(animate=False)
        except AttributeError:  # old versions scroll automatically
            pass

    def _render_with_partial(self) -> None:
        """Render conversation + **streaming** assistant text."""
        tmp = self.context_manager.conversation_history + [{"role": "assistant", "content": self.partial}]
        # Filter out system messages and format the rest
        non_system_messages = (msg for msg in tmp if msg["role"] != "system")
        body = "\n\n".join(map(self.format_message, non_system_messages))

        self.view.update(body)
        try:
            self.view.scroll_end(animate=False)
        except AttributeError:
            pass

    # ─────────────────── streaming coroutine ─────────────────
    async def _stream_reply(self) -> None:
        try:
            def _worker() -> str:
                partial_local = ""
                self.call_from_thread(self._update_status, "🔄 Connecting...")
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=self.context_manager.conversation_history,
                    stream=True,
                    max_completion_tokens=2000,
                )
                self.call_from_thread(self._update_status, "⏳ Receiving response...")
                for chunk in stream:
                    # Skip chunks with no choices (metadata/filtering chunks)
                    if not chunk.choices or len(chunk.choices) == 0:
                        continue
                    # Skip chunks without delta content
                    if not hasattr(chunk.choices[0], 'delta') or not chunk.choices[0].delta:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        partial_local += delta
                        self.call_from_thread(self._append_delta, delta)
                        self.call_from_thread(self._update_status, f"📝 Received {len(partial_local)} chars...")
                return partial_local

            final_reply = await asyncio.to_thread(_worker)
            self.context_manager.conversation_history.append({"role": "assistant", "content": final_reply})
            self.status.update("")
            self._render()
        except asyncio.CancelledError:
            self.status.update("Operation cancelled.")
        except Exception as e:
            self.status.update(f"Error: {str(e)}")

    # ------------------------------------------------------------------
    #  UI thread: incremental delta handler
    # ------------------------------------------------------------------
    def _append_delta(self, delta: str) -> None:
        self.partial += delta
        self._render_with_partial()

    def _update_status(self, message: str) -> None:
        """Update status message from worker thread."""
        self.status.update(message)


# ────────────────────────── standalone test ─────────────────────────
if __name__ == "__main__":
    HelpBotApp().run()