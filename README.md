# ValBot-CLI

## Overview

ValBot-CLI is a powerful, extensible command-line interface for AI-assisted development and interaction. ValBot combines conversational AI with autonomous agent workflows to help you code, debug, analyze specifications, and automate complex tasks—all from your terminal.

**Key Capabilities:**

- **💬 Interactive Chat**: Real-time AI conversation with context-aware responses, multiline input, and rich formatting
- **🤖 Autonomous Agents**: Pre-built and custom agents that can execute multi-step workflows, use tools, and interact with files, git, terminals, and more
- **🔌 Extensible Plugin System**: Add custom agents from local files or Git repositories, create your own agents, and share them with others
- **⚙️ Flexible Configuration**: Customize models, prompts, commands, and agent behaviors to match your workflow
- **🔄 Built-in Updates**: Keep ValBot and installed plugins up to date with the `/update` command
- **📝 Context Management**: Load conversation context from files to give the AI deep understanding of your codebase

Whether you're asking quick questions, analyzing complex specifications, or running sophisticated agent workflows, ValBot adapts to your needs with a simple, terminal-first interface.

![image](https://github.com/user-attachments/assets/10b07c71-3fb4-40c9-a8eb-2a26366cefb4)

An illustration of Valbot's capabilities in understanding and explaining multiple files within a checker framework, followed by the creation of a brand-new checker file upon the user's request.

https://github.com/user-attachments/assets/f82c59b6-c8d4-4296-9223-01dc5b5f87ba

---

## Getting Started

### Requirements

- Python 3.11+

### Quick Setup Scripts (Recommended)

Prefer an automated setup? Use the included scripts for Windows or Intel EC Linux to create a virtual environment, install requirements, and configure your VALBOT_CLI_KEY.

#### Windows: setup.bat

- How to run:
  - Open Command Prompt or PowerShell in the repository root and run:
    ```bat
    .\setup.bat
    ```
  - Or double-click `setup.bat` in File Explorer.

- What it does:
  - Verifies Python 3.10+ is available.
  - Ensures a `.env` file exists and optionally prompts for `VALBOT_CLI_KEY`.
  - Creates a fresh `venv` directory (removes an existing one if present).
  - Upgrades `pip` and installs requirements from `requirements.txt`.
  - Builds a standalone `valbot.exe` with PyInstaller and copies it to `%USERPROFILE%\bin`.

- After it finishes:
  - If `%USERPROFILE%\bin` is on your PATH, you can run ValBot from anywhere:
    ```bat
    valbot "Hello"
    ```
  - From the repo root, you can also use the helper batch script with examples:
    ```bat
    .\valbot.bat "Your message here"
    .\valbot.bat -m "Your message" -c file1.txt file2.txt
    .\valbot.bat -a agent_name -p param1=value1 param2=value2
    .\valbot.bat --config your_config.json "Your message"
    ```
  - You can update `VALBOT_CLI_KEY` later by editing the `.env` file created in the repo root if you used a placeholder.

#### Intel EC Linux: ec_linux_setup.sh

- How to run:
  - Make the script executable and launch it from the repository root:
    ```bash
    chmod +x ec_linux_setup.sh
    ./ec_linux_setup.sh
    ```

- What it does (interactive):
  - Lets you select a Python 3.10+ interpreter (suggests the EC default).
  - Creates a new virtual environment or reuses an existing one.
  - Installs requirements (optionally using the EC proxy: `http://proxy-chain.intel.com:911`).
  - Prompts for your `VALBOT_CLI_KEY` and can write it to a `.env` in the project root.
  - Generates runner scripts: `valbot.sh` (bash/zsh) and `valbot.csh` (tcsh).
  - Optionally appends an alias `valbot` to `~/.aliases`.

- After it finishes:
  - Run via the generated script:
    ```bash
    ./valbot.sh "Hello"
    ```
  - Or, if you added the alias:
    ```bash
    valbot -m "Analyze this code" -c myfile.py
    ```

- Where to get `VALBOT_CLI_KEY`:
  - Visit https://genai-proxy.intel.com/ and choose "Manage Your API Tokens" to create a token.

### Manual Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/intel-innersource/applications.ai.valbot-cli valbot-cli
   cd valbot-cli
   ```

2. **Install Python packages**:

    - Note: It is recommended to setup a python virtual env as to not conflict with your personal python package requirements. [Python venv documentation](https://docs.python.org/3/library/venv.html). For EC, setting up a virtual env is a requirement, as you can not install packages directly.
         - quick refrence for setting up a python venv in EC:
           ```tcsh
               setenv VENV_PATH /path/to/create/your/valbot-venv
               /usr/intel/pkgs/python3/3.11.1/bin/python3 -m venv $VENV_PATH
               source $VENV_PATH/bin/activate.csh
           ```

    - After your venv is setup and activated, you can now pip install the required python packages:
   ```bash
   pip3 install -r requirements.txt
   # On Linux EC specify with proxy:
   pip3 install --proxy="http://proxy-chain.intel.com:911" -r path/to/requirements.txt
   ```

### Endpoint Setup

There are 3 endpoints supported, you must setup at least 1 of these:

#### Valbot proxy endpoint (*Recommended*)

Steps to get API Key:

  - Navigate to: [https://genai-proxy.intel.com/](https://genai-proxy.intel.com/) and login with your intel credentials.
  - Click `Manage Your API Tokens` > `Create New Token` > Fill in details > `Create Token`

Steps to configure Valbot to use this endpoint:
  - Copy this token and set it in your env as `VALBOT_CLI_KEY` (see below bash script for BKM on making valbot invocation easier)
  - Configure your ~/.valbot_config.json file to use the "valbot-proxy" endpoint. Place this somewhere in your config file:
    ```json
      {
      <...>
      "agent_model_config": {
          "default_endpoint": "valbot-proxy",
          "default_model": "gpt-4.1",
          "small_model": "gpt-4.1-mini"
      },
      "chat_model_config": {
          "default_endpoint": "valbot-proxy",
          "default_model": "gpt-5"
      },
      <...>
      }
    ```

#### AZURE Endpoint
  <TBD>


#### ~~IGPT (Deprecated)~~

~~Steps to get API Key~~:
  - ~~See instructions at: [https://wiki.ith.intel.com/spaces/GenAI/pages/3590613857/Using+the+Inference+API](https://wiki.ith.intel.com/spaces/GenAI/pages/3590613857/Using+the+Inference+API)~~

~~Steps to configure Valbot to use this endpoint:~~
  - ~~Set the igpt client id and secret as `CLIENT_ID` and `CLIENT_SECRET` in your env~~
  - ~~Configure your ~/.valbot_config.json file to use the "igpt" endpoint. Place this somewhere in your config file:
    Note: igpt only supports `gpt-4o`model~~
    ```json
      {
      <...>
      "agent_model_config": {
          "default_endpoint": "igpt",
          "default_model": "gpt-4o",
          "small_model": "gpt-4o"
      },
      "chat_model_config": {
          "default_endpoint": "igpt",
          "default_model": "gpt-4o"
      },
      <...>
      }
    ```

### Quick Start Helper Script

For easier usage, you can create a helper script and alias:

```bash
#!/bin/bash
VENV_PATH=/path/to/your/valbot-venv
VALBOT_PATH=/path/to/your/cloned/valbot-cli
source "$VENV_PATH/bin/activate"
export VALBOT_CLI_KEY="<YOUR VALBOT_CLI_KEY>"
python $VALBOT_PATH/app.py "$@"
deactivate
```

```bash
# Add to your .aliases file
alias valbot '~/scripts/valbot.sh'
```

---

## ValBot TUI (Terminal User Interface)

ValBot includes a beautiful, feature-rich Terminal User Interface (TUI) built with Textual, providing a modern visual experience alongside the traditional CLI mode.

### TUI Features

🎨 **Modern Material Design Interface**
- Beautiful dark theme with gradient accents
- Smooth animations and responsive layout
- Real-time message streaming with progress indicators
- Syntax-highlighted code blocks with copy buttons
- Full markdown rendering support

💬 **Interactive Chat Experience**
- Real-time AI conversation with streaming responses
- Message history with user/assistant differentiation
- Multi-line input support via system editor
- Reasoning display for GPT-5 models (when enabled)
- Visual feedback on loaded files and context

🖥️ **Integrated Terminal**
- Execute shell commands directly in the TUI
- Real-time output streaming
- Directory navigation and environment management
- Command history and error handling

📁 **File System Integration**
- Built-in file explorer with directory tree view
- Load files into conversation context with visual feedback
- File content preview with syntax highlighting
- Support for glob patterns and multiple files

⌨️ **Keyboard Shortcuts**
- `Ctrl+Q` - Quit application
- `Ctrl+C` - Clear chat history
- `Ctrl+T` - Toggle terminal panel
- `Ctrl+F` - Toggle file explorer
- `Ctrl+M` - Change AI model
- `Esc` - Cancel current operation

✅ **Full CLI Feature Parity**
- All slash commands: `/clear`, `/help`, `/model`, `/agent`, `/context`, `/file`, `/terminal`, `/multi`, `/prompts`, `/commands`, `/settings`, `/reload`, `/update`, `/add_agent`, `/add_tool`
- CommandManager integration with custom prompts and commands
- Agent system with interactive picker and descriptions
- Context management with ContextManager integration
- System prompt loading from configuration
- GPT-5 reasoning support with configurable effort levels

### Installing TUI Dependencies

The TUI requires additional Python packages. Install them with:

```bash
pip install textual textual-dev
```

Or reinstall all requirements (includes TUI libraries):

```bash
pip install -r requirements.txt
```

For Linux EC with proxy:
```bash
pip install --proxy="http://proxy-chain.intel.com:911" textual textual-dev
# Or reinstall all requirements
pip install --proxy="http://proxy-chain.intel.com:911" -r requirements.txt
```

### Running the TUI

#### Windows

Using the provided batch file:
```cmd
valbot_tui.bat
```

Using Python directly:
```cmd
python valbot_tui_launcher.py
```

With custom configuration:
```cmd
python app.py --tui --config my_config.json
```

#### Linux/macOS

Make the launcher executable and run:
```bash
chmod +x valbot_tui.sh
./valbot_tui.sh
```

Using Python directly:
```bash
python valbot_tui_launcher.py
```

With custom configuration:
```bash
python app.py --tui --config my_config.json
```

### TUI Slash Commands

All CLI commands are supported in the TUI:

**Chat Commands:**
- `/clear` - Clear conversation history
- `/new` - Start new conversation
- `/help` - Show comprehensive help
- `/quit` or `/exit` - Exit application

**Model & Agent Commands:**
- `/model` - Interactive model picker with arrow keys
- `/agent` - Interactive agent selection with descriptions

**File & Context Commands:**
- `/context <files>` - Load files into conversation (supports globs)
- `/file <path>` - Display file contents with syntax highlighting

**Terminal Commands:**
- `/terminal <command>` - Execute shell commands

**Advanced Commands:**
- `/multi` - Multi-line input via system editor
- `/prompts` - Show custom prompts
- `/commands` - Show all available commands
- `/settings` - Display settings information
- `/reload` - Reinitialize chatbot
- `/update` - Check for updates
- `/add_agent` - Add new agent from Git repo or local path
- `/add_tool` - Add new tool extension

### TUI Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ ValBot TUI                                    🕐 14:23:45        │
├─────────────────────────────────────────┬───────────────────────┤
│                                         │                       │
│         Chat Panel                      │   File Explorer       │
│   (Messages and responses)              │   (Optional panel)    │
│                                         │                       │
├─────────────────────────────────────────┴───────────────────────┤
│                                                                 │
│         Terminal Panel (Collapsible)                            │
│   (Command output and execution)                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Type your message or /command...                               │
├─────────────────────────────────────────────────────────────────┤
│ Model: gpt-4o | Session: new | Mode: chat                      │
└─────────────────────────────────────────────────────────────────┘
```

### TUI Tips

1. **Best Terminal Experience**
   - Use modern terminals: Windows Terminal, iTerm2, GNOME Terminal, or Alacritty
   - Recommended size: 120 columns × 40 rows (minimum 80×24)
   - Enable 256-color or true color support
   - Use fonts with ligatures (Fira Code, JetBrains Mono)

2. **Performance Optimization**
   - Clear chat history regularly with `/clear` or `Ctrl+C`
   - Close terminal panel when not in use with `Ctrl+T`
   - Use specific models appropriate for your needs

3. **Development Mode**
   ```bash
   # Run with hot reload
   textual run --dev valbot_tui.py
   
   # View debug console
   textual console
   ```

---

## CLI Mode (Command Line Interface)

ValBot's traditional CLI mode provides a powerful text-based REPL for AI interaction.

### CLI Chat and Prompts

ValBot provides an interactive chat interface for real-time AI interaction with rich text formatting.

#### Basic Usage

Run the ChatBot with optional initial message and context files:
```bash
python app.py -m "Hello, AI!" -c context1.txt context2.txt
```

With the helper script/alias:
```bash
valbot "How do I list all files in a directory using Linux commands?"
valbot -m "Analyze this code" -c myfile.py
```

#### Interactive Commands

The following commands are available during a chat session:

- **/prompts**: Display available prompts with their descriptions and arguments.
- **/commands**: Display available slash commands with their descriptions.
- **/clear**: Clear the conversation history.
- **/quit** (or **/exit**): Exit the chat.
- **/context**: Load more context from a file or list of files.
- **/help**: Chat with the help bot.
- **/settings**: Display and modify settings.
- **/multi**: Handle multiline input.
- **/agent**: Select an agentic flow.
- **/model**: Model picker.
- **/add_agent**: Add a new agent from a Git repo or local path (updates `~/.valbot_config.json`).
- **/add_tool**: Add a new tool extension from a Git repo or local path (updates `~/.valbot_config.json`).
- **/reload**: Restart the application to reload configuration.
- **/update**: Check for updates to the main app and installed plugins.

#### Custom Prompts

Custom prompts allow you to define reusable prompt templates for specific interactions. Define them in your `~/.valbot_config.json`:

```json
{
  "custom_prompts": [
    {
      "prompt_cmd": "jarvis",
      "description": "Reply as a Jarvis - the helpful butler.",
      "prompt": "Reply as a butler, always formal. To this request: {request}",
      "args": ["request"]
    },
    {
      "prompt_cmd": "motivate",
      "description": "Provide a motivational quote.",
      "prompt": "Share a motivational quote to inspire the user.",
      "args": []
    }
  ]
}
```

Use the `/prompts` command during a chat to see available custom prompts.

#### Command-Line Arguments for Chat

- **Positional Message**: You can provide an initial message directly as a positional argument.
- **`-m`, `--message`**: Initial message for the AI.
- **`-c`, `--context`**: Files to load context from.

#### Chat Usage Examples

- **No initial message or context**:
  ```bash
  python app.py
  # or with alias:
  valbot
  ```

- **With initial message**:
  ```bash
  python app.py -m "How do I list all files in a directory using Linux commands?"
  # or with alias:
  valbot "How do I list all files in a directory using Linux commands?"
  ```

- **With context files**:
  ```bash
  python app.py -c intro.txt history.txt
  # or with alias:
  valbot -c intro.txt history.txt
  ```

### CLI Agents

Agents are autonomous workflows that can perform complex tasks using tools and multi-step reasoning. ValBot comes with built-in agents and supports custom agent extensions.

#### What Are Agents?

Agents are AI-powered assistants that can:
- Use tools to interact with files, git, terminals, and more
- Follow multi-step workflows to accomplish complex tasks
- Ask for human input when needed
- Return structured results

Think of agents as specialized AI workers that can autonomously complete tasks, from answering questions about specifications to debugging code or editing files.

#### Running Agents

Directly invoke a specific agent on startup and pass it parameters via the command line:
```bash
python app.py -a my_agent_name -p param1=value1 param2=value2
```

Where:
- `-a`, `--agent` specifies the agent flow to invoke by name
- `-p`, `--params` lets you pass one or more parameters to the agent as key=value pairs. Each pair should be provided as an individual argument (i.e., space-separated).

You can also select an agent interactively during a chat session using the `/agent` command.

#### Agent Usage Examples

- **Invoke an agent with parameters**:
  ```bash
  python app.py -a "spec_expert" -p spec_path=specs/pmc_spec.pdf project=abcd
  # or with alias:
  valbot -a "spec_expert" -p spec_path=specs/pmc_spec.pdf project=abcd
  ```

- **Select agent interactively**:
  ```bash
  python app.py
  # Then type: /agent
  # Choose from the list of available agents
  ```

#### Built-in Agents

ValBot includes several built-in agents defined in `default_config.json`:
- **Spec Expert**: Q&A with specification documents
- **Code Expert**: Code analysis and generation
- **File Edit Agent**: Edit files with AI assistance
- **Terminal Agent**: Execute terminal commands
- **Project Agent**: Manage project-wide tasks
- And more...

Check your `\agent` command to see all available built-in agents.

---

## Creating and Adding Custom Agents

### Adding Agents Interactively with `/add_agent`

The easiest way to add agents is using the in-chat command `/add_agent`:

1. **Choose a source**: `local` or `github`.

2. **For local agents**, enter:
   - Name of the agent
   - Description (optional)
   - Path to the agent's `.py` file (will be stored as an absolute path)

3. **For GitHub agents**, enter:
   - Git repo URL (e.g., `https://github.com/org/repo.git`)
   - Valbot will clone the repo to a temporary directory and read `registry.json` at the repo root to discover available agents
   - Select the agent from the registry list
   - The selection stores `repo`, `path`, and optionally `ref` into your config
   - You can also provide a custom `install_path`

4. **Handle name conflicts**:
   - If an agent with the same name exists, choose to overwrite or enter a new name

5. **Restart to load**:
   - The entry is saved to `~/.valbot_config.json`
   - Restart Valbot to load newly added agents

### Adding Tools with `/add_tool`

You can manually add tool extensions using `/add_tool`, similar to `/add_agent`. Choose `local` or `github` source, provide the tool details, and ValBot will add it to your config. Tools are typically auto-installed when adding agents that require them.

### Getting Agents from Others

#### GitHub Repository Format

When adding from GitHub, the repository should include a `registry.json` at its root listing one or more agents:

```json
{
  "agents": [
    {
      "name": "CoolAgent",
      "path": "agents/cool_agent.py",
      "description": "Does cool things",
      "ref": "v1.2.3",
      "required_tools": ["git_tools", "file_tools"]
    }
  ],
  "tools": [
    {
      "name": "custom_tool",
      "path": "tools/custom_tool.py",
      "description": "A custom tool"
    }
  ]
}
```

Each agent must provide at least `name` and `path` (or `location`). Optional fields include `description`, `ref`, and `required_tools` (list of tool names the agent needs). Tools can also be listed in the registry and will be auto-installed when adding agents that require them.

#### How Remote Agent Installation Works

- By default, repositories are cloned into `<valbot_repo>/agent_plugins/my_agents/` under a stable, hashed subfolder name derived from `repo@ref`
- You can override the install directory by setting `install_path` in your `agent_extensions` entry (absolute path or relative to the repo root)
- On first load, Valbot prints: `Cloning plugin repo <url> into <dir>...`
- The plugin loader expects the `path` to point to a `.py` file; non-existent or non-`.py` paths are skipped with a warning

**Automatic Tool Installation:** When adding an agent, if the agent's registry entry includes a `required_tools` list, ValBot will:
- Check if each tool is already available (in `common_tools` or `tool_extensions`)
- Search the same registry for matching tool definitions
- Prompt you to automatically install any missing tools found in the registry
- This ensures agents have their dependencies without manual setup

### Adding Agents Manually

You can also add agents by editing your `~/.valbot_config.json` file directly:

```json
{
  "agent_extensions": [
    {
      "name": "My Custom Agent",
      "description": "Custom agent functionality",
      "location": "/my/drive/my_plugins/my_custom_agent.py"
    },
    {
      "name": "Cool Remote Agent",
      "description": "Agent sourced from a Git repository",
      "repo": "https://github.com/org/repo.git",
      "path": "agents/cool_agent.py",
      "ref": "v1.2.3",
      "install_path": "agent_plugins/my_agents"
    }
  ],
  "tool_extensions": [
    {
      "name": "my_custom_tool",
      "description": "Custom tool functionality",
      "location": "/my/drive/my_tools/my_custom_tool.py"
    }
  ]
}
```

- **Local agents/tools**: specify `location` as an absolute path (recommended) or a path relative to the repo root.
- **Remote agents/tools**: specify `repo` (Git URL) and `path` (path to the plugin .py inside the repo). Optional: `ref` (branch, tag, or commit) and `install_path` (where to install the repo; defaults to `agent_plugins/my_agents`).

### Creating Your Own Agents

To create your own agent, subclass `AgentPlugin` and implement the `run_agent_flow` method:

```python
from agent_plugins.agent_plugin import AgentPlugin
from pydantic_ai import Agent, RunContext
import asyncio

class MyCustomPlugin(AgentPlugin):
    # If you define REQUIRED_ARGS, these arguments will be prompted to be provided by the user
    # at the start of running the agent flow
    # If you don't need user provided info at the start of the agent flow, you can omit defining REQUIRED_ARGS
    # All REQUIRED_ARGS can be provided with --params <arg_name>=<value> <arg2>=<val2> when starting the agent with --agent
    REQUIRED_ARGS = {
        # 'arg_name': (AgentPlugin.<prompt_method>, '<prompt string>')
        # See agent_plugin.py for already defined prompt_methods, or create your own
        'users_name': (AgentPlugin.simple_prompt, 'Hello what is your name?')
     }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Usually you would create your agent(s) here and provide them with tools:
        self.my_agent = self.create_address_agent(model)
        self.apply_tools()

    def create_address_agent(self, model):
      return Agent(
          model=model,
          system_prompt=(
              'You are responsible for getting a users address. '
              'You must confirm with the user that their address is correct after you fetch it'
          ),
          result_type=str
      )

    def apply_tools(self):
      # This is where you give your agent access to tools to perform its tasks
      # There are many predefined tools for your agent to utilize in agent_plugins/common_tools/
      # You just need to import them and give them to your agent
      from agent_plugins.common_tools.ask_human import ask_human
      self.my_agent.tool(ask_human)

      # Or you can make your own custom tools to provide your agent
      self.my_agent.tool(self.lookup_address)

    # There are many ways to define tools and many constraints you can impose like
    # custom schemas and structured inputs.
    # Read more advanced tool definitions here: https://ai.pydantic.dev/tools/#function-tools-and-schema
    # At a minimum, tools need to use python typing and provide docstrings,
    # so the agent knows what the tool does and how to use it.
    def lookup_address(self, ctx: RunContext, person_name: str) -> str:
      """Looks up a persons address given their name"""
      # This is where you'd implement your tools logic, but for simplicity everyone lives here:
      return "123 E. Main Street, Phoenix Arizona, 12345"

    def run_agent_flow(self, context, **kwargs):
        # Implement custom behavior here, this is the entry point when an agent flow is started
        users_name = self.initializer_args['users_name'] # All REQUIRED_ARGS will be captured in self.initializer_args
        result = asyncio.run(self.my_agent.run(
          f"The User said their name was: {users_name}"
        ))
        print(f"Users address: {result.data}")
```

Add the plugin to `.valbot_config.json`:

```json
{
  "agent_extensions": [
    {
      "name": "My Custom Plugin",
      "description": "Custom functionality",
      "location": "/my/drive/my_plugins/my_custom_plugin.py"
    }
  ]
}
```

#### Available Tools for Agents

ValBot provides several common tools in `agent_plugins/common_tools/`:
- **ask_human**: Ask the user for input
- **file_tools**: Read, write, and manipulate files
- **git_tools**: Git operations
- **github_tools**: GitHub API operations
- **terminal_tools**: Execute terminal commands
- **hsd_tools**: Intel HSD-specific tools

You can import and use these in your custom agents, or create your own tools following the Pydantic AI tool format. You can also add **tool extensions** from repositories or local files using the `/add_tool` command.

### Custom Agent Commands

You can create custom commands that invoke specific agents with predefined arguments:

```json
{
  "custom_agent_commands": [
    {
      "command": "/pmc_expert",
      "uses_agent": "Spec Expert",
      "description": "Ask questions about PMC specs.",
      "agent_args": {
        "spec_path": "path/to/the/pmc/specs"
      }
    },
    {
      "command": "/my_command",
      "uses_agent": "My Custom Agent",
      "description": "Run my custom command.",
      "agent_args": {
        "arg1": "value1",
        "arg2": "value2"
      }
    }
  ]
}
```

---

## Keeping ValBot Updated

ValBot includes a built-in update system accessible via the `/update` command during a chat session.

### What Gets Updated

- **Main Application**: Checks for new tagged releases and updates the ValBot codebase
- **Plugins**: Checks for updates to agent plugins installed from Git repositories (branch-based only, not pinned commits)

### Update Process

When you run `/update`:

1. ValBot checks for available updates to both the main app and installed plugins
2. Shows current and latest versions/commits
3. Warns if dependencies have changed
4. Prompts you to select which updates to install

**Note**: If multiple plugins share the same Git repository, updating one will update all plugins from that repo. ValBot will prompt you to either update them together or pin the others to their current commit.

### After Updating

- **Running from source**: ValBot reloads automatically. If dependencies changed, run `pip install -r requirements.txt`
- **Running from .exe**: Follow the displayed instructions to rebuild the executable using `setup.bat`

---

## Advanced Configuration

### Configuration Files

ValBotTerminal uses two main configuration files:
- **`default_config.json`**: Contains built-in agents (under `default_agents`), built-in commands, and custom prompts provided by the application. This file is part of the ValBot installation.
- **`~/.valbot_config.json`**: Located in the user's home directory, this file allows users to override settings (including model and endpoint), manage agent extensions, and add custom commands and prompts.

The application merges `default_config.json` and `.valbot_config.json`, with `.valbot_config.json` taking precedence for customizations.

### Environment Variables

Configure these environment variables for authentication:

- **`VALBOT_CLI_KEY`**: Your API key for the Valbot proxy endpoint (recommended).
- ~~**`CLIENT_ID`**: Your client ID for authentication~~ (deprecated IGPT endpoint).
- ~~**`CLIENT_SECRET`**: Your client secret for authentication~~ (deprecated IGPT endpoint).

These must be set in your environment to authenticate with the Intel API. If you utilize the helper scripts/alias this is handled automatically for you each time you invoke `valbot`

### Model and Endpoint Configuration

Configure your default models and endpoints in `~/.valbot_config.json`:

```json
{
  "agent_model_config": {
    "default_endpoint": "valbot-proxy",
    "default_model": "gpt-4.1",
    "small_model": "gpt-4.1-mini"
  },
  "chat_model_config": {
    "default_endpoint": "valbot-proxy",
    "default_model": "gpt-5"
  }
}
```

#### Available Endpoints

1. **valbot-proxy** (Recommended)
   - Models: `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `gpt-oss:20b`
   - Requires: `VALBOT_CLI_KEY` environment variable
   - Get key at: https://genai-proxy.intel.com/

2. **azure** (To Be Documented)
   - Configuration details coming soon

3. **igpt** (Deprecated)
   - Model: `gpt-4o`
   - Requires: `CLIENT_ID` and `CLIENT_SECRET` environment variables

### Agent Extensions Configuration

Configure custom agents in `~/.valbot_config.json`:

```json
{
  "agent_extensions": [
    {
      "name": "My Custom Agent",
      "description": "Custom agent functionality",
      "location": "/my/drive/my_plugins/my_custom_agent.py"
    },
    {
      "name": "Cool Remote Agent",
      "description": "Agent sourced from a Git repository",
      "repo": "https://github.com/org/repo.git",
      "path": "agents/cool_agent.py",
      "ref": "v1.2.3",
      "install_path": "agent_plugins/my_agents"
    }
  ],
  "tool_extensions": [
    {
      "name": "my_custom_tool",
      "description": "Custom tool functionality",
      "location": "/my/drive/my_tools/my_custom_tool.py"
    }
  ]
}
```

**Configuration Options:**
- **For local agents/tools:**
  - `name`: Display name for the agent or tool
  - `description`: Brief description of what the agent/tool does
  - `location`: Absolute path (recommended) or relative path to the `.py` file

- **For remote (Git) agents/tools:**
  - `name`: Display name for the agent or tool
  - `description`: Brief description of what the agent/tool does
  - `repo`: Git repository URL
  - `path`: Path to the agent/tool `.py` file within the repo
  - `ref` (optional): Branch, tag, or commit to checkout (defaults to default branch)
  - `install_path` (optional): Where to clone the repo (defaults to `agent_plugins/my_agents`)

### Custom Agent Commands Configuration

Define shortcuts that invoke agents with preset arguments:

```json
{
  "custom_agent_commands": [
    {
      "command": "/pmc_expert",
      "uses_agent": "Spec Expert",
      "description": "Ask questions about PMC specs.",
      "agent_args": {
        "spec_path": "path/to/the/pmc/specs",
        "project": "pmc"
      }
    }
  ]
}
```

**Configuration Options:**
- `command`: The command name (must start with `/`)
- `uses_agent`: Name of the agent to invoke (must match an agent name)
- `description`: Brief description shown in help
- `agent_args`: Dictionary of arguments to pass to the agent

### Custom Prompts Configuration

Define reusable prompt templates:

```json
{
  "custom_prompts": [
    {
      "prompt_cmd": "jarvis",
      "description": "Reply as a Jarvis - the helpful butler.",
      "prompt": "Reply as a butler, always formal. To this request: {request}",
      "args": ["request"]
    },
    {
      "prompt_cmd": "motivate",
      "description": "Provide a motivational quote.",
      "prompt": "Share a motivational quote to inspire the user.",
      "args": []
    },
    {
      "prompt_cmd": "explain",
      "description": "Explain code in simple terms",
      "prompt": "Explain the following {language} code in simple terms, suitable for a beginner: {code}",
      "args": ["language", "code"]
    }
  ]
}
```

**Configuration Options:**
- `prompt_cmd`: The command name to invoke this prompt
- `description`: Brief description shown in the `/prompts` list
- `prompt`: The prompt template with `{arg}` placeholders
- `args`: List of argument names that match the placeholders in the prompt

### Complete Configuration Example

<details>
<summary> Expand for a comprehensive example of `~/.valbot_config.json`</summary>

```json
{
  "agent_model_config": {
    "default_endpoint": "valbot-proxy",
    "default_model": "gpt-4.1",
    "small_model": "gpt-4.1-mini"
  },
  "chat_model_config": {
    "default_endpoint": "valbot-proxy",
    "default_model": "gpt-5"
  },
  "agent_extensions": [
    {
      "name": "File Edit Agent",
      "description": "Edit files with AI assistance",
      "location": "/home/user/my_agents/file_edit_agent.py"
    },
    {
      "name": "Cool Remote Agent",
      "description": "Agent sourced from a Git repository",
      "repo": "https://github.com/org/repo.git",
      "path": "agents/cool_agent.py",
      "ref": "v1.2.3",
      "install_path": "agent_plugins/my_agents"
    }
  ],
  "tool_extensions": [
    {
      "name": "my_custom_tool",
      "description": "Custom tool functionality",
      "location": "/home/user/my_tools/my_custom_tool.py"
    },
    {
      "name": "remote_tool",
      "description": "Tool from a Git repository",
      "repo": "https://github.com/org/repo.git",
      "path": "tools/remote_tool.py",
      "ref": "v1.2.3"
    }
  ],
  "custom_agent_commands": [
    {
      "command": "/pmc_expert",
      "uses_agent": "Spec Expert",
      "description": "Ask questions about PMC specs.",
      "agent_args": {
        "spec_path": "/path/to/pmc/specs",
        "project": "pmc"
      }
    },
    {
      "command": "/edit_code",
      "uses_agent": "File Edit Agent",
      "description": "Edit code files with AI assistance.",
      "agent_args": {
        "mode": "interactive"
      }
    }
  ],
  "custom_prompts": [
    {
      "prompt_cmd": "jarvis",
      "description": "Reply as a Jarvis - the helpful butler.",
      "prompt": "Reply as a butler, always formal. To this request: {request}",
      "args": ["request"]
    },
    {
      "prompt_cmd": "motivate",
      "description": "Provide a motivational quote.",
      "prompt": "Share a motivational quote to inspire the user.",
      "args": []
    },
    {
      "prompt_cmd": "explain",
      "description": "Explain code in simple terms",
      "prompt": "Explain the following {language} code in simple terms: {code}",
      "args": ["language", "code"]
    }
  ]
}
```
</details>

### Plugin System Details

The `PluginManager` handles loading and executing plugins based on the configuration files. It supports running custom commands, agent flows, and prompts.

**How remote agent loading works:**
- If an agent entry includes `repo` and `path`, Valbot clones the repository (once) into the install root (default `agent_plugins/my_agents`)
- A stable directory name is used based on the repo URL and ref
- If `ref` is provided, that branch/tag/commit is checked out after cloning
- The final plugin file path is resolved as `<install_root>/<repo_dir>/<path>`
- If an entry instead includes `location`, it is used directly (relative paths are resolved against the repository root of this application)

### Command-Line Arguments Reference

Complete list of command-line arguments:

- **Positional Message**: Directly provide an initial message when running the application
  ```bash
  python app.py "Your message here"
  ```

- **`-m`, `--message`**: Specify an initial message for the AI interaction
  ```bash
  python app.py -m "Analyze this code"
  ```

- **`-c`, `--context`**: Load context from one or more files
  ```bash
  python app.py -c file1.txt file2.py
  ```

- **`-a`, `--agent`**: Specify an agent to invoke at startup
  ```bash
  python app.py -a "spec_expert"
  ```

- **`-p`, `--params`**: Pass key=value formatted parameters to your agent
  ```bash
  python app.py -a "spec_expert" -p spec_path=/path/to/spec project=myproject
  ```

- **`--config`**: Specify a custom configuration file (Windows only with valbot.bat)
  ```bash
  valbot.bat --config custom_config.json "Your message"
  ```

### Troubleshooting

**Agent not found:**
- Ensure the agent is defined in either `default_config.json` or `~/.valbot_config.json`
- Check that the agent name matches exactly (case-sensitive)
- For remote agents, verify the repository was cloned successfully

**Authentication errors:**
- Verify `VALBOT_CLI_KEY` is set correctly in your environment
- Check that your API key is valid at https://genai-proxy.intel.com/
- For IGPT (deprecated), ensure both `CLIENT_ID` and `CLIENT_SECRET` are set

**Plugin loading errors:**
- Check that the plugin file path exists and is accessible
- Ensure the plugin file has a `.py` extension
- Verify the plugin class inherits from `AgentPlugin`
- Check console output for specific error messages
