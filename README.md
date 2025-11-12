# ValBot TUI

## Overview

ValBot TUI is a powerful, extensible AI assistant with a beautiful Terminal User Interface built with Textual. ValBot combines conversational AI with autonomous agent workflows to help you code, debug, analyze specifications, and automate complex tasks—all through an intuitive visual interface.

**Key Capabilities:**

- **🎨 Modern Material Design Interface**: Beautiful dark theme with gradient accents, real-time streaming, syntax highlighting, integrated terminal, file explorer, and full markdown rendering
- **💬 Interactive Chat**: Real-time AI conversation with streaming responses, context-aware replies, multiline input, and rich formatting
- **🤖 Autonomous Agents**: Pre-built and custom agents that can execute multi-step workflows, use tools, and interact with files, git, terminals, and more
- **🔌 Extensible Plugin System**: Add custom agents from local files or Git repositories, create your own agents, and share them with others
- **⚙️ Flexible Configuration**: Customize models, prompts, commands, and agent behaviors to match your workflow
- **🔄 Built-in Updates**: Keep ValBot and installed plugins up to date with the `/update` command
- **📝 Context Management**: Load conversation context from files to give the AI deep understanding of your codebase
- **⌨️ Keyboard-Driven Workflow**: Extensive keyboard shortcuts and command palette for efficient navigation

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Automated Setup (Recommended)](#automated-setup-recommended)
  - [Alias Setup for Reusability & Team Use](#alias-setup-for-reusability--team-use)
  - [Running the TUI](#running-the-tui)
  - [Manual Installation](#manual-installation)
  - [Python Virtual Environment Setup](#python-virtual-environment-setup-recommended)
  - [Configuration](#configuration)
- [ValBot TUI - Your AI Assistant Interface](#valbot-tui---your-ai-assistant-interface)
  - [Why Use the TUI?](#why-use-the-tui)
  - [TUI Features](#tui-features)
  - [TUI Slash Commands](#tui-slash-commands)
  - [TUI Usage Examples](#tui-usage-examples)
  - [TUI Tips & Best Practices](#tui-tips--best-practices)
  - [Troubleshooting TUI](#troubleshooting-tui)
- [CLI Mode (Alternative Text Interface)](#cli-mode-alternative-text-interface)
- [Agents - Autonomous AI Workflows](#agents---autonomous-ai-workflows)
  - [Adding Agents Interactively](#adding-agents-interactively-with-add_agent)
  - [Adding Tools](#adding-tools-with-add_tool)
  - [Getting Agents from Others](#getting-agents-from-others)
  - [Adding Agents Manually](#adding-agents-manually)
  - [Creating Your Own Agents](#creating-your-own-agents)
  - [Custom Agent Commands](#custom-agent-commands)
- [Keeping ValBot Updated](#keeping-valbot-updated)
- [Advanced Configuration](#advanced-configuration)
  - [Configuration Files](#configuration-files)
  - [Environment Variables](#environment-variables)
  - [Model and Endpoint Configuration](#model-and-endpoint-configuration)
  - [Agent Extensions Configuration](#agent-extensions-configuration)
  - [Custom Agent Commands Configuration](#custom-agent-commands-configuration)
  - [Custom Prompts Configuration](#custom-prompts-configuration)
  - [Command-Line Arguments Reference](#command-line-arguments-reference)
  - [Troubleshooting](#troubleshooting)

---

## Getting Started

### Requirements

- Python 3.10+ (Python 3.11+ recommended)
- Modern terminal emulator (Windows Terminal, iTerm2, GNOME Terminal, or Alacritty recommended)
- Terminal with 256-color or true color support
- API key from endpoint provider (see [Configuration](#configuration) for setup instructions)

### Automated Setup (Recommended)

For the easiest setup experience with automatic virtual environment creation, dependency installation, and configuration, use our automated setup scripts:

#### Intel EC Linux: ec_linux_setup.sh

```bash
git clone https://github.com/lingaisw/valbot-tui valbot-tui
cd valbot-tui
chmod +x ec_linux_setup.sh
./ec_linux_setup.sh
```

This will:
- Let you select Python interpreter (auto-detects EC default paths)
- Interactively create or use an existing virtual environment with flexible path selection
- Option to use existing venv, delete and recreate, or choose a different path
- Validate existing virtual environments before use
- Install dependencies with proxy support
- Detect and optionally reuse existing API key from .env file
- Generate launcher script (valbot_tui.sh that launches the TUI)
- Set up aliases for convenient access to ValBot TUI

#### Windows: setup.bat

```bat
git clone https://github.com/lingaisw/valbot-tui valbot-tui
cd valbot-tui
.\setup.bat
```

This will:
- Interactively create or use an existing virtual environment with flexible path selection
- Option to use existing venv, delete and recreate, or choose a different path
- Validate existing virtual environments before use
- Install all dependencies with optional proxy support
- Detect and optionally reuse existing API key from .env file
- Prompt for your API key if not already configured
- Build a standalone executable (optional)
- Set up the `valbot` alias for easy access

---

### Alias Setup for Reusability & Team Use

The automated setup scripts configure this automatically, enabling easy reusability and team-wide deployment. Once set up, the same ValBot installation can be used by your entire team with personalized settings per user.

**Team Benefits:**
- **Single Installation**: One ValBot installation can serve multiple team members
- **Personal Profiles**: Each user maintains their own configuration in `~/.valbot_config.json`
- **Theme Persistence**: UI themes automatically save to each user's profile (`~/.valbot_theme.json`)
- **Shared Agents**: Teams can share custom agents while maintaining individual preferences
- **Easy Onboarding**: New team members just need to run the alias setup script

#### Quick Team Setup (Linux/macOS)

For team members joining an existing ValBot installation:

```bash
# Navigate to the shared ValBot installation
cd /path/to/valbot-cli-main

# Run the alias setup script
source valbot_tui_alias_setup.sh

# Open a new terminal and run
valbot
```

This script will:
- Add the `valbot` alias to your `~/.aliases` file
- Handle existing alias conflicts automatically
- Provide clear instructions for activation

After setup, simply run `valbot` from anywhere to launch the TUI!

#### Manual Alias Setup

To set up the alias manually:

#### Quick Setup (once per terminal)

Use these commands for temporary access in your current terminal session:

**EC Linux/Linux/macOS (Bash):**
```bash
alias valbot="/path/to/valbot-cli-main/valbot_tui.sh"
```

**Windows (PowerShell):**
```powershell
function valbot { & "C:\path\to\valbot-cli-main\valbot_tui.bat" $args }
```

**Windows (Command Prompt):**
```cmd
doskey valbot="C:\path\to\valbot-cli-main\valbot_tui.bat" $*
```

#### Permanent Setup (once per session)

For persistent access across all terminal sessions:

**EC Linux/Linux (Bash):**

Add to your `~/.bashrc` or `~/.bash_profile`:
```bash
# Add this line (replace with your actual path):
alias valbot="/path/to/valbot-cli-main/valbot_tui.sh"
```

Then reload your shell configuration:
```bash
source ~/.bashrc
```

**EC Linux (tcsh/csh):**

Add to your `~/.tcshrc` or `~/.cshrc`:
```tcsh
# Add this line (replace with your actual path):
alias valbot "/path/to/valbot-cli-main/valbot_tui.sh"
```

Then reload your shell configuration:
```tcsh
source ~/.tcshrc
```

**macOS (Bash):**

Add to your `~/.bash_profile`:
```bash
# Add this line (replace with your actual path):
alias valbot="/path/to/valbot-cli-main/valbot_tui.sh"
```

Then reload your shell configuration:
```bash
source ~/.bash_profile
```

**macOS (Zsh):**

Add to your `~/.zshrc`:
```bash
# Add this line (replace with your actual path):
alias valbot="/path/to/valbot-cli-main/valbot_tui.sh"
```

Then reload your shell configuration:
```bash
source ~/.zshrc
```

**Windows (PowerShell):**

Add to your PowerShell profile (`$PROFILE`):
```powershell
# Open your profile for editing
notepad $PROFILE

# Add this line (replace with your actual path):
function valbot { & "C:\path\to\valbot-cli-main\valbot_tui.bat" $args }
```

After saving, reload your profile:
```powershell
. $PROFILE
```

**Windows (Command Prompt):**

Create a batch file named `valbot.bat` in a directory that's in your PATH (e.g., `C:\Windows\System32` or a custom bin directory):
```cmd
@echo off
"C:\path\to\valbot-cli-main\valbot_tui.bat" %*
```

**Note:** If you used the automated setup scripts (`setup.bat` or `ec_linux_setup.sh`), the alias should already be configured for you.

**User Profile Files:**
- `~/.valbot_config.json` - Personal configuration (models, agents, API keys)
- `~/.valbot_theme.json` - UI theme preferences (auto-saved on theme change)
- `.env` - API credentials (in ValBot installation directory)

This architecture allows teams to share a single ValBot installation while each member maintains their own personalized settings and theme preferences.

---

### Running the TUI

Once setup is complete, launch ValBot with:

```bash
valbot
```

**Alternative launch methods:**

**EC Linux/Linux/macOS:**

Using the launcher script:
```bash
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

**Windows:**

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

**Note:** The launcher scripts automatically detect virtual environments in common locations (`venv`, `valbot-venv`, `.venv`). If a virtual environment is found, it will be activated automatically.

---

### Manual Installation

If you prefer manual setup or need more control over the installation process:

**1. Clone the repository:**
```bash
git clone https://github.com/lingaisw/valbot-tui valbot-tui
cd valbot-tui
```

**2. Install dependencies:**

For EC Linux with proxy:
```bash
pip install --proxy="http://proxy-chain.intel.com:911" -r requirements.txt
```

For Linux/macOS:
```bash
pip install -r requirements.txt
```

For Windows:
```bash
pip install -r requirements.txt
```

**3. Configure your API key:**

For EC Linux/Linux/macOS, set it in your environment:
```bash
export VALBOT_CLI_KEY="your_api_key_here"
```

Or create a `.env` file in the repository root:
```bash
VALBOT_CLI_KEY=your_api_key_here
```

For Windows, create a `.env` file in the repository root:
```bash
VALBOT_CLI_KEY=your_api_key_here
```

Get your API key at: https://genai-proxy.intel.com/ → "Manage Your API Tokens" → "Create New Token"

**4. Set up the valbot alias** (see [Setting Up the valbot Alias](#setting-up-the-valbot-alias))

**5. Launch ValBot TUI:**

```bash
valbot
```

Or use the platform-specific scripts (see [Running the TUI](#running-the-tui))

---

### Python Virtual Environment Setup (Recommended)

Using a virtual environment prevents package conflicts:

**Create and activate virtual environment:**

**Intel EC Linux (tcsh):**
```tcsh
setenv VENV_PATH /path/to/create/your/valbot-venv
/usr/intel/pkgs/python3/3.11.1/bin/python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate.csh
```

**Intel EC Linux/Linux (Bash):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

#### API Key Setup

Get your API key from https://genai-proxy.intel.com/:
1. Login with your Intel credentials
2. Click "Manage Your API Tokens"
3. Create New Token
4. Copy the token

Set the key in your environment or `.env` file:
```bash
VALBOT_CLI_KEY=your_api_key_here
```

#### Model Configuration (Optional)

Configure your preferred models in `~/.valbot_config.json`:

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

**Available Models:**
- `gpt-5` - Latest reasoning model (recommended for chat)
- `gpt-4.1` - Powerful general model (recommended for agents)
- `gpt-4.1-mini` - Faster, lightweight model
- `gpt-5-mini` - Compact reasoning model
- `gpt-oss:20b` - Open source model option

---

## ValBot TUI - Your AI Assistant Interface

ValBot's primary interface is a beautiful, feature-rich Terminal User Interface (TUI) built with Textual. The TUI provides a modern visual experience with real-time streaming, integrated tools, and an intuitive layout—making it the best way to interact with ValBot.

### Why Use the TUI?

- **Visual and Intuitive**: Beautiful Material Design interface with real-time message streaming, clear message organization, and instant visual feedback
- **Intelligent Autocomplete**: Smart command and file path autocomplete with fuzzy matching—type `/` for command suggestions or start typing file paths for automatic completion
- **Integrated Tools**: Built-in terminal panel and interactive file explorer eliminate context switching—execute commands and browse files without leaving the chat
- **Enhanced Productivity**: Extensive keyboard shortcuts, multi-line input support, command history, and session management streamline your workflow
- **Superior Code Experience**: Syntax-highlighted code blocks with one-click copy buttons, full markdown rendering with tables/lists/quotes, and collapsible sections
- **Customizable Interface**: Multiple theme support with Material Design colors, adjustable panel layouts, and personalized keyboard bindings
- **Real-Time Streaming**: Watch AI responses appear token-by-token with progress indicators, reasoning display for GPT-5 models, and live agent status updates
- **Same Power**: Full feature parity with CLI mode—all commands, agents, and capabilities available with better visualization

### TUI Features

🎨 **Modern Material Design Interface**
- Beautiful dark theme with gradient accents (ValBot Dark theme built-in)
- Customizable color themes via Textual theme system
- **Auto-saving themes**: Your theme selection automatically saves to user's profile at `~/.valbot_theme.json` and loads on startup
- Smooth animations and responsive layout that adapts to terminal size
- Real-time message streaming with token-by-token display
- Syntax-highlighted code blocks with one-click copy buttons
- Full markdown rendering with tables, lists, quotes, emphasis, and links
- Collapsible sections for better organization of long responses
- Clean, distraction-free interface with optional panel toggling

💬 **Interactive Chat Experience**
- **Real-time streaming**: Watch AI responses appear character-by-character as they're generated
- **Message organization**: Clear visual differentiation between user and assistant messages with timestamps
- **Multi-line input**: Built-in TextArea widget with auto-expanding height (3-10 lines)
  - **Smart enter behavior**: Enter submits, Shift+Enter or ↓ for new lines
  - **Auto-height adjustment**: Input box grows/shrinks based on content (3-10 lines)
  - **Busy state protection**: Blocks submissions while AI is processing
- **Command autocomplete**: Type `/` to see smart suggestions for all available commands with descriptions
  - **Live filtering**: Suggestions update as you type
  - **Keyboard navigation**: Use ↑↓ arrows or Tab to navigate, Enter to select
  - **Context-aware**: Shows relevant commands based on your input
- **File path autocomplete**: Automatic path completion with fuzzy matching as you type file paths
  - **Tab-triggered**: Press Tab on any word to get path suggestions
  - **Smart detection**: Recognizes file path patterns automatically
  - **Real-time filtering**: Updates suggestions as you continue typing
  - **Works in all commands**: Autocomplete for `/context`, `/file`, and any path parameter
- **Reasoning display**: Shows GPT-5 thinking process in dedicated panel when `display_reasoning` enabled
- **Visual feedback**: Loading animations, progress indicators, and status messages for all operations
- **Configurable reasoning**: Set effort levels (low/medium/high) for AI reasoning depth
- **Response streaming**: Proper event handling with ResponseAudioDeltaEvent and ResponseTextDeltaEvent
- **Session management**: Save and load chat sessions (in development)
- **Message history**: Scroll through complete conversation history with syntax preservation
- **Notification system**: Toast notifications for errors, warnings, and important events

🖥️ **Integrated Terminal Panel**
- **Execute commands directly**: Run any shell command with `/terminal` without leaving the chat
- **Real-time output streaming**: See command output appear instantly as it's generated
- **Dedicated terminal panel**: Separate, toggleable panel that doesn't interfere with chat
- **Directory navigation**: Full support for `cd` and environment changes that persist
- **Error handling**: Clear display of stdout, stderr, and return codes
- **Command history**: Access previously executed terminal commands
- **Toggle with hotkey**: Press `Ctrl+T` to show/hide terminal panel and maximize workspace
- **Background execution**: Terminal panel updates while you continue chatting

📁 **File System Integration**
- **Interactive file explorer**: Built-in DirectoryTree widget with expandable folders
- **Full navigation controls**: Back/forward history, up directory, refresh, and address bar
  - **Browser-style navigation**: Navigate with ← Back, → Forward, ↑ Up buttons
  - **Address bar**: Type or paste paths directly for instant navigation
  - **Keyboard shortcuts**: Use arrow keys to browse folders
  - **Smart path detection**: Auto-complete and navigate as you type in address bar
  - **History tracking**: Full back/forward navigation history like a web browser
- **File path autocomplete**: Smart autocomplete with fuzzy matching for file paths in commands
  - Press `Tab` to trigger path suggestions for current word
  - Works with relative and absolute paths
  - Automatically filters as you type
- **Direct file loading**: Click any file in explorer to load into conversation context
- **Glob pattern support**: Use patterns like `src/**/*.py` to load multiple files at once
- **Visual feedback**: See loaded files confirmed in chat with file count and syntax preview
- **Syntax highlighting**: Automatic language detection and highlighting for file content
- **File content preview**: View file contents in formatted panels before loading
- **Toggle with hotkey**: Press `Ctrl+F` to show/hide file explorer panel
- **Current directory aware**: File explorer starts at your working directory
- **Large file handling**: Intelligent chunked reading with size limits (100MB default)
  - Prevents memory issues with large files
  - Shows file size warnings before loading
  - Reads files in 8KB chunks for efficiency

⌨️ **Keyboard Shortcuts & Navigation**
- **`Ctrl+Q`** - Quit application gracefully
- **`Ctrl+C` or `Ctrl+N`** - Clear chat history and start new conversation
- **`Ctrl+T`** - Toggle terminal panel visibility
- **`Ctrl+F`** - Toggle file explorer panel visibility
- **`Ctrl+M`** - Open interactive model picker with arrow key navigation
- **`Ctrl+A`** - Open interactive agent picker with descriptions
- **`Ctrl+S`** - Save current session (when implemented)
- **`Esc`** - Cancel current operation, close pickers, or dismiss overlays
- **`Enter`** - Submit message or confirm selection
- **Arrow keys** - Navigate through autocomplete suggestions and pickers
- **Tab** - Accept autocomplete suggestion (when available)

🎯 **Smart Autocomplete System**
- **Command autocomplete**: Type `/` to see all available commands with live filtering
- **File path autocomplete**: Smart completion for file paths with fuzzy matching
  - Works in `/context`, `/file`, and any command accepting file paths
  - Shows up to 20 matching files as you type
  - Supports relative and absolute paths
  - Filters based on partial matches
- **Fuzzy matching**: Find commands even with typos or partial names
- **Context-aware**: Autocomplete adapts based on what you're typing
- **Visual overlay**: Floating autocomplete panel with syntax highlighting
- **Keyboard navigation**: Use arrow keys to select, Enter to accept, Esc to dismiss

✅ **Complete Feature Parity with CLI**
- **All slash commands**: `/clear`, `/new`, `/help`, `/quit`, `/model`, `/agent`, `/context`, `/file`, `/terminal`, `/multi`, `/prompts`, `/commands`, `/settings`, `/reload`, `/update`, `/add_agent`, `/add_tool`
- **CommandManager integration**: Full support for custom prompts with argument parsing and validation
- **Custom commands**: From agent plugins with automatic delegation to plugin manager
- **Agent system**: Interactive agent picker with arrow navigation, full descriptions, and real-time workflow execution
- **Context management**: Complete ContextManager integration with visual confirmation and file previews
- **System prompt**: Automatically loads `system_prompt` from config on startup for consistent behavior
- **GPT-5 reasoning**: Displays thinking process with configurable effort levels in dedicated panel
- **Tool integration**: All common tools (ask_human, file_tools, git_tools, github_tools, terminal_tools, hsd_tools)
- **Rich text output**: Handles Rich markup for colored output from agents and tools
- **Code block copy**: One-click copy buttons on all code blocks with success feedback

🎨 **Advanced UI Features**
- **Copy buttons**: Hover over any code block to see copy button with click feedback
- **Syntax highlighting**: Automatic language detection for 50+ programming languages
- **Markdown tables**: Full support for tables with proper formatting and borders
- **Hybrid text output**: Handles both Rich markup (colored/styled output from agents) and standard markdown formats seamlessly
- **Live updates**: UI updates in real-time as AI generates responses
- **Panel management**: Resize, show/hide panels without losing state
- **Scroll preservation**: Smart scrolling keeps new content visible while allowing manual scroll
- **Error display**: Beautiful error messages with clear formatting and context
- **Welcome message**: Informative welcome screen with quick start tips
- **Status bar**: Live status indicators showing model, session state, and connection status
- **Loading animations**: Smooth loading indicators with animated messages

### TUI Slash Commands

All CLI commands are fully supported in the TUI with enhanced visual feedback and intelligent autocomplete:

**🔍 Smart Autocomplete Features:**
- **Command autocomplete**: Type `/` to see all available commands with live filtering
- **File path autocomplete**: Start typing file paths and see smart suggestions appear automatically
  - Works in `/context`, `/file`, and any command that accepts paths
  - Fuzzy matching finds files even with partial names
  - Shows up to 20 matching results with full paths
  - Use arrow keys to navigate, Enter to select, Esc to dismiss
- **Interactive pickers**: For `/model` and `/agent`, use arrow keys to browse options visually
- **Tab completion**: Press Tab to accept the top suggestion (when available)

**Chat Commands:**
- `/clear` or `/new` - Clear conversation history and start fresh
  - Keyboard shortcut: `Ctrl+C` or `Ctrl+N`
  - Confirms before clearing to prevent accidental loss
- `/help` - Show comprehensive help with all available commands
  - Displays command list with descriptions and usage examples
- `/quit` or `/exit` - Exit the application gracefully
  - Keyboard shortcut: `Ctrl+Q`

**Model & Agent Commands:**
- `/model` - Interactive model picker with arrow key navigation
  - Keyboard shortcut: `Ctrl+M`
  - Shows available models from your configured endpoints
  - Displays current model with visual indicator
  - Arrow keys to navigate, Enter to select
  - Visual confirmation of model change with success message
- `/agent` - Interactive agent selection with descriptions
  - Keyboard shortcut: `Ctrl+A`
  - Browse all available agents with full descriptions
  - Shows agent names, descriptions, and required arguments
  - Execute agent workflows with full context and real-time status
  - Error handling with clear error messages and retry options
  - Watch agent progress with live updates

**File & Context Commands:**
- `/context <files>` - Load files into conversation context with autocomplete
  - **File path autocomplete**: Start typing and see suggestions appear
  - Supports single files: `/context myfile.py`
  - Supports glob patterns: `/context src/**/*.py` or `/context *.{js,ts}`
  - Supports multiple files: `/context file1.py file2.js README.md`
  - Visual feedback showing loaded files with count and size
  - File content preview with syntax highlighting
  - Shows which files were successfully loaded vs skipped
- `/file <path>` - Display file contents with syntax highlighting and autocomplete
  - **File path autocomplete** helps you find files quickly
  - Shows file in a beautifully formatted panel
  - Automatic language detection for 50+ languages
  - Syntax highlighting for better readability
  - Line numbers and file metadata displayed

**Terminal Commands:**
- `/terminal <command>` - Execute shell commands in integrated terminal panel
  - Real-time output streaming with instant feedback
  - Support for any shell command (bash, PowerShell, cmd)
  - Error output clearly displayed with color coding
  - Panel toggles automatically when command is run
  - Toggle panel visibility with `Ctrl+T`
  - Supports directory changes (`cd`) that persist
  - Environment variables maintained across commands
  - Command history accessible via terminal panel

**Advanced Commands:**
- `/multi` - Multi-line input via system editor
  - Opens your default system editor (vim, nano, notepad, VS Code, etc.)
  - Allows composing complex multi-line prompts with full formatting
  - Preserves indentation, code blocks, and markdown
  - Automatically detects default editor from environment
  - Falls back to TextArea for long input if editor fails
- `/prompts` - Show custom prompts from config
  - Lists all available custom prompts with descriptions
  - Shows required arguments for each prompt
  - Invoke with `/prompt_name [args]`
  - Example: `/jarvis "Hello there"`
  - Autocomplete suggests available custom prompts
- `/commands` - Show all available commands
  - Lists slash commands, custom commands, and agent commands
  - Shows descriptions, usage, and examples
  - Includes both built-in and plugin commands
  - Organized by category for easy navigation
- `/settings` - Display current settings and configuration
  - Shows model configuration with endpoint details
  - Displays endpoint information and API status
  - Shows system prompt status and length
  - Lists loaded plugins and extensions
  - Configuration file paths displayed
- `/reload` - Reinitialize chatbot and reload configuration
  - Reloads configuration from default_config.json and ~/.valbot_config.json
  - Refreshes agent and tool plugins
  - Resets conversation history
  - Useful after editing config files or adding plugins
  - Shows confirmation of what was reloaded
- `/update` - Check for and install updates
  - Checks main app for new tagged releases
  - Checks plugins for updates (branch-based only)
  - Shows current and latest version/commit information
  - Warns if dependencies changed
  - Interactive update selection with visual confirmation
  - Automatic reload after successful update
- `/add_agent` - Add new agent from Git repo or local path
  - Interactive prompts for agent details with validation
  - Supports local and GitHub sources
  - Automatic tool dependency installation
  - Reads registry.json from GitHub repos
  - Saves to `~/.valbot_config.json`
  - Shows confirmation and next steps
- `/add_tool` - Add new tool extension
  - Similar workflow to `/add_agent`
  - Local and remote sources supported
  - Integrates with plugin system
  - Automatic installation and configuration

### TUI Configuration

#### Model and Endpoint Setup

Configure your preferred models and endpoints for the TUI in `~/.valbot_config.json`:

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

**Available Endpoints:**
1. **valbot-proxy** (Recommended)
   - Models: `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `gpt-oss:20b`
   - Requires: `VALBOT_CLI_KEY` environment variable
   - Get key at: https://genai-proxy.intel.com/

2. **azure** (To Be Documented)
   - Configuration details coming soon

3. **igpt** (Deprecated)
   - Model: `gpt-4o`
   - Requires: `CLIENT_ID` and `CLIENT_SECRET` environment variables

#### Environment Variables

Set these in your `.env` file or system environment:

```bash
# Required: API key for valbot-proxy endpoint
VALBOT_CLI_KEY=your_api_key_here

# Optional: For igpt endpoint (deprecated)
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
```

**Getting Your API Key:**
1. Visit https://genai-proxy.intel.com/
2. Login with your Intel credentials
3. Click "Manage Your API Tokens"
4. Create New Token
5. Copy the token and set it as `VALBOT_CLI_KEY`

You can switch models interactively in the TUI using `Ctrl+M` or `/model` command.

### TUI Usage Examples

**Basic Chat:**
```
1. Launch TUI: valbot
2. Type your message: "How do I read files in Python?"
3. Press Enter
4. Watch response stream in real-time with syntax highlighting
```

**Using Agents:**
```
1. Press Ctrl+A or type /agent
2. Navigate with arrow keys to select an agent
3. Press Enter to select
4. Provide any required arguments when prompted
5. Watch agent execute workflow with visual feedback
```

**Loading Context:**
```
1. Type: /context src/**/*.py
2. See visual confirmation of loaded files
3. Ask questions about the code
4. Files remain in context for the conversation
```

**Integrated Terminal:**
```
1. Type: /terminal ls -la
2. Terminal panel opens automatically
3. See real-time command output
4. Press Ctrl+T to toggle terminal visibility
```

**File Explorer:**
```
1. Press Ctrl+F to open file explorer
2. Navigate with arrow keys
3. Press Enter to load file into context
4. File content appears in chat with syntax highlighting
```

### TUI Tips & Best Practices

**1. Terminal Experience**
- **Recommended terminals**: Windows Terminal, iTerm2, GNOME Terminal, Alacritty, Konsole
- **Minimum size**: 80 columns × 24 rows
- **Recommended size**: 120 columns × 40 rows
- **Optimal size**: Full screen or maximized window
- **Color support**: Enable 256-color or true color (24-bit) in terminal settings
- **Font recommendations**: 
  - Fira Code (with ligatures)
  - JetBrains Mono
  - Cascadia Code
  - Source Code Pro

**2. Performance Optimization**
- Clear chat history regularly with `/clear` or `Ctrl+C` to free memory
- Close terminal panel when not in use with `Ctrl+T`
- Close file explorer with `Ctrl+F` when not needed
- Use appropriate models for your task:
  - `gpt-4.1-mini` or `gpt-5-mini` for simple queries
  - `gpt-4.1` or `gpt-5` for complex reasoning
- Enable reasoning display only when needed (config: `display_reasoning: true`)

**3. Keyboard Shortcuts Workflow**
- Use `Ctrl+M` for quick model switching during conversation
- Use `Ctrl+A` for quick agent access without typing commands
- Use `Esc` to cancel any picker or operation
- Use `Ctrl+T` to toggle terminal without losing focus
- Use `Ctrl+F` to quickly load files from explorer

**4. Code Block Management**
- Hover over code blocks to see copy button
- Click copy button to copy code to clipboard
- Use syntax highlighting to verify code language detection
- Collapsed sections help manage long responses

**5. Development & Debugging**

**Run with hot reload for development:**
```bash
textual run --dev valbot_tui.py
```

**View debug console in separate terminal:**
```bash
# Terminal 1: Start debug console
textual console

# Terminal 2: Run TUI
python valbot_tui_launcher.py
```

**Enable debug logging:**
```python
# Add to valbot_tui.py temporarily
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Troubleshooting TUI

**TUI won't start:**
```bash
# Check if textual is installed
pip show textual

# Reinstall if needed
pip install textual textual-dev

# Try with explicit Python version
python3.11 valbot_tui_launcher.py
```

**Display rendering issues:**
- Update to latest textual: `pip install --upgrade textual`
- Try a different terminal emulator
- Check terminal color support: `echo $COLORTERM`
- Disable terminal transparency/blur effects

**Keyboard shortcuts not working:**
- Check if terminal is intercepting shortcuts
- Try alternative shortcuts (some terminals may conflict)
- Check terminal key binding settings
- On macOS, ensure Terminal preferences allow shortcuts

**Slow performance:**
- Clear chat history with `/clear`
- Close unused panels (terminal, file explorer)
- Reduce terminal window size slightly
- Use smaller models for simple queries
- Check system resources (CPU, memory)

**Copy button not working:**
- Ensure clipboard support is available on your system
- On Linux, install `xclip` or `xsel`: `sudo apt install xclip`
- Check terminal clipboard integration settings

**File explorer not showing files:**
- Check file permissions in the directory
- Verify you're in the correct working directory
- Try collapsing and expanding folder nodes
- Restart TUI if file system changed

---

## Agents - Autonomous AI Workflows

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

ValBot uses two main configuration files:
- **`default_config.json`**: Contains built-in agents (under `default_agents`), built-in commands, and custom prompts provided by the application. This file is part of the ValBot installation.
- **`~/.valbot_config.json`**: Located in the user's home directory, this file allows users to override settings (including model and endpoint), manage agent extensions, and add custom commands and prompts.

The application merges `default_config.json` and `.valbot_config.json`, with `.valbot_config.json` taking precedence for customizations.

### Environment Variables

Configure these environment variables for authentication:

- **`VALBOT_CLI_KEY`**: Your API key for the Valbot proxy endpoint (recommended).

Set this in your `.env` file or system environment to authenticate with the API.

### Theme Persistence & Team Usage

ValBot automatically saves your theme preference to `~/.valbot_theme.json` in your home directory. This enables:

- **Automatic theme restoration**: Your chosen theme loads every time you start ValBot
- **Team-wide installation**: Multiple users can share the same ValBot installation
- **Personal preferences**: Each user maintains their own theme settings
- **No configuration needed**: Theme persistence works automatically—just select a theme and it's saved

**How it works:**
1. Use `/theme` command or theme picker in TUI to select a theme
2. Theme choice automatically saves to `~/.valbot_theme.json`
3. Next time you launch ValBot, your theme is restored automatically
4. Each user on the system has their own separate theme file

**Team deployment:**
- Install ValBot once in a shared location
- Each team member sets up the `valbot` alias pointing to the installation
- Users maintain personal configs in their home directories:
  - `~/.valbot_config.json` - Personal settings, agents, API keys
  - `~/.valbot_theme.json` - UI theme preference
- Shared agents and tools can be added to the main installation
- Updates benefit the entire team from a single update

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

These arguments are primarily for CLI mode. For TUI mode, use the `valbot` alias or pass `--tui` flag:

```bash
# Launch TUI mode (recommended)
valbot

# Alternative: Launch TUI with Python
python app.py --tui

# Launch TUI with custom config
python app.py --tui --config custom_config.json
```

**CLI Mode Arguments:**

- **Positional Message**: Directly provide an initial message when running in CLI mode
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

- **`--config`**: Specify a custom configuration file
  ```bash
  python app.py --config custom_config.json
```

---

## CLI Mode (Alternative Text Interface)

For users who prefer a traditional command-line REPL interface, ValBot can also run in CLI mode without the TUI.

### Running CLI Mode

**Basic usage:**
```bash
python app.py
```

**With initial message:**
```bash
python app.py -m "Your question here"
python app.py "Your question here"  # Positional argument
```

**With context files:**
```bash
python app.py -c file1.py file2.txt
python app.py -m "Analyze this" -c mycode.py
```

**With agents:**
```bash
python app.py -a agent_name -p param1=value1 param2=value2
```

**With custom config:**
```bash
python app.py --config custom_config.json
```

### CLI Features

The CLI mode supports all the same commands and features as the TUI:
- All slash commands (`/help`, `/agent`, `/model`, `/context`, etc.)
- Agent workflows
- Custom prompts
- Context management
- Model switching
- File operations

The main difference is the interface—CLI uses a text-based REPL while TUI provides a visual interface with panels, syntax highlighting, and keyboard shortcuts.

**Note:** For the best experience with syntax highlighting, code blocks, file explorer, integrated terminal, and visual feedback, we recommend using the TUI mode.

---

### Troubleshooting**TUI won't start:**
- Ensure textual is installed: `pip install textual textual-dev`
- Try updating textual: `pip install --upgrade textual`
- Use a modern terminal emulator (Windows Terminal, iTerm2, etc.)
- Check terminal size (minimum 80×24, recommended 120×40)

**Agent not found:**
- Ensure the agent is defined in either `default_config.json` or `~/.valbot_config.json`
- Check that the agent name matches exactly (case-sensitive)
- For remote agents, verify the repository was cloned successfully
- Use `/agent` command in TUI to see all available agents

**Authentication errors:**
- Verify `VALBOT_CLI_KEY` is set correctly in your `.env` file or environment
- Check that your API key is valid at https://genai-proxy.intel.com/
- Restart the TUI after setting the API key

**Plugin loading errors:**
- Check that the plugin file path exists and is accessible
- Ensure the plugin file has a `.py` extension
- Verify the plugin class inherits from `AgentPlugin`
- Check console output for specific error messages
- Use `/reload` command to reload plugins after changes

**Display or rendering issues in TUI:**
- Update to latest textual: `pip install --upgrade textual`
- Try a different terminal emulator
- Check terminal color support settings
- Disable terminal transparency/blur effects
- Ensure terminal supports 256 colors or true color
