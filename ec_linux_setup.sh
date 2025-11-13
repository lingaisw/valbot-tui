#!/usr/bin/env bash
# ==============================================================================
#  ValBot CLI - EC Linux Interactive Setup
# ------------------------------------------------------------------------------
#  This script walks you through setting up ValBot CLI on Intel EC Linux.
#  It will:
#    - Detect or let you choose a Python 3.10+ interpreter (EC default shown)
#    - Use an existing virtual environment or create a new one
#    - Activate the venv and install requirements (with optional proxy)
#    - Prompt for your VALBOT_CLI_KEY
#    - Generate convenient launcher scripts:
#         - valbot.sh (bash/zsh)
#         - valbot.csh (tcsh)
#    - Optionally write a .env file with VALBOT_CLI_KEY
#
#  Notes from README (EC specifics):
#    - EC users cannot install Python packages globally; a venv is required
#    - Typical EC Python path: /usr/intel/pkgs/python3/3.12.3/bin/python3
#    - Typical proxy: http://proxy-chain.intel.com:911
# ------------------------------------------------------------------------------
#  Usage:
#     chmod +x ec_linux_setup.sh
#     ./ec_linux_setup.sh
# ==============================================================================

set -Eeuo pipefail

# --- Colors ---
if [[ -t 1 ]]; then
  BOLD='\e[1m'; RED='\e[31m'; GREEN='\e[32m'; YELLOW='\e[33m'; BLUE='\e[34m'; NC='\e[0m'
else
  BOLD=''; RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

MIN_MAJOR=3
MIN_MINOR=10

say() { echo -e "$1"; }
ok()  { say "${GREEN}✔${NC} $1"; }
warn(){ say "${YELLOW}⚠${NC} $1"; }
err() { say "${RED}✖${NC} $1"; }

default_read() {
  # $1 prompt, $2 default -> sets REPLY
  local prompt="$1"; local def_val="${2-}"
  if [[ -n "${def_val}" ]]; then
    read -r -p "$prompt [$def_val]: " REPLY
    REPLY=${REPLY:-$def_val}
  else
    read -r -p "$prompt: " REPLY
  fi
}

confirm() {
  # $1 prompt, default Y
  local prompt="$1"
  read -r -p "$prompt [Y/n]: " ans || true
  case "${ans:-Y}" in
    [Yy]*) return 0 ;;
    *)     return 1 ;;
  esac
}

check_python_version() {
  local py="$1"
  if ! command -v "$py" >/dev/null 2>&1; then
    return 1
  fi
  local ver
  ver="$($py -c 'import sys; print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || true)"
  if [[ -z "$ver" ]]; then
    return 1
  fi
  local maj min
  maj="${ver%%.*}"
  min="${ver#*.}"; min="${min%%.*}"
  if (( maj > MIN_MAJOR )) || { (( maj == MIN_MAJOR )) && (( min >= MIN_MINOR )); }; then
    echo "$ver"
    return 0
  fi
  return 2
}

pick_python() {
  say "${BOLD}Step 1:${NC} Select Python interpreter (needs ${MIN_MAJOR}.${MIN_MINOR}+)."
  local ec_default="/usr/intel/pkgs/python3/3.12.3/bin/python3"
  local candidates=()
  [[ -x "$ec_default" ]] && candidates+=("$ec_default")
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")
  command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
  candidates=($(printf "%s\n" "${candidates[@]}" | awk '!x[$0]++'))

  local chosen="" ver=""
  for c in "${candidates[@]}"; do
    if ver=$(check_python_version "$c"); then
      chosen="$c"; break
    fi
  done

  if [[ -z "$chosen" ]]; then
    warn "No suitable Python ${MIN_MAJOR}.${MIN_MINOR}+ auto-detected."
    while true; do
      default_read "Enter path to Python ${MIN_MAJOR}.${MIN_MINOR}+ interpreter" "$ec_default"
      local try="$REPLY"
      if ver=$(check_python_version "$try"); then
        chosen="$try"; break
      else
        err "'$try' is not a valid Python ${MIN_MAJOR}.${MIN_MINOR}+ interpreter. Try again."
      fi
    done
  fi
  ok "Using Python at: $chosen (version $ver)"
  PYTHON_BIN="$chosen"
}

choose_or_create_venv() {
  say "
${BOLD}Step 2:${NC} Create a new virtual environment or use an existing one. (Default: create new)"
  if confirm "Create a new virtual environment now?"; then
    local create_new="yes"
    local reuse_existing="no"
    while true; do
      default_read "Enter a path where we should create the venv" "$PROJECT_ROOT/valbot-venv"
      local p="$REPLY"
      if [[ -e "$p" && ! -d "$p" ]]; then
        err "Path exists and is not a directory: $p"
        continue
      fi
      if [[ -d "$p" && "$(ls -A "$p" 2>/dev/null | wc -l)" != "0" ]]; then
        say "Directory '$p' exists and is not empty."
        while true; do
          read -r -p "How would you like to proceed? [U]se as is, [D]elete and create new, or [C]hoose a different path [u/d/c]: " action
          case "${action,,}" in
            u)
              if [[ -f "$p/bin/activate" || -f "$p/bin/activate.csh" ]]; then
                VENV_PATH="$p"
                reuse_existing="yes"
                create_new="no"
                ok "Using existing venv: $VENV_PATH"
                break 2
              else
                warn "No activate script found under '$p/bin'. This may not be a Python virtual environment."
                if confirm "Continue anyway and use this directory as is?"; then
                  VENV_PATH="$p"
                  reuse_existing="yes"
                  create_new="no"
                  break 2
                else
                  continue
                fi
              fi
              ;;
            d)
              read -r -p "Really delete '$p' and create a new venv here? This will remove ALL contents. [y/N]: " ans
              case "${ans:-N}" in
                [Yy]*)
                  rm -rf "$p"
                  VENV_PATH="$p"
                  create_new="yes"
                  reuse_existing="no"
                  break
                  ;;
                *)
                  continue
                  ;;
              esac
              ;;
            c)
              continue
              ;;
            *)
              warn "Please enter 'u' (use), 'd' (delete and create new), or 'c' (choose a different path)."
              ;;
          esac
        done
      else
        VENV_PATH="$p"
        create_new="yes"
        reuse_existing="no"
        break
      fi
    done
    if [[ "$create_new" == "yes" ]]; then
      say "Creating virtual environment at: $VENV_PATH"
      "$PYTHON_BIN" -m venv "$VENV_PATH"
      ok "Virtual environment created."
    else
      ok "Skipping creation; will use existing directory as venv: $VENV_PATH"
    fi
  else
    while true; do
      default_read "Enter existing venv path" "$PROJECT_ROOT/valbot-venv"
      local p="$REPLY"
      if [[ -f "$p/bin/activate" || -f "$p/bin/activate.csh" ]]; then
        VENV_PATH="$p"
        ok "Using existing venv: $VENV_PATH"
        break
      else
        err "No activate script found under '$p/bin'. Please provide a valid venv path."
      fi
    done
  fi
}

pip_install_requirements() {
  say "
${BOLD}Step 3:${NC} Install Python requirements into the venv."
  if [[ ! -f "$REQ_FILE" ]]; then
    err "requirements.txt not found at $REQ_FILE"
    say "Make sure you run this from the project root where requirements.txt exists."
    exit 1
  fi

  local pip_bin="$VENV_PATH/bin/pip"
  if [[ ! -x "$pip_bin" ]]; then
    err "pip not found in venv: $pip_bin"
    exit 1
  fi

  # Ask about proxy BEFORE upgrading pip so the upgrade also uses the proxy
  local use_proxy="no" proxy_url="http://proxy-chain.intel.com:911"
  if confirm "Are you behind the EC proxy and want to use it for pip installs?"; then
    use_proxy="yes"
    default_read "Enter proxy URL" "$proxy_url"
    proxy_url="$REPLY"
    # Export common proxy env vars for completeness
    export HTTP_PROXY="$proxy_url" HTTPS_PROXY="$proxy_url" \
           http_proxy="$proxy_url" https_proxy="$proxy_url"
  fi

  say "Upgrading pip..."
  if [[ "$use_proxy" == "yes" ]]; then
    "$VENV_PATH/bin/python" -m pip install --proxy="$proxy_url" --upgrade pip
  else
    "$VENV_PATH/bin/python" -m pip install --upgrade pip
  fi

  say "Installing requirements..."
  if [[ "$use_proxy" == "yes" ]]; then
    "$pip_bin" install --proxy="$proxy_url" -r "$REQ_FILE" --disable-pip-version-check --upgrade
  else
    "$pip_bin" install -r "$REQ_FILE" --disable-pip-version-check --upgrade
  fi
  ok "Requirements installed."
}

collect_key_and_env() {
  say "\n${BOLD}Step 4:${NC} Configure your VALBOT_CLI_KEY."
  say "Visit https://genai-proxy.intel.com/ -> Manage Your API Tokens to create a token."

  local key=""
  local existing_key=""

  # Check if .env file exists and try to extract the key
  if [[ -f "$PROJECT_ROOT/.env" ]]; then
    existing_key=$(grep -E '^VALBOT_CLI_KEY=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2- | tr -d '\n' || true)
    if [[ -n "$existing_key" && "$existing_key" != "your_valbot_cli_key_here" ]]; then
      say "Found existing .env file with VALBOT_CLI_KEY."
      if confirm "Use the existing key from .env file?"; then
        VALBOT_CLI_KEY_VAL="$existing_key"
        ok "Using existing key from .env file."
        return
      else
        say "You can enter a new key below."
      fi
    fi
  fi

  default_read "Enter your VALBOT_CLI_KEY (leave blank to set a placeholder)" ""
  key="$REPLY"
  VALBOT_CLI_KEY_VAL="${key:-your_valbot_cli_key_here}"

  if confirm "Also write this value to a .env file in the project root?"; then
    printf 'VALBOT_CLI_KEY=%s\n' "$VALBOT_CLI_KEY_VAL" > "$PROJECT_ROOT/.env"
    ok ".env created at $PROJECT_ROOT/.env"
  else
    warn "Skipping .env creation. Runner scripts will still export VALBOT_CLI_KEY."
  fi
}

write_runner_scripts() {
  say "
${BOLD}Step 5:${NC} Create convenience launcher script."
  local bash_runner="$PROJECT_ROOT/valbot_tui.sh"

  if [[ -f "$bash_runner" ]] && ! confirm "Overwrite existing $bash_runner?"; then
    warn "Skipping $bash_runner"
  else
    cat > "$bash_runner" <<EOF
#!/bin/bash
# ValBot TUI Launcher for Unix/Linux/macOS
# This script launches the Terminal User Interface version of ValBot

# Get the directory where this script is located
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# Don't change to script directory - stay in caller's directory
# cd "\$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

# Hardcoded virtual environment path from setup
VENV_PATH="$VENV_PATH"

# Activate virtual environment if it exists
if [ -f "\$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment at \$VENV_PATH..."
    source "\$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at \$VENV_PATH"
    echo "Running with system Python..."
fi

# Launch the TUI in a new xfce4-terminal window
# use xfce4-terminal because support for 256-color & more emojis
echo "Starting ValBot TUI in new terminal..."
xfce4-terminal --command="bash -c '
    # Re-activate virtual environment in the new terminal
    if [ -f \"\$VENV_PATH/bin/activate\" ]; then
        source \"\$VENV_PATH/bin/activate\"
    fi
    
    # Launch the TUI
    python3 \"\$SCRIPT_DIR/valbot_tui_launcher.py\" \$@
'" &

echo "ValBot TUI launched in new terminal window (PID: \$!)"
EOF
    chmod +x "$bash_runner"
    ok "Wrote $bash_runner"
  fi

  # Create alias setup script
  local alias_setup_script="$PROJECT_ROOT/valbot_tui_alias_setup.sh"
  if [[ -f "$alias_setup_script" ]] && ! confirm "Overwrite existing $alias_setup_script?"; then
    warn "Skipping $alias_setup_script"
  else
    cat > "$alias_setup_script" <<'EOFSCRIPT'
#!/usr/bin/env bash
# ==============================================================================
#  ValBot CLI - Alias Setup Script
# ------------------------------------------------------------------------------
#  This script sets up the 'valbot' alias in your ~/.aliases file
#  to launch ValBot TUI from anywhere.
# ==============================================================================

set -Eeuo pipefail

# --- Colors ---
if [[ -t 1 ]]; then
  BOLD='\e[1m'; RED='\e[31m'; GREEN='\e[32m'; YELLOW='\e[33m'; BLUE='\e[34m'; NC='\e[0m'
else
  BOLD=''; RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

say() { echo -e "$1"; }
ok()  { say "${GREEN}✔${NC} $1"; }
warn(){ say "${YELLOW}⚠${NC} $1"; }
err() { say "${RED}✖${NC} $1"; }

confirm() {
  # $1 prompt, default Y
  local prompt="$1"
  read -r -p "$prompt [Y/n]: " ans || true
  case "${ans:-Y}" in
    [Yy]*) return 0 ;;
    *)     return 1 ;;
  esac
}

main() {
  say "${BOLD}========================================${NC}"
  say "${BOLD}   ValBot CLI - Alias Setup Script     ${NC}"
  say "${BOLD}========================================${NC}\n"

  local aliases_file="$HOME/.aliases"
  local bash_runner="$PROJECT_ROOT/valbot_tui.sh"

  # Check if valbot_tui.sh exists
  if [[ ! -f "$bash_runner" ]]; then
    err "Error: valbot_tui.sh not found at $bash_runner"
    say "Please run the main setup script (ec_linux_setup.sh) first."
    exit 1
  fi

  say "This script will add the 'valbot' alias to your ~/.aliases file."
  say "The alias will point to: $bash_runner"
  echo ""

  if [[ -f "$aliases_file" ]]; then
    say "Detected existing $aliases_file"
    
    # Check if alias already exists
    if grep -q 'alias valbot' "$aliases_file"; then
      warn "Alias 'valbot' already exists in $aliases_file."
      if confirm "Override the existing alias?"; then
        sed -i '/alias valbot/d' "$aliases_file"
      else
        warn "Alias setup cancelled."
        exit 0
      fi
    fi

    if confirm "Add alias 'valbot' to $aliases_file?"; then
      {
        echo ""
        echo "alias valbot \"$bash_runner\""
      } >> "$aliases_file"
      ok "Alias successfully added to $aliases_file"
      say "\n${BOLD}Setup Complete!${NC}"
      say "To start using the alias:"
      say "  1. Open a new terminal, OR"
      say "  2. Run: source \"$aliases_file\""
      say "\nThen you can run 'valbot' from anywhere!"
    else
      warn "Alias setup cancelled."
      exit 0
    fi
  else
    warn "No $aliases_file found."
    if confirm "Create $aliases_file and add the alias?"; then
      {
        echo "alias valbot \"$bash_runner\""
      } > "$aliases_file"
      ok "Created $aliases_file and added alias"
      say "\n${BOLD}Setup Complete!${NC}"
      say "To start using the alias:"
      say "  1. Add this line to your ~/.bashrc or ~/.zshrc:"
      say "     source \"$aliases_file\""
      say "  2. Open a new terminal, OR"
      say "  3. Run: source \"$aliases_file\""
      say "\nThen you can run 'valbot' from anywhere!"
    else
      warn "Alias setup cancelled."
      say "You can still create an alias manually in your shell configuration:"
      say "  - Bash: echo 'alias valbot \"$bash_runner\"' >> ~/.bashrc"
      say "  - Zsh: echo 'alias valbot \"$bash_runner\"' >> ~/.zshrc"
      exit 0
    fi
  fi
}

main "$@"
EOFSCRIPT
    chmod +x "$alias_setup_script"
    ok "Wrote $alias_setup_script"
  fi

  say "
You can create an alias for convenience (add to your shell rc file):"
  say "  - Bash/Zsh: alias valbot \"$bash_runner\""
  say "  - Or run: $alias_setup_script (to set up the alias automatically)"
}

add_aliases_file() {
  say "
${BOLD}Step 6:${NC} Optional: add an alias to run valbot-tui from anywhere."
  local aliases_file="$HOME/.aliases"
  local bash_runner="$PROJECT_ROOT/valbot_tui.sh"
  if [[ -f "$aliases_file" ]]; then
    if confirm "Detected $aliases_file. Add alias 'valbot' pointing to $bash_runner?"; then
      if grep -q 'alias valbot ' "$aliases_file"; then
        warn "Alias 'valbot' already exists in $aliases_file. It will be overridden."
        sed -i '/alias valbot /d' "$aliases_file"
      fi
      {
        echo ""
        echo "alias valbot \"$bash_runner\""
      } >> "$aliases_file"
      ok "Alias added to $aliases_file"
      say "Run: source \"$aliases_file\" or open a new shell to use 'valbot'."
    else
      warn "Skipping alias creation in $aliases_file."
      say "You can add it manually later with:"
      say "  echo 'alias valbot \"$bash_runner\"' >> $aliases_file"
    fi
  else
    warn "No $aliases_file found."
    say "You can still create an alias manually in your shell configuration, for example:"
    say "  - Bash/Zsh: echo 'alias valbot \"$bash_runner\"' >> ~/.bashrc  # or ~/.zshrc"
  fi
}

final_notes() {
  say "
${BOLD}Setup Complete!${NC}"
  say "How to run ValBot TUI:"
  say "  - Run: $PROJECT_ROOT/valbot_tui.sh"
  say "  - Or use the alias 'valbot' if you added it to your shell config"
  say "
Extra tips:"
  say "  - The alias now launches the TUI (Text User Interface) by default"
  say "  - To adjust models/endpoints, edit ~/.valbot_config.json as per README."
  say "  - To use EC proxy later for pip: pip install --proxy=\"http://proxy-chain.intel.com:911\" -r requirements.txt"
}

main() {
  say "${BOLD}========================================${NC}"
  say "${BOLD}      ValBot CLI - EC Linux Setup      ${NC}"
  say "${BOLD}========================================${NC}\n"

  pick_python
  choose_or_create_venv
  pip_install_requirements
  collect_key_and_env
  write_runner_scripts
  add_aliases_file
  final_notes
}

main "$@"