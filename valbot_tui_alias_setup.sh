#!/usr/bin/env bash
# ==============================================================================
#  ValBot CLI - Alias Setup Script (Standalone)
# ------------------------------------------------------------------------------
#  This script sets up the 'valbot' alias in your ~/.aliases file
#  to launch ValBot TUI from anywhere.
#
#  Usage:
#    1. Edit the VALBOT_INSTALL_PATH variable below to point to your 
#       ValBot installation directory
#    2. Run: chmod +x valbot_tui_alias_setup.sh
#    3. Run: ./valbot_tui_alias_setup.sh
# ==============================================================================

# --- CONFIGURATION: Edit this path to match your ValBot installation ---
# This should be the directory where valbot_tui.sh is located
VALBOT_INSTALL_PATH="/usr/intel/pkgs/python3/3.12.3/bin"

# --- Colors ---
if [[ -t 1 ]]; then
  BOLD='\e[1m'
  RED='\e[31m'
  GREEN='\e[32m'
  YELLOW='\e[33m'
  BLUE='\e[34m'
  NC='\e[0m'
else
  BOLD=''
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  NC=''
fi

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
  local bash_runner="$VALBOT_INSTALL_PATH/valbot_tui.sh"

  # Check if valbot_tui.sh exists
  if [[ ! -f "$bash_runner" ]]; then
    err "Error: valbot_tui.sh not found at $bash_runner"
    say "Please edit the VALBOT_INSTALL_PATH variable in this script to point to"
    say "the directory where valbot_tui.sh is installed."
    say "\nCurrent path: $VALBOT_INSTALL_PATH"
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
        echo "# ValBot CLI alias added by valbot_tui_alias_setup.sh on $(date)"
        echo "alias valbot=\"$bash_runner\""
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
        echo "# ValBot CLI alias added by valbot_tui_alias_setup.sh on $(date)"
        echo "alias valbot=\"$bash_runner\""
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
      say "  - Bash: echo 'alias valbot=\"$bash_runner\"' >> ~/.bashrc"
      say "  - Zsh: echo 'alias valbot=\"$bash_runner\"' >> ~/.zshrc"
      exit 0
    fi
  fi
}

main "$@"
