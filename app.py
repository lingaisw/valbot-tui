import platform_setup

platform_setup.setup_proxy_env()

import argparse
import sys
from client_setup import initialize_agent_model
from chatbot import ChatBot
from config import ConfigManager
from agent_plugins import utilities
from agent_plugins.plugin_manager import PluginManager

def parse_key_value_pairs(pairs):
    """Parse a list of key=value pairs into a dictionary."""
    result = {}
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key] = value
        else:
            print(f"Invalid parameter format: {pair}. Expected format: key=value")
    return result


def main():
    parser = argparse.ArgumentParser(description="ChatBot with initial message and context support")
    parser.add_argument("positional_message", type=str, nargs='?', help="Initial message to send to the AI (can be positional or with -m)")
    parser.add_argument("-m", "--message", type=str, help="Initial message to send to the AI")
    parser.add_argument("-c", "--context", type=str, nargs='+', help="A file or list of files to load context from")
    parser.add_argument("-a", "--agent", type=str, help="Agent flow to invoke at startup")
    parser.add_argument("-p", "--params", type=str, nargs='+', help="Parameters for the agent flow. Format: key=value")
    parser.add_argument("--config", type=str, help="Path to a custom configuration file")
    parser.add_argument("--tui", action="store_true", help="Launch Terminal User Interface mode")
    args = parser.parse_args()
    
    # Launch TUI mode if requested
    if args.tui:
        try:
            from valbot_tui import ValbotTUI
            app = ValbotTUI(config_path=args.config)
            app.run()
            return
        except ImportError as e:
            print(f"Error: TUI dependencies not installed. Run: pip install textual textual-dev")
            print(f"Details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error launching TUI: {e}")
            sys.exit(1)
    
    # Determine the initial message, prioritizing the flag if both are provided
    initial_message = args.message if args.message is not None else args.positional_message

    config_manager = ConfigManager(args.config)
    utilities.set_config_manager(config_manager)
    agent_config = config_manager.get_setting('agent_model_config')
    agent_model = initialize_agent_model(endpoint=agent_config['default_endpoint'], model_name=agent_config['default_model'])
    plugin_manager = PluginManager(config_manager, model=agent_model)
    bot = ChatBot(agent_model=agent_model, config_manager=config_manager, plugin_manager=plugin_manager, is_cli=True)

    if args.agent:
        bot.plugin_manager.run_plugin(args.agent, bot.context_manager.conversation_history, **parse_key_value_pairs(args.params) if args.params else {})
    else:
        bot.run(initial_message=initial_message, context=args.context)


if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        print("quitting...")