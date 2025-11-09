from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
from client_setup import initialize_agent_model
from config import ConfigManager

# Global variable to hold the config_manager instance (optional)
_global_config_manager = None

def set_config_manager(cfg_manager):
    global _global_config_manager
    _global_config_manager = cfg_manager

def get_config_manager():
    """Returns the currently set global config_manager, if set."""
    return _global_config_manager

def initialize_model(model_name: str = None, endpoint: str = None) -> OpenAIResponsesModel:
    """Utility to initialize and return a model.
    If endpoint is not provided, it will be looked up in the config manager (cfg).
    """
    cfg = get_config_manager()
    if endpoint is None:
        endpoint = cfg.get_setting('agent_model_config.default_endpoint', None)
    if model_name is None:
        model_name = cfg.get_setting('agent_model_config.default_model', None)
    if model_name == 'small_model':
        model_name = cfg.get_setting('agent_model_config.small_model', None)
    return initialize_agent_model(endpoint=endpoint, model_name=model_name)
