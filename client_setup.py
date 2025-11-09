import os
import logging
import httpx
import requests
import msal
from typing import Optional

from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
from openai import AsyncOpenAI, OpenAI, AzureOpenAI
from rich.console import Console
from dotenv import load_dotenv
import sys

# ────────────────────────────────────────────────────────────────────────────────
# 1. Load environment variables
# ────────────────────────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    # Inside the exe
    base_dir = sys._MEIPASS
else:
    # when executing raw python files, or debugging
    base_dir = os.path.dirname(__file__)

dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

AZURE_API_KEY       = os.getenv("AZURE_API_KEY") # optional
VALBOT_CLI_KEY      = os.getenv("VALBOT_CLI_KEY")
VALBOT_PROXY_URL    = os.getenv("VALBOT_PROXY_URL", "https://genai-proxy.intel.com/api/")
ENDPOINT_PROXY_URL  = os.getenv("ENDPOINT_PROXY_URL", "http://localhost:8000")

PROXIES = {
    "http":  "http://proxy-dmz.intel.com:912",
    "https": "http://proxy-dmz.intel.com:912",
}

BASE_URL = "https://apis-internal.intel.com/generativeaiinference/v4"

console = Console()

def _env_report() -> None:
    """Print what credentials are present and exit if *none* are useable."""

    def _status(name: str, value: Optional[str], optional: bool = False) -> None:
        if not value:
            colour = "yellow" if optional else "red"
            note   = "(optional)" if optional else ""
            console.print(f"[{colour}]✗ {name} is missing {note}[/{colour}]")

    # Require *either* the valbot proxy key or the Azure key
    if (not AZURE_API_KEY) and (not VALBOT_CLI_KEY):
        console.print(
            "\n[bold red]No usable credentials found.[/bold red]\n"
            " • Set VALBOT_CLI_KEY to use models deployed via valbot-proxy (Recommended).\n"
            "    • See instructions at: https://github.com/intel-innersource/applications.ai.valbot-cli/?tab=readme-ov-file#valbot-proxy-endpoint-recommended \n"
            " • Or set AZURE_API_KEY to use models deployed via Azure OpenAI.\n"
        )
        _status("AZURE_API_KEY", AZURE_API_KEY, optional=True)
        _status("VALBOT_CLI_KEY", VALBOT_CLI_KEY)
        raise SystemExit(1)

_env_report()

httpx_async_client = httpx.AsyncClient(proxy=PROXIES["http"])

def get_bearer_token(client_id: str, client_secret: str) -> Optional[str]:
    """Exchange CLIENT_ID / CLIENT_SECRET for a bearer token."""
    url     = "https://apis-internal.intel.com/v1/auth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data    = {
        "grant_type":    "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    }
    resp = requests.post(url, headers=headers, data=data, proxies=PROXIES)
    if resp.status_code == 200:
        return resp.json().get("access_token")

    logging.error("Token request failed: %s %s", resp.status_code, resp.text)
    return None

def get_igpt_model(model_name: Optional[str] = None) -> OpenAIResponsesModel:
    # Print a warning that the igpt endpoint is deprecated and will be removed at the end of September 2024. Users should migrate to valbot-proxy in their ~/.valbot_config.json and set VALBOT_CLI_KEY in their environment.
    console.print("[bold yellow]Warning: The igpt endpoint is deprecated and will be removed at the end of September 2024.\nPlease migrate to valbot-proxy in your ~/.valbot_config.json and set VALBOT_CLI_KEY in your environment.[/bold yellow]")
    console.print("Go to [bold blue]https://github.com/intel-innersource/applications.ai.valbot-cli/tree/main?tab=readme-ov-file#setup[/bold blue] for migration instructions.\n")
    token = get_bearer_token(CLIENT_ID, CLIENT_SECRET)
    agent_client = AsyncOpenAI(
        api_key=token,
        base_url=BASE_URL,
        http_client=httpx_async_client,
    )
    return OpenAIResponsesModel(
        model_name=model_name or "gpt-4o",
        provider=OpenAIProvider(openai_client=agent_client),
    )

def get_azure_model(model_name: Optional[str] = None) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(
        model_name=model_name or "gpt-4.1",
        provider=AzureProvider(
            azure_endpoint="https://paivvalboteastus2.openai.azure.com",
            api_version="2025-01-01-preview",
            api_key=AZURE_API_KEY,
            http_client=httpx_async_client,
        ),
    )

def get_valbot_proxy_model(model_name: Optional[str] = None) -> OpenAIResponsesModel:
    """Get valbot proxy model with dynamic provider selection.

    For gpt-oss* models, uses OpenAI provider (compatible with ollama).
    For other models, uses Azure provider.
    """
    if model_name and model_name.startswith("gpt-oss"):
        return get_valbot_proxy_model_ollama(model_name)
    else:
        return get_valbot_proxy_model_azure(model_name)

def get_valbot_proxy_model_azure(model_name: Optional[str] = None) -> OpenAIResponsesModel:
    """Get valbot proxy model using Azure provider."""
    httpx_async_client = httpx.AsyncClient(verify=False)
    return OpenAIResponsesModel(
        model_name=model_name or "gpt-4.1",
        provider=AzureProvider(
            api_key=VALBOT_CLI_KEY,
            azure_endpoint=VALBOT_PROXY_URL,
            api_version="2025-04-01-preview",
            http_client=httpx_async_client,
        )
    )

def get_valbot_proxy_model_ollama(model_name: Optional[str] = None) -> OpenAIResponsesModel:
    """Get valbot proxy model using OpenAI provider (compatible with ollama)."""
    httpx_async_client = httpx.AsyncClient(verify=False)
    return OpenAIResponsesModel(
        model_name=model_name or "gpt-oss:20b",
        provider=OpenAIProvider(
            openai_client=AsyncOpenAI(
                api_key=VALBOT_CLI_KEY,
                base_url=VALBOT_PROXY_URL + "v1/",
                http_client=httpx_async_client,
            )
        )
    )

def initialize_agent_model(
    endpoint: str = "igpt", model_name: Optional[str] = None
) -> OpenAIResponsesModel:
    """Factory returning an OpenAIResponsesModel for either Intel iGPT or Azure."""
    if endpoint == "igpt":
        return get_igpt_model(model_name)
    if endpoint == "azure":
        return get_azure_model(model_name)
    if endpoint == "valbot-proxy":
        return get_valbot_proxy_model(model_name)

    raise ValueError(f"Unknown endpoint: {endpoint!r}")


def initialize_chat_client(endpoint: str = "valbot-proxy", model_name: Optional[str] = None):
    """Factory returning an OpenAI client for various endpoints."""
    if endpoint == "igpt":
        return initialize_chat_client_igpt(CLIENT_ID, CLIENT_SECRET)
    if endpoint == "azure":
        return initialize_chat_client_azure()
    if endpoint == "valbot-proxy":
        # Dynamic selection based on model name
        if model_name and model_name.startswith("gpt-oss"):
            return initialize_chat_client_valbot_proxy_ollama()
        else:
            return initialize_chat_client_valbot_proxy()

    raise ValueError(f"Unknown endpoint: {endpoint!r}")

def initialize_chat_client_igpt(client_id: str, client_secret: str) -> OpenAI:
    """Low‑level `OpenAI` client (v3 Intel endpoint)."""
    console.print("[bold yellow]Warning: The igpt endpoint is deprecated and will be removed at the end of September 2025.\nPlease migrate to valbot-proxy in your ~/.valbot_config.json and set VALBOT_CLI_KEY in your environment.[/bold yellow]")
    console.print("Go to [bold blue]https://github.com/intel-innersource/applications.ai.valbot-cli/tree/main?tab=readme-ov-file#setup[/bold blue] for migration instructions.\n")
    token = get_bearer_token(client_id, client_secret)
    if not token:
        console.print("[bold red]Error obtaining token. Exiting.[/bold red]")
        raise SystemExit(1)

    httpx_client = httpx.Client(proxy=PROXIES["http"])
    return OpenAI(
        api_key     = token,
        base_url    = "https://apis-internal.intel.com/generativeaiinference/v4",
        http_client = httpx_client,
    )

def initialize_chat_client_azure() -> OpenAI:
    httpx_client = httpx.Client(proxy=PROXIES["http"])
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint="https://paivvalboteastus2.openai.azure.com",
        api_version="2025-01-01-preview",
        http_client=httpx_client,
    )

def initialize_chat_client_valbot_proxy() -> OpenAI:
    httpx_client = httpx.Client(verify=False)
    return AzureOpenAI(
        api_key=VALBOT_CLI_KEY,
        azure_endpoint=VALBOT_PROXY_URL,
        api_version="2025-04-01-preview",
        http_client=httpx_client,
    )

def initialize_chat_client_valbot_proxy_ollama() -> OpenAI:
    httpx_client = httpx.Client( verify=False)
    return OpenAI(
        api_key=VALBOT_CLI_KEY,
        base_url=VALBOT_PROXY_URL+"v1/",
        http_client=httpx_client,
    )