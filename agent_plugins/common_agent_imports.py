import asyncio
import logging
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown