from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import AnthropicModelConnector


class AnthropicAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = AnthropicModelConnector
