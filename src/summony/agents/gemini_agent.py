from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import GeminiModelConnector


class GeminiAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = GeminiModelConnector
