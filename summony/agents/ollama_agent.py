from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import OllamaModelConnector


class OllamaAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = OllamaModelConnector
