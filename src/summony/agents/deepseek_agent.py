from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import DeepSeekModelConnector


class DeepSeekAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = DeepSeekModelConnector
