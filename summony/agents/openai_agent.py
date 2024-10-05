from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import OpenAIModelConnector


class OpenAIAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = OpenAIModelConnector
