from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import XAIModelConnector


class XAIAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = XAIModelConnector
