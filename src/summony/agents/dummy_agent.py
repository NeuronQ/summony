from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import BaseAgent
from ..model_connectors import DummyModelConnector


class DummyAgent(BaseAgent):
    MODEL_CONNECTOR_CLASS = DummyModelConnector
