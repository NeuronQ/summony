from abc import abstractmethod
import logging
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self, Tuple


g_logger = logging.getLogger(__name__)


class ModelConnectorInterface:
    logger: logging.Logger

    @abstractmethod
    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ): ...

    @abstractmethod
    def generate_completion(self, messages, model, **kwargs) -> Tuple[str, dict]: ...

    @abstractmethod
    async def async_generate_completion(
        self, messages, model, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]: ...

    @abstractmethod
    def get_base_url(self) -> str: ...
