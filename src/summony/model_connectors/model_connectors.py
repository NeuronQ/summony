from abc import abstractmethod
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Literal,
    Self,
    Tuple,
    TypedDict,
)


g_logger = logging.getLogger(__name__)


class MessageDict(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


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
    def generate(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]: ...

    @abstractmethod
    async def generate_async(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]: ...

    @abstractmethod
    def generate_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Iterator[Tuple[str, dict]]: ...

    @abstractmethod
    async def generate_async_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]: ...

    @abstractmethod
    def get_base_url(self) -> str: ...
