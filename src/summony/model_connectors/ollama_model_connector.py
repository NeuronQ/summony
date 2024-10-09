import os
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
)

from ollama import Client, AsyncClient


from .model_connectors import ModelConnectorInterface, MessageDict


g_logger = logging.getLogger(__name__)


class OllamaModelConnector(ModelConnectorInterface):
    logger: logging.Logger

    client: Client
    async_client: AsyncClient

    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        api_key = creds.get("api_key") if creds else os.getenv("OLLAMA_API_KEY")
        if client_args is None:
            client_args = {}
        if api_key:
            client_args["api_key"] = api_key
        self.client = Client(**client_args)
        self.async_client = AsyncClient(**client_args)
        self.logger = logger if logger is not None else g_logger

    def generate(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        response = self.client.chat(messages=messages, model=model, options=kwargs)
        return response["message"]["content"], response

    async def generate_async(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        response = await self.async_client.chat(
            messages=messages, model=model, options=kwargs
        )
        return response["message"]["content"], response

    def _process_chunk(self, chunk: dict, chunk_idx: int) -> Tuple[str, dict]:
        try:
            chunk_text = chunk["message"]["content"]
        except Exception as exc:
            chunk_text = None
            self.logger.warning(
                "OllamaModelConnector: Failed to get content from chunk %d: %s",
                chunk_idx,
                exc,
                exc_info=True,
            )
        return chunk_text, chunk

    def generate_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Iterator[Tuple[str, dict]]:
        stream = self.client.chat(
            messages=messages, model=model, stream=True, options=kwargs
        )
        i = 0
        for chunk in stream:
            chunk_text, chunk = self._process_chunk(chunk, i)
            if chunk_text:
                yield chunk_text, chunk
            i += 1

    async def generate_async_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        stream = await self.async_client.chat(
            messages=messages, model=model, stream=True, options=kwargs
        )
        i = 0
        async for chunk in stream:
            chunk_text, chunk = self._process_chunk(chunk, i)
            if chunk_text:
                yield chunk_text, chunk
            i += 1

    def get_base_url(self) -> str:
        _client = getattr(self.client, "_client", None)
        return str(getattr(_client, "base_url", ""))
