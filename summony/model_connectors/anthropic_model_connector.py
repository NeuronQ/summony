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

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage, MessageStreamEvent

from .model_connectors import ModelConnectorInterface, MessageDict


g_logger = logging.getLogger(__name__)


class AnthropicModelConnector(ModelConnectorInterface):
    logger: logging.Logger

    client: Anthropic
    async_client: AsyncAnthropic

    # static
    _DEFAULT_MAX_TOKENS = 4096
    _CURRENT_SONNET_MODEL = "claude-3-5-sonnet-20240620"
    _CURRENT_OPUS_MODEL = "claude-3-opus-20240229"
    _CURRENT_HAIKU_MODEL = "claude-3-haiku-20240307"
    _MODEL_SHORTCUTS = {
        "claude": _CURRENT_SONNET_MODEL,
        "claude-sonnet": _CURRENT_SONNET_MODEL,
        "claude-3-5-sonnet": _CURRENT_SONNET_MODEL,
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-opus": _CURRENT_OPUS_MODEL,
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-haiku": _CURRENT_HAIKU_MODEL,
        "claude-3-haiku": _CURRENT_HAIKU_MODEL,
    }

    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        api_key = creds.get("api_key") if creds else os.environ["ANTHROPIC_API_KEY"]
        if client_args is None:
            client_args = {}
        self.client = Anthropic(api_key=api_key, **client_args)
        self.async_client = AsyncAnthropic(api_key=api_key, **client_args)
        self.logger = logger if logger is not None else g_logger

    def generate(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        client_message = self.client.messages.create(
            **self._make_message_create_args(messages, model, kwargs)
        )
        return self._process_created_message(client_message)

    async def generate_async(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        client_message = await self.async_client.messages.create(
            **self._make_message_create_args(messages, model, kwargs)
        )
        return self._process_created_message(client_message)

    @staticmethod
    def _process_created_message(message: AnthropicMessage) -> Tuple[str, dict]:
        return message.content[0].text, message.model_dump(mode="json")

    def generate_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Iterator[Tuple[str, dict]]:
        stream = self.async_client.messages.create(
            **self._make_message_create_args(messages, model, kwargs), stream=True
        )
        i = 0
        for event in stream:
            chunk_text, chunk_dict = self._process_stream_event(event, i)
            if chunk_text:
                yield chunk_text, chunk_dict
            i += 1

    async def generate_async_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        stream = await self.async_client.messages.create(
            **self._make_message_create_args(messages, model, kwargs), stream=True
        )
        i = 0
        async for event in stream:
            chunk_text, chunk_dict = self._process_stream_event(event, i)
            if chunk_text:
                yield chunk_text, chunk_dict
            i += 1

    def _process_stream_event(
        self, event: MessageStreamEvent, event_idx: int
    ) -> Tuple[str, dict]:
        try:
            if event.type == "content_block_start":
                chunk_text = event.content_block.text
            elif event.type == "content_block_delta":
                chunk_text = event.delta.text
            else:
                chunk_text = None
        except Exception as exc:
            chunk_text = None
            self.logger.warning(
                "AnthropicModelConnector: Failed to get content from chunk %d: %s",
                event_idx,
                exc,
                exc_info=True,
            )
        return chunk_text, event.model_dump(mode="json")

    def get_base_url(self) -> str:
        return str(self.client.base_url)

    @classmethod
    def _make_message_create_args(
        cls, messages: list[dict], model: str, extra_args: dict
    ) -> dict:
        out = dict(messages=messages, model=cls._MODEL_SHORTCUTS.get(model, model))
        if len(messages) and messages[0]["role"] == "system":
            out["messages"] = messages[1:]
            out["system"] = messages[0]["content"]
        out.update(extra_args)
        if "max_tokens" not in out:
            out["max_tokens"] = cls._DEFAULT_MAX_TOKENS
        return out
