from collections import defaultdict
import os
import logging
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self, Tuple

from anthropic import Anthropic, AsyncAnthropic

from ..loggers import AgentLoggerInterface, DefaultAgentLogger


g_logger = logging.getLogger(__name__)


class AnthropicModelConnector:
    model_name: str
    logger: AgentLoggerInterface
    client: Anthropic
    async_client: AsyncAnthropic

    # static
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
        self.client = Anthropic(
            api_key=(
                creds.get("api_key") if creds else os.environ["ANTHROPIC_API_KEY"]
            ),
            **(client_args or {}),
        )
        self.async_client = AsyncAnthropic(
            api_key=(
                creds.get("api_key") if creds else os.environ["ANTHROPIC_API_KEY"]
            ),
            **(client_args or {}),
        )
        self.logger = logger if logger is not None else g_logger

    def generate_completion(self, messages, model, **kwargs) -> Tuple[str, dict]:
        client_message = self.client.messages.create(
            **self._make_message_create_args(messages, model), **kwargs
        )
        completion_text = client_message.content[0].text
        try:
            completion_dict = client_message.model_dump(mode="json")
        except Exception as exc:
            completion_dict = {}
            self.logger.warning(
                "AnthropicModelConnector.generate_completion: Failed to convert completion to JSON-serializable dict: %s",
                exc,
                exc_info=True,
            )
        return completion_text, completion_dict

    async def async_generate_completion(
        self, messages, model, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        stream = await self.async_client.messages.create(
            **self._make_message_create_args(messages, model), **kwargs, stream=True
        )
        i = 0
        async for event in stream:
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
                    "AnthropicModelConnector.async_generate_completion: Failed to get content from chunk %d: %s",
                    i,
                    exc,
                    exc_info=True,
                )

            try:
                chunk_dict = event.model_dump(mode="json")
            except Exception as exc:
                chunk_dict = {}
                self.logger.warning(
                    "AnthropicModelConnector.async_generate_completion: Failed to make JSON-serializable dict from chunk %d: %s",
                    i,
                    exc,
                    exc_info=True,
                )

            if chunk_text:
                yield chunk_text, chunk_dict

            i += 1

    def get_base_url(self) -> str:
        return str(self.client.base_url)

    @classmethod
    def _make_message_create_args(cls, messages: list[dict], model: str) -> dict:
        out = dict(messages=messages, model=cls._MODEL_SHORTCUTS.get(model, model))
        if len(messages) and messages[0]["role"] == "system":
            out["messages"] = messages[1:]
            out["system"] = messages[0]["content"]
        return out
