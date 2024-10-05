import os
import logging
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self, Tuple

from openai import OpenAI, AsyncOpenAI

from .model_connectors import ModelConnectorInterface


g_logger = logging.getLogger(__name__)


class OpenAIModelConnector(ModelConnectorInterface):
    logger: logging.Logger

    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.client = OpenAI(
            api_key=(creds.get("api_key") if creds else os.environ["OPENAI_API_KEY"]),
            **(client_args or {}),
        )
        self.async_client = AsyncOpenAI(
            api_key=(creds.get("api_key") if creds else os.environ["OPENAI_API_KEY"]),
            **(client_args or {}),
        )
        self.logger = logger if logger is not None else g_logger

    def generate_completion(self, messages, model, **kwargs) -> Tuple[str, dict]:
        completion = self.client.chat.completions.create(
            messages=messages, model=model, **kwargs
        )
        completion_text = completion.choices[0].message.content
        try:
            completion_dict = completion.model_dump(mode="json")
        except Exception as exc:
            completion_dict = {}
            self.logger.warning(
                "OpenAIModelConnector.generate_completion: Failed to convert completion to JSON-serializable dict: %s",
                exc,
                exc_info=True,
            )
        return completion_text, completion_dict

    async def async_generate_completion(
        self, messages, model, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        stream = await self.async_client.chat.completions.create(
            messages=messages, model=model, stream=True, **kwargs
        )
        i = 0
        async for chunk in stream:
            try:
                chunk_text = chunk.choices[0].delta.content
            except Exception as exc:
                chunk_text = None
                self.logger.warning(
                    "OpenAIAgent.async_generate_completion: Failed to get content from chunk %d: %s",
                    i,
                    exc,
                    exc_info=True,
                )

            try:
                chunk_dict = chunk.model_dump(mode="json")
            except Exception as exc:
                chunk_dict = {}
                self.logger.warning(
                    "OpenAIModelConnector.async_generate_completion: Failed to make JSON-serializable dict from chunk %d: %s",
                    i,
                    exc,
                    exc_info=True,
                )

            if chunk_text:
                yield chunk_text, chunk_dict

            i += 1

    def get_base_url(self) -> str:
        return str(self.client.base_url)
