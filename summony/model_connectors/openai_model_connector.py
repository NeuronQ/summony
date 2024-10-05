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
        api_key = creds.get("api_key") if creds else os.environ["OPENAI_API_KEY"]
        if client_args is None:
            client_args = {}
        self.client = OpenAI(
            api_key=api_key,
            **client_args,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            **client_args,
        )
        self.logger = logger if logger is not None else g_logger

    def generate_completion(self, messages, model, **kwargs) -> Tuple[str, dict]:
        completion = self.client.chat.completions.create(
            messages=messages, model=model, **kwargs
        )
        completion_text = completion.choices[0].message.content
        completion_dict = completion.model_dump(mode="json")
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

            chunk_dict = chunk.model_dump(mode="json")

            if chunk_text:
                yield chunk_text, chunk_dict

            i += 1

    def get_base_url(self) -> str:
        return str(self.client.base_url)
