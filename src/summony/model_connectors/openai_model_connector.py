from copy import deepcopy
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

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .model_connectors import ModelConnectorInterface, MessageDict


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

    def generate(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        completion_create_args = self._make_completion_create_args(
            messages, model, kwargs
        )
        completion = self.client.chat.completions.create(**completion_create_args)
        return self._process_completion(completion)

    async def generate_async(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        completion_create_args = self._make_completion_create_args(
            messages, model, kwargs
        )
        completion = await self.async_client.chat.completions.create(
            **completion_create_args
        )
        return self._process_completion(completion)

    @staticmethod
    def _process_completion(completion: ChatCompletion) -> Tuple[str, dict]:
        completion_text = completion.choices[0].message.content
        completion_dict = completion.model_dump(mode="json")
        return completion_text, completion_dict

    def generate_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Iterator[Tuple[str, dict]]:
        completion_create_args = self._make_completion_create_args(
            messages, model, kwargs
        )
        if model.startswith("o1"):
            yield self.generate_async(**completion_create_args)
            return
        stream = self.client.chat.completions.create(
            **completion_create_args, stream=True
        )
        i = 0
        for chunk in stream:
            chunk_text, chunk_dict = self._process_chunk(chunk, i)
            if chunk_text:
                yield chunk_text, chunk_dict
            i += 1

    async def generate_async_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        completion_create_args = self._make_completion_create_args(
            messages, model, kwargs
        )
        if model.startswith("o1"):
            completion_text, completion_dict = await self.generate_async(
                **completion_create_args
            )
            yield completion_text, completion_dict
            return
        stream = await self.async_client.chat.completions.create(
            **completion_create_args, stream=True
        )
        i = 0
        async for chunk in stream:
            chunk_text, chunk_dict = self._process_chunk(chunk, i)
            if chunk_text:
                yield chunk_text, chunk_dict
            i += 1

    def _process_chunk(
        self, chunk: ChatCompletionChunk, chunk_idx: int
    ) -> Tuple[str, dict]:
        try:
            chunk_text = chunk.choices[0].delta.content
        except Exception as exc:
            chunk_text = None
            self.logger.warning(
                "OpenAIAgent: Failed to get content from chunk %d: %s",
                chunk_idx,
                exc,
                exc_info=True,
            )
        chunk_dict = chunk.model_dump(mode="json")
        return chunk_text, chunk_dict

    def get_base_url(self) -> str:
        return str(self.client.base_url)

    @classmethod
    def _make_completion_create_args(
        cls, messages: list[dict], model: str, extra_args: dict
    ) -> dict:
        out = dict(messages=messages, model=model)
        if model.startswith("o1"):
            if len(out["messages"]) and out["messages"][0]["role"] == "system":
                out["messages"] = deepcopy(messages)
                out["messages"][0]["role"] = "user"
        out.update(extra_args)
        return out
