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
    TypedDict,
)

import google.generativeai as genai

from .model_connectors import ModelConnectorInterface, MessageDict


g_logger = logging.getLogger(__name__)


class GeminiMessageDict(TypedDict):
    role: Literal["user", "model"]
    parts: Any


class GeminiModelConnector(ModelConnectorInterface):
    logger: logging.Logger

    client_args: dict

    # static
    _ROLES_MAP = {
        "user": "user",
        "assistant": "model",
    }

    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        api_key = creds.get("api_key") if creds else os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        self.client_args = client_args or {}
        self.logger = logger if logger is not None else g_logger

    def generate(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        chat_session, last_message_text = self._make_chat_session(
            messages, model, **kwargs
        )
        response = chat_session.send_message(last_message_text)
        return response.text, response.to_dict()

    async def generate_async(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        chat_session, last_message_text = self._make_chat_session(
            messages, model, **kwargs
        )
        response = await chat_session.send_message_async(last_message_text)
        return response.text, response.to_dict()

    def _make_chat_session(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[genai.ChatSession, str]:
        system_prompt, history, last_message_text = self._process_messages(messages)
        model: genai.GenerativeModel = genai.GenerativeModel(
            model,
            system_instruction=system_prompt,
            generation_config=kwargs,
            **self.client_args,
        )
        chat_session = model.start_chat(history=history)
        return chat_session, last_message_text

    def _process_chunk(
        self, chunk: genai.types.GenerateContentResponse, chunk_idx: int
    ) -> Tuple[str, dict]:
        try:
            chunk_text = chunk.text
        except Exception as exc:
            chunk_text = None
            self.logger.warning(
                "GeminiAgent: Failed to get content from chunk %d: %s",
                chunk_idx,
                exc,
                exc_info=True,
            )
        chunk_dict = chunk.to_dict()
        return chunk_text, chunk_dict

    def generate_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Iterator[Tuple[str, dict]]:
        chat_session, last_message_text = self._make_chat_session(
            messages, model, **kwargs
        )
        response = chat_session.send_message(last_message_text, stream=True)
        i = 0
        for chunk in response:
            chunk_text, chunk_dict = self._process_chunk(chunk, i)
            if chunk_text:
                yield chunk_text, chunk_dict
            i += 1

    async def generate_async_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        chat_session, last_message_text = self._make_chat_session(
            messages, model, **kwargs
        )
        response = await chat_session.send_message_async(last_message_text, stream=True)
        i = 0
        async for chunk in response:
            chunk_text, chunk_dict = self._process_chunk(chunk, i)
            if chunk_text:
                yield chunk_text, chunk_dict
            i += 1

    def _process_messages(
        self, messages: list[MessageDict]
    ) -> Tuple[str, list[GeminiMessageDict], str]:
        if messages[0]["role"] == "system":
            messages = messages[1:]
            system_prompt = messages[0]["content"]
        else:
            system_prompt = None
        history = [
            {"role": self._ROLES_MAP[m["role"]], "parts": m["content"]}
            for m in messages[:-1]
        ]
        last_message = messages[-1]
        assert last_message["role"] == "user"
        last_message_text = last_message["content"]
        return system_prompt, history, last_message_text

    def get_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/"
