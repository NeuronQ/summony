import os
import logging
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self, Tuple

# from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai


from .model_connectors import ModelConnectorInterface


g_logger = logging.getLogger(__name__)


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

    def generate_completion(self, messages, model, **kwargs) -> Tuple[str, dict]:
        system_prompt, history, last_message_text = self._process_messages(messages)

        model = genai.GenerativeModel(
            model,
            system_instruction=system_prompt,
            generation_config=kwargs,
            **self.client_args,
        )

        chat = model.start_chat(history=history)

        response = chat.send_message(last_message_text)

        completion_text = response.text
        completion_dict = response.to_dict()

        return completion_text, completion_dict

    async def async_generate_completion(
        self, messages, model, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        system_prompt, history, last_message_text = self._process_messages(messages)

        model = genai.GenerativeModel(
            model,
            system_instruction=system_prompt,
            generation_config=kwargs,
            **self.client_args,
        )

        chat = model.start_chat(history=history)

        response = await chat.send_message_async(last_message_text, stream=True)

        i = 0
        async for chunk in response:
            try:
                chunk_text = chunk.text
            except Exception as exc:
                chunk_text = None
                self.logger.warning(
                    "GeminiAgent.async_generate_completion: Failed to get content from chunk %d: %s",
                    i,
                    exc,
                    exc_info=True,
                )
            chunk_dict = chunk.to_dict()
            yield chunk_text, chunk_dict
            i += 1

    def _process_messages(self, messages):
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
