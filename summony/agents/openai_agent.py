from collections import defaultdict
import os
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from openai import OpenAI, AsyncOpenAI

from .agents import AgentInterface, Message
from ..utils import separate_prefixed, HashableDict


class OpenAIAgent(AgentInterface):
    name: str
    messages: list[Message]
    model_name: str
    params: dict[str, Any]
    params_versions: list[HashableDict]

    raw_responses: dict[list]

    client: OpenAI
    async_client: AsyncOpenAI

    _active_stream = None

    # static
    _DEFAULT_PARAMS: dict[str, Any] = {
        "max_tokens": 1024,
    }

    def __init__(
        self,
        model_name: str,
        name: str | None = None,
        system_prompt: str | None = None,
        creds: dict | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.model_name = model_name

        self.name = name if name is not None else model_name

        self.client = OpenAI(
            api_key=(creds.get("api_key") if creds else os.environ["OPENAI_API_KEY"])
        )
        self.async_client = AsyncOpenAI(
            api_key=(creds.get("api_key") if creds else os.environ["OPENAI_API_KEY"])
        )

        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message.system(system_prompt))

        if params is not None:
            self.params = self._DEFAULT_PARAMS.copy()
        else:
            self.params = {}

        self.params_versions = []
        self._store_params_version(self.params)

        self.raw_responses = defaultdict(list)

    def ask(self, question: str, prefill: str | None = None, **kwargs) -> str:
        params_from_kwargs, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            raise ValueError(
                f"ERROR in OpenAIAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        params = {**self.params, **params_from_kwargs}
        params_version = self._store_params_version(params)

        chat_completion = self.client.chat.completions.create(
            messages=self._make_agent_messages(self.messages),
            model=self.model_name,
            **params,
        )

        self.raw_responses[len(self.messages)].append(chat_completion)

        client_message = chat_completion.choices[0].message
        message = Message(
            role=client_message.role,
            content=client_message.content,
            params=params_version,
        )
        self.messages.append(message)

        return client_message.content

    def reask(self, **kwargs) -> str:
        params_from_kwargs, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            raise ValueError(
                f"ERROR in OpenAIAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        params = {**self.params, **params_from_kwargs}
        params_version = self._store_params_version(params)

        chat_completion = self.client.chat.completions.create(
            messages=self._make_agent_messages(self.messages[:-1]),
            model=self.model_name,
            **params,
        )

        self.raw_responses[len(self.messages)].extend(["<reask>", chat_completion])

        client_message = chat_completion.choices[0].message
        message = Message(
            role=client_message.role,
            content=client_message.content,
            params=params_version,
        )

        if not isinstance(self.messages[-1], (list, tuple)):
            self.messages[-1] = [self.messages[-1]]
        self.messages[-1].append(message)

        return client_message.content

    async def ask_async_stream(
        self, question: str, prefill: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        params_from_kwargs, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            raise ValueError(
                f"ERROR in OpenAIAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        params = {**self.params, **params_from_kwargs}
        params_version = self._store_params_version(params)

        self._active_stream = await self.async_client.chat.completions.create(
            messages=self._make_agent_messages(self.messages),
            model=self.model_name,
            stream=True,
            **params,
        )

        reply_message = Message.assistant("")
        reply_message.params = params_version
        self.messages.append(reply_message)

        async for chunk in self._active_stream:
            self.raw_responses[len(self.messages) - 1].append(chunk)
            chunk_content = None
            try:
                chunk_content = chunk.choices[0].delta.content
            except:
                # TODO log error/warning or smth.
                pass
            if chunk_content:
                reply_message.content += chunk_content
                yield chunk_content

    async def reask_async_stream(self, **kwargs) -> AsyncIterator[str]:
        params_from_kwargs, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            raise ValueError(
                f"ERROR in OpenAIAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        params = {**self.params, **params_from_kwargs}
        params_version = self._store_params_version(params)

        self._active_stream = await self.async_client.chat.completions.create(
            messages=self._make_agent_messages(self.messages[:-1]),
            model=self.model_name,
            stream=True,
            **params,
        )

        message = Message.assistant("")
        message.params = params_version
        if not isinstance(self.messages[-1], (list, tuple)):
            self.messages[-1] = [self.messages[-1]]
        self.messages[-1].append(message)

        self.raw_responses[len(self.messages) - 1].append("<reask>")

        async for chunk in self._active_stream:
            self.raw_responses[len(self.messages) - 1].append(chunk)
            chunk_content = None
            try:
                chunk_content = chunk.choices[0].delta.content
            except:
                # TODO log error/warning or smth.
                pass
            if chunk_content:
                message.content += chunk_content
                yield chunk_content

    def _make_agent_messages(self, messages: list[Message]) -> None:
        out = []
        for i, m in enumerate(messages):
            if not isinstance(m, (list, tuple)):
                to_append = m
            else:
                assert len(m) > 0
                chosen = [alt_m for alt_m in m if alt_m.chosen]
                to_append = chosen[0] if chosen else m[-1]
            out.append({"role": to_append.role, "content": to_append.content})
        return out

    def _store_params_version(self, params: dict[str, Any]) -> int:
        hparams = HashableDict(params)
        if hparams in self.params_versions:
            return self.params_versions.index(hparams)
        self.params_versions.append(hparams)
        return len(self.params_versions) - 1
