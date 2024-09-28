from collections import defaultdict
import os
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from openai import OpenAI, AsyncOpenAI

from .agents import AgentInterface, Message
from ..utils import separate_prefixed


class OpenAIAgent(AgentInterface):
    messages: list[Message]
    model_name: str

    raw_responses: dict[list]

    client: OpenAI
    async_client: AsyncOpenAI

    _active_stream = None

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        creds: dict | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.model_name = model_name

        self.client = OpenAI(api_key=(creds.get('api_key') if creds else os.environ["OPENAI_API_KEY"]))
        self.async_client = AsyncOpenAI(api_key=(creds.get('api_key') if creds else os.environ["OPENAI_API_KEY"]))

        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message.system(system_prompt))

        if params is not None:
            self.params = params.copy()
        else:
            self.params = {}

        self.raw_responses = defaultdict(list)
        
    def ask(self, question: str, prefill: str | None = None, **kwargs) -> str:
        params_from_kwarg, left_kwargs = separate_prefixed(kwargs, 'p_')
        if left_kwargs:
            raise ValueError(f"ERROR in OpenAIAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}")

        self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        chat_completion = self.client.chat.completions.create(
            messages=[dict(m) for m in self.messages],
            model=self.model_name,
            **{**self.params, **params_from_kwarg},
        )

        self.raw_responses[len(self.messages)].append(chat_completion)

        client_message = chat_completion.choices[0].message
        message = Message(role=client_message.role, content=client_message.content)
        self.messages.append(message)

        return client_message.content

    async def ask_async_stream(self, question: str, prefill: str | None = None, **kwargs) -> AsyncIterator[str]:
        params_from_kwarg, left_kwargs = separate_prefixed(kwargs, 'p_')
        if left_kwargs:
            raise ValueError(f"ERROR in OpenAIAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}")

        self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        self._active_stream = await self.async_client.chat.completions.create(
            messages=[dict(m) for m in self.messages],
            model=self.model_name,
            stream=True,
            **{**self.params, **params_from_kwarg},
        )

        self.messages.append(Message.assistant(''))

        async for chunk in self._active_stream:
            self.raw_responses[len(self.messages) - 1].append(chunk)
            chunk_content = None
            try:
                chunk_content = chunk.choices[0].delta.content
            except:
                # TODO log error/warning or smth.
                pass
            if chunk_content:
                self.messages[-1].content += chunk_content
                yield chunk_content
