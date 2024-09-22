from collections import defaultdict
import os
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from openai import OpenAI, AsyncOpenAI

from .agents import AgentInterface, Message


class OpenAIAgent(AgentInterface):
    messages: list[Message]
    model_name: str

    raw_responses: dict[list]

    client: OpenAI
    async_client: AsyncOpenAI

    _active_stream = None

    def __init__(self, model_name: str, system_prompt: str | None = None, creds: dict | None = None):
        self.model_name = model_name

        self.client = OpenAI(api_key=(creds.get('api_key') if creds else os.environ["OPENAI_API_KEY"]))
        self.async_client = AsyncOpenAI(api_key=(creds.get('api_key') if creds else os.environ["OPENAI_API_KEY"]))

        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message.system(system_prompt))

        self.raw_responses = defaultdict(list)
        
    def ask(self, question: str) -> str:
        self.messages.append(Message.user(question))

        chat_completion = self.client.chat.completions.create(
            messages=[dict(m) for m in self.messages],
            model=self.model_name,
        )

        self.raw_responses[len(self.messages)].append(chat_completion)

        client_message = chat_completion.choices[0].message
        message = Message(role=client_message.role, content=client_message.content)
        self.messages.append(message)

        return client_message.content

    async def ask_async_stream(self, question: str) -> AsyncIterator[str]:
        self.messages.append(Message.user(question))

        self._active_stream = await self.async_client.chat.completions.create(
            messages=[dict(m) for m in self.messages],
            model=self.model_name,
            stream=True,
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
