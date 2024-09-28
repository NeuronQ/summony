from collections import defaultdict
import os
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from anthropic import Anthropic, AsyncAnthropic

from .agents import AgentInterface, Message
from ..utils import separate_prefixed


class AnthropicAgent(AgentInterface):
    messages: list[Message]
    model_name: str
    params: dict[str, Any]

    raw_responses: dict[list]

    client: Anthropic
    async_client: AsyncAnthropic

    # specific
    provided_model_name: str

    _active_stream = None

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
        model_name: str,
        system_prompt: str | None = None,
        creds: dict | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.provided_model_name = model_name
        self.model_name = self._MODEL_SHORTCUTS.get(model_name, model_name)

        self.client = Anthropic(
            api_key=(creds.get("api_key") if creds else os.environ["ANTHROPIC_API_KEY"])
        )
        self.async_client = AsyncAnthropic(
            api_key=(creds.get("api_key") if creds else os.environ["ANTHROPIC_API_KEY"])
        )

        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message.system(system_prompt))

        if params is not None:
            self.params = params.copy()
        else:
            self.params = {}

        self.raw_responses = defaultdict(list)

    def ask(self, question: str, prefill: str | None = None, **kwargs) -> str:
        params_from_kwarg, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            raise ValueError(
                f"ERROR in AnthropicAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        client_message = self.client.messages.create(
            **self._make_message_creat_args(
                {**self.params, **params_from_kwarg},
            )
        )

        self.raw_responses[len(self.messages)].append(client_message)

        content = client_message.content[0].text
        message = Message(role=client_message.role, content=content)
        self.messages.append(message)

        return content

    async def ask_async_stream(
        self, question: str, prefill: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        params_from_kwarg, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            raise ValueError(
                f"ERROR in AnthropicAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        self.messages.append(Message.assistant(""))

        self._active_stream = await self.async_client.messages.create(
            **self._make_message_creat_args({**self.params, **params_from_kwarg}),
            stream=True,
        )

        async for event in self._active_stream:
            self.raw_responses[len(self.messages) - 1].append(event)
            chunk_content = None
            try:
                if event.type == "content_block_start":
                    chunk_content = event.content_block.text
                elif event.type == "content_block_delta":
                    chunk_content = event.delta.text
            except:
                # TODO
                pass
            if chunk_content:
                self.messages[-1].content += chunk_content
                yield chunk_content

    def _make_message_creat_args(self, params):
        message_create_args = dict(
            messages=[dict(m) for m in self.messages],
            model=self.model_name,
            max_tokens=1024,
            **params,
        )
        if self.messages[0].role == "system":
            message_create_args["messages"] = message_create_args["messages"][1:]
            message_create_args["system"] = self.messages[0].content
        return message_create_args