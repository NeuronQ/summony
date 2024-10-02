from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self, Sequence
from uuid import uuid4

from dataclasses_json import dataclass_json

from ..utils import HashableDict


@dataclass_json
@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    chosen: bool | None = None

    # :: <params_idx> | <agent_idx> -> <params_idx>
    params: dict[int, int] | int | None = None

    def __iter__(self):
        # converting to dict with dict(my_msg) uses this (and .to_dict is added by @dataclass_json)
        for k, v in self.to_dict().items():
            yield k, v

    @classmethod
    def system(cls, content: str, **kwargs) -> Self:
        return cls(role="system", content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs) -> Self:
        return cls(role="user", content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> Self:
        return cls(role="assistant", content=content, **kwargs)


class AgentInterface:
    name: str
    messages: list[Message]
    model_name: str
    params: dict[str, Any]
    params_versions: list[HashableDict]

    raw_responses: dict[list]
    client: Any
    async_client: Any

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        name: str | None = None,
        system_prompt: str | None = None,
        creds: dict | None = None,
    ): ...

    @abstractmethod
    def ask(self, question: str, prefill: str | None = None, **kwargs) -> str: ...

    # TODO: fix ALL reask implementations to use prefill if any
    @abstractmethod
    def reask(self, **kwargs) -> str: ...

    @abstractmethod
    async def ask_async_stream(
        self, question: str, prefill: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""

    @abstractmethod
    async def reask_async_stream(self, **kwargs) -> AsyncIterator[str]:
        yield ""
