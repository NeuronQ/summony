from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Message:
    role: Literal['system', 'user', 'assistant']
    content: str

    def __iter__(self):
        # converting to dict with dict(my_msg) uses this (and .to_dict is added by @dataclass_json)
        for k, v in self.to_dict().items():
            yield k, v

    @classmethod
    def system(cls, content: str, **kwargs) -> Self:
        return cls(role='system', content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs) -> Self:
        return cls(role='user', content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: str, **kwargs) -> Self:
        return cls(role='assistant', content=content, **kwargs)


class AgentInterface:
    messages: list[Message]
    model_name: str
    params: dict[str, Any]

    raw_responses: dict[list]

    @abstractmethod
    def __init__(self, model_name: str, system_prompt: str | None = None, creds: dict | None = None):
        ...

    @abstractmethod
    def ask(self, question: str) -> str:
        ...

    @abstractmethod
    async def ask_async_stream(self, question: str) -> AsyncIterator[str]:
        yield ''
