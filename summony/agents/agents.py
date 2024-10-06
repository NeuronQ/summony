from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Literal,
    Self,
    Sequence,
    Type,
)
from uuid import uuid4

from dataclasses_json import dataclass_json

from ..utils import separate_prefixed, HashableDict
from ..loggers import XLoggerInterface, DefaultXLogger
from ..model_connectors import ModelConnectorInterface


@dataclass_json
@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    chosen: bool | None = None

    # :: <params_idx> | <agent_idx> -> <params_idx>
    params: dict[int, int] | int | None = None
    log_path: str | None = None

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

    logger: XLoggerInterface
    raw_responses: dict[list]
    connector: ModelConnectorInterface

    MODEL_CONNECTOR_CLASS: Type[ModelConnectorInterface] = None

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        name: str | None = None,
        system_prompt: str | None = None,
        creds: dict | None = None,
        params: dict[str, Any] | None = None,
        logger: XLoggerInterface | None = None,
        client_args: dict[str, Any] | None = None,
    ): ...

    @abstractmethod
    def ask(
        self, question: str | None = None, prefill: str | None = None, **kwargs
    ) -> str: ...

    @abstractmethod
    async def ask_async_stream(
        self, question: str | None = None, prefill: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""


class BaseAgent(AgentInterface):
    name: str
    messages: list[Message]
    model_name: str
    params: dict[str, Any]
    params_versions: list[HashableDict]

    logger: XLoggerInterface
    raw_responses: dict[list]
    connector: ModelConnectorInterface

    # static
    MODEL_CONNECTOR_CLASS: Type[ModelConnectorInterface] = None

    _DEFAULT_PARAMS: dict[str, Any] = {
        # "max_tokens": 1024,
    }

    def __init__(
        self,
        model_name: str,
        name: str | None = None,
        system_prompt: str | None = None,
        creds: dict | None = None,
        params: dict[str, Any] | None = None,
        logger: XLoggerInterface | None = None,
        client_args: dict[str, Any] | None = None,
    ):
        self.model_name = model_name

        self.name = name if name is not None else model_name

        self.logger = logger if logger is not None else DefaultXLogger(name=self.name)

        self.connector = self.MODEL_CONNECTOR_CLASS(
            creds=creds, logger=logger, client_args=client_args
        )

        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message.system(system_prompt))

        if params is None:
            self.params = self._DEFAULT_PARAMS.copy()
        else:
            self.params = deepcopy(params)

        self.params_versions = []
        self._store_params_version(self.params)

        self.raw_responses = defaultdict(list)

    def ask(
        self, question: str | None = None, prefill: str | None = None, **kwargs
    ) -> str:
        if question is None:
            assert (
                prefill is None
            ), "When re-asking (question is None), prefill must also be None"

        params_from_kwargs, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            self.logger.warning(
                f"Warning in BaseAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        if question is not None:
            self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        params = {**self.params, **params_from_kwargs}
        params_version = self._store_params_version(params)

        model_call_params = dict(
            messages=self._make_agent_messages(
                self.messages if question is not None else self.messages[:-1]
            ),
            model=self.model_name,
            **params,
            **left_kwargs,
        )
        try:
            completion_text, completion_dict = self.connector.generate(
                **model_call_params
            )

            reply_message = Message.assistant(completion_text, params=params_version)

            if question is not None:
                self.messages.append(reply_message)

            else:
                if not isinstance(self.messages[-1], (list, tuple)):
                    self.messages[-1] = [self.messages[-1]]
                self.messages[-1].append(reply_message)

                self.raw_responses[len(self.messages)].append("<reask>")

            self.raw_responses[len(self.messages)].append(completion_dict)

            log_path = self.logger.log_model_call(
                req_content=model_call_params,
                req_base_url=self.connector.get_base_url(),
                res_content=completion_dict,
            )
            reply_message.log_path = log_path

        except Exception as exc:
            self.logger.exception("Error in BaseAgent.ask: %s", exc, exc_info=True)
            self.logger.log_model_call(
                req_content=model_call_params,
                req_base_url=self.connector.get_base_url(),
                error=exc,
            )
            raise exc

        return completion_text

    async def ask_async_stream(
        self, question: str | None = None, prefill: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        if question is None:
            assert (
                prefill is None
            ), "When re-asking (question is None), prefill must also be None"

        params_from_kwargs, left_kwargs = separate_prefixed(kwargs, "p_")
        if left_kwargs:
            self.logger.warning(
                f"Warning in BaseAgent.ask: unexpected kwargs: {list(left_kwargs.keys())}"
            )

        if question is not None:
            self.messages.append(Message.user(question))

        if prefill is not None:
            self.messages.append(Message.assistant(prefill))

        params = {**self.params, **params_from_kwargs}
        params_version = self._store_params_version(params)

        model_call_params = dict(
            messages=self._make_agent_messages(
                self.messages if question is not None else self.messages[:-1]
            ),
            model=self.model_name,
            **params,
            **left_kwargs,
        )
        try:
            reply_message = Message.assistant("")
            reply_message.params = params_version

            if question is not None:
                self.messages.append(reply_message)
            else:
                if not isinstance(self.messages[-1], (list, tuple)):
                    self.messages[-1] = [self.messages[-1]]
                self.messages[-1].append(reply_message)

                self.raw_responses[len(self.messages) - 1].append("<reask>")

            chunks_dicts = []
            async for (
                chunk_text,
                chunk_dict,
            ) in self.connector.generate_async_stream(**model_call_params):
                chunks_dicts.append(chunk_dict)
                self.raw_responses[len(self.messages) - 1].append(chunk_dict)
                reply_message.content += chunk_text
                yield chunk_text

            log_path = self.logger.log_model_call(
                req_content=model_call_params,
                req_base_url=self.connector.get_base_url(),
                res_content={"chunks": chunks_dicts},
            )
            reply_message.log_path = log_path

        except Exception as exc:
            self.logger.exception(
                "Error in BaseAgent.ask_async_stream: %s", exc, exc_info=True
            )
            self.logger.log_model_call(
                req_content=model_call_params,
                req_base_url=self.connector.get_base_url(),
                error=exc,
            )
            raise exc

    def _store_params_version(self, params: dict[str, Any]) -> int:
        hparams = HashableDict(params)
        if hparams in self.params_versions:
            return self.params_versions.index(hparams)
        self.params_versions.append(hparams)
        return len(self.params_versions) - 1

    @staticmethod
    def _make_agent_messages(messages: list[Message]) -> list[dict]:
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
