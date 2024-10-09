from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha1
import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Literal,
    Self,
    Sequence,
    TypedDict,
)

from dataclasses_json import dataclass_json

from .agents import AgentInterface, Message
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent
from .gemini_agent import GeminiAgent
from .ollama_agent import OllamaAgent
from .dummy_agent import DummyAgent


agent_classes = {
    "OpenAIAgent": OpenAIAgent,
    "AnthropicAgent": AnthropicAgent,
    "GeminiAgent": GeminiAgent,
    "OllamaAgent": OllamaAgent,
    "DummyAgent": DummyAgent,
}


class ConversationData(TypedDict):
    agents: list[dict]

    # :: <message_id> -> <Message>
    messages: dict[str, Message]

    # :: <agent_idx> -> <list of message ids>
    agent_messages: dict[int, list[str | list[str]]]

    # :: <agent_idx> -> <list of prams dicts>
    params: dict[int, list[dict[str, Any]]]


def hash_msg(m: Message) -> str:
    return sha1(str((m.role, m.content)).encode("utf-8")).hexdigest()


def conversation_to_dict(agents: list[AgentInterface]) -> ConversationData:
    agents_data = []
    messages_data = {}
    agent_messages = {}
    params = {}

    def add_to_messages_data(m: Message, ag_idx: int) -> str:
        m_clone = deepcopy(m)
        if type(m_clone.params) is int:
            m_clone.params = {ag_idx: m_clone.params}
        m_id = hash_msg(m)
        if m_id in messages_data:
            if messages_data[m_id]["params"] is None:
                messages_data[m_id]["params"] = m_clone.params
            elif m_clone.params:
                messages_data[m_id]["params"].update(m_clone.params)
        else:
            messages_data[m_id] = dict(m_clone)
        return m_id

    for ag_idx, ag in enumerate(agents):
        agents_data.append(
            {
                "name": ag.name,
                "model_name": ag.model_name,
                "class": ag.__class__.__name__,
                "params": ag.params,
            }
        )
        params[ag_idx] = ag.params_versions
        agent_messages[ag_idx] = []
        for m in ag.messages:
            if not isinstance(m, (tuple, list)):
                m_id = add_to_messages_data(m, ag_idx)
                agent_messages[ag_idx].append(m_id)
            else:
                m_list = []
                for mm in m:
                    mm_id = add_to_messages_data(mm, ag_idx)
                    m_list.append(mm_id)
                agent_messages[ag_idx].append(m_list)

    return dict(
        agents=agents_data,
        messages=messages_data,
        agent_messages=agent_messages,
        params=params,
    )


def conversation_from_dict(data: ConversationData) -> list[AgentInterface]:
    ags = []

    for ag_idx, ag_data in enumerate(data["agents"]):
        ag_cls = agent_classes[ag_data["class"]]
        ag = ag_cls(
            name=ag_data["name"],
            model_name=ag_data["model_name"],
            params=ag_data["params"],
        )
        ag.params_versions = data["params"][str(ag_idx)]
        ags.append(ag)

    for ag_idx, messages in data["agent_messages"].items():
        ag = ags[int(ag_idx)]
        for msg_or_list in messages:
            if not isinstance(msg_or_list, (list, tuple)):
                msg = msg_or_list
                ag.messages.append(Message(**data["messages"][msg]))
            else:
                ag.messages.append(
                    [Message(**data["messages"][msg]) for msg in msg_or_list]
                )

    return ags
