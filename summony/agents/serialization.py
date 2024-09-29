from copy import deepcopy
from dataclasses import dataclass
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

from . import agents
from .agents import AgentInterface, Message


class ConversationData(TypedDict):
    agents: list[dict]

    # :: <message_id> -> <Message>
    messages: dict[str, Message]

    # :: <agent_idx> -> <list of message ids>
    agent_messages: dict[int, list[str | list[str]]]

    # :: <agent_idx> -> <list of prams dicts>
    params: dict[int, list[dict[str, Any]]]


def conversation_to_dict(agents: list[AgentInterface]) -> ConversationData:
    agents_data = []
    messages_data = {}
    agent_messages = {}
    params = {}

    for ag_idx, ag in enumerate(agents):
        agents_data.append(
            {
                "name": ag.name,
                "model_name": ag.model_name,
                "class": ag.__class__.__name__,
            }
        )
        params[ag_idx] = [ag.params]
        agent_messages[ag_idx] = []
        for m in ag.messages:
            if isinstance(m, Message):
                m_clone = deepcopy(m)
                m_clone.params_idx = (ag_idx, 0)
                m_id = hash((m.role, m.content))
                messages_data[m_id] = m_clone
                agent_messages[ag_idx].append(m_id)
            else:
                m_list = []
                for mm in m:
                    mm_clone = deepcopy(mm)
                    mm_clone.params_idx = (ag_idx, 0)
                    mm_id = hash((mm.role, mm.content))
                    messages_data[mm_id] = mm_clone
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
        ag_cls = getattr(agents, ag_data["class"])
        params = data["params"][ag_idx][-1]
        ag = ag_cls(
            name=ag_data["name"], model_name=ag_data["model_name"], params=params
        )
        ags.append(ag)

    for ag_idx, mid in data["agent_messages"].items():
        ag = ags[ag_idx]
        if not isinstance(mid, (list, tuple)):
            ag.messages.append(deepcopy(data["messages"][mid]))
        else:
            ag.messages.append([deepcopy(data["messages"][mi]) for mi in mid])

    return ags
