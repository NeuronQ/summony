import asyncio
from collections import defaultdict
from IPython.display import Markdown, HTML, display
import ipywidgets as widgets
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from ..agents import AgentInterface, Message, get_default_agent_for_model
from ..agents.serialization import hash_msg


class NBUI:
    agents: list[AgentInterface]
    is_agent_active: list[bool]

    _agent_coros: list

    def __init__(
        self,
        *,
        models: list[str] | None = None,
        agents: list[AgentInterface] | None = None,
        system_prompt: str | None = None,
        system_prompts: list[str] | None = None,
        mode: Literal["ipywidgets.table", "ipywidgets.grid"] = "ipywidgets.grid",
        **kwargs,
    ):
        assert (models is not None) or (agents is not None)
        if models is not None:
            assert agents is None
            self.agents = [get_default_agent_for_model(m) for m in models]
        elif agents is not None:
            assert models is None
            self.agents = agents

        self.is_agent_active = [True] * len(self.agents)

        if system_prompt is not None:
            assert system_prompts is None
            msg_system_prompt = Message.system(system_prompt)
            for ag in self.agents:
                ag.messages.append(msg_system_prompt)
        elif system_prompts is not None:
            assert system_prompt is None
            for i, sp in enumerate(system_prompts):
                if i < len(self.agents):
                    self.agents[i].messages.append(Message.system(sp))
                else:
                    break

        for k, v in kwargs.items():
            if k.startswith("p_"):
                if isinstance(v, (list, tuple)):
                    for i, vv in enumerate(v):
                        self.agents[i].params[k[2:]] = vv
                elif isinstance(v, dict):
                    for i, vv in v.items():
                        self.agents[i].params[k[2:]] = vv
                else:
                    for ag in self.agents:
                        ag.params[k[2:]] = v

        self.mode = mode

    async def __call__(
        self,
        q: str | None = None,
        prefill: str | None = None,
        to: list[int] | None = None,
    ):
        await self.ask(q, prefill, to)

    async def ask(
        self,
        q: str | None = None,
        prefill: str | None = None,
        to: list[int] | None = None,
    ):
        self._begin_show_reply_streams(to)

        self._agent_coros = []
        for i in range(len(self.agents)):
            if self.is_agent_active[i]:
                if to is None or i in to:
                    self._agent_coros.append(
                        self._ask_and_update_reply_stream_display(i, q, prefill, to)
                    )
            else:
                if to is not None and i in to:
                    raise ValueError(
                        f"ERROR in NBUI.ask: IGNORING agent {i} it's not active, but was requested to reply"
                    )

        await asyncio.gather(*self._agent_coros)

        self._agent_coros = []

        self._end_show_reply_streams(to)

        self._show_last_replies(to)

    def set_active_agents(self, active_agent_idxs):
        self.is_agent_active = [
            (i in active_agent_idxs) for i in range(len(self.agents))
        ]

    async def _ask_and_update_reply_stream_display(self, ag_idx, q, prefill, to):
        ag = self.agents[ag_idx]
        stream = ag.ask_async_stream(q, prefill)

        async for _ in stream:
            texts = [
                (
                    ag.messages[-1].content
                    if not isinstance(ag.messages[-1], (list, tuple))
                    else ag.messages[-1][-1].content
                ).replace("\n", "<br>")
                for i, ag in enumerate(self.agents)
                if self.is_agent_active[i] and (to is None or i in to)
            ]
            if self.mode == "ipywidgets.table":
                self._render_reply_streams_mode_ipwtable(texts)
            elif self.mode == "ipywidgets.grid":
                self._render_reply_streams_mode_ipwgridbox(texts)
            else:
                raise ValueError(
                    f"ERROR in NBUI._update_reply_stream_display: Unknown mode: {self.mode}"
                )

    def _begin_show_reply_streams(self, to):
        self._show_reply_stream_style()
        self._current_reply_streams_container = self._build_reply_streams_container(to)
        self._current_reply_streams_accordion = widgets.Accordion(
            children=[self._current_reply_streams_container],
            titles=["raw replies"],
            selected_index=0,  # to expand
        )
        display(self._current_reply_streams_accordion)

    def set_params(self, *args, **kwargs):
        if len(args) == 0:
            for k, v in kwargs.items():
                if isinstance(v, (list, tuple)):
                    for i, vv in enumerate(v):
                        if i < len(self.agents):
                            self.agents[i].params[k] = vv
                        else:
                            break
                else:
                    for ag in self.agents:
                        ag.params[k] = v
        elif len(args) == 1:
            ag_idx = args[0]
            self.agents[ag_idx].params.update(kwargs)
        else:
            raise ValueError(
                "ERROR in NBUI.set_params: only one positional argument expected (the agent index)"
            )

    def unset_params(self, *args):
        ag_idx = None
        if len(args) > 1 and type(args[0]) is int:
            ag_idx = args[0]
            args = args[1:]
        if ag_idx is None:
            for k in args:
                for ag in self.agents:
                    if k in ag.params:
                        del ag.params[k]
        else:
            ag = self.agents[ag_idx]
            for k in args:
                if k in ag.params:
                    del ag.params[k]

    def _build_reply_streams_container(self, to):
        if self.mode == "ipywidgets.table":
            return self._build_reply_streams_container_mode_ipwtable(to)
        elif self.mode == "ipywidgets.grid":
            return self._build_reply_streams_container_mode_ipwgridbox(to)
        else:
            raise ValueError(
                f"ERROR in NBUI._build_reply_streams_container: Unknown mode: {self.mode}"
            )

    def _build_reply_streams_container_mode_ipwtable(self):
        return widgets.HTML()

    def _build_reply_streams_container_mode_ipwgridbox(self, to):
        dispalyed_agents_count = sum(
            1
            for i in range(len(self.agents))
            if (self.is_agent_active[i] and (to is None or i in to))
        )
        message_heads = [
            widgets.HTML(self._make_message_head_html(i))
            for i in range(len(self.agents))
            if self.is_agent_active[i] and (to is None or i in to)
        ]
        self._current_message_bodies = [
            widgets.HTML()
            for i in range(len(self.agents))
            if self.is_agent_active[i] and (to is None or i in to)
        ]
        items = [
            *message_heads,
            *self._current_message_bodies,
        ]
        return widgets.GridBox(
            items,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({dispalyed_agents_count}, 1fr)",
                grid_gap="0 0.5rem",
            ),
        )

    def _show_reply_stream_style(self):
        style = """
            <style>
                .S6-Avatar {
                    background: black;
                    color: white;
                    font-weight: bold;
                    padding: 0.2rem 0.5rem;
                    border-radius: 8px;
                }
                .S6-Avatar.S6-AgentIdx-0 {
                    background: rebeccapurple;
                }
                .S6-Avatar.S6-AgentIdx-1 {
                    background: darkcyan;
                }
                .S6-Avatar.S6-AgentIdx-2 {
                    background: magenta;
                }
                .S6-Avatar.S6-AgentIdx-3 {
                    background: blueviolet;
                }
                .S6-Avatar.S6-AgentIdx-4 {
                    background: darkgoldenrod;
                }
                .S6-Avatar.S6-AgentIdx-5 {
                    background: darkorange;
                }
                .S6-Avatar.S6-AgentIdx-6 {
                    background: green;
                }
                .S6-Avatar.S6-AgentIdx-7 {
                    background: cornflowerblue;
                }
                

                .S6-Message-Head {
                    margin: 0.2rem 0;
                }

                .S6-ReplyBlock {
                    padding: 0.2rem 0.5rem;
                    border-radius: 12px;
                }

                .S6-ReplyBlock,
                .S6-ReplyBlock * {
                    text-align: left;
                    vertical-align: top;
                    line-height: 1.3rem;
                    font-family: monospace;

                }
                .S6-ReplyBlock-0 {
                    background: #6633991a;
                }
                .S6-ReplyBlock-1 {
                    background: #008b8b24;
                }
                .S6-ReplyBlock-2 {
                    background: #ff00ff1f;
                }
                .S6-ReplyBlock-3 {
                    background: #8a2be21a;
                }
                .S6-ReplyBlock-4 {
                    background: #b8860b2b;
                }
                .S6-ReplyBlock-5 {
                    background: #ff8c002b;
                }
                .S6-ReplyBlock-6 {
                    background: rgba(0,255,50,0.1);
                }
                .S6-ReplyBlock-7 {
                    background: #6495ed24;
                }
            </style>
            """
        display(HTML(style))

    def _render_reply_streams_mode_ipwtable(self, texts):
        cols = (
            f"""
            <td class="S6-ReplyBlock S6-ReplyBlock-{i}">
                {self._make_message_head_html(self._active_agent_idx_to_agent_idx(i))}
                <div>{t}</div>
            </td>
            """
            for i, t in enumerate(texts)
        )
        self._current_reply_streams_container.value = f"""
            <table style="width: 100%; table-layout:fixed">
                <tr>{"\n".join(cols)}</tr>
            </table>
            """

    def _active_agent_idx_to_agent_idx(self, active_agent_idx):
        return [i for i, is_active in enumerate(self.is_agent_active) if is_active][
            active_agent_idx
        ]

    def _render_reply_streams_mode_ipwgridbox(self, texts):
        for i, t in enumerate(texts):
            self._current_message_bodies[
                i
            ].value = f'<div class="S6-ReplyBlock S6-ReplyBlock-{i}">{t}</div>'

    def _end_show_reply_streams(self, to):
        self._current_reply_streams_accordion.selected_index = None  # to collapse

    def _show_last_replies(self, to):
        for i, ag in enumerate(self.agents):
            if self.is_agent_active[i] and (to is None or i in to):
                display(HTML(self._make_avatar_html(i, "Agent " + ag.model_name)))
                last_msg = (
                    ag.messages[-1]
                    if not isinstance(ag.messages[-1], (list, tuple))
                    else ag.messages[-1][-1]
                )
                display(Markdown(last_msg.content.replace("\n", "\n\n")))
                if i == len(self.agents) - 1:
                    display(HTML("<hr>"))

    def _make_message_head_html(self, agent_idx):
        return f"""
            <div class="S6-Message-Head">
                {self._make_avatar_html(agent_idx, 'Agent ' + self.agents[agent_idx].model_name)}
            </div>
            """

    def _make_avatar_html(self, idx, name):
        return f'<span class="S6-Avatar S6-AgentIdx-{idx}" style="border: 1px solid goldenrod">🤖 {name}</span>'

    def _make_message_display_levels(self, msg_by_id, msg_in_agents_count) -> dict:
        levels = {}

        for ag_idx, ag in enumerate(self.agents):
            curr_level_idx = -1
            for m in ag.messages:
                if (
                    not isinstance(m, (list, tuple))
                    and m.role != "assistant"
                    and msg_in_agents_count[hash_msg(m)] == len(self.agents)
                ):
                    curr_level_idx += 1

                # add to levels
                if curr_level_idx not in levels:
                    levels[curr_level_idx] = [[], defaultdict(list)]
                is_seq = isinstance(m, (list, tuple))
                h = hash_msg(m) if not is_seq else tuple(hash_msg(mm) for mm in m)
                if (
                    not isinstance(m, (list, tuple))
                    and m.role != "assistant"
                    and msg_in_agents_count[h] == len(self.agents)
                ):
                    h = hash_msg(m)
                    if h not in levels[curr_level_idx][0]:
                        levels[curr_level_idx][0].append(h)
                else:
                    if h not in levels[curr_level_idx][1][ag_idx]:
                        levels[curr_level_idx][1][ag_idx].append(h)

        return levels

    def show_conversation(self, short=None, mode="html"):
        msg_by_id = {}
        msg_in_agents_count = defaultdict(int)
        for ag in self.agents:
            for m in ag.messages:
                if not isinstance(m, (list, tuple)):
                    h = hash_msg(m)
                    msg_by_id[h] = m
                    msg_in_agents_count[h] += 1
                else:
                    for mm in m:
                        h = hash_msg(mm)
                        msg_by_id[h] = mm
                        msg_in_agents_count[h] += 1
        levels = self._make_message_display_levels(msg_by_id, msg_in_agents_count)
        if mode == "md":
            return self._make_conversation_md(levels, msg_by_id, short)
        else:
            return self._show_conversation(levels, msg_by_id, short)

    def _make_conversation_md(self, levels, msg_by_id, short=None):
        def msg2txt(msg):
            c = msg.content[:500] if short else msg.content
            return f"# {level_idx}:{i}:{j} << {msg.role} >> {hash_msg(msg)}\n```md\n{c}\n```\n"

        out = ""

        j = 0
        for level_idx, (head, body) in levels.items():
            i = 0
            out += "-" * 80 + "\n"
            out += "-" * 80 + "\n"
            for msg_id in head:
                msg = msg_by_id[msg_id]
                out += "\n" + msg2txt(msg) + "\n"
                i += 1
            for ag_idx, msg_ids in body.items():
                out += "-" * 80 + "\n"
                out += f"# --- << {self.agents[ag_idx].name} >> ---\n"
                for mid_or_group in msg_ids:
                    if not isinstance(mid_or_group, (list, tuple)):
                        msg = msg_by_id[mid_or_group]
                        out += "\n#" + msg2txt(msg) + "\n"
                    else:
                        for mid_idx, mid in enumerate(mid_or_group):
                            out += f"\n## --- (( variant {mid_idx} )) ---\n"
                            out += "\n" + msg2txt(msg) + "\n"
                    j += 1
                j = 0
                i += 1
            out += "\n"

        return out

    def _show_conversation(self, levels, msg_by_id, short=80):
        def _show_msg(msg, idx=None, ag_name=None):
            display(
                HTML(
                    self._make_avatar_html(
                        idx if msg.role == "assistant" else None,
                        f"Agent {ag_name}"
                        if ag_name and msg.role == "assistant"
                        else msg.role.upper(),
                    )
                    + f" &nbsp;&nbsp;{level_idx}:{i}:{j}"
                )
            )
            c = msg.content[:short] if short else msg.content
            display(Markdown(c.replace("\n", "\n\n")))

        j = 0
        ag_idx = None
        for level_idx, (head, body) in levels.items():
            i = 0
            display(HTML("<hr>"))
            for msg_id in head:
                msg = msg_by_id[msg_id]
                _show_msg(msg)
                i += 1
            for ag_idx, msg_ids in body.items():
                display(HTML('<hr style="border-style: dashed; border-bottom: 0">'))
                ag_name = self.agents[ag_idx].name
                for mid_or_group in msg_ids:
                    if not isinstance(mid_or_group, (list, tuple)):
                        msg = msg_by_id[mid_or_group]
                        _show_msg(msg, ag_idx, ag_name)
                    else:
                        for mid_idx, mid in enumerate(mid_or_group):
                            msg = msg_by_id[mid]
                            _show_msg(
                                msg,
                                ag_idx,
                                ag_name + f" #{mid_idx}",
                            )
                    j += 1
                j = 0
                i += 1
            ag_idx = None
