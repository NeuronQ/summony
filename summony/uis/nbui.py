import asyncio
from IPython.display import Markdown, HTML, display
import ipywidgets as widgets
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from ..agents import AgentInterface, Message, get_default_agent_for_model


class NBUI:
    agents: list[AgentInterface]
    is_agent_active: list[bool]

    _agent_coros: list

    def __init__(
        self,
        *,
        models: list[str] | None,
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
            for ag in self.agents:
                ag.messages.append(Message.system(system_prompt))
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
                        if i < len(self.agents):
                            self.agents[i].params[k[2:]] = vv
                        else:
                            break
                else:
                    for ag in self.agents:
                        ag.params[k[2:]] = v

        self.mode = mode

    async def __call__(self, q=None, prefill=None):
        await self.ask(q, prefill)

    async def ask(self, q=None, prefill=None):
        self._begin_show_reply_streams()
        self._agent_coros = [
            self._update_reply_stream_display(i, q, prefill)
            for i in range(len(self.agents))
            if self.is_agent_active[i]
        ]
        await asyncio.gather(*self._agent_coros)
        self._agent_coros = []
        self._end_show_reply_streams()
        self._show_last_replies()

    def set_active_agents(self, active_agent_idxs):
        self.is_agent_active = [
            (i in active_agent_idxs) for i in range(len(self.agents))
        ]

    async def _update_reply_stream_display(self, ag_idx, q, prefill):
        ag = self.agents[ag_idx]
        if q is not None:
            stream = ag.ask_async_stream(q, prefill)
        else:
            assert prefill is None
            stream = ag.reask_async_stream()
        async for _ in stream:
            texts = [
                (
                    ag.messages[-1].content
                    if not isinstance(ag.messages[-1], (list, tuple))
                    else ag.messages[-1][-1].content
                ).replace("\n", "<br>")
                for i, ag in enumerate(self.agents)
                if self.is_agent_active[i]
            ]
            if self.mode == "ipywidgets.table":
                self._render_reply_streams_mode_ipwtable(texts)
            elif self.mode == "ipywidgets.grid":
                self._render_reply_streams_mode_ipwgridbox(texts)
            else:
                raise ValueError(
                    f"ERROR in NBUI._update_reply_stream_display: Unknown mode: {self.mode}"
                )

    def _begin_show_reply_streams(self):
        self._show_reply_stream_style()
        self._current_reply_streams_container = self._build_reply_streams_container()
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

    def _build_reply_streams_container(self):
        if self.mode == "ipywidgets.table":
            return self._build_reply_streams_container_mode_ipwtable()
        elif self.mode == "ipywidgets.grid":
            return self._build_reply_streams_container_mode_ipwgridbox()
        else:
            raise ValueError(
                f"ERROR in NBUI._build_reply_streams_container: Unknown mode: {self.mode}"
            )

    def _build_reply_streams_container_mode_ipwtable(self):
        return widgets.HTML()

    def _build_reply_streams_container_mode_ipwgridbox(self):
        message_heads = [
            widgets.HTML(self._make_message_head_html(i))
            for i in range(len(self.agents))
            if self.is_agent_active[i]
        ]
        self._current_message_bodies = [
            widgets.HTML() for i in range(len(self.agents)) if self.is_agent_active[i]
        ]
        items = [
            *message_heads,
            *self._current_message_bodies,
        ]
        return widgets.GridBox(
            items,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({sum(self.is_agent_active)}, 1fr)",
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
                    background: orange;
                }
                .S6-Avatar.S6-AgentIdx-1 {
                    background: green;
                }

                .S6-Message-Head {
                    margin: 0.2rem 0;
                }

                .S6-ReplyBlock {
                    padding: 0.2rem 0.5rem;
                    border-radius: 8px;
                }

                .S6-ReplyBlock,
                .S6-ReplyBlock * {
                    text-align: left;
                    vertical-align: top;
                    line-height: 1.3rem;
                    font-family: monospace;

                }
                .S6-ReplyBlock-0 {
                    background: rgba(255,0,50,0.1)
                }
                .S6-ReplyBlock-1 {
                    background: rgba(0,255,50,0.1)
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

    def _end_show_reply_streams(self):
        self._current_reply_streams_accordion.selected_index = None  # to collapse

    def _show_last_replies(self):
        for i, ag in enumerate(self.agents):
            if self.is_agent_active[i]:
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
        return f'<span class="S6-Avatar S6-AgentIdx-{idx}">🤖 {name}</span>'
