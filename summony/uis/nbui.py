import asyncio
from IPython.display import Markdown, HTML, display
import ipywidgets as widgets
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from ..agents import AgentInterface, Message, get_default_agent_for_model


class NBUI:
    agents: list[AgentInterface]

    _agent_coros: list

    def __init__(self, models: list[str] | None , agents: list[AgentInterface] | None = None):
        assert (models is not None) or (agents is not None)
        if models is not None:
            assert agents is None
            self.agents = [get_default_agent_for_model(m) for m in models]
        elif agents is not None:
            assert models is None
            self.agents = agents

    async def __call__(self, q):
        await self.ask(q)

    async def ask(self, q):
        self._begin_show_reply_streams()
        self._agent_coros = [self._update_reply_stream_display(i, q) for i in range(len(self.agents))]
        await asyncio.gather(*self._agent_coros)
        self._agent_coros = []
        self._end_show_reply_streams()
        self._show_last_replies()
    
    async def _update_reply_stream_display(self, ag_idx, q):
        ag = self.agents[ag_idx]
        async for _ in ag.ask_async_stream(q):
            self._render_reply_streams()

    def _begin_show_reply_streams(self):
        self._show_reply_stream_style()
        self._current_reply_streams_html = widgets.HTML()
        self._current_reply_streams_accordion = widgets.Accordion(
            children=[self._current_reply_streams_html],
            titles=['raw replies'],
            selected_index=0,  # to expand
        )
        display(self._current_reply_streams_accordion)

    def _show_reply_stream_style(self):
        style = '''
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
                    margin-bottom: 0.5rem;
                }

                .S6-ReplyBlock td {
                    text-align: left;
                    vertical-align: top;
                    line-height: 1.3rem;
                    font-family: monospace;
                }
                .S6-ReplyBlock td:nth-child(1) {
                    background: rgba(255,0,50,0.1)
                }
                .S6-ReplyBlock td:nth-child(2) {
                    background: rgba(0,255,50,0.1)
                }
            </style>
            '''
        display(HTML(style))

    def _render_reply_streams(self):
        texts = [ag.messages[-1].content.replace("\n", "<br>")
                 for ag in self.agents]
        cols = (
            f'''
            <td>
                <div class="S6-Message-Head">
                    {self._make_avatar_html(i, 'Agent ' + self.agents[i].model_name)}
                </div>
                <div>{t}</div>
            </td>
            '''
            for i, t in enumerate(texts)
        )
        self._current_reply_streams_html.value = f'''
            <table class="S6-ReplyBlock" style="width: 100%; table-layout:fixed">
                <tr>{"\n".join(cols)}</tr>
            </table>
            '''
        
    def _end_show_reply_streams(self):
        self._current_reply_streams_accordion.selected_index = None  # to collapse

    def _show_last_replies(self):
        for i, ag in enumerate(self.agents):
            display(HTML(self._make_avatar_html(i, 'Agent ' + ag.model_name)))
            display(Markdown(ag.messages[-1].content.replace("\n", "\n\n")))
            if i == len(self.agents) - 1:
                display(HTML("<hr>"))

    def _make_avatar_html(self, idx, name):
        return f'<span class="S6-Avatar S6-AgentIdx-{idx}">🤖 {name}</span>'
    