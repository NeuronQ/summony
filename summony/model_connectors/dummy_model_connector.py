import asyncio
import json
import logging
from pathlib import Path
import random
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Literal,
    Self,
    Tuple,
)

from openai import OpenAI, AsyncOpenAI

from .model_connectors import ModelConnectorInterface, MessageDict


g_logger = logging.getLogger(__name__)


class DummyModelConnector(ModelConnectorInterface):
    model_name: str
    logger: logging.Logger
    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger if logger is not None else g_logger

    def generate(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        data_dir = (
            Path(__file__).parent.resolve() / "dummy_model_connector_data" / "simple"
        )
        json_files = list(data_dir.glob("*.json"))
        picked_file = random.choice(json_files)
        with open(picked_file, "r") as f:
            data = json.load(f)
        completion_dict = data["response"]
        completion_text = completion_dict["choices"][0]["message"]["content"]
        time.sleep(random.uniform(0.3, 0.6))
        return completion_text, completion_dict

    async def generate_async(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Tuple[str, dict]:
        return self.generate(messages, model, **kwargs)

    def generate_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> Iterator[Tuple[str, dict]]:
        for c in self._load_random_chunks():
            time.sleep(random.uniform(0.1, 0.4))
            yield c["choices"][0]["delta"]["content"], c

    async def generate_async_stream(
        self, messages: list[MessageDict], model: str, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        for c in self._load_random_chunks():
            await asyncio.sleep(random.uniform(0.1, 0.4))
            yield c["choices"][0]["delta"]["content"], c

    @staticmethod
    def _load_random_chunks() -> list[dict]:
        data_dir = (
            Path(__file__).parent.resolve() / "dummy_model_connector_data" / "streaming"
        )
        json_files = list(data_dir.glob("*.json"))
        picked_file = random.choice(json_files)
        with open(picked_file, "r") as f:
            data = json.load(f)
        return data["response"]["chunks"]

    def get_base_url(self) -> str:
        return "https://api.openai.com/v1/"
