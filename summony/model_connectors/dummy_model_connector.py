import asyncio
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self, Tuple

from openai import OpenAI, AsyncOpenAI

from .model_connectors import ModelConnectorInterface


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

    def generate_completion(self, messages, model, **kwargs) -> Tuple[str, dict]:
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

    async def async_generate_completion(
        self, messages, model, **kwargs
    ) -> AsyncIterator[Tuple[str, dict]]:
        data_dir = (
            Path(__file__).parent.resolve() / "dummy_model_connector_data" / "streaming"
        )
        json_files = list(data_dir.glob("*.json"))
        picked_file = random.choice(json_files)
        with open(picked_file, "r") as f:
            data = json.load(f)
        chunks = data["response"]["chunks"]
        for c in chunks:
            await asyncio.sleep(random.uniform(0.1, 0.4))
            yield c["choices"][0]["delta"]["content"], c

    def get_base_url(self) -> str:
        return "https://api.openai.com/v1/"
