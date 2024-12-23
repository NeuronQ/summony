import logging
import os
from typing import Any

from openai import OpenAI, AsyncOpenAI

from .openai_model_connector import OpenAIModelConnector


g_logger = logging.getLogger(__name__)


class XAIModelConnector(OpenAIModelConnector):
    def __init__(
        self,
        creds: dict | None = None,
        client_args: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        api_key = creds.get("api_key") if creds else os.environ["XAI_API_KEY"]
        if client_args is None:
            client_args = {}
        if "base_url" not in client_args:
            client_args["base_url"] = "https://api.x.ai/v1"
        self.client = OpenAI(
            api_key=api_key,
            **client_args,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            **client_args,
        )
        self.logger = logger if logger is not None else g_logger
