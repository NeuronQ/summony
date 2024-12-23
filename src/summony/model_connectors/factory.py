from .model_connectors import ModelConnectorInterface
from .anthropic_model_connector import AnthropicModelConnector
from .openai_model_connector import OpenAIModelConnector
from .xai_model_connector import XAIModelConnector
from .gemini_model_connector import GeminiModelConnector
from .ollama_model_connector import OllamaModelConnector
from .dummy_model_connector import DummyModelConnector


def get_default_connector_for_model(model: str) -> ModelConnectorInterface:
    if model.startswith("dummy"):
        return DummyModelConnector()
    elif model.startswith("gpt") or model.startswith("o1"):
        return OpenAIModelConnector()
    elif model.startswith("grok"):
        return XAIModelConnector()
    elif model.startswith("claude"):
        return AnthropicModelConnector()
    elif model.startswith("gemini"):
        return GeminiModelConnector()
    elif model.startswith("ollama::"):
        return OllamaModelConnector()
    else:
        raise ValueError(
            f"Don't know how to create default model connector for model: {model!r}"
        )
