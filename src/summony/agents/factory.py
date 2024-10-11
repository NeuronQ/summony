from .agents import AgentInterface
from .anthropic_agent import AnthropicAgent
from .openai_agent import OpenAIAgent
from .gemini_agent import GeminiAgent
from .ollama_agent import OllamaAgent
from .dummy_agent import DummyAgent


def get_default_agent_for_model(model: str) -> AgentInterface:
    if model.startswith("dummy"):
        return DummyAgent(model_name=model)
    elif model.startswith("gpt") or model.startswith("o1"):
        return OpenAIAgent(model_name=model)
    elif model.startswith("claude"):
        return AnthropicAgent(model_name=model)
    elif model.startswith("gemini"):
        return GeminiAgent(model_name=model)
    elif model.startswith("ollama::"):
        return OllamaAgent(model_name=model.split("::", 1)[-1])
    else:
        raise ValueError(f"Don't know how to create default agent for model: {model!r}")
