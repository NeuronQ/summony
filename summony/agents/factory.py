from .agents import AgentInterface
from .anthropic_agent import AnthropicAgent
from .openai_agent import OpenAIAgent
from .gemini_agent import GeminiAgent
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
    else:
        raise ValueError(f"Don't know how to create default agent for model: {model!r}")
