from .agents import AgentInterface, Message

from .openai_agent import OpenAIAgent
from .xai_agent import XAIAgent
from .deepseek_agent import DeepSeekAgent
from .anthropic_agent import AnthropicAgent
from .gemini_agent import GeminiAgent
from .ollama_agent import OllamaAgent
from .dummy_agent import DummyAgent

from .factory import get_default_agent_for_model
