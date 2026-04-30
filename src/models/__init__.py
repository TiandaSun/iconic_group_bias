"""VLM wrapper classes for local and API-based models."""

from src.models.base import (
    BaseVLM,
    VLMAPIError,
    VLMError,
    VLMInferenceError,
    retry_with_exponential_backoff,
)

# Open-source models (local inference)
from src.models.llama_vision import LLaMAVision
from src.models.qwen_vl import QwenVL

# Proprietary models (API)
from src.models.claude_api import ClaudeAPI
from src.models.gemini_api import GeminiAPI
from src.models.gpt4_api import GPT4VisionAPI

__all__ = [
    # Base classes and utilities
    "BaseVLM",
    "VLMError",
    "VLMAPIError",
    "VLMInferenceError",
    "retry_with_exponential_backoff",
    # Open-source models
    "QwenVL",
    "LLaMAVision",
    # API models
    "GPT4VisionAPI",
    "GeminiAPI",
    "ClaudeAPI",
]
