"""LLM module for Jarvis - Ollama-backed chat engine with fallback.

Provides:
    - LLMEngine - abstract base for all engines.
    - OllamaEngine - production engine.
    - UnavailableLLMEngine - fallback when server is down.
    - create_llm_engine - factory from Config.
    - get_default - default engine convenience.
"""

from __future__ import annotations

from .base import LLMEngine
from .ollama_engine import OllamaEngine
from .factory import UnavailableLLMEngine, create_llm_engine, get_default

__all__ = [
    "LLMEngine",
    "OllamaEngine",
    "UnavailableLLMEngine",
    "create_llm_engine",
    "get_default",
]

