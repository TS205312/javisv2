"""Base abstract class for LLM engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class LLMEngine(ABC):
    """Abstract base class for all LLM engines in Jarvis.

    Concrete engines must implement chat, health, and close.
    The base class provides convenience methods generate
    and list_models that delegate to the concrete implementation
    or return sensible defaults.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Send a chat message and return the model response.

        Args:
            messages: List of message dicts with role and content keys.
            system_prompt: Optional system-level instruction.
            stream: If True, yield response chunks. If False, return full text.

        Returns:
            Full response string when stream=False, or an iterator of
            response chunks when stream=True.
        """

    @abstractmethod
    def health(self) -> bool:
        """Check whether the engine is reachable and operational."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the engine."""

    def generate(self, prompt: str) -> str:
        """Convenience method: send a single user prompt and get a response.

        Equivalent to:
            engine.chat([{"role": "user", "content": prompt}])
        """
        return self.chat([{"role": "user", "content": prompt}], stream=False)

    def list_models(self) -> list[str]:
        """List available models on the server.

        Default implementation returns an empty list; subclasses may
        override to query the server for available models.
        """
        return []

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
