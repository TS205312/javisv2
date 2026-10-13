"""Factory for creating LLM engines.

Provides create_llm_engine (builds an engine from a Config, with
fallback to UnavailableLLMEngine when Ollama is unreachable)
and get_default (loads default Config and returns an engine).
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from src.core.config import Config

from .base import LLMEngine
from .ollama_engine import OllamaEngine

logger = logging.getLogger(__name__)


class UnavailableLLMEngine(LLMEngine):
    """Fallback LLM engine returned when the Ollama server is unreachable.

    health() always returns False so callers know the server is
    down.  generate() / chat() return a deterministic Vietnamese
    response so that the assistant can still produce output and
    the UI / pipeline does not crash.
    """

    _RESPONSE: str = (
        "Xin loi, may chu Ollama hien khong kha dung. "
        "Vui long kiem tra dich vu Ollama va thu lai sau."
    )

    def __init__(self, reason: str = "Ollama server unreachable") -> None:
        self._reason = reason
        self._closed = False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Return the deterministic Vietnamese fallback message."""
        if self._closed:
            raise RuntimeError("UnavailableLLMEngine is closed.")
        if stream:
            return self._stream_response()
        return self._RESPONSE

    def health(self) -> bool:
        """Always False - this engine represents an unavailable backend."""
        return False

    def close(self) -> None:
        self._closed = True

    def list_models(self) -> list[str]:
        return []

    def _stream_response(self) -> Iterator[str]:
        """Yield the fallback response in a single chunk."""
        yield self._RESPONSE

    def generate(self, prompt: str) -> str:
        """Convenience method: return fallback response."""
        return self.chat([{"role": "user", "content": prompt}])

    def __enter__(self) -> UnavailableLLMEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _try_ollama_health(base_url: str) -> bool:
    """Probe the Ollama server for liveness with a short timeout."""
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed; cannot verify Ollama health.")
        return False

    for endpoint in (
        f"{base_url}/api/version",
        f"{base_url}/api/tags",
    ):
        try:
            with httpx.Client(base_url=base_url, timeout=5) as client:
                resp = client.get(endpoint, timeout=5)
                if resp.status_code < 500:
                    return True
        except Exception as exc:
            logger.debug("Health check failed for %s: %s", endpoint, exc)
            continue
    return False


def create_llm_engine(config: Optional[Config] = None) -> LLMEngine:
    """Create an LLM engine from a Config.

    If config is None, the default config is loaded.

    If the Ollama server is reachable (verified via a short health
    probe), returns an OllamaEngine.  Otherwise returns an
    UnavailableLLMEngine so the assistant can continue to run.
    """
    if config is None:
        config = Config.from_yaml()

    base_url = config.ollama_host

    if _try_ollama_health(base_url):
        logger.info("Ollama server reachable at %s", base_url)
        return OllamaEngine(
            base_url=base_url,
            model=config.llm_model,
            num_ctx=config.num_ctx,
            num_predict=config.num_predict,
            temperature=config.temperature,
            timeout=config.timeout,
        )

    logger.warning(
        "Ollama server unreachable at %s; using UnavailableLLMEngine",
        base_url,
    )
    return UnavailableLLMEngine(reason="Ollama unreachable at " + base_url)


def get_default() -> LLMEngine:
    """Return the default LLM engine using loaded Config.

    Equivalent to create_llm_engine(None).
    """
    return create_llm_engine(None)


__all__ = [
    "UnavailableLLMEngine",
    "create_llm_engine",
    "get_default",
]

