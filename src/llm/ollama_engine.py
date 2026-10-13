"""Ollama LLM engine using httpx (no ollama package required).

Communicates with the Ollama REST API at base_url.  Supports
both streaming and non-streaming chat completion, health checks via
/api/version or /api/tags, and automatic retry on connection
failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5
"""Short timeout for health-check requests."""

_HOST_TIMEOUT_SECONDS = 5
"""Timeout in seconds used when attempting to establish a connection
during health checks and model listing."""


class OllamaEngine:
    """Ollama chat engine backed by httpx.

    Parameters are sourced from Config but overridable at construction
    time.  No network activity or model pulling happens at construction
    time; the HTTP client is created lazily on first use.

    CPU-friendly: limits message history length, makes no GPU
    assumptions, and never auto-pulls models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
        num_ctx: int = 2048,
        num_predict: int = 512,
        temperature: float = 0.3,
        timeout: int = 90,
        max_messages: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = max(1, num_ctx)
        self.num_predict = max(1, num_predict)
        self.temperature = float(temperature)
        self.timeout = max(1, timeout)
        self.max_messages = max(1, max_messages)
        self._client: Optional[Any] = None
        self._closed = False

    def _import_httpx(self) -> Any:
        """Lazy-import httpx and return the module."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for OllamaEngine but is not installed. "
                "Install it with: pip install httpx"
            ) from exc
        return httpx

    def _get_client(self) -> Any:
        """Return the lazy-created httpx.Client (created on first call)."""
        if self._client is None:
            httpx = self._import_httpx()
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _trim_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Trim message history to stay within max_messages.

        Keeps the most recent messages; drops the oldest first.
        Always preserves at least the last message so the context
        is never empty.
        """
        if len(messages) <= self.max_messages:
            return list(messages)
        return list(messages[-self.max_messages :])

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }
        if system_prompt is not None:
            payload["system"] = system_prompt
        return payload

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Any:
        """Send an HTTP request with retry-once semantics.

        Retries the request exactly once on httpx.ConnectError
        or httpx.ConnectTimeout.  Other exceptions are not
        retried.
        """
        client = self._get_client()
        httpx = self._import_httpx()
        timeout = (
            timeout_seconds if timeout_seconds is not None else self.timeout
        )

        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < 2:
            try:
                return client.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=timeout,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                attempts += 1
                logger.warning(
                    "Connection attempt %d/2 failed for %s %s: %s",
                    attempts,
                    method,
                    url,
                    exc,
                )
                if attempts >= 2:
                    raise
        raise last_exc or RuntimeError("Unexpected request failure")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Send chat messages to Ollama and return the response.

        Args:
            messages: List of dicts with role and content keys.
            system_prompt: Optional system prompt prepended to context.
            stream: If True, yield response tokens as they arrive.

        Returns:
            Full response string when stream=False, or an iterator
            of response chunks when stream=True.

        Raises:
            RuntimeError: If httpx is not installed or the request fails.
        """
        if self._closed:
            raise RuntimeError(
                "OllamaEngine is closed; recreate via factory."
            )

        trimmed = self._trim_messages(messages)
        url = self._chat_endpoint()
        payload = self._build_payload(trimmed, system_prompt, stream)

        if stream:
            return self._chat_stream(url, payload)

        response = self._request("POST", url, json_body=payload)
        data = self._parse_json(response)
        return str(data.get("response", ""))

    def _chat_stream(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> Iterator[str]:
        """Stream chat response via chunked JSON lines."""
        response = self._request("POST", url, json_body=payload)
        try:
            for raw_line in response.iter_lines():
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping non-JSON stream line: %s", line[:200]
                    )
                    continue
                text = chunk.get("response", "")
                if text:
                    yield text
                if chunk.get("done", False):
                    break
        finally:
            if hasattr(response, "close"):
                try:
                    response.close()
                except Exception:
                    pass

    def _parse_json(self, response: Any) -> dict[str, Any]:
        """Parse JSON from an HTTP response, raising on failure."""
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse Ollama response as JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected Ollama response format: "
                f"expected dict, got {type(data).__name__}"
            )
        return data

    def health(self) -> bool:
        """Check Ollama server availability via /api/version or /api/tags.

        Uses a short timeout so the check does not block for long.
        Returns False on any error (including httpx not being installed).
        """
        if self._closed:
            return False

        self._import_httpx()
        endpoints = [
            f"{self.base_url}/api/version",
            f"{self.base_url}/api/tags",
        ]

        for endpoint in endpoints:
            try:
                response = self._request(
                    "GET", endpoint, timeout_seconds=_HOST_TIMEOUT_SECONDS
                )
                if response.status_code < 500:
                    return True
            except Exception:
                continue

        return False

    def list_models(self) -> list[str]:
        """List models available on the Ollama server.

        Returns an empty list if the server is unreachable or httpx
        is not installed.
        """
        if self._closed:
            return []

        self._import_httpx()
        url = f"{self.base_url}/api/tags"
        try:
            response = self._request(
                "GET", url, timeout_seconds=_HOST_TIMEOUT_SECONDS
            )
            data = self._parse_json(response)
            models = data.get("models", [])
            if isinstance(models, list):
                return [
                    m.get("model", "")
                    for m in models
                    if isinstance(m, dict)
                ]
        except Exception as exc:
            logger.debug("Failed to list models: %s", exc)

        return []

    def close(self) -> None:
        """Close the underlying httpx client if it was created."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._closed = True

    def __enter__(self) -> OllamaEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["OllamaEngine"]
