"""Live hello checks for the configured provider models."""
import time
import unittest
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from core import config
from core import llm_helper


ALLOWED_CLASSIFICATIONS = {
    "PASS",
    "KEY_INVALID",
    "MODEL_UNAVAILABLE",
    "RATE_LIMIT",
    "NETWORK_ERROR",
    "EMPTY_RESPONSE",
}


def _send_hello(provider, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
    }
    headers = {"Content-Type": "application/json"}
    params = None
    if provider == "Kilo":
        url = "https://api.kilo.ai/api/gateway/chat/completions"
        headers["Authorization"] = f"Bearer {config.KILO_API_KEY}"
    elif provider == "OpenRouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers["Authorization"] = f"Bearer {config.OPENROUTER_API_KEY}"
    elif provider == "Groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {config.GROQ_API_KEY}"
    elif provider == "Gemini":
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        params = {"key": config.GEMINI_API_KEY}
        payload = {
            "contents": [{"parts": [{"text": "hello"}]}],
            "generationConfig": {"maxOutputTokens": 32},
        }
    elif provider == "Ollama":
        url = f"{config.OLLAMA_HOST.rstrip('/')}/api/generate"
        payload = {"model": model, "prompt": "hello", "stream": False}
    else:
        raise ValueError(f"Unknown provider: {provider}")

    response = requests.post(
        url,
        headers=headers,
        params=params,
        json=payload,
        timeout=180 if provider == "Ollama" else 60,
    )
    status = response.status_code
    if status != 200:
        raise requests.HTTPError(response=response)

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        return status, ""
    if provider == "Gemini":
        candidates = data.get("candidates", [])
        if not candidates:
            return status, ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return status, " ".join(
            str(part.get("text", "")) for part in parts if part.get("text")
        )
    if provider == "Ollama":
        return status, data.get("response", "")
    choices = data.get("choices", [])
    if not choices:
        return status, ""
    return status, choices[0].get("message", {}).get("content") or ""


def _classify_error(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    status = status or getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        status_code = None
    if status_code in (401, 403):
        return status_code, "KEY_INVALID"
    if status_code == 404:
        return status_code, "MODEL_UNAVAILABLE"
    if status_code == 429:
        return status_code, "RATE_LIMIT"
    if isinstance(exc, requests.RequestException):
        return status_code or "N/A", "NETWORK_ERROR"
    if status_code and status_code >= 500:
        return status_code, "NETWORK_ERROR"
    return status or "N/A", "NETWORK_ERROR"


class ProviderLiveTests(unittest.TestCase):
    results = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        print("\nProvider | Model | HTTP | Content? | Latency | Phân loại | Ghi chú")
        print("--- | --- | --- | --- | ---: | --- | ---")
        for provider, model, status, has_content, latency, classification, note in cls.results:
            print(
                f"{provider} | {model} | {status} | "
                f"{'yes' if has_content else 'no'} | {latency:.0f} ms | "
                f"{classification} | {note}"
            )

    def _request(
        self, provider, model, call, missing_key=False, assert_pass=False
    ):
        started = time.perf_counter()
        if missing_key:
            result = ("SKIP", "KEY_INVALID", "API key trống", False)
        else:
            try:
                status, content = call()
                if status != 200:
                    result = (
                        str(status), "NETWORK_ERROR",
                        "HTTP status is not 200", False,
                    )
                elif not isinstance(content, str) or not content.strip():
                    result = (
                        str(status), "EMPTY_RESPONSE",
                        "response không có content", False,
                    )
                elif "thinking process" in content.casefold() or "<thinking>" in content.casefold():
                    result = (
                        str(status), "EMPTY_RESPONSE",
                        "content bị lọc do thinking", False,
                    )
                else:
                    result = (str(status), "PASS", "có content", True)
            except Exception as exc:
                status, classification = _classify_error(exc)
                result = (str(status), classification, type(exc).__name__, False)

        latency = (time.perf_counter() - started) * 1000
        self.results.append((
            provider, model, result[0], result[3], latency, result[1], result[2]
        ))
        print(f"[{provider}] model={model} status={result[0]} {result[1]}")
        self.assertIn(result[1], ALLOWED_CLASSIFICATIONS)
        if not missing_key and assert_pass:
            self.assertEqual(
                result[1],
                "PASS",
                f"{provider} model {model} did not return non-empty hello content",
            )
        return result[1]

    def test_kilo(self):
        self._request(
            "Kilo",
            config.KILO_MODEL,
            lambda: _send_hello("Kilo", config.KILO_MODEL),
            missing_key=not config.KILO_API_KEY.strip(),
        )

    def test_openrouter(self):
        for model in (
            config.OPENROUTER_PRIMARY_MODEL,
            config.OPENROUTER_FALLBACK_MODEL,
        ):
            if not config.OPENROUTER_API_KEY.strip():
                self._request(
                    "OpenRouter", model, lambda: (0, ""),
                    missing_key=True, assert_pass=False,
                )
                continue

            available, reason = llm_helper.openrouter_model_is_available(model)
            if not available:
                self.results.append((
                    "OpenRouter", model, "SKIP", False, 0,
                    "MODEL_UNAVAILABLE", reason,
                ))
                print(
                    f"[OpenRouter] model={model} HTTP=SKIP "
                    f"MODEL_UNAVAILABLE ({reason})"
                )
                continue
            self._request(
                "OpenRouter",
                model,
                lambda model=model: _send_hello("OpenRouter", model),
                assert_pass=False,
            )

    def test_gemini(self):
        self._request(
            "Gemini",
            config.GEMINI_MODEL,
            lambda: _send_hello("Gemini", config.GEMINI_MODEL),
            missing_key=not config.GEMINI_API_KEY.strip(),
        )

    def test_groq(self):
        for model in (config.GROQ_MODEL, config.GROQ_FALLBACK_MODEL):
            self._request(
                "Groq",
                model,
                lambda model=model: _send_hello("Groq", model),
                missing_key=not config.GROQ_API_KEY.strip(),
                assert_pass=False,
            )

    def test_ollama(self):
        installed, reason = llm_helper.ollama_model_is_installed(config.OLLAMA_MODEL)
        if installed:
            classification = self._request(
                "Ollama",
                config.OLLAMA_MODEL,
                lambda: _send_hello("Ollama", config.OLLAMA_MODEL),
                assert_pass=True,
            )
            self.assertIn(classification, ALLOWED_CLASSIFICATIONS)
        else:
            self.results.append(
                (
                    "Ollama",
                    config.OLLAMA_MODEL,
                    "SKIP",
                    False,
                    0,
                    "MODEL_UNAVAILABLE",
                    reason,
                )
            )
            print(f"[Ollama] model={config.OLLAMA_MODEL} MODEL_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
