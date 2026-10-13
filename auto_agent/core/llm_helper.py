"""LLM client with free-first Ollama, Groq, and OpenRouter providers."""
import copy
import json
import logging
import re
import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path

import requests

from core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_FALLBACK_MODEL,
    GROQ_MODEL,
    KILO_API_KEY,
    KILO_MODEL,
    OLLAMA_HOST,
    OLLAMA_NUM_PARALLEL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_FREE_POOL,
    OPENROUTER_MODEL,
    OPENROUTER_PRIMARY_MODEL,
)

REQUEST_TIMEOUT = 60
OLLAMA_REQUEST_TIMEOUT = 180
logger = logging.getLogger(__name__)
ROUTER_STATE_PATH = Path("logs/router_state.json")
OPENROUTER_MODELS_CACHE_PATH = Path("logs/openrouter_models.json")
OPENROUTER_MODELS_CACHE_TTL = 60 * 60
OPENROUTER_MODEL_FAILURE_WINDOW = 5 * 60
OPENROUTER_MODEL_BLOCK_DURATION = 30 * 60
_STATE_LOCK = threading.RLock()
_OLLAMA_SEMAPHORE = threading.BoundedSemaphore(max(1, OLLAMA_NUM_PARALLEL))
_DEFAULT_ROUTER_STATE = {
    "last_good_tier": 1,
    "last_good_model": GROQ_MODEL,
    "blocked_tiers": {},
    "blocked_models": {},
    "model_failures": {},
    "daily_count": {},
    "last_reset_day": date.today().isoformat(),
}


def _save_router_state(state):
    ROUTER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ROUTER_STATE_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(ROUTER_STATE_PATH)


def _load_router_state():
    with _STATE_LOCK:
        state = copy.deepcopy(_DEFAULT_ROUTER_STATE)
        try:
            loaded = json.loads(ROUTER_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Router state could not be loaded (%s).", type(exc).__name__)

        if state.get("last_reset_day") != date.today().isoformat():
            state["daily_count"] = {}
            state["last_reset_day"] = date.today().isoformat()
        if not isinstance(state.get("blocked_tiers"), dict):
            state["blocked_tiers"] = {}
        # OpenRouter cooldowns are model-specific now, not tier-wide.
        state["blocked_tiers"].pop("2", None)
        state["blocked_tiers"].pop("5", None)
        if not isinstance(state.get("blocked_models"), dict):
            state["blocked_models"] = {}
        if not isinstance(state.get("model_failures"), dict):
            state["model_failures"] = {}
        if not isinstance(state.get("daily_count"), dict):
            state["daily_count"] = {}
        _save_router_state(state)
        return state


_ROUTER_STATE = _load_router_state()


def _mutate_router_state(mutator):
    with _STATE_LOCK:
        today = date.today().isoformat()
        if _ROUTER_STATE.get("last_reset_day") != today:
            _ROUTER_STATE["daily_count"] = {}
            _ROUTER_STATE["last_reset_day"] = today
        mutator(_ROUTER_STATE)
        _save_router_state(_ROUTER_STATE)


def _increment_daily_count(tier):
    def increment(state):
        key = str(tier)
        state["daily_count"][key] = int(state["daily_count"].get(key, 0)) + 1

    _mutate_router_state(increment)


def _set_blocked(tier, until):
    _mutate_router_state(
        lambda state: state["blocked_tiers"].__setitem__(str(tier), until)
    )


def _clear_blocked(tier):
    def clear(state):
        state["blocked_tiers"].pop(str(tier), None)

    _mutate_router_state(clear)


def _blocked_until(tier):
    with _STATE_LOCK:
        value = _ROUTER_STATE["blocked_tiers"].get(str(tier))
        if value is None:
            return None
        if float(value) <= time.time():
            _clear_blocked(tier)
            return None
        return float(value)


def _record_success(tier, model):
    def update(state):
        state["last_good_tier"] = tier
        state["last_good_model"] = model
        state["blocked_tiers"].pop(str(tier), None)

    _mutate_router_state(update)


def _record_429(tier):
    _set_blocked(tier, time.time() + 60)


def _openrouter_blocked_until(model):
    with _STATE_LOCK:
        blocked = _ROUTER_STATE["blocked_models"].get(model)
        if blocked is None:
            return None
        blocked_until = float(blocked)
        if blocked_until <= time.time():
            _mutate_router_state(
                lambda state: state["blocked_models"].pop(model, None)
            )
            return None
        return blocked_until


def _record_openrouter_failure(model):
    now = time.time()

    def update(state):
        recent = [
            float(timestamp)
            for timestamp in state["model_failures"].get(model, [])
            if now - float(timestamp) <= OPENROUTER_MODEL_FAILURE_WINDOW
        ]
        recent.append(now)
        state["model_failures"][model] = recent
        if len(recent) >= 3:
            state["blocked_models"][model] = (
                now + OPENROUTER_MODEL_BLOCK_DURATION
            )

    _mutate_router_state(update)


def _record_openrouter_success(model):
    def update(state):
        state["model_failures"].pop(model, None)
        state["blocked_models"].pop(model, None)

    _mutate_router_state(update)


def _cached_openrouter_models():
    cached = None
    try:
        cache = json.loads(
            OPENROUTER_MODELS_CACHE_PATH.read_text(encoding="utf-8")
        )
        if isinstance(cache, dict) and isinstance(cache.get("models"), list):
            cached = cache
            age = time.time() - float(cache.get("fetched_at", 0))
            if age < OPENROUTER_MODELS_CACHE_TTL:
                return set(cache["models"]), ""
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "OpenRouter model cache could not be read (%s).",
            type(exc).__name__,
        )

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("unexpected model catalog response")
        models = {
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        cache_data = {
            "fetched_at": time.time(),
            "models": sorted(models),
        }
        try:
            OPENROUTER_MODELS_CACHE_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )
            temporary_path = OPENROUTER_MODELS_CACHE_PATH.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(cache_data, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(OPENROUTER_MODELS_CACHE_PATH)
        except OSError as exc:
            logger.warning(
                "OpenRouter model cache could not be saved (%s).",
                type(exc).__name__,
            )
        return models, ""
    except (requests.RequestException, ValueError, TypeError) as exc:
        reason = _exception_reason(exc)
        logger.warning(
            "OpenRouter model catalog request failed (%s); %s.",
            reason,
            "using stale cache" if cached else "no model cache is available",
        )
        if cached:
            return set(cached["models"]), "stale catalog cache"
        return set(), f"model catalog unavailable ({reason})"


def openrouter_model_is_available(model):
    """Check model IDs against the public catalog cached for one hour."""
    models, catalog_reason = _cached_openrouter_models()
    if model in models:
        return True, catalog_reason
    if not models:
        return False, catalog_reason or "OpenRouter model catalog is empty"
    return False, "model not present in OpenRouter catalog"


def reset_router_state():
    global _ROUTER_STATE
    with _STATE_LOCK:
        _ROUTER_STATE = {
            **copy.deepcopy(_DEFAULT_ROUTER_STATE),
            "blocked_tiers": {},
            "blocked_models": {},
            "model_failures": {},
            "daily_count": {},
            "last_reset_day": date.today().isoformat(),
        }
        _save_router_state(_ROUTER_STATE)


def _extract_json(text):
    """Trích xuất JSON từ text, kể cả khi bọc trong markdown fences."""
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def ollama_model_is_installed(model=None, timeout=10):
    """Check the configured Ollama model using the local `ollama list` command."""
    model = model or OLLAMA_MODEL
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"ollama list failed: {type(exc).__name__}"
    if result.returncode != 0:
        return False, f"ollama list exited with code {result.returncode}"

    installed = {
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.split()
    }
    if model in installed:
        return True, ""
    return False, f"Ollama thiếu model {model}, skip"


def _record_provider_event(event_type, details):
    from core.logger import session_logger

    session_logger.log(event_type, details)


def ollama_is_available(timeout=3):
    """Compatibility helper; report whether the local Ollama API responds."""
    try:
        return requests.get(
            f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=timeout
        ).status_code == 200
    except requests.RequestException:
        return False


def _call_ollama(prompt, model, json_mode):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    with _OLLAMA_SEMAPHORE:
        response = requests.post(
            f"{OLLAMA_HOST.rstrip('/')}/api/generate",
            json=payload,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json().get("response", "")


def _call_groq(prompt, model, json_mode):
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY, timeout=REQUEST_TIMEOUT)
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**request)
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


def _call_openrouter(prompt, model, json_mode):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    choices = response.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content") or ""


def _call_gemini(prompt, model, json_mode):
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel(model)
    options = {"timeout": REQUEST_TIMEOUT}
    if json_mode:
        options["generation_config"] = {"response_mime_type": "application/json"}
    response = client.generate_content(prompt, request_options=options)
    return getattr(response, "text", "") or ""


def _call_kilo(prompt, model, json_mode):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(
        "https://api.kilo.ai/api/gateway/chat/completions",
        headers={
            "Authorization": f"Bearer {KILO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    choices = response.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content") or ""


def _router_log(tier, name, model, status, reason=""):
    message = f"[ROUTER] Tier {tier} ({name}) model={model} → {status}"
    if reason:
        message += f" ({reason})"
    print(message)
    logger.info(message)


def _tier_model(provider, default_model, requested_model):
    """Allow per-agent provider:model overrides without changing tier ordering."""
    prefix = f"{provider}:"
    if requested_model.casefold().startswith(prefix):
        override = requested_model[len(prefix):].strip()
        if override:
            return override
    return default_model


def _response_failure_reason(text):
    if not isinstance(text, str) or not text.strip():
        return "empty response"
    lowered = text.casefold()
    if "thinking process" in lowered or "<thinking>" in lowered:
        return "rejected thinking content"
    return None


def _exception_status(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status or getattr(exc, "status_code", None) or getattr(exc, "code", None)


def _exception_reason(exc):
    status = _exception_status(exc)
    if status == 429 or str(status) == "429":
        return "429 rate limit"
    if status == 503 or str(status) == "503":
        return "503 service unavailable"
    if status == 404 or str(status) == "404":
        return "404 model unavailable"
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection error"
    if status:
        return f"HTTP {status}"
    return type(exc).__name__


def _tier_models(name, configured_model, fallback_model, requested_model):
    provider = name.casefold()
    main_model = _tier_model(provider, configured_model, requested_model)
    if fallback_model is None:
        return (main_model,)
    fallback = _tier_model(provider, fallback_model, requested_model)
    return (main_model,) if fallback == main_model else (main_model, fallback)


ROUTER_TIER_ORDER = (
    (1, "Groq", GROQ_MODEL),
    (2, "Kilo", KILO_MODEL),
    (3, "OpenRouter", OPENROUTER_PRIMARY_MODEL),
    (4, "Groq", GROQ_FALLBACK_MODEL),
    (5, "OpenRouter fallback", OPENROUTER_FALLBACK_MODEL),
    (6, "Gemini", GEMINI_MODEL),
    (7, "Ollama", OLLAMA_MODEL),
)


def _canonical_provider(name):
    lowered = name.casefold()
    if lowered.startswith("openrouter"):
        return "openrouter"
    if lowered.startswith("groq"):
        return "groq"
    return lowered


def _requested_route(requested_model):
    provider, separator, model_name = requested_model.partition(":")
    provider = provider.casefold()
    if separator and provider in {"groq", "kilo", "openrouter", "gemini", "ollama"}:
        return provider, model_name.strip()
    return None, ""


def call_llm(prompt, model="auto", json_mode=False):
    """Try provider tiers in fixed order, with optional persisted router state."""
    requested_model = (model or "auto").strip()
    requested_provider, requested_route_model = _requested_route(requested_model)
    tried = []
    providers = (
        (1, "Groq", GROQ_API_KEY, GROQ_MODEL, None, _call_groq),
        (2, "Kilo", KILO_API_KEY, KILO_MODEL, None, _call_kilo),
        (3, "OpenRouter", OPENROUTER_API_KEY, OPENROUTER_PRIMARY_MODEL, None, _call_openrouter),
        (4, "Groq", GROQ_API_KEY, GROQ_FALLBACK_MODEL, None, _call_groq),
        (5, "OpenRouter fallback", OPENROUTER_API_KEY, OPENROUTER_FALLBACK_MODEL, None, _call_openrouter),
        (6, "Gemini", GEMINI_API_KEY, GEMINI_MODEL, None, _call_gemini),
        (7, "Ollama", "configured", OLLAMA_MODEL, None, _call_ollama),
    )

    by_tier = {provider[0]: provider for provider in providers}
    tier_order = list(range(1, 8))
    if requested_provider:
        matching_tiers = [
            entry[0]
            for entry in providers
            if _canonical_provider(entry[1]) == requested_provider
            and (
                not requested_route_model
                or entry[3] == requested_route_model
            )
        ]
        if matching_tiers:
            preferred_tier = matching_tiers[0]
            cloud_tiers = list(range(1, 7))
            cloud_order = (
                cloud_tiers[preferred_tier - 1:]
                + cloud_tiers[:preferred_tier - 1]
            )
        else:
            cloud_order = list(range(1, 7))
        unique_order = []
        used_providers = set()
        for tier in cloud_order:
            provider_name = _canonical_provider(by_tier[tier][1])
            if provider_name not in used_providers:
                unique_order.append(tier)
                used_providers.add(provider_name)
        tier_order = unique_order + [7]

    for tier in tier_order:
        _, name, key, configured_model, fallback_model, request = by_tier[tier]
        is_openrouter = name.startswith("OpenRouter")
        if requested_provider and tier == tier_order[0] and requested_route_model:
            models = (requested_route_model,)
        elif is_openrouter:
            if tier == 3 and OPENROUTER_FREE_POOL:
                models = (configured_model,) + tuple(
                    pool_model
                    for pool_model in OPENROUTER_FREE_POOL
                    if pool_model
                    not in {OPENROUTER_PRIMARY_MODEL, OPENROUTER_FALLBACK_MODEL}
                )
            elif tier == 3 and not OPENROUTER_FREE_POOL:
                models = (OPENROUTER_MODEL or configured_model,)
            else:
                models = (configured_model,)
        else:
            models = _tier_models(
                name, configured_model, fallback_model, requested_model
            )
        if name != "Ollama" and not key.strip():
            reason = "no key"
            _router_log(tier, name, models[0], "SKIP", reason)
            tried.append(f"Tier {tier} {name}: SKIP ({reason})")
            continue

        blocked_until = _blocked_until(tier) if not is_openrouter else None
        if blocked_until is not None:
            local_deadline = datetime.fromtimestamp(
                blocked_until
            ).strftime("%H:%M:%S")
            reason = f"blocked đến {local_deadline}"
            _router_log(tier, name, models[0], "SKIP", reason)
            tried.append(f"Tier {tier} {name}: SKIP ({reason})")
            continue

        if name == "Ollama":
            selected_model = models[0]
            installed, reason = ollama_model_is_installed(selected_model)
            if not installed:
                _router_log(tier, name, selected_model, "SKIP", reason)
                tried.append(f"Tier {tier} {name}: SKIP ({reason})")
                _record_provider_event("llm_provider_failed", {
                    "provider": "ollama",
                    "reason": reason,
                    "tier": tier,
                })
                continue

        for selected_model in models:
            if is_openrouter:
                model_blocked_until = _openrouter_blocked_until(selected_model)
                if model_blocked_until is not None:
                    deadline = datetime.fromtimestamp(
                        model_blocked_until
                    ).strftime("%H:%M:%S")
                    reason = f"model blocked đến {deadline}"
                    logger.warning(
                        "OpenRouter model %s is blocked until %s; skipping.",
                        selected_model,
                        deadline,
                    )
                    print(
                        f"[ROUTER] OpenRouter model={selected_model} → "
                        f"SKIP ({reason})"
                    )
                    tried.append(
                        f"Tier {tier} OpenRouter model={selected_model}: "
                        f"SKIP ({reason})"
                    )
                    continue
                available, catalog_reason = openrouter_model_is_available(
                    selected_model
                )
                if not available:
                    reason = catalog_reason or "model not in catalog"
                    logger.warning(
                        "OpenRouter model %s was skipped: %s.",
                        selected_model,
                        reason,
                    )
                    print(
                        f"[ROUTER] OpenRouter model={selected_model} → "
                        f"SKIP ({reason})"
                    )
                    tried.append(
                        f"Tier {tier} OpenRouter model={selected_model}: "
                        f"SKIP ({reason})"
                    )
                    continue

            _increment_daily_count(tier)
            request_started = time.perf_counter()
            try:
                text = request(prompt, selected_model, json_mode)
                request_elapsed = time.perf_counter() - request_started
                if name == "Ollama" and request_elapsed > 30:
                    logger.warning(
                        "[OLLAMA] Chậm (%.1fs); cloud tiers đã thử hết.",
                        request_elapsed,
                    )
                response_failure = _response_failure_reason(text)
                if response_failure:
                    raise ValueError(response_failure)
                if json_mode:
                    _extract_json(text)
            except Exception as exc:
                reason = _exception_reason(exc)
                if isinstance(exc, ValueError) and str(exc) in {
                    "empty response",
                    "rejected thinking content",
                }:
                    reason = str(exc)
                if is_openrouter:
                    _record_openrouter_failure(selected_model)
                    logger.warning(
                        "OpenRouter model %s failed (%s); trying next pool model.",
                        selected_model,
                        reason,
                    )
                    print(
                        f"[ROUTER] OpenRouter model={selected_model} → "
                        f"FAIL ({reason})"
                    )
                elif reason.startswith("429"):
                    _record_429(tier)
                _router_log(tier, name, selected_model, "FAIL", reason)
                tried.append(
                    f"Tier {tier} {name} model={selected_model}: FAIL ({reason})"
                )
                logger.warning(
                    "LLM tier %d (%s) failed: %s; continuing.",
                    tier,
                    name,
                    reason,
                )
                _record_provider_event("llm_provider_failed", {
                    "provider": name.casefold(),
                    "model": selected_model,
                    "reason": reason,
                    "tier": tier,
                })
                continue

            if is_openrouter:
                _record_openrouter_success(selected_model)
                print(
                    f"[ROUTER] OpenRouter model={selected_model} → OK"
                )
            else:
                _clear_blocked(tier)
            _router_log(tier, name, selected_model, "OK")
            tried.append(
                f"Tier {tier} {name} model={selected_model}: OK"
            )
            _record_success(tier, selected_model)
            _record_provider_event("llm_provider", {
                "provider": name.casefold(),
                "model": selected_model,
                "tier": tier,
            })
            return text

    raise RuntimeError(
        "All LLM tiers failed or were skipped:\n" + "\n".join(tried)
    )


class LLMClient:
    """Provider-fallback client retained for existing agent call sites."""

    def __init__(self, model_name="auto"):
        self.model_name = model_name or "auto"

    def ask(self, prompt, json_mode=False):
        try:
            text = call_llm(prompt, model=self.model_name, json_mode=json_mode)
            if json_mode:
                return _extract_json(text)
            return text
        except Exception as e:
            if json_mode:
                return {"error": str(e)}
            return f"Error: {str(e)}"


_default_client = LLMClient("auto")


def ask_llm(prompt, json_mode=False):
    """Backward-compatible helper used by the single-agent pipeline."""
    return _default_client.ask(prompt, json_mode=json_mode)
