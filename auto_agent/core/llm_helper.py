"""LLM client with free-first Ollama, Groq, and OpenRouter providers."""
import copy
import hashlib
import json
import logging
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path

import requests
from core.sensitive_data import redact_text

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
    OLLAMA_PREFER_LOCAL,
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
_GEMINI_SEMAPHORE = threading.BoundedSemaphore(2)
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


_CACHE_LOCK = threading.RLock()
_LLM_RESPONSE_CACHE = {}
_CACHE_TTL_SECONDS = 180.0
_MAX_CACHE_ENTRIES = 256


def clear_llm_cache():
    with _CACHE_LOCK:
        _LLM_RESPONSE_CACHE.clear()


def _get_cached(prompt, model, json_mode, max_output_tokens):
    now = time.time()
    key = (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        str(model),
        bool(json_mode),
        max_output_tokens,
    )
    with _CACHE_LOCK:
        if key in _LLM_RESPONSE_CACHE:
            ts, val = _LLM_RESPONSE_CACHE[key]
            if now - ts <= _CACHE_TTL_SECONDS:
                return copy.deepcopy(val)
            _LLM_RESPONSE_CACHE.pop(key, None)
    return None


def _set_cached(prompt, model, json_mode, max_output_tokens, val):
    now = time.time()
    key = (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        str(model),
        bool(json_mode),
        max_output_tokens,
    )
    with _CACHE_LOCK:
        if len(_LLM_RESPONSE_CACHE) >= _MAX_CACHE_ENTRIES:
            oldest = min(
                _LLM_RESPONSE_CACHE.keys(),
                key=lambda k: _LLM_RESPONSE_CACHE[k][0],
            )
            _LLM_RESPONSE_CACHE.pop(oldest, None)
        _LLM_RESPONSE_CACHE[key] = (now, copy.deepcopy(val))


def reset_router_state():
    global _ROUTER_STATE
    clear_llm_cache()
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
    """Parse JSON returned alone, in markdown fences, or surrounded by prose."""
    if not isinstance(text, str):
        raise TypeError("JSON response must be text")

    candidate = text.strip()
    match = re.search(
        r"```(?:json)?\s*(.*?)```",
        candidate,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()

    decoder = json.JSONDecoder()
    try:
        return decoder.decode(candidate)
    except json.JSONDecodeError:
        pass

    for index, character in enumerate(candidate):
        if character not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    raise json.JSONDecodeError("No JSON value found in model response", candidate, 0)


def ollama_model_is_installed(model=None, timeout=10):
    """Check whether the local Ollama API has the requested model installed."""
    model = model or OLLAMA_MODEL
    try:
        response = requests.get(
            f"{OLLAMA_HOST.rstrip('/')}/api/tags",
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, f"Ollama API unavailable ({type(exc).__name__})"

    try:
        payload = response.json()
    except ValueError as exc:
        return False, f"Invalid Ollama /api/tags response ({type(exc).__name__})"
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return False, "Invalid Ollama /api/tags response (missing models list)"

    installed = {
        model_name
        for item in payload["models"]
        if isinstance(item, dict)
        for model_name in (item.get("name"), item.get("model"))
        if isinstance(model_name, str)
    }
    if model in installed:
        return True, ""
    return False, f"Ollama thiếu model {model}, skip"


def ollama_installed_models(timeout=3):
    """Return installed Ollama model names, collapsing aliases with the same digest."""
    try:
        response = requests.get(
            f"{OLLAMA_HOST.rstrip('/')}/api/tags",
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return []

    entries = [item for item in payload["models"] if isinstance(item, dict)]
    entries.sort(
        key=lambda item: (item.get("name") or item.get("model")) != OLLAMA_MODEL
    )
    models = []
    seen_digests = set()
    seen_names = set()
    for item in entries:
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name:
            continue
        digest = item.get("digest")
        identity = digest if isinstance(digest, str) and digest else name
        if identity in seen_digests or name in seen_names:
            continue
        models.append(name)
        seen_digests.add(identity)
        seen_names.add(name)
    return models


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


def _call_ollama(prompt, model, json_mode, max_output_tokens=None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    if max_output_tokens is not None:
        payload["options"] = {"num_predict": max_output_tokens}
    with _OLLAMA_SEMAPHORE:
        response = requests.post(
            f"{OLLAMA_HOST.rstrip('/')}/api/generate",
            json=payload,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json().get("response", "")


def _call_groq(prompt, model, json_mode, max_output_tokens=None):
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY, timeout=REQUEST_TIMEOUT)
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _max_output_tokens(json_mode, max_output_tokens),
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        if not json_mode or not _is_response_format_rejection(exc):
            raise
        request.pop("response_format")
        logger.warning(
            "Groq rejected JSON response_format for model %s; retrying without it.",
            model,
        )
        response = client.chat.completions.create(**request)
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


def _call_openrouter(prompt, model, json_mode, max_output_tokens=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _max_output_tokens(json_mode, max_output_tokens),
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


def _call_gemini(prompt, model, json_mode, max_output_tokens=None):
    from google import genai

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"timeout": REQUEST_TIMEOUT * 1000},
    )
    generation_config = {
        "max_output_tokens": _max_output_tokens(json_mode, max_output_tokens)
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    with _GEMINI_SEMAPHORE:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=generation_config,
        )
    return getattr(response, "text", "") or ""


def _call_kilo(prompt, model, json_mode, max_output_tokens=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _max_output_tokens(json_mode, max_output_tokens),
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


def _max_output_tokens(json_mode, requested=None):
    if requested is None:
        return 2048 if json_mode else 1024
    return min(8192, max(1, int(requested)))


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
    if isinstance(exc, json.JSONDecodeError):
        return "invalid JSON response"
    if status:
        detail = str(exc).replace("\n", " ").strip()
        return f"HTTP {status}" + (f": {detail[:160]}" if detail else "")
    detail = str(exc).replace("\n", " ").strip()
    return type(exc).__name__ + (f": {detail[:160]}" if detail else "")


def _is_bad_request(exc):
    return str(_exception_status(exc)) == "400"


def _is_response_format_rejection(exc):
    if not _is_bad_request(exc):
        return False
    detail = str(exc).casefold()
    return any(
        marker in detail
        for marker in ("response_format", "json_object", "json mode", "json-mode")
    )


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


def call_llm(
    prompt, model="auto", json_mode=False, max_output_tokens=None,
    allow_fallback=True,
):
    """Try provider tiers in fixed order, with optional persisted router state."""
    prompt = redact_text(str(prompt))
    requested_model = (model or "auto").strip()
    requested_provider, requested_route_model = _requested_route(requested_model)
    if not allow_fallback and not requested_provider:
        raise ValueError("Strict model checks require a provider:model route")
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
    cloud_tiers = list(range(1, 7))
    default_cloud_order = [6, 1, 2, 3, 4, 5]
    requested_tier = None
    if requested_provider == "ollama":
        requested_tier = 7
        cloud_order = cloud_tiers
    elif requested_provider:
        provider_tiers = [
            entry[0]
            for entry in providers
            if _canonical_provider(entry[1]) == requested_provider
        ]
        matching_tiers = [
            tier
            for tier in provider_tiers
            if not requested_route_model
            or by_tier[tier][3] == requested_route_model
        ]
        if not matching_tiers:
            matching_tiers = provider_tiers[:1]
        if matching_tiers:
            requested_tier = matching_tiers[0]
            preferred_index = cloud_tiers.index(requested_tier)
            cloud_order = cloud_tiers[preferred_index:] + cloud_tiers[:preferred_index]
        else:
            cloud_order = cloud_tiers
    else:
        cloud_order = default_cloud_order

    if not allow_fallback and requested_tier is None:
        raise ValueError(f"No provider route found for {requested_model}")

    if requested_provider:
        unique_order = []
        used_providers = set()
        for tier in cloud_order:
            provider_name = _canonical_provider(by_tier[tier][1])
            if provider_name not in used_providers:
                unique_order.append(tier)
                used_providers.add(provider_name)
    else:
        unique_order = cloud_order
    local_first = OLLAMA_PREFER_LOCAL and not requested_provider
    if requested_provider == "ollama":
        local_first = False
    tier_order = (
        [requested_tier]
        if not allow_fallback
        else (
            ([7] if local_first or requested_provider == "ollama" else [])
            + unique_order
            + ([] if local_first or requested_provider == "ollama" else [7])
        )
    )

    for tier in tier_order:
        _, name, key, configured_model, fallback_model, request = by_tier[tier]
        is_openrouter = name.startswith("OpenRouter")
        if requested_tier == tier and requested_route_model:
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
                if max_output_tokens is None:
                    text = request(prompt, selected_model, json_mode)
                else:
                    text = request(
                        prompt,
                        selected_model,
                        json_mode,
                        _max_output_tokens(json_mode, max_output_tokens),
                    )
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

    def ask(self, prompt, json_mode=False, max_output_tokens=None, use_cache=True):
        try:
            prompt = redact_text(str(prompt))
            if use_cache:
                cached = _get_cached(
                    prompt, self.model_name, json_mode, max_output_tokens
                )
                if cached is not None:
                    return cached
            if max_output_tokens is None:
                text = call_llm(
                    prompt, model=self.model_name, json_mode=json_mode
                )
            else:
                text = call_llm(
                    prompt,
                    model=self.model_name,
                    json_mode=json_mode,
                    max_output_tokens=max_output_tokens,
                )
            text = redact_text(str(text))
            res = _extract_json(text) if json_mode else text
            if use_cache and res and not (isinstance(res, dict) and "error" in res):
                _set_cached(
                    prompt, self.model_name, json_mode, max_output_tokens, res
                )
            return res
        except Exception as e:
            if json_mode:
                return {"error": redact_text(str(e))}
            return f"Error: {redact_text(str(e))}"


_default_client = LLMClient("auto")


def ask_llm(prompt, json_mode=False, max_output_tokens=None):
    """Backward-compatible helper used by the single-agent pipeline."""
    return _default_client.ask(
        prompt,
        json_mode=json_mode,
        max_output_tokens=max_output_tokens,
    )
