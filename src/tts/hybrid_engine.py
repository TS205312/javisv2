"""Hybrid TTS engine: edge-tts (online) with pyttsx3 (offline) fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

from .base import TTSEngine, _play_wav_file
from .pyttsx3_engine import Pyttsx3Engine
from .edge_tts_engine import EdgeTTSEngine

logger = logging.getLogger(__name__)

_ONLINE_HEALTH_TTL = 5.0


class HybridTTSEngine(TTSEngine):
    """Hybrid TTS engine that prefers online synthesis with offline fallback.

    Selection logic:
        1. If `config.tts_prefer_online` is True and edge-tts health check
           passes (cached for a short TTL to avoid a network probe every
           sentence), use edge-tts.
        2. If edge-tts synthesis or playback fails, fall back to pyttsx3.
        3. If online is not preferred, start directly with pyttsx3.

    Circuit behavior:
        - Failing providers are short-circuited for a cooldown period
          (via internal circuit breaker on each sub-engine).
        - No infinite retries; each call attempts at most once per provider.
    """

    def __init__(
        self,
        config=None,
        paths=None,
        online_engine: Optional[EdgeTTSEngine] = None,
        offline_engine: Optional[Pyttsx3Engine] = None,
    ) -> None:
        self._config = config
        self._paths = paths
        self._online = online_engine or EdgeTTSEngine(config=config, paths=paths)
        self._offline = offline_engine or Pyttsx3Engine(config=config, paths=paths)
        self._online_enabled = bool(getattr(config, "tts_prefer_online", True))
        self._offline_used_count = 0
        self._online_health_cached = False
        self._online_health_ts = 0.0
        self._closed = False
        self._online_breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)
        self._offline_breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)

    def _should_use_online(self) -> bool:
        if not self._online_enabled or self._closed:
            return False
        if not self._online_breaker.allow():
            logger.debug("Online circuit breaker open; using offline")
            return False
        now = time.monotonic()
        if now - self._online_health_ts >= _ONLINE_HEALTH_TTL:
            try:
                healthy = self._online.health
            except Exception:
                healthy = False
            self._online_health_cached = healthy
            self._online_health_ts = now
            if not healthy:
                self._online_breaker.record_failure()
                return False
        else:
            if not self._online_health_cached:
                return False
        return True

    @property
    def health(self) -> bool:
        if self._closed:
            return False
        if self._should_use_online():
            return True
        return self._offline.health

    @property
    def available_voices(self) -> List[str]:
        voices: List[str] = []
        if self._should_use_online():
            try:
                voices = self._online.available_voices
            except Exception:
                pass
        if not voices:
            try:
                voices = self._offline.available_voices
            except Exception:
                pass
        return voices

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            return b""
        if self._closed:
            return b""
        if self._should_use_online():
            try:
                result = self._online.synthesize(text)
                if result and len(result) > 0:
                    self._online_breaker.record_success()
                    return result
                logger.warning("Online synthesis returned empty; falling back")
                self._online_breaker.record_failure()
            except Exception as exc:
                logger.warning("Online synthesis failed: %s; falling back", exc)
                self._online_breaker.record_failure()
        if not self._offline_breaker.allow():
            logger.warning("Offline circuit breaker open")
            return b""
        try:
            result = self._offline.synthesize(text)
            if result and len(result) > 0:
                self._offline_breaker.record_success()
                return result
        except Exception as exc:
            logger.error("Offline synthesis also failed: %s", exc)
            self._offline_breaker.record_failure()
        logger.warning("Hybrid synthesis: all providers returned empty")
        return b""

    def play(self, text: str) -> Union[bool, str]:
        if not text or not text.strip():
            return False
        if self._closed:
            return False
        online_result = None
        used_online = False
        if self._should_use_online():
            used_online = True
            try:
                online_result = self._online.play(text)
                if online_result is True:
                    self._online_breaker.record_success()
                    return True
                if isinstance(online_result, str):
                    logger.info("Online playback returned path; trying offline")
                    self._online_breaker.record_failure()
                elif online_result is False:
                    self._online_breaker.record_failure()
            except Exception as exc:
                logger.warning("Online play failed: %s; trying offline", exc)
                self._online_breaker.record_failure()
        if not self._offline_breaker.allow():
            logger.warning("Offline circuit breaker open")
            if used_online and isinstance(online_result, str):
                return online_result
            return False
        try:
            offline_result = self._offline.play(text)
            if offline_result is True:
                self._offline_used_count += 1
                self._offline_breaker.record_success()
                return True
            if isinstance(offline_result, str):
                return offline_result
            self._offline_breaker.record_failure()
        except Exception as exc:
            logger.error("Offline play also failed: %s", exc)
            self._offline_breaker.record_failure()
        if isinstance(online_result, str):
            return online_result
        return False

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        if self._closed:
            return False
        if self._should_use_online():
            try:
                result = self._online.speak(text)
                if result:
                    self._online_breaker.record_success()
                    return True
                logger.warning("Online speak failed; trying offline")
                self._online_breaker.record_failure()
            except Exception as exc:
                logger.warning("Online speak error: %s; trying offline", exc)
                self._online_breaker.record_failure()
        if not self._offline_breaker.allow():
            return False
        try:
            result = self._offline.speak(text)
            if result:
                self._offline_breaker.record_success()
                return result
        except Exception as exc:
            logger.error("Offline speak also failed: %s", exc)
            self._offline_breaker.record_failure()
            return False
        return False

    def close(self) -> None:
        if self._online is not None:
            try:
                self._online.close()
            except Exception:
                pass
        if self._offline is not None:
            try:
                self._offline.close()
            except Exception:
                pass
        self._closed = True
        logger.info("HybridTTSEngine closed")


class _CircuitBreaker:
    """Simple circuit breaker for internal use."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = 0.0

    def allow(self) -> bool:
        if self._state == "open":
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._cooldown_seconds:
                self._state = "half_open"
                self._failure_count = 0
            else:
                return False
        return True

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()


__all__ = ["HybridTTSEngine"]