"""Generic fallback chain with health checks, caching, and circuit breaker.

The FallbackChain wraps a list of providers. Each call to invoke() tries
providers in order until one succeeds. A provider that fails repeatedly
is tripped (circuit breaker) and skipped for a cooldown period. Successful
results are cached so repeated calls with the same key are served from
memory without invoking any provider.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, TypeVar

T = TypeVar('T')
Key = TypeVar('Key')


class HealthStatus:
    """Constants for provider health states."""

    CLOSED = 'closed'      # Circuit closed: calls pass through.
    OPEN = 'open'          # Circuit open: calls fail fast.
    HALF_OPEN = 'half_open'  # Probing: a single call is allowed.


class CircuitBreaker:
    """Simple circuit breaker for a single provider.

    After `failure_threshold` consecutive failures the circuit opens and
    every call fails fast until `cooldown_seconds` elapses, at which point
    a single probe call is allowed (half-open).
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._state = HealthStatus.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_state()
            return self._state

    def _refresh_state(self) -> None:
        if self._state == HealthStatus.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.cooldown_seconds:
                self._state = HealthStatus.HALF_OPEN
                self._failure_count = 0

    def allow(self) -> bool:
        """Return True if a call should be attempted right now."""
        with self._lock:
            self._refresh_state()
            if self._state == HealthStatus.OPEN:
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = HealthStatus.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = HealthStatus.OPEN
                self._opened_at = time.monotonic()


class HealthCheck:
    """Callable wrapper that pings a provider and reports liveness."""

    def __init__(self, name: str, check: Callable[[], bool], interval_seconds: float = 60.0) -> None:
        self.name = name
        self._check = check
        self.interval_seconds = max(0.0, interval_seconds)
        self._last_check = 0.0
        self._last_result = False
        self._lock = threading.Lock()

    def is_healthy(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if now - self._last_check >= self.interval_seconds:
                try:
                    self._last_result = bool(self._check())
                except Exception:
                    self._last_result = False
                self._last_check = now
            return self._last_result


class FallbackChain(Generic[T]):
    """Execute a callable across multiple providers with fallback semantics.

    Each provider is a callable returning a value of type T. The chain
    stops at the first provider that returns a non-None result. Results are
    cached by key. A provider that repeatedly fails is circuit-broken.
    """

    def __init__(self, providers: Optional[Sequence[Callable[[], T]]] = None,
                 failure_threshold: int = 3,
                 cooldown_seconds: float = 30.0,
                 cache_ttl: float = 300.0) -> None:
        self._providers: List[Callable[[], T]] = list(providers) if providers else []
        self._breakers: List[CircuitBreaker] = [
            CircuitBreaker(failure_threshold, cooldown_seconds)
            for _ in self._providers
        ]
        self._health_checks: List[Optional[HealthCheck]] = [None] * len(self._providers)
        self._cache: Dict[Any, Any] = {}
        self._cache_ts: Dict[Any, float] = {}
        self._cache_ttl = max(0.0, cache_ttl)
        self._lock = threading.Lock()

    def add_provider(self, provider: Callable[[], T],
                     health_check: Optional[HealthCheck] = None) -> None:
        """Append a provider and its circuit breaker."""
        self._providers.append(provider)
        self._breakers.append(CircuitBreaker())
        self._health_checks.append(health_check)

    def set_health_check(self, index: int, health_check: HealthCheck) -> None:
        if 0 <= index < len(self._health_checks):
            self._health_checks[index] = health_check

    def _cache_get(self, key: Any) -> Optional[T]:
        if key is None:
            return None
        with self._lock:
            ts = self._cache_ts.get(key)
            if ts is None:
                return None
            if self._cache_ttl > 0 and (time.monotonic() - ts) > self._cache_ttl:
                self._cache.pop(key, None)
                self._cache_ts.pop(key, None)
                return None
            return self._cache.get(key)

    def _cache_set(self, key: Any, value: T) -> None:
        if key is None:
            return
        with self._lock:
            self._cache[key] = value
            self._cache_ts[key] = time.monotonic()

    def invoke(self, *args, **kwargs) -> T:
        """Invoke the chain.

        The first positional argument, if hashable, is used as the cache key.
        """
        cache_key = args[0] if args else None
        if cache_key is not None:
            try:
                hash(cache_key)
            except TypeError:
                cache_key = None

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        last_error: Optional[Exception] = None
        for idx, provider in enumerate(self._providers):
            breaker = self._breakers[idx]
            if not breaker.allow():
                continue
            health = self._health_checks[idx]
            if health is not None and not health.is_healthy():
                continue
            try:
                result = provider(*args, **kwargs)
                if result is not None:
                    breaker.record_success()
                    self._cache_set(cache_key, result)
                    return result
            except Exception as exc:
                last_error = exc
                breaker.record_failure()
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError('FallbackChain: all providers failed or were tripped')

    def __call__(self, *args, **kwargs) -> T:
        return self.invoke(*args, **kwargs)

    def reset(self) -> None:
        """Reset all circuit breakers and the cache."""
        with self._lock:
            for breaker in self._breakers:
                breaker._state = HealthStatus.CLOSED
                breaker._failure_count = 0
            self._cache.clear()
            self._cache_ts.clear()


__all__ = ['FallbackChain', 'CircuitBreaker', 'HealthCheck', 'HealthStatus']
