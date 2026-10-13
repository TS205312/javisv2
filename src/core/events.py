"""Thread-safe event bus for Jarvis.

The EventBus decouples components through a simple publish/subscribe
pattern. Listeners are callables registered with a topic; publishing a
message fans out to all matching listeners on a dedicated worker thread
so that slow listeners never block the publisher.
"""

from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Event:
    """A single message published on the bus."""

    topic: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)


EventListener = Callable[[Event], None]


class EventBus:
    """In-process pub/sub bus with async delivery.

    Publishing enqueues the event; a background thread dispatches it to
    subscribers. This keeps the publisher responsive even when listeners
    are slow. Delivery is best-effort: exceptions in listeners are caught
    and routed to `on_error` if set.
    """

    def __init__(self, queue_size: int = 1000) -> None:
        self._queue: 'queue.Queue[Event]' = queue.Queue(maxsize=queue_size)
        self._subscribers: Dict[str, List[EventListener]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_error: Optional[Callable[[EventListener, Event, Exception], None]] = None

    def subscribe(self, topic: str, listener: EventListener) -> None:
        """Register `listener` to receive events matching `topic`."""
        with self._lock:
            self._subscribers.setdefault(topic, []).append(listener)

    def unsubscribe(self, topic: str, listener: EventListener) -> None:
        """Remove a previously registered listener."""
        with self._lock:
            listeners = self._subscribers.get(topic)
            if not listeners:
                return
            try:
                listeners.remove(listener)
            except ValueError:
                pass
            if not listeners:
                self._subscribers.pop(topic, None)

    def on_error(self, callback: Callable[[EventListener, Event, Exception], None]) -> None:
        """Set the error handler invoked when a listener raises."""
        self._on_error = callback

    def publish(self, topic: str, payload: Any = None) -> bool:
        """Enqueue an event for async delivery.

        Returns:
            True if the event was enqueued, False if the queue is full.
        """
        event = Event(topic=topic, payload=payload)
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            return False

    def publish_sync(self, topic: str, payload: Any = None) -> None:
        """Deliver an event immediately to current subscribers (blocking)."""
        event = Event(topic=topic, payload=payload)
        with self._lock:
            listeners = list(self._subscribers.get(topic, []))
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                if self._on_error is not None:
                    try:
                        self._on_error(listener, event, exc)
                    except Exception:
                        pass

    def start(self) -> None:
        """Start the background delivery thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name='eventbus')
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the delivery thread to stop and wait for it."""
        if not self._running:
            return
        self._running = False
        # Wake up the worker by pushing a sentinel.
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is None:
                break
            with self._lock:
                listeners = list(self._subscribers.get(event.topic, []))
            for listener in listeners:
                try:
                    listener(event)
                except Exception as exc:
                    if self._on_error is not None:
                        try:
                            self._on_error(listener, event, exc)
                        except Exception:
                            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def create_event_bus(queue_size: int = 1000) -> EventBus:
    """Convenience factory for a started EventBus."""
    bus = EventBus(queue_size=queue_size)
    bus.start()
    return bus


__all__ = ['EventBus', 'Event', 'EventListener', 'create_event_bus']
