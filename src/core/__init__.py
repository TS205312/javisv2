"""Core subsystems: config, paths, lazy loading, fallback, events."""

from .config import Config
from .paths import PathManager
from .lazy import LazyLoader
from .fallback import FallbackChain
from .events import EventBus

__all__ = ["Config", "PathManager", "LazyLoader", "FallbackChain", "EventBus"]
