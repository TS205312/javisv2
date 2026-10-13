"""Factory for creating STT engine instances.

Provides a centralized way to instantiate available STT engines.
If a required dependency is missing, a stub engine with a clear
error message is returned instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

from .base import STTEngine
from .vosk_engine import VoskEngine

logger = logging.getLogger(__name__)


class STTEngineStub(STTEngine):
    """Stub engine returned when the real engine is unavailable.

    All operations raise a clear error message indicating which
    dependency is missing.
    """

    def __init__(self, engine_name: str, missing_dep: str) -> None:
        self._engine_name = engine_name
        self._missing_dep = missing_dep

    @property
    def health(self) -> bool:
        return False

    def transcribe(self, audio: Any) -> str:
        raise RuntimeError(
            "{0} engine unavailable: {1} is not installed.".format(
                self._engine_name, self._missing_dep
            )
        )

    def listen(self, duration: float, prompt: str = "") -> str:
        raise RuntimeError(
            "{0} engine unavailable: {1} is not installed.".format(
                self._engine_name, self._missing_dep
            )
        )

    def close(self) -> None:
        pass


def _resolve_model_path(path: Union[str, None], paths: Any) -> str:
    """Resolve Vosk model path from config, paths manager, or defaults."""
    if path:
        return str(path)
    if paths is not None and hasattr(paths, "vosk_model_path"):
        model_path = paths.vosk_model_path()
        if model_path is not None:
            return str(model_path)
    return ""


def _validate_model_path(model_path: Optional[str]) -> bool:
    """Validate that the Vosk model directory exists and has required files."""
    if not model_path:
        return False
    try:
        path = Path(model_path)
        if not path.exists() or not path.is_dir():
            return False
        # Check for essential Vosk model files
        required = ["am", "conf", "graph"]
        return all((path / r).exists() for r in required)
    except Exception:
        return False


def create_stt_engine(config: Any = None, paths: Any = None) -> STTEngine:
    """Create an STT engine from configuration.

    Attempts to create a VoskEngine. If Vosk or its dependencies are
    unavailable, returns an STTEngineStub with a clear error message.

    Args:
        config: A Config-like object with ``stt_model_path`` attribute.
        paths: A PathManager-like object for resolving model paths.

    Returns:
        An STTEngine instance (VoskEngine or STTEngineStub).
    """
    model_path = None
    if config is not None:
        model_path = getattr(config, "stt_model_path", None)

    resolved = _resolve_model_path(model_path, paths)
    resolved_path = resolved if resolved else None

    # Validate model path before attempting to create engine
    if resolved_path and not _validate_model_path(resolved_path):
        logger.warning(
            "Vosk model not found or invalid at %s; using stub",
            resolved_path,
        )
        return STTEngineStub(
            "VoskEngine",
            "Vosk model not found at {0}".format(resolved_path),
        )

    try:
        import vosk  # noqa: F401
        engine = VoskEngine(model_path=resolved_path)
        logger.info("STT engine created: VoskEngine")
        return engine
    except ImportError:
        logger.warning("Vosk not installed, using STTEngineStub")
        return STTEngineStub("VoskEngine", "vosk (pip install vosk)")
    except Exception as exc:
        logger.warning("Vosk engine init failed: %s, using stub", exc)
        return STTEngineStub(
            "VoskEngine",
            "vosk model or configuration ({0})".format(exc),
        )


__all__ = ["STTEngineStub", "create_stt_engine"]