"""Factory for creating TTS engine instances."""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from .base import TTSEngine
from .pyttsx3_engine import Pyttsx3Engine
from .edge_tts_engine import EdgeTTSEngine
from .hybrid_engine import HybridTTSEngine

logger = logging.getLogger(__name__)

_AVAILABLE_ENGINES: Dict[str, Type[TTSEngine]] = {
    "hybrid": HybridTTSEngine,
    "edge": EdgeTTSEngine,
    "pyttsx3": Pyttsx3Engine,
}


def create_tts_engine(config, paths) -> TTSEngine:
    """Create a TTS engine from configuration.

    Args:
        config: A Config-like object with `tts_prefer_online`, `tts_voice`,
            `tts_rate`, `tts_volume` attributes.
        paths: A PathManager-like object for resolving file paths.

    Returns:
        A configured TTSEngine instance. Currently returns
        HybridTTSEngine which auto-selects between online and offline.
    """
    prefer_online = getattr(config, "tts_prefer_online", True)
    voice = getattr(config, "tts_voice", "vi-VN-NamMinhNeural")
    rate = getattr(config, "tts_rate", 200)
    volume = getattr(config, "tts_volume", 1.0)
    logger.info(
        "Creating TTS engine: prefer_online=%s, voice=%s, rate=%s, volume=%s",
        prefer_online, voice, rate, volume,
    )

    online_engine = EdgeTTSEngine(config=config, paths=paths, voice=voice)
    offline_engine = Pyttsx3Engine(config=config, paths=paths, rate=rate, volume=volume)
    return HybridTTSEngine(config=config, paths=paths, online_engine=online_engine, offline_engine=offline_engine)


def list_available_engines() -> List[str]:
    """Return the names of all available TTS engine types."""
    return list(_AVAILABLE_ENGINES.keys())


__all__ = ["create_tts_engine", "list_available_engines"]