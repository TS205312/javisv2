"""Factory and fallback implementations for Jarvis vision engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import VisionEngine


class VisionEngineStub(VisionEngine):
    """Non-operational engine returned when a required dependency is absent."""

    def __init__(
        self,
        engine_name: str,
        missing_dep: str,
        reason: str | None = None,
    ) -> None:
        self._engine_name = engine_name
        self._missing_dep = missing_dep
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def health(self) -> bool:
        return False

    def analyze(self, image: Path | bytes | Any, prompt: str) -> str:
        raise RuntimeError(self._message())

    def capture(self, prompt: str, camera_index: int = 0) -> str:
        raise RuntimeError(self._message())

    def close(self) -> None:
        return None

    def _message(self) -> str:
        if self._reason:
            return "{0} engine unavailable: {1}.".format(
                self._engine_name, self._reason
            )
        return "{0} engine unavailable: {1} is not installed.".format(
            self._engine_name, self._missing_dep
        )


def create_opencv_engine(
    config: Any = None,
    paths: Any = None,
) -> VisionEngine:
    """Create an OpenCV engine, or a stub when its dependencies are absent."""
    try:
        import cv2
        import numpy
        from .opencv_engine import OpenCVFrameCapture

        return OpenCVFrameCapture(config=config, paths=paths)
    except ImportError as exc:
        return VisionEngineStub(
            "OpenCVFrameCapture",
            "opencv-python (cv2) and numpy are not installed",
            "opencv-python (cv2) and numpy are not installed: {0}".format(exc),
        )
    except Exception as exc:
        return VisionEngineStub(
            "OpenCVFrameCapture",
            "OpenCV configuration is invalid",
            "OpenCV configuration is invalid: {0}".format(exc),
        )


def create_moondream_engine(
    config: Any = None,
    paths: Any = None,
) -> VisionEngine:
    """Create an Ollama Moondream engine, or a stub when httpx is absent."""
    try:
        import httpx
        from .moondream_engine import OllamaMoondreamEngine

        return OllamaMoondreamEngine(config=config, paths=paths)
    except ImportError as exc:
        return VisionEngineStub(
            "OllamaMoondreamEngine",
            "httpx is not installed",
            "httpx is not installed: {0}".format(exc),
        )
    except Exception as exc:
        return VisionEngineStub(
            "OllamaMoondreamEngine",
            "Ollama configuration is invalid",
            "Ollama configuration is invalid: {0}".format(exc),
        )


def create_vision_engines(
    config: Any = None,
    paths: Any = None,
) -> dict[str, VisionEngine]:
    """Create the OpenCV capture and Moondream analysis engines.

    Missing optional dependencies are represented by ``VisionEngineStub``
    instances so callers can keep running and inspect ``health``/``reason``.
    """
    return {
        "opencv": create_opencv_engine(config=config, paths=paths),
        "moondream": create_moondream_engine(config=config, paths=paths),
    }


def get_default(
    config: Any = None,
    paths: Any = None,
) -> dict[str, VisionEngine]:
    """Return the standard vision engine set."""
    return create_vision_engines(config=config, paths=paths)


def list_available_engines() -> list[str]:
    """Return names of engine slots provided by this factory."""
    return ["opencv", "moondream"]


UnavailableVisionEngine = VisionEngineStub


__all__ = [
    "VisionEngineStub",
    "UnavailableVisionEngine",
    "create_opencv_engine",
    "create_moondream_engine",
    "create_vision_engines",
    "get_default",
    "list_available_engines",
]