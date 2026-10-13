r"""pyttsx3-based offline TTS engine for Jarvis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from .base import TTSEngine, _play_wav_file

logger = logging.getLogger(__name__)

DEFAULT_RATE = 200
DEFAULT_VOLUME = 1.0
_FALLBACK_DIR = Path(r"D:\AI_System\audio")


class Pyttsx3Engine(TTSEngine):
    r"""Offline TTS engine using pyttsx3 (SAPI5 on Windows).

    Features:
        - Lazy import of pyttsx3 (not required at import time).
        - Vietnamese voice selection: prefers names containing "vietnam" or
          "vi", then a vi-VN locale voice, then the first available voice.
        - Sane rate/volume defaults.
        - All output/temp WAV files live under D:\AI_System\audio (never C:).
        - `synthesize()` exports WAV via pyttsx3 natively (no comtypes required).
        - `close()` only calls methods that exist on the engine and always
          releases the underlying object.
    """

    def __init__(
        self,
        config=None,
        paths=None,
        rate: int = DEFAULT_RATE,
        volume: float = DEFAULT_VOLUME,
    ) -> None:
        self._config = config
        self._paths = paths
        # Use config values if provided, otherwise use passed parameters
        if config is not None:
            self._rate = max(1, int(getattr(config, "tts_rate", rate)))
            self._volume = max(0.0, min(1.0, float(getattr(config, "tts_volume", volume))))
        else:
            self._rate = max(1, int(rate))
            self._volume = max(0.0, min(1.0, float(volume)))
        self._engine = None
        self._selected_voice = None
        self._voices: List = []
        self._closed = False

    def _ensure_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
            except ImportError as exc:
                raise ImportError(
                    "pyttsx3 is not installed; install with: pip install pyttsx3"
                ) from exc
            try:
                self._engine = pyttsx3.init()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialise pyttsx3 engine: {0}".format(exc)
                ) from exc
            self._configure_voice()
            try:
                self._engine.setProperty("rate", self._rate)
                self._engine.setProperty("volume", self._volume)
            except Exception as exc:
                logger.warning("pyttsx3 property set failed: %s", exc)
        return self._engine

    def _configure_voice(self):
        engine = self._engine
        try:
            self._voices = list(engine.getProperty("voices") or [])
        except Exception as exc:
            logger.warning("Cannot retrieve pyttsx3 voices: %s", exc)
            self._voices = []
        self._selected_voice = self._select_vietnamese_voice()
        if self._selected_voice is not None:
            try:
                engine.setProperty("voice", self._selected_voice.id)
            except Exception as exc:
                logger.warning("Cannot set pyttsx3 voice: %s", exc)

    def _select_vietnamese_voice(self):
        scored: List[tuple] = []
        for voice in self._voices:
            try:
                name = (voice.name or "").lower()
                vid = (voice.id or "").lower()
            except Exception:
                continue
            score = 0
            if "vietnam" in name or "vi\u1ec7t" in name:
                score = 3
            elif "vi" in name:
                score = 2
            elif vid.startswith("vi-") or "vi-vn" in vid:
                score = 1
            if score:
                scored.append((score, voice))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            chosen = scored[0][1]
            logger.info("Selected pyttsx3 voice: %s", chosen.name)
            return chosen
        if self._voices:
            logger.info("pyttsx3: no Vietnamese voice found; using first voice")
            return self._voices[0]
        return None

    def _audio_dir(self) -> Path:
        if self._paths is not None:
            try:
                return self._paths.get("audio")
            except Exception:
                pass
        return _FALLBACK_DIR

    def _ensure_audio_dir(self) -> Path:
        audio_dir = self._audio_dir()
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Cannot create audio dir %s: %s", audio_dir, exc)
        return audio_dir

    @property
    def health(self) -> bool:
        if self._closed:
            return False
        try:
            self._ensure_engine()
            return True
        except Exception:
            return False

    @property
    def available_voices(self) -> List[str]:
        try:
            self._ensure_engine()
        except Exception:
            return []
        names: List[str] = []
        for v in self._voices:
            try:
                names.append(v.name)
            except Exception:
                pass
        return names

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        if self._closed:
            logger.warning("Pyttsx3Engine is closed")
            return False
        try:
            engine = self._ensure_engine()
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as exc:
            logger.error("pyttsx3 speak failed: %s", exc)
            return False

    def _synthesize_to_wav(self, text: str) -> Optional[Path]:
        try:
            engine = self._ensure_engine()
        except Exception as exc:
            logger.warning("pyttsx3 engine not available: %s", exc)
            return None
        audio_dir = self._ensure_audio_dir()
        path = audio_dir / "pyttsx3_{0}.wav".format(hash(text) & 0xFFFFFFFF)
        try:
            engine.save_to_file(text, str(path))
            engine.runAndWait()
        except Exception as exc:
            logger.warning("pyttsx3 save_to_file failed: %s", exc)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        return path

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            return b""
        if self._closed:
            logger.warning("Pyttsx3Engine is closed")
            return b""
        wav_path = self._synthesize_to_wav(text)
        if wav_path is not None and wav_path.exists():
            try:
                data = wav_path.read_bytes()
                wav_path.unlink(missing_ok=True)
                return data
            except Exception as exc:
                logger.warning("Cannot read WAV %s: %s", wav_path, exc)
                return b""
        return b""

    def play(self, text: str) -> Union[bool, str]:
        if not text or not text.strip():
            return False
        if self._closed:
            logger.warning("Pyttsx3Engine is closed")
            return False
        if self.speak(text):
            return True
        return False

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
            try:
                disconnect = getattr(self._engine, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                pass
        self._engine = None
        self._selected_voice = None
        self._voices = []
        self._closed = True
        logger.info("Pyttsx3Engine closed")


__all__ = ["Pyttsx3Engine"]