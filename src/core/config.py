"""Configuration loader for Jarvis.

Reads config.yaml and overlays values from environment variables.
No network access is performed at import time; all I/O is wrapped in
defensive try/except blocks so a missing or malformed file still yields
a usable configuration with documented defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Runtime configuration for the Jarvis platform.

    Values are sourced from (in order of precedence):

    1. Environment variables (JARVIS_*).
    2. config.yaml in the project root.
    3. Hard-coded defaults defined in _defaults.
    """

    # Ollama / LLM
    ollama_host: str = 'http://localhost:11434'
    llm_model: str = 'qwen2.5:0.5b'
    num_ctx: int = 2048
    num_predict: int = 512
    temperature: float = 0.3
    timeout: int = 90

    # Speech-to-text
    stt_sample_rate: int = 16000
    stt_model_path: str = ''

    # Text-to-speech
    tts_prefer_online: bool = True
    tts_voice: str = 'vi-VN-NamMinhNeural'
    tts_rate: int = 200
    tts_volume: float = 1.0

    # RAG
    rag_model: str = 'all-MiniLM-L6-v2'
    rag_chunk_size: int = 512
    rag_overlap: int = 80
    rag_top_k: int = 4

    # Vision
    vision_model: str = 'moondream:latest'
    vision_timeout: int = 90
    vision_max_width: int = 640
    vision_jpeg_quality: int = 80

    # Resource limits
    resource_max_workers: int = 2
    resource_max_ram_gb: int = 12

    @classmethod
    def _env_str(cls, key: str, default: str) -> str:
        value = os.environ.get('JARVIS_' + key)
        if value is None or value == '':
            return default
        return value

    @classmethod
    def _env_int(cls, key: str, default: int) -> int:
        raw = os.environ.get('JARVIS_' + key)
        if raw is None or raw == '':
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _env_float(cls, key: str, default: float) -> float:
        raw = os.environ.get('JARVIS_' + key)
        if raw is None or raw == '':
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _env_bool(cls, key: str, default: bool) -> bool:
        raw = os.environ.get('JARVIS_' + key)
        if raw is None or raw == '':
            return default
        return raw.strip().lower() in {'1', 'true', 'yes', 'on'}

    @classmethod
    def _clamp_int(cls, value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, value))

    @classmethod
    def _clamp_float(cls, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @classmethod
    def from_yaml(cls, path: str = 'config.yaml') -> 'Config':
        defaults = cls._defaults()
        data = {}
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as fh:
                loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, Exception):
            data = {}

        merged = asdict(defaults)
        for key, value in data.items():
            if hasattr(defaults, key):
                merged[key] = value

        return cls(
            ollama_host=cls._env_str('OLLAMA_HOST', merged.get('ollama_host', defaults.ollama_host)),
            llm_model=cls._env_str('LLM_MODEL', merged.get('llm_model', defaults.llm_model)),
            num_ctx=cls._clamp_int(cls._env_int('NUM_CTX', merged.get('num_ctx', defaults.num_ctx)), 1, 32768),
            num_predict=cls._clamp_int(cls._env_int('NUM_PREDICT', merged.get('num_predict', defaults.num_predict)), 1, 8192),
            temperature=cls._clamp_float(cls._env_float('TEMPERATURE', merged.get('temperature', defaults.temperature)), 0.0, 2.0),
            timeout=cls._clamp_int(cls._env_int('TIMEOUT', merged.get('timeout', defaults.timeout)), 1, 600),
            stt_sample_rate=cls._clamp_int(cls._env_int('STT_SAMPLE_RATE', merged.get('stt_sample_rate', defaults.stt_sample_rate)), 8000, 48000),
            stt_model_path=cls._env_str('STT_MODEL_PATH', merged.get('stt_model_path', defaults.stt_model_path)),
            tts_prefer_online=cls._env_bool('TTS_PREFER_ONLINE', merged.get('tts_prefer_online', defaults.tts_prefer_online)),
            tts_voice=cls._env_str('TTS_VOICE', merged.get('tts_voice', defaults.tts_voice)),
            tts_rate=cls._clamp_int(cls._env_int('TTS_RATE', merged.get('tts_rate', defaults.tts_rate)), 50, 500),
            tts_volume=cls._clamp_float(cls._env_float('TTS_VOLUME', merged.get('tts_volume', defaults.tts_volume)), 0.0, 1.0),
            rag_model=cls._env_str('RAG_MODEL', merged.get('rag_model', defaults.rag_model)),
            rag_chunk_size=cls._clamp_int(cls._env_int('RAG_CHUNK_SIZE', merged.get('rag_chunk_size', defaults.rag_chunk_size)), 64, 4096),
            rag_overlap=cls._clamp_int(cls._env_int('RAG_OVERLAP', merged.get('rag_overlap', defaults.rag_overlap)), 0, 1024),
            rag_top_k=cls._clamp_int(cls._env_int('RAG_TOP_K', merged.get('rag_top_k', defaults.rag_top_k)), 1, 50),
            vision_model=cls._env_str('VISION_MODEL', merged.get('vision_model', defaults.vision_model)),
            vision_timeout=cls._clamp_int(cls._env_int('VISION_TIMEOUT', merged.get('vision_timeout', defaults.vision_timeout)), 5, 300),
            vision_max_width=cls._clamp_int(cls._env_int('VISION_MAX_WIDTH', merged.get('vision_max_width', defaults.vision_max_width)), 160, 1920),
            vision_jpeg_quality=cls._clamp_int(cls._env_int('VISION_JPEG_QUALITY', merged.get('vision_jpeg_quality', defaults.vision_jpeg_quality)), 1, 100),
            resource_max_workers=cls._clamp_int(cls._env_int('RESOURCE_MAX_WORKERS', merged.get('resource_max_workers', defaults.resource_max_workers)), 1, 16),
            resource_max_ram_gb=cls._clamp_int(cls._env_int('RESOURCE_MAX_RAM_GB', merged.get('resource_max_ram_gb', defaults.resource_max_ram_gb)), 1, 128),
        )

    @classmethod
    def _defaults(cls) -> 'Config':
        return cls()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str = 'config.yaml') -> Config:
    return Config.from_yaml(path=path)


__all__ = ['Config', 'load_config']