"""Path management for Jarvis.

PathManager is the single source of truth for all filesystem locations.
All paths default to the D: drive so that heavy data never lands on the
system C: drive. Directories are created safely on first access;
failures are swallowed and reported via the returned path object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class PathInfo:
    """Result of resolving a managed path."""

    key: str
    path: Path
    created: bool = False


class PathManager:
    """Central registry and creator of Jarvis directories."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        brain_dir: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(
            base_dir or os.environ.get("JARVIS_BASE_DIR", r"D:\AI_System")
        )
        self.brain_dir = Path(
            brain_dir or os.environ.get("JARVIS_BRAIN_DIR", r"D:\AI_Brain")
        )
        self._defaults: Dict[str, Path] = {
            "root": self.base_dir,
            "brain_root": self.brain_dir,
            "models": self.brain_dir / "models",
            "cache": self.base_dir / "cache",
            "logs": self.base_dir / "logs",
            "faiss": self.brain_dir / "faiss",
            "audio": self.base_dir / "audio",
            "vision": self.base_dir / "vision",
            "huggingface": self.brain_dir / "huggingface",
        }

    def get(self, key: str) -> Path:
        if key not in self._defaults:
            raise KeyError("Unknown path key: {0}".format(key))
        return self._defaults[key]

    def ensure(self, key: str) -> PathInfo:
        path = self.get(key)
        created = False
        try:
            path.mkdir(parents=True, exist_ok=True)
            created = True
        except OSError:
            pass
        return PathInfo(key=key, path=path, created=created)

    def ensure_all(self) -> Dict[str, PathInfo]:
        results: Dict[str, PathInfo] = {}
        for key in self._defaults:
            results[key] = self.ensure(key)
        return results

    def vosk_model_path(self, configured: Optional[str] = None) -> Path:
        """Resolve the Vosk speech-to-text model directory.

        Resolution order (first existing wins):
        1. Explicit `configured` argument or `JARVIS_STT_MODEL_PATH` env var.
        2. D:\\AI_Brain\\models\\vosk-model-small-vn-0.4.
        3. D:\\jarvis-data\\models\\vosk\\vosk-model-small-vn-0.4.

        The returned path is always absolute. No silent fallback to C: is
        performed; callers receive the intended D: path even if it does not
        exist yet so they can surface a clear error to the user.
        """
        if configured:
            return Path(configured).resolve()
        env_override = os.environ.get("JARVIS_STT_MODEL_PATH")
        if env_override:
            return Path(env_override).resolve()
        preferred = (self.brain_dir / "models" /
                     "vosk-model-small-vn-0.4").resolve()
        if preferred.exists():
            return preferred
        legacy = Path(r"D:\jarvis-data\models\vosk\vosk-model-small-vn-0.4").resolve()
        if legacy.exists():
            return legacy
        return preferred

    def all_paths(self) -> Dict[str, Path]:
        return dict(self._defaults)


def default_path_manager() -> PathManager:
    return PathManager()


__all__ = ["PathManager", "PathInfo", "default_path_manager"]