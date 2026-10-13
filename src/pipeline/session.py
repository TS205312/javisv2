"""Session management for Jarvis pipeline.

SessionManager maintains a bounded conversation history (max 12 turns)

and persists it atomically as JSON at D:\\AI_System\\session.json.

All file operations are defensive; failures are swallowed and logged

rather than raised, ensuring the assistant remains usable.

"""

from __future__ import annotations

import json

import logging

import os

import tempfile

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SESSION_PATH = Path(r"D:\\AI_System\session.json")

@dataclass

class Turn:

    """A single conversation turn."""

    user: str

    assistant: str

@dataclass

class SessionState:

    """Serializable session state."""

    turns: List[Dict[str, str]] = field(default_factory=list)

class SessionManager:

    """Bounded, persistent conversation session.

    History is capped at `max_turns` (default 12). When the cap is

    exceeded, oldest turns are dropped. State is persisted atomically to

    `session_path` on every modification.

    All paths used are on the D: drive; no session data lands on C:.

    """

    def __init__(

        self,

        session_path: Optional[Path | str] = None,

        max_turns: int = 12,

    ) -> None:

        self.max_turns = max(1, max_turns)

        if session_path is None:

            session_path = DEFAULT_SESSION_PATH

        elif isinstance(session_path, str):

            session_path = Path(session_path)

        self._session_path: Path = Path(session_path)

        self._history: List[Turn] = []

        self._loaded: bool = False

    @property

    def session_path(self) -> Path:

        """Return the persistence file path."""

        return self._session_path

    @property

    def length(self) -> int:

        """Return the number of turns currently stored."""

        return len(self._history)

    @property

    def history(self) -> List[Turn]:

        """Return a shallow copy of the turn list."""

        return list(self._history)

    def add_turn(self, user: str, assistant: str) -> None:

        """Append a user-assistant turn and persist.

        If `max_turns` is exceeded, the oldest turns are trimmed.

        Errors during persistence are logged but never raised.

        """

        self._history.append(Turn(user=user, assistant=assistant))

        while len(self._history) > self.max_turns:

            self._history.pop(0)

        self._save_defensive()

    def history_for_prompt(self) -> List[Dict[str, str]]:

        """Return history as a list of `{"role": ..., "content": ...}` dicts.

        Suitable for injection into an LLM chat call. Returns an empty

        list if no history exists.

        """

        result: List[Dict[str, str]] = []

        for turn in self._history:

            result.append({"role": "user", "content": turn.user})

            result.append({"role": "assistant", "content": turn.assistant})

        return result

    def clear(self) -> None:

        """Remove all stored turns and persist the empty state."""

        self._history.clear()

        self._save_defensive()

    def load(self) -> None:

        """Load session state from disk defensively.

        If the file does not exist, is corrupted, or causes any error,

        the session remains empty and the error is logged.

        """

        if not self._session_path.exists():

            self._history.clear()

            self._loaded = True

            return

        try:

            with open(self._session_path, "r", encoding="utf-8") as fh:

                data: Dict[str, Any] = json.load(fh)

            turns_raw = data.get("turns", [])

            if not isinstance(turns_raw, list):

                logger.warning("Invalid session format: turns is not a list")

                self._history.clear()

                self._loaded = True

                return

            loaded: List[Turn] = []

            for item in turns_raw:

                if isinstance(item, dict) and "user" in item and "assistant" in item:

                    loaded.append(Turn(user=str(item["user"]), assistant=str(item["assistant"])))

            self._history = loaded[-self.max_turns :]

            self._loaded = True

            logger.info("Loaded %d turns from session", len(self._history))

        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:

            logger.warning("Failed to load session: %s; starting fresh", exc)

            self._history.clear()

            self._loaded = True

        except Exception as exc:

            logger.warning("Unexpected session load error: %s", exc)

            self._history.clear()

            self._loaded = True

    def save(self) -> None:

        """Persist current history to disk."""

        self._save_defensive()

    def _save_defensive(self) -> None:

        """Internal: write session JSON atomically, swallowing errors."""

        session_dir = self._session_path.parent

        try:

            session_dir.mkdir(parents=True, exist_ok=True)

        except OSError as exc:

            logger.warning("Cannot create session directory: %s", exc)

            return

        payload: Dict[str, Any] = {

            "turns": [

                {"user": t.user, "assistant": t.assistant} for t in self._history

            ]

        }

        tmp_fd: Optional[int] = None

        tmp_path: Optional[Path] = None

        try:

            tmp_fd, tmp_path_str = tempfile.mkstemp(

                dir=str(session_dir), suffix=".tmp", prefix="session_"

            )

            tmp_path = Path(tmp_path_str)

            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:

                json.dump(payload, fh, ensure_ascii=False, indent=2)

            os.replace(str(tmp_path), str(self._session_path))

        except (OSError, TypeError) as exc:

            logger.warning("Failed to save session: %s", exc)

        except Exception as exc:

            logger.warning("Unexpected session save error: %s", exc)

        finally:

            if tmp_path is not None and tmp_path.exists():

                try:

                    os.remove(str(tmp_path))

                except OSError:

                    logger.debug("Failed to remove temp file")

    def __repr__(self) -> str:

        return "<SessionManager turns={0} path={1}>".format(

            len(self._history), self._session_path

        )

