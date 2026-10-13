"""Command parsing and routing for Jarvis pipeline.

Provides a registry of local commands (Vietnamese/English) that the
assistant can handle without invoking the LLM. Commands are matched
against user input via regex patterns; unmatched input returns an
unhandled result so the orchestrator falls through to normal chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CommandResult:
    """Result of a command routing attempt.

    Attributes:
        command: The canonical command name (e.g. `exit`, `rag`),
            or `None` when no command matched.
        handled: `True` if a registered command matched the input.
        response: The response text to present to the user.
        action: The action key that the orchestrator can dispatch on,
            or `None` when unhandled.
        args: Extracted arguments from the input.
    """

    command: Optional[str]
    handled: bool
    response: str
    action: Optional[str] = None
    args: Dict[str, str] = field(default_factory=dict)


CommandHandler = Callable[[Dict[str, str]], CommandResult]


class _CommandDef:
    """Internal: a registered command with its pattern and handler."""

    def __init__(
        self,
        name: str,
        action: str,
        pattern: re.Pattern[str],
        handler: CommandHandler,
        description: str,
    ) -> None:
        self.name = name
        self.action = action
        self.pattern = pattern
        self.handler = handler
        self.description = description


class CommandRegistry:
    """Registry of local commands (Vietnamese and English).

    Commands are matched in registration order; the first match wins.
    Each command maps user input to a canonical action plus extracted
    arguments, then invokes its handler to produce a response.
    """

    def __init__(self) -> None:
        self._commands: List[_CommandDef] = []

    def register(
        self,
        name: str,
        action: str,
        triggers: List[str],
        handler: CommandHandler,
        description: str = "",
        needs_args: bool = False,
    ) -> None:
        """Register a command.

        Args:
            name: Display name of the command.
            action: Canonical action key dispatched by the orchestrator.
            triggers: List of trigger pattern strings (raw, not regex).
                Each is converted to a regex that matches the whole line.
            handler: Called with `{"cmd": matched_text, "args": arg_dict}`
                when the command matches.
            description: Human-readable one-liner.
            needs_args: If `True`, captures the remainder of the line
                as `{"arg0": ...}`.
        """
        if needs_args:
            inner = "|".join(re.escape(t) for t in triggers)
            compiled = re.compile(
                f"(?:{inner})\\s+(.+)$",
                re.IGNORECASE,
            )
        else:
            compiled = re.compile(
                "|".join(f"({re.escape(t)})" for t in triggers),
                re.IGNORECASE,
            )
        self._commands.append(
            _CommandDef(
                name=name,
                action=action,
                pattern=compiled,
                handler=handler,
                description=description,
            )
        )

    def match(self, text: str) -> CommandResult:
        """Try to match *text* against registered commands.

        Returns a `CommandResult`; if nothing matches, returns an
        unhandled result (`handled=False`, `command=None`).
        """
        stripped = text.strip()
        if not stripped:
            return CommandResult(
                command=None, handled=False, response="", action=None
            )
        for cmd in self._commands:
            m = cmd.pattern.fullmatch(stripped)
            if m:
                try:
                    arg_val = m.group(1) if m.lastindex else ""
                    if arg_val:
                        return cmd.handler(
                            {"cmd": m.group(0), "args": {"arg0": arg_val}}
                        )
                    return cmd.handler(
                        {"cmd": m.group(0), "args": {}}
                    )
                except Exception as exc:
                    return CommandResult(
                        command=cmd.name,
                        handled=True,
                        response="L\u01ed7i khi th\u1ef1c thi l\u1ec7nh: {0}".format(exc),
                        action=cmd.action,
                    )
        return CommandResult(
            command=None, handled=False, response="", action=None
        )

    @property
    def commands(self) -> List[Tuple[str, str, str]]:
        """Return list of (name, action, description) for introspection."""
        return [(c.name, c.action, c.description) for c in self._commands]


def _ingest_handler(args: Dict[str, str]) -> CommandResult:
    raw = args.get("arg0", "")
    if not raw:
        return CommandResult(
            command="ingest",
            handled=True,
            response="C\u1ea7n ch\u1ec9 \u0111\u1ec3 \u0111\u01b0\u1eddng d\u1eabn: nạp <\u0111\u01b0\u1eddng d\u1eabn>",
            action="ingest",
            args={"path": ""},
        )
    return CommandResult(
        command="ingest",
        handled=True,
        response="S\u1eb5n s\u00e0ng n\u1ea1p t\u00e0i li\u1ec7u t\u1eeb: {0}".format(raw),
        action="ingest",
        args={"path": raw},
    )


def _rag_handler(args: Dict[str, str]) -> CommandResult:
    query = args.get("arg0", "").strip()
    if not query:
        return CommandResult(
            command="rag",
            handled=True,
            response="C\u1ea7n c\u1ea5u h\u1ec7i: h\u1ec7i t\u00e0i li\u1ec7u <c\u1ea5u h\u1ec7i>",
            action="rag",
            args={"query": ""},
        )
    return CommandResult(
        command="rag",
        handled=True,
        response="H\u1ec7i t\u00e0i li\u1ec7u: {0}".format(query),
        action="rag",
        args={"query": query},
    )


def _vision_handler(args: Dict[str, str]) -> CommandResult:
    prompt = args.get("arg0", "").strip()
    if not prompt:
        return CommandResult(
            command="vision",
            handled=True,
            response="C\u1ea7n m\u00f4 t\u1ea3: ch\u1ea5p \u1ea3nh <m\u00f4 t\u1ea3>",
            action="vision",
            args={"prompt": ""},
        )
    return CommandResult(
        command="vision",
        handled=True,
        response="Ch\u1ea5p \u1ea3nh: {0}".format(prompt),
        action="vision",
        args={"prompt": prompt},
    )


def _tts_online_handler(args: Dict[str, str]) -> CommandResult:
    return CommandResult(
        command="tts",
        handled=True,
        response="B\u1eadt TTS online.",
        action="tts_online",
        args={},
    )


def _tts_offline_handler(args: Dict[str, str]) -> CommandResult:
    return CommandResult(
        command="tts",
        handled=True,
        response="T\u1eaft TTS online, chuy\u1ec3n offline.",
        action="tts_offline",
        args={},
    )


def _exit_handler(args: Dict[str, str]) -> CommandResult:
    return CommandResult(
        command="exit",
        handled=True,
        response="T\u1ea1m bi\u1ebet!",
        action="exit",
        args={},
    )


def _help_handler(args: Dict[str, str]) -> CommandResult:
    help_text = (
        "C\u00e1c l\u1ec7nh c\u00f3 s\u1eb5n:\n"
        "  tho\u00e1t / exit / quit  - \u0110\u00f3ng tr\u1ee3 l\u00fd\n"
        "  tr\u1ee3 gi\u00fap / help     - Hi\u1ec3n th\u1ecb danh s\u00e1ch l\u1ec7nh\n"
        "  tr\u1ea1ng th\u00e1i / status - Ki\u1ec3m tra t\xecnh tr\u1ea1ng h\u1ec7 th\u1ed1ng\n"
        "  n\u1ea1p <path> / ingest <path> - N\u1ea1p t\u00e0i li\u1ec7u v\xe0o RAG\n"
        "  h\u1ec7i t\u00e0i li\u1ec7u <query> / rag <query> - H\u1ec7i v\u1ec1 t\u00e0i li\u1ec7u \u0111\xe3 n\u1ea1p\n"
        "  ch\u1ea5p \u1ea3nh <prompt> / see <prompt> - Ch\u1ea5p \u1ea3nh v\xe0 ph\xe2n t\xedch\n"
        "  tts online/offline  - B\u1eadt/t\u1eaft TTS online\n"
        "  qu\u1eb1n / clear        - X\u00f3a l\u1ecbch s\u1eed h\u1ec7i th\u01b0\u1edbc"
    )
    return CommandResult(
        command="help",
        handled=True,
        response=help_text.strip(),
        action="help",
        args={},
    )


def _status_handler(args: Dict[str, str]) -> CommandResult:
    return CommandResult(
        command="status",
        handled=True,
        response="\u0110ang ki\u1ec3m tra tr\u1ea1ng th\u00e1i...",
        action="status",
        args={},
    )


def _clear_handler(args: Dict[str, str]) -> CommandResult:
    return CommandResult(
        command="clear",
        handled=True,
        response="\u0110\xe3 x\u00f3a l\u1ecbch s\u1eed h\u1ec7i th\u01b0\u1edbc.",
        action="clear",
        args={},
    )


def build_default_registry() -> CommandRegistry:
    """Build and return the default command registry with all local commands."""
    registry = CommandRegistry()

    registry.register(
        name="Tho\u00e1t",
        action="exit",
        triggers=["tho\u00e1t", "exit", "quit"],
        handler=_exit_handler,
        description="\u0110\u00f3ng tr\u1ee3 l\u00fd",
    )
    registry.register(
        name="Tr\u1ee3 gi\u00fap",
        action="help",
        triggers=["tr\u1ee3 gi\u00fap", "help"],
        handler=_help_handler,
        description="Hi\u1ec3n th\u1ecb danh s\u00e1ch l\u1ec7nh",
    )
    registry.register(
        name="Tr\u1ea1ng th\u00e1i",
        action="status",
        triggers=["tr\u1ea1ng th\u00e1i", "status"],
        handler=_status_handler,
        description="Ki\u1ec3m tra t\xecnh tr\u1ea1ng h\u1ec7 th\u1ed1ng",
    )
    registry.register(
        name="N\u1ea1p t\u00e0i li\u1ec7u",
        action="ingest",
        triggers=["n\u1ea1p", "ingest"],
        handler=_ingest_handler,
        description="N\u1ea1p t\u00e0i li\u1ec7u v\xe0o RAG: nạp <\u0111\u01b0\u1eddng d\u1eabn>",
        needs_args=True,
    )
    registry.register(
        name="H\u1ec7i t\u00e0i li\u1ec7u",
        action="rag",
        triggers=["h\u1ec7i t\u00e0i li\u1ec7u", "rag"],
        handler=_rag_handler,
        description="H\u1ec7i v\u1ec1 t\u00e0i li\u1ec7u \u0111\xe3 n\u1ea1p",
        needs_args=True,
    )
    registry.register(
        name="Ch\u1ea5p \u1ea3nh",
        action="vision",
        triggers=["ch\u1ea5p \u1ea3nh", "see"],
        handler=_vision_handler,
        description="Ch\u1ea5p \u1ea3nh v\xe0 ph\xe2n t\xedch",
        needs_args=True,
    )
    registry.register(
        name="TTS online",
        action="tts_online",
        triggers=["tts online"],
        handler=_tts_online_handler,
        description="B\u1eadt TTS online",
    )
    registry.register(
        name="TTS offline",
        action="tts_offline",
        triggers=["tts offline"],
        handler=_tts_offline_handler,
        description="T\u1eaft TTS online",
    )
    registry.register(
        name="Qu\u1eb1n",
        action="clear",
        triggers=["qu\u1eb1n", "clear"],
        handler=_clear_handler,
        description="X\u00f3a l\u1ecbch s\u1eed h\u1ec7i th\u01b0\u1edbc",
    )

    return registry
