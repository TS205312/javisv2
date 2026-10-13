"""Pipeline package for Jarvis.

The pipeline package ties together all subsystems into a unified
assistant orchestration layer. It provides:

- `AssistantOrchestrator` - main entry point for chat, voice,
  vision, and document ingestion.
- `SessionManager` - bounded, persistent conversation history.
- `CommandRegistry` - local command parsing and routing.
"""

from __future__ import annotations

from .commands import CommandResult, CommandRegistry, build_default_registry
from .orchestrator import AssistantOrchestrator
from .session import SessionManager, Turn

__all__ = [
    "AssistantOrchestrator",
    "CommandResult",
    "CommandRegistry",
    "SessionManager",
    "Turn",
    "build_default_registry",
]
