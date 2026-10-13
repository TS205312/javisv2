"""Terminal rendering helpers with a Rich fallback."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Optional, Sequence, TextIO

try:
    from rich.console import Console as RichConsole
    from rich.table import Table
    from rich.text import Text
except ImportError:
    RichConsole = None
    Table = None
    Text = None


def rich_available() -> bool:
    return RichConsole is not None


def render_value(value: Any) -> str:
    if isinstance(value, Mapping):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return render_value(asdict(value))
        except Exception:
            pass
    if value is None:
        return ""
    return str(value)


class Console:
    """Small output adapter that uses Rich when it is installed."""

    def __init__(
        self,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        use_rich: Optional[bool] = None,
    ) -> None:
        if use_rich is not None:
            enabled = use_rich
        self.enabled = enabled and RichConsole is not None
        self._rich = RichConsole(file=stream) if self.enabled else None
        self._rich_error = RichConsole(file=sys.stderr) if self.enabled else None

    @property
    def using_rich(self) -> bool:
        return self.enabled

    def print(self, message: Any = "", *, style: Optional[str] = None) -> None:
        text = render_value(message)
        if self._rich is not None:
            self._rich.print(Text(text), style=style)
            return
        print(text)

    def error(self, message: Any) -> None:
        text = render_value(message)
        if self._rich_error is not None:
            self._rich_error.print(Text(text), style="bold red")
            return
        print(text, file=sys.stderr)

    def warning(self, message: Any) -> None:
        text = render_value(message)
        if self._rich is not None:
            self._rich.print(Text(text), style="bold yellow")
            return
        print(text, file=sys.stderr)

    def success(self, message: Any) -> None:
        text = render_value(message)
        if self._rich is not None:
            self._rich.print(Text(text), style="bold green")
            return
        print(text)

    def rows(
        self,
        rows: Iterable[Sequence[Any] | Mapping[str, Any]],
        *,
        title: Optional[str] = None,
        columns: Sequence[str] = ("Check", "Status", "Detail"),
    ) -> None:
        materialized = list(rows)
        if self._rich is not None and Table is not None:
            table = Table(title=title, show_header=True, header_style="bold")
            for column in columns:
                table.add_column(column)
            for row in materialized:
                if isinstance(row, Mapping):
                    values = [row.get(column, "") for column in columns]
                else:
                    values = list(row)
                    values.extend("" for _ in range(len(columns) - len(values)))
                cells = [Text(str(value)) for value in values[: len(columns)]]
                if len(cells) > 1:
                    status = str(values[1]).upper()
                    if status in {"OK", "PASS"}:
                        cells[1] = Text(str(values[1]), style="bold green")
                    elif status in {"WARN", "WARNING"}:
                        cells[1] = Text(str(values[1]), style="bold yellow")
                    elif status in {"FAIL", "ERROR"}:
                        cells[1] = Text(str(values[1]), style="bold red")
                table.add_row(*cells)
            self._rich.print(table)
            return

        if title:
            print(title)
        print("  ".join(columns))
        print("  ".join("-" * len(column) for column in columns))
        for row in materialized:
            if isinstance(row, Mapping):
                values = [row.get(column, "") for column in columns]
            else:
                values = list(row)
            print("  ".join(str(value) for value in values[: len(columns)]))


__all__ = ["Console", "render_value", "rich_available"]
