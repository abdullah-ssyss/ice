from __future__ import annotations

from collections.abc import Iterable, Sequence
from shutil import get_terminal_size
from typing import Any


ConsoleSection = tuple[str, Sequence[tuple[str, Any]]]


def format_table(
    sections: Iterable[ConsoleSection],
    *,
    max_value_width: int = 88,
    terminal_width: int | None = None,
) -> str:
    """Render aligned metrics within the current terminal's available columns."""

    if max_value_width < 1:
        raise ValueError("max_value_width must be at least 1")
    if terminal_width is None:
        terminal_width = get_terminal_size(fallback=(80, 24)).columns
    # Avoid writing into the last column: some terminals wrap immediately there.
    available_width = max(1, terminal_width - 1)

    rows: list[tuple[str, str]] = []
    for section, metrics in sections:
        rows.append((f"{section}/", ""))
        rows.extend((f"    {name}", _stringify(value)) for name, value in metrics)

    if not rows:
        return ""

    if available_width < 12:
        # A bordered two-column layout is unreadable at this size.
        return "\n".join(
            _ellipsize(f"{key.strip()}: {value}" if value else key, available_width)
            for key, value in rows
        )

    content_width = available_width - 7  # borders, separator, and padding
    value_width = min(
        max((len(value) for _, value in rows), default=0),
        max_value_width,
    )
    value_width = max(value_width, 1)
    # Reserve space for numeric values before letting long metric names expand.
    key_width = min(
        max(len(key) for key, _ in rows),
        content_width - min(value_width, max(1, content_width // 2)),
    )
    value_width = min(value_width, content_width - key_width)
    border = "-" * (key_width + value_width + 7)
    lines = [border]
    for key, value in rows:
        lines.append(
            f"| {_ellipsize(key, key_width):<{key_width}} | "
            f"{_ellipsize(value, value_width):<{value_width}} |"
        )
    lines.append(border)
    return "\n".join(lines)


def print_table(sections: Iterable[ConsoleSection]) -> None:
    print(format_table(sections), flush=True)


def _stringify(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    # Paths and external strings must not inject new rows or tab-dependent spacing.
    return " ".join(str(value).split())


def _ellipsize(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."
