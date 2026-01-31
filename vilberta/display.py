"""Display adapter — routes all output through a queue to the TUI."""

from __future__ import annotations

from queue import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vilberta.tui import DisplayEvent

_event_queue: Queue[DisplayEvent] | None = None


def init_display(queue: Queue[DisplayEvent]) -> None:
    global _event_queue
    _event_queue = queue


def _emit(event_type: str, content: str) -> None:
    from vilberta.tui import DisplayEvent

    if _event_queue is not None:
        _event_queue.put(DisplayEvent(type=event_type, content=content))
    else:
        # Fallback before TUI is initialized (e.g. preflight checks)
        print(content)


def print_speak(text: str) -> None:
    _emit("speak", text)


def print_text(text: str) -> None:
    _emit("text", text)


def print_transcript(text: str) -> None:
    _emit("transcript", text)


def print_status(message: str) -> None:
    _emit("status", message)


def print_error(message: str) -> None:
    _emit("error", message)


def print_vad(*, up: bool) -> None:
    _emit("vad", "up" if up else "down")


def print_stats(stats: object) -> None:
    from vilberta.tui import DisplayEvent

    if _event_queue is not None:
        _event_queue.put(DisplayEvent(type="stats", content="", stats=stats))  # type: ignore[arg-type]
