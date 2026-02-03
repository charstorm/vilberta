"""Display adapter — routes all output through a queue to the UI."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue


@dataclass
class RequestStats:
    """Stats for a single request."""

    audio_duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class DisplayEvent:
    """Event for display."""

    type: str
    content: str
    stats: RequestStats | None = None


_event_queue: Queue[DisplayEvent] | None = None


def init_display(queue: Queue[DisplayEvent]) -> None:
    global _event_queue
    _event_queue = queue


def _emit(event_type: str, content: str) -> None:
    if _event_queue is not None:
        _event_queue.put(DisplayEvent(type=event_type, content=content))
    else:
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


def print_stats(stats: RequestStats) -> None:
    if _event_queue is not None:
        _event_queue.put(DisplayEvent(type="stats", content="", stats=stats))


# display.py
