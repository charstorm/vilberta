"""Display adapter — routes all output through a queue to the UI."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Any


@dataclass
class DisplayEvent:
    """Event for display."""

    type: str
    content: str


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


def print_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    _emit("tool_call", f"{tool_name}({args_str})")


def print_tool_result(tool_name: str, success: bool, result: str) -> None:
    status = "OK" if success else "ERR"
    _emit("tool_result", f"{tool_name}|{status}|{result}")


def print_subsystem_ready(subsystem: str) -> None:
    _emit("subsystem_ready", subsystem)


# display.py
