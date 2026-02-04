"""Simple CLI interface for vilberta without curses/textual dependencies."""

from __future__ import annotations

import threading
from queue import Queue, Empty

from vilberta.display import DisplayEvent


class SimpleCLI:
    """Simple CLI interface using plain print statements."""

    def __init__(self) -> None:
        self.event_queue: Queue[DisplayEvent] | None = None
        self.shutdown_event: threading.Event | None = None
        self.exchange_count = 0

    def run(
        self, event_queue: Queue[DisplayEvent], shutdown_event: threading.Event
    ) -> None:
        """Run the CLI event loop."""
        self.event_queue = event_queue
        self.shutdown_event = shutdown_event

        while not shutdown_event.is_set():
            try:
                event = event_queue.get(timeout=0.1)
                self._handle_event(event)
            except Empty:
                continue

    def _handle_event(self, event: DisplayEvent) -> None:
        """Handle a single display event."""
        if event.type == "boot":
            print(event.content)

        elif event.type == "speak":
            print(f"🤖 {event.content}")

        elif event.type == "text":
            print(f"\n{event.content}\n")

        elif event.type == "transcript":
            self.exchange_count += 1
            print(f"👤 {event.content}")

        elif event.type == "status":
            print(f"[{event.content}]")

        elif event.type == "error":
            print(f"❌ Error: {event.content}")

        elif event.type == "vad":
            status = "▲ speech" if event.content == "up" else "▼ silence"
            print(f"[VAD: {status}]")

    def cleanup(self) -> None:
        """Cleanup method (no-op for CLI)."""
        pass


# cli.py
