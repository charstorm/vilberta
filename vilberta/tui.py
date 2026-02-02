"""Futuristic Textual-based TUI for vilberta with animations."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, RichLog, Footer
from textual.reactive import reactive
from rich.text import Text
from rich.console import Group
from rich.table import Table

from vilberta.config import MODEL_NAME, TTS_VOICE, SAMPLE_RATE


@dataclass
class RequestStats:
    audio_duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    ttft_s: float = 0.0
    total_latency_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class DisplayEvent:
    type: str
    content: str
    stats: RequestStats | None = None


class WaveformWidget(Static):
    """Animated waveform visualization."""

    vad_active = reactive(False)
    frame = reactive(0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.waveform_data: deque[float] = deque([0.0] * 40, maxlen=40)

    def on_mount(self) -> None:
        self.set_interval(1 / 30, self.update_waveform)

    def update_waveform(self) -> None:
        import math

        self.frame += 1

        if self.vad_active:
            new_val = abs(math.sin(self.frame * 0.2)) * 8
            self.waveform_data.append(new_val)
        else:
            self.waveform_data.append(max(0, self.waveform_data[-1] * 0.8))

        self.refresh()

    def render(self) -> Text:
        bars = " ▁▂▃▄▅▆▇█"
        waveform = ""

        for val in list(self.waveform_data):
            idx = min(len(bars) - 1, int(val))
            waveform += bars[idx]

        style = "bold cyan" if self.vad_active else "dim cyan"
        return Text(waveform, style=style)


class SystemPanel(Container):
    """Left panel showing system information and stats."""

    status_text = reactive("INITIALIZING")
    last_stats: reactive[RequestStats | None] = reactive(None)
    exchange_count = reactive(0)
    session_cost = reactive(0.0)
    session_tokens_in = reactive(0)
    session_tokens_out = reactive(0)
    vad_active = reactive(False)

    def __init__(self) -> None:
        super().__init__(id="system-panel")
        self.response_times: deque[float] = deque(maxlen=20)
        self.pulse_frame = 0

    def compose(self) -> ComposeResult:
        yield Static(id="system-header")
        yield Static(id="system-info")
        yield WaveformWidget(id="waveform")
        yield Static(id="stats-display")
        yield Static(id="session-display")
        yield Static(id="status-display")

    def on_mount(self) -> None:
        self.set_interval(1 / 10, self.update_pulse)
        self.update_display()

    def update_pulse(self) -> None:
        self.pulse_frame += 1
        self.update_header()

    def update_header(self) -> None:
        pulse = "●" if (self.pulse_frame // 10) % 2 == 0 else "○"
        header = Text()
        header.append(f"{pulse} ", style="bold bright_cyan")
        header.append("VILBERTA", style="bold bright_magenta")
        self.query_one("#system-header", Static).update(header)

    def update_display(self) -> None:
        self.update_header()
        self.update_system_info()
        self.update_stats()
        self.update_session()
        self.update_status()

    def update_system_info(self) -> None:
        info = Text()
        info.append(f"{MODEL_NAME[:30]}\n", style="cyan")
        info.append(f"{TTS_VOICE} • {SAMPLE_RATE // 1000}kHz\n", style="dim")
        info.append("\n")
        info.append("AUDIO WAVEFORM", style="dim")
        self.query_one("#system-info", Static).update(info)

    def update_stats(self) -> None:
        if not self.last_stats:
            self.query_one("#stats-display", Static).update("")
            return

        stats = self.last_stats
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")
        table.add_column(style="cyan")

        table.add_row("Duration", f"{stats.audio_duration_s:.2f}s")
        table.add_row("TTFT", f"{stats.ttft_s:.2f}s")
        table.add_row("Latency", f"{stats.total_latency_s:.2f}s")
        table.add_row("Input", f"{stats.input_tokens:,}")
        table.add_row("Output", f"{stats.output_tokens:,}")

        if stats.cache_read_tokens:
            table.add_row("Cache R", f"{stats.cache_read_tokens:,}")
        if stats.cache_write_tokens:
            table.add_row("Cache W", f"{stats.cache_write_tokens:,}")

        display = Group(
            Text("LAST REQUEST", style="dim"), Text("─" * 28, style="dim"), table
        )

        self.query_one("#stats-display", Static).update(display)

    def update_session(self) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")
        table.add_column(style="bright_magenta")

        table.add_row("Turns", str(self.exchange_count))
        table.add_row("Cost", f"${self.session_cost:.4f}")
        table.add_row("Tok In", f"{self.session_tokens_in:,}")
        table.add_row("Tok Out", f"{self.session_tokens_out:,}")

        sparkline = self.create_sparkline()

        display = Group(
            Text("\nSESSION", style="dim"),
            Text("─" * 28, style="dim"),
            table,
            Text("\nRESPONSE TIMES", style="dim"),
            sparkline,
        )

        self.query_one("#session-display", Static).update(display)

    def create_sparkline(self) -> Text:
        if not self.response_times:
            return Text("", style="cyan")

        bars = "▁▂▃▄▅▆▇█"
        times = list(self.response_times)
        max_val = max(times) if max(times) > 0 else 1.0

        sparkline = ""
        for val in times:
            idx = min(len(bars) - 1, int((val / max_val) * len(bars)))
            sparkline += bars[idx]

        return Text(sparkline, style="cyan")

    def update_status(self) -> None:
        style = "bold bright_cyan" if "LISTEN" in self.status_text else "bold cyan"
        status = Text("▸ ", style=style)
        status.append(self.status_text, style=style)

        display = Group(Text("\nSTATUS", style="dim"), status)

        self.query_one("#status-display", Static).update(display)

    def watch_status_text(self, value: str) -> None:
        self.update_status()

    def watch_last_stats(self, value: RequestStats | None) -> None:
        if value:
            self.response_times.append(value.total_latency_s)
        self.update_stats()
        self.update_session()

    def watch_exchange_count(self, value: int) -> None:
        self.update_session()

    def watch_session_cost(self, value: float) -> None:
        self.update_session()

    def watch_session_tokens_in(self, value: int) -> None:
        self.update_session()

    def watch_session_tokens_out(self, value: int) -> None:
        self.update_session()

    def watch_vad_active(self, value: bool) -> None:
        waveform = self.query_one("#waveform", WaveformWidget)
        waveform.vad_active = value


class ConversationPanel(Vertical):
    """Center panel showing conversation history."""

    def __init__(self) -> None:
        super().__init__(id="conversation-panel")

    def compose(self) -> ComposeResult:
        yield RichLog(
            id="conversation-log",
            highlight=False,
            markup=True,
            auto_scroll=True,
            wrap=True,
        )


class EventsPanel(Vertical):
    """Right panel showing event log."""

    def __init__(self) -> None:
        super().__init__(id="events-panel")

    def compose(self) -> ComposeResult:
        yield RichLog(
            id="events-log",
            highlight=False,
            markup=True,
            max_lines=100,
            auto_scroll=True,
        )
        yield Static(id="shortcuts")

    def on_mount(self) -> None:
        shortcuts = Text()
        shortcuts.append("SHORTCUTS\n", style="dim")
        shortcuts.append("─────────\n", style="dim")
        shortcuts.append("  q  ", style="cyan")
        shortcuts.append("Quit\n", style="dim")
        shortcuts.append("  ↑↓ ", style="cyan")
        shortcuts.append("Scroll\n", style="dim")
        shortcuts.append("Home ", style="cyan")
        shortcuts.append("Top\n", style="dim")
        shortcuts.append(" End ", style="cyan")
        shortcuts.append("Bottom", style="dim")

        self.query_one("#shortcuts", Static).update(shortcuts)


class VilbertaTUI(App[None]):
    """Futuristic TUI for Vilberta voice assistant."""

    CSS = """
    Screen {
        background: #0a0a0a;
    }
    
    #system-panel {
        width: 35;
        height: 100%;
        border: heavy #00ffff;
        border-title-color: #ff00ff;
        border-title-style: bold;
        padding: 1 2;
        background: #0a0a14;
    }
    
    #conversation-panel {
        width: 1fr;
        height: 100%;
        border: heavy #00ffff;
        border-title-color: #ff00ff;
        border-title-style: bold;
        padding: 1 2;
        background: #0a0a14;
        margin: 0 1;
    }
    
    #events-panel {
        width: 32;
        height: 100%;
        border: heavy #00ffff;
        border-title-color: #ff00ff;
        border-title-style: bold;
        padding: 1 2;
        background: #0a0a14;
    }
    
    #conversation-log {
        height: 1fr;
        border: none;
        background: transparent;
        color: #e0e0e0;
    }
    
    #events-log {
        height: 1fr;
        border: none;
        background: transparent;
        scrollbar-color: #00ffff;
        scrollbar-color-hover: #ff00ff;
    }
    
    #shortcuts {
        height: auto;
        margin-top: 1;
        border-top: solid #00ffff;
        padding-top: 1;
    }
    
    #waveform {
        height: 3;
        content-align: center middle;
        background: #050510;
        border: solid #00ffff;
        margin: 1 0;
    }
    
    Footer {
        background: #00ffff;
        color: #000000;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.event_queue: Queue[DisplayEvent] | None = None
        self.shutdown_event: threading.Event | None = None
        self.in_ai_voice_block = False
        self.in_ai_text_block = False

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield SystemPanel()
            yield ConversationPanel()
            yield EventsPanel()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "VILBERTA"
        self.sub_title = "Voice Intelligence System"

        self.query_one("#system-panel").border_title = "SYSTEM"
        self.query_one("#conversation-panel").border_title = "CONVERSATION"
        self.query_one("#events-panel").border_title = "EVENTS"

        self.set_interval(1 / 30, self.process_events)

    def process_events(self) -> None:
        if self.shutdown_event and self.shutdown_event.is_set():
            self.exit()
            return

        if not self.event_queue:
            return

        try:
            while True:
                event = self.event_queue.get_nowait()
                self.handle_event(event)
        except Empty:
            pass

    def handle_event(self, event: DisplayEvent) -> None:
        conversation = self.query_one("#conversation-log", RichLog)
        events_log = self.query_one("#events-log", RichLog)
        system_panel = self.query_one("#system-panel", SystemPanel)

        if event.type == "speak":
            if self.in_ai_text_block:
                conversation.write("")
                self.in_ai_text_block = False

            if not self.in_ai_voice_block:
                conversation.write(Text(f"🤖 > {event.content}", style="bold yellow"))
                self.in_ai_voice_block = True
            else:
                conversation.write(Text(f"    │ {event.content}", style="yellow"))

            events_log.write(Text(f"SPEAK {event.content[:30]}", style="yellow"))

        elif event.type == "text":
            if self.in_ai_voice_block:
                conversation.write("")
                self.in_ai_voice_block = False

            lines = event.content.splitlines()
            for line in lines:
                if not self.in_ai_text_block:
                    conversation.write(Text(f"💬 > {line}", style="bold magenta"))
                    self.in_ai_text_block = True
                else:
                    conversation.write(Text(f"    │ {line}", style="magenta"))

            events_log.write(Text(f"TEXT  {event.content[:30]}", style="magenta"))

        elif event.type == "transcript":
            self.end_ai_blocks(conversation)
            conversation.write("")
            conversation.write(Text(f"👤 > {event.content}", style="bold bright_cyan"))
            conversation.write("")

            system_panel.exchange_count += 1
            events_log.write(Text(f"USER  {event.content[:30]}", style="bright_cyan"))

        elif event.type == "status":
            msg = event.content.strip()
            if msg.startswith("[") and msg.endswith("]"):
                conversation.write(Text(f"       {msg}", style="dim"))

            status_map = {
                "Listening...": "LISTENING",
                "Processing...": "PROCESSING",
                "Continuing recording...": "RECORDING",
                "Loading TTS model...": "LOADING TTS",
                "TTS ready.": "TTS READY",
                "Initializing LLM service...": "INIT LLM",
                "Ready. Listening...": "LISTENING",
            }

            for key, val in status_map.items():
                if key in event.content:
                    system_panel.status_text = val
                    break

            events_log.write(Text(f"STAT  {msg[:30]}", style="dim"))

        elif event.type == "error":
            conversation.write(Text(f"❌ > {event.content}", style="bold red"))
            events_log.write(Text(f"ERROR {event.content[:30]}", style="red"))

        elif event.type == "vad":
            system_panel.vad_active = event.content == "up"
            status = "▲ speech" if event.content == "up" else "▼ silence"
            style = "green" if event.content == "up" else "dim"
            events_log.write(Text(f"VAD   {status}", style=style))

        elif event.type == "stats":
            if event.stats:
                system_panel.last_stats = event.stats
                system_panel.session_cost += event.stats.cost_usd
                system_panel.session_tokens_in += event.stats.input_tokens
                system_panel.session_tokens_out += event.stats.output_tokens

                events_log.write(
                    Text(
                        f"STATS ttft={event.stats.ttft_s:.2f}s "
                        f"in={event.stats.input_tokens} out={event.stats.output_tokens}",
                        style="cyan",
                    )
                )

        elif event.type == "boot":
            events_log.write(Text(f"BOOT  {event.content.strip()[:30]}", style="cyan"))

    def end_ai_blocks(self, conversation: RichLog) -> None:
        if self.in_ai_voice_block or self.in_ai_text_block:
            conversation.write("")
            self.in_ai_voice_block = False
            self.in_ai_text_block = False

    async def action_quit(self) -> None:
        """Quit the application."""
        if self.shutdown_event:
            self.shutdown_event.set()
        self.exit()

    def setup(
        self, event_queue: Queue[DisplayEvent], shutdown_event: threading.Event
    ) -> None:
        """Setup the TUI with the given event queue and shutdown event."""
        self.event_queue = event_queue
        self.shutdown_event = shutdown_event


def run_tui(event_queue: Queue[DisplayEvent], shutdown_event: threading.Event) -> None:
    app = VilbertaTUI()
    app.setup(event_queue, shutdown_event)
    app.run()


# Compatibility wrapper class for backward compatibility with main.py
class CursesTUI:
    """Wrapper class that mimics the old CursesTUI interface."""

    def __init__(self) -> None:
        self.app = VilbertaTUI()

    def run(
        self, event_queue: Queue[DisplayEvent], shutdown_event: threading.Event
    ) -> None:
        """Run the TUI with the given event queue and shutdown event."""
        self.app.setup(event_queue, shutdown_event)
        self.app.run()

    def cleanup(self) -> None:
        """Cleanup method for compatibility (Textual handles cleanup automatically)."""
        pass


# vilberta_textual_tui.py
