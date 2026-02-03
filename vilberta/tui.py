"""Textual-based TUI for vilberta with theme support and minimal CSS."""

from __future__ import annotations

import threading
import re
from collections import deque
from queue import Queue, Empty
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container, VerticalScroll
from textual.widgets import Static, Footer
from textual.reactive import reactive

from vilberta.config import get_config
from vilberta.display import DisplayEvent, RequestStats


def parse_markdown_to_textual(text: str) -> str:
    """Convert markdown formatting to Textual markup."""
    # Bold: **text** -> [bold]text[/]
    text = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/]", text)

    # Italic: *text* (but not ** which was already handled)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"[italic]\1[/]", text)

    # Underline: __text__
    text = re.sub(r"__(.+?)__", r"[underline]\1[/]", text)

    return text


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

    def render(self) -> str:
        bars = " ▁▂▃▄▅▆▇█"
        waveform = "".join(
            bars[min(len(bars) - 1, int(val))] for val in self.waveform_data
        )
        return waveform


class ScrollingLog(VerticalScroll):
    """A scrolling log widget that auto-scrolls to bottom."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.messages: list[tuple[str, str]] = []

    def write(
        self, content: str, style: str = "", parse_markdown: bool = False
    ) -> None:
        """Add a message to the log."""
        if parse_markdown:
            content = parse_markdown_to_textual(content)

        self.messages.append((content, style))

        new_static = Static(content, markup=parse_markdown, classes=style)
        self.mount(new_static)

        self.scroll_end(animate=False)


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
        header = f"{pulse} VILBERTA"
        self.query_one("#system-header", Static).update(header)

    def update_display(self) -> None:
        self.update_header()
        self.update_system_info()
        self.update_stats()
        self.update_session()
        self.update_status()

    def update_system_info(self) -> None:
        cfg = get_config()
        info = (
            f"{cfg.model_name[:30]}\n"
            f"{cfg.tts_voice} • {cfg.sample_rate // 1000}kHz\n"
            "\n"
            "AUDIO WAVEFORM"
        )
        widget = self.query_one("#system-info", Static)
        widget.update(info)
        widget.add_class("muted")

    def update_stats(self) -> None:
        if not self.last_stats:
            self.query_one("#stats-display", Static).update("")
            return

        stats = self.last_stats

        lines = [
            "LAST REQUEST",
            "─" * 28,
            f"Duration    {stats.audio_duration_s:.2f}s",
            f"Latency     {stats.latency_s:.2f}s",
            f"Input       {stats.input_tokens:,}",
            f"Output      {stats.output_tokens:,}",
        ]

        if stats.cache_read_tokens:
            lines.append(f"Cache R     {stats.cache_read_tokens:,}")
        if stats.cache_write_tokens:
            lines.append(f"Cache W     {stats.cache_write_tokens:,}")

        self.query_one("#stats-display", Static).update("\n".join(lines))

    def update_session(self) -> None:
        sparkline = self.create_sparkline()

        lines = [
            "",
            "SESSION",
            "─" * 28,
            f"Turns       {self.exchange_count}",
            f"Cost        ${self.session_cost:.4f}",
            f"Tok In      {self.session_tokens_in:,}",
            f"Tok Out     {self.session_tokens_out:,}",
            "",
            "RESPONSE TIMES",
            sparkline,
        ]

        self.query_one("#session-display", Static).update("\n".join(lines))

    def create_sparkline(self) -> str:
        if not self.response_times:
            return ""

        bars = "▁▂▃▄▅▆▇█"
        times = list(self.response_times)
        max_val = max(times) if max(times) > 0 else 1.0

        sparkline = "".join(
            bars[min(len(bars) - 1, int((val / max_val) * len(bars)))] for val in times
        )

        return sparkline

    def update_status(self) -> None:
        status = f"▸ {self.status_text}"
        display = f"\nSTATUS\n{status}"

        widget = self.query_one("#status-display", Static)
        widget.update(display)

        if "LISTEN" in self.status_text:
            widget.add_class("highlight")
        else:
            widget.remove_class("highlight")

    def watch_status_text(self, value: str) -> None:
        self.update_status()

    def watch_last_stats(self, value: RequestStats | None) -> None:
        if value:
            self.response_times.append(value.latency_s)
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
        if value:
            waveform.add_class("active")
            waveform.remove_class("inactive")
        else:
            waveform.add_class("inactive")
            waveform.remove_class("active")


class ConversationPanel(Vertical):
    """Center panel showing conversation history."""

    def __init__(self) -> None:
        super().__init__(id="conversation-panel")

    def compose(self) -> ComposeResult:
        yield ScrollingLog(id="conversation-log")


class EventsPanel(Vertical):
    """Right panel showing event log."""

    def __init__(self) -> None:
        super().__init__(id="events-panel")

    def compose(self) -> ComposeResult:
        yield ScrollingLog(id="events-log")
        yield Static(id="shortcuts")

    def on_mount(self) -> None:
        shortcuts = (
            "SHORTCUTS\n─────────\n  q   Quit\n  ↑↓  Scroll\nHome  Top\n End  Bottom"
        )
        widget = self.query_one("#shortcuts", Static)
        widget.update(shortcuts)
        widget.add_class("muted")


class VilbertaTUI(App[None]):
    """TUI for Vilberta voice assistant with theme support."""

    CSS = """
    #system-panel {
        width: 35;
        padding: 1 2;
    }
    
    #conversation-panel {
        width: 1fr;
        padding: 1 2;
        margin: 0 1;
    }
    
    #events-panel {
        width: 32;
        padding: 1 2;
    }
    
    #waveform {
        height: 3;
        content-align: center middle;
        margin: 1 0;
        border: solid $primary;
        color: $primary;
    }
    
    #waveform.active {
        color: $primary;
        text-style: bold;
    }
    
    #waveform.inactive {
        color: $primary-darken-2;
    }
    
    #shortcuts {
        height: auto;
        margin-top: 1;
        border-top: solid $primary;
        padding-top: 1;
    }
    
    #system-header {
        text-style: bold;
        color: $primary;
    }
    
    .muted {
        color: $foreground-muted;
    }
    
    .highlight {
        color: $primary;
        text-style: bold;
    }
    
    .ai-voice {
        color: $warning;
    }
    
    .ai-voice.first {
        text-style: bold;
    }
    
    .ai-text {
        color: $secondary;
    }
    
    .ai-text.first {
        text-style: bold;
    }
    
    .user {
        color: $primary;
        text-style: bold;
    }
    
    .status {
        color: $foreground-muted;
    }
    
    .error {
        color: $error;
        text-style: bold;
    }
    
    .vad-active {
        color: $success;
    }
    
    .vad-inactive {
        color: $foreground-muted;
    }
    
    .event {
        color: $primary;
    }
    
    ScrollingLog {
        height: 1fr;
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
        conversation = self.query_one("#conversation-log", ScrollingLog)
        events_log = self.query_one("#events-log", ScrollingLog)
        system_panel = self.query_one("#system-panel", SystemPanel)

        if event.type == "speak":
            if self.in_ai_text_block:
                conversation.write("")
                self.in_ai_text_block = False

            if not self.in_ai_voice_block:
                conversation.write(
                    f"🤖 > {event.content}", "ai-voice first", parse_markdown=True
                )
                self.in_ai_voice_block = True
            else:
                conversation.write(
                    f"    │ {event.content}", "ai-voice", parse_markdown=True
                )

            events_log.write(f"SPEAK {event.content[:30]}", "ai-voice")

        elif event.type == "text":
            if self.in_ai_voice_block:
                conversation.write("")
                self.in_ai_voice_block = False

            lines = event.content.splitlines()
            for line in lines:
                if not self.in_ai_text_block:
                    conversation.write(
                        f"💬 > {line}", "ai-text first", parse_markdown=True
                    )
                    self.in_ai_text_block = True
                else:
                    conversation.write(f"    │ {line}", "ai-text", parse_markdown=True)

            events_log.write(f"TEXT  {event.content[:30]}", "ai-text")

        elif event.type == "transcript":
            self.end_ai_blocks(conversation)
            conversation.write("")
            conversation.write(f"👤 > {event.content}", "user")
            conversation.write("")

            system_panel.exchange_count += 1
            events_log.write(f"USER  {event.content[:30]}", "user")

        elif event.type == "status":
            msg = event.content.strip()
            if msg.startswith("[") and msg.endswith("]"):
                conversation.write(f"       {msg}", "status")

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

            events_log.write(f"STAT  {msg[:30]}", "status")

        elif event.type == "error":
            conversation.write(f"❌ > {event.content}", "error")
            events_log.write(f"ERROR {event.content[:30]}", "error")

        elif event.type == "vad":
            system_panel.vad_active = event.content == "up"
            status = "▲ speech" if event.content == "up" else "▼ silence"
            style = "vad-active" if event.content == "up" else "vad-inactive"
            events_log.write(f"VAD   {status}", style)

        elif event.type == "stats":
            if event.stats:
                system_panel.last_stats = event.stats
                system_panel.session_cost += event.stats.cost_usd
                system_panel.session_tokens_in += event.stats.input_tokens
                system_panel.session_tokens_out += event.stats.output_tokens

                events_log.write(
                    f"STATS latency={event.stats.latency_s:.2f}s "
                    f"in={event.stats.input_tokens} out={event.stats.output_tokens}",
                    "event",
                )

        elif event.type == "boot":
            events_log.write(f"BOOT  {event.content.strip()[:30]}", "event")

    def end_ai_blocks(self, conversation: ScrollingLog) -> None:
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


class CursesTUI:
    """Wrapper class for backward compatibility."""

    def __init__(self) -> None:
        self.app = VilbertaTUI()

    def run(
        self, event_queue: Queue[DisplayEvent], shutdown_event: threading.Event
    ) -> None:
        """Run the TUI with the given event queue and shutdown event."""
        self.app.setup(event_queue, shutdown_event)
        self.app.run()

    def cleanup(self) -> None:
        """Cleanup method for compatibility."""
        pass


# tui.py
