"""Curses-based split-panel TUI for vilberta."""

from __future__ import annotations

import curses
import textwrap
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty

from vilberta.config import MODEL_NAME, TTS_VOICE, SAMPLE_RATE


@dataclass
class DisplayEvent:
    type: str  # "speak", "text", "transcript", "status", "error", "vad", "boot"
    content: str


class CursesTUI:
    def __init__(self) -> None:
        self._stdscr: curses.window | None = None
        self._lock = threading.Lock()

        # Left panel — current speech
        self._speaking_text = ""
        self._speaking_active = False

        # Right panel — conversation log
        self._right_lines: list[tuple[int, str]] = []

        # Status
        self._status_text = "INITIALIZING"
        self._vad_active = False
        self._exchange_count = 0

    def _init_curses(self) -> curses.window:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.nodelay(True)

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            # 1: dim — borders, labels
            curses.init_pair(1, 8, -1)
            # 2: bold white — speech text (left panel)
            curses.init_pair(2, curses.COLOR_WHITE, -1)
            # 3: default — text responses (right panel)
            curses.init_pair(3, -1, -1)
            # 4: red — errors
            curses.init_pair(4, curses.COLOR_RED, -1)
            # 5: dim — user transcripts
            curses.init_pair(5, 8, -1)
            # 6: green — active VAD / status highlights
            curses.init_pair(6, curses.COLOR_GREEN, -1)
            # 7: cyan — section labels
            curses.init_pair(7, curses.COLOR_CYAN, -1)

        self._stdscr = stdscr
        return stdscr

    def _dims(self) -> tuple[int, int]:
        assert self._stdscr is not None
        return self._stdscr.getmaxyx()

    def _put(self, row: int, col: int, text: str, attr: int = 0) -> None:
        scr = self._stdscr
        assert scr is not None
        h, w = self._dims()
        if row < 0 or row >= h or col >= w:
            return
        try:
            scr.addstr(row, col, text[: w - col], attr)
        except curses.error:
            pass

    # ── Layout ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        scr = self._stdscr
        assert scr is not None
        h, w = self._dims()
        if h < 4 or w < 40:
            return

        scr.erase()

        left_w = max(w * 2 // 5, 24)
        if left_w > w - 20:
            left_w = w // 2

        # Divider column
        div = left_w
        dim = curses.color_pair(1)

        # Vertical divider
        for row in range(0, h - 1):
            self._put(row, div, "│", dim)

        # Horizontal separator above status bar
        sep_row = h - 2
        self._put(sep_row, 0, "─" * div, dim)
        self._put(sep_row, div, "┴", dim)
        self._put(sep_row, div + 1, "─" * (w - div - 1), dim)

        self._draw_left(left_w, h, dim)
        self._draw_right(left_w, h, w, dim)
        self._draw_status(h, w, dim)
        scr.refresh()

    def _draw_left(self, left_w: int, h: int, dim: int) -> None:
        """Left panel: model info at top, current speech below."""
        cw = left_w - 3  # content width (1 padding + content + 1 gap before divider)
        if cw < 10:
            return

        body_end = h - 3  # last usable row before separator

        # ── Header ──
        self._put(0, 1, "vilberta", curses.color_pair(2) | curses.A_BOLD)
        self._put(1, 1, MODEL_NAME, dim)
        self._put(2, 1, f"voice: {TTS_VOICE}  rate: {SAMPLE_RATE // 1000}kHz", dim)

        # Thin separator
        self._put(3, 1, "─" * cw, dim)

        # ── Speaking section ──
        label_row = 4
        if self._speaking_active:
            self._put(label_row, 1, "speaking", curses.color_pair(6) | curses.A_BOLD)
        else:
            self._put(label_row, 1, "idle", dim)

        # Speech text
        content_top = label_row + 1
        available = body_end - content_top + 1
        if available < 1:
            return

        with self._lock:
            text = self._speaking_text

        if text:
            wrapped = textwrap.wrap(text, width=cw) or [""]
        else:
            wrapped = []

        pair = curses.color_pair(2) if self._speaking_active else dim
        for i in range(available):
            row = content_top + i
            if i < len(wrapped):
                self._put(row, 1, wrapped[i].ljust(cw)[:cw], pair)
            else:
                self._put(row, 1, " " * cw, 0)

    def _draw_right(self, left_w: int, h: int, w: int, dim: int) -> None:
        """Right panel: scrolling conversation."""
        start_col = left_w + 2
        cw = w - start_col - 1
        if cw < 10:
            return

        body_top = 0
        body_end = h - 3
        conv_height = body_end - body_top + 1

        # Wrap lines
        display_lines: list[tuple[int, str]] = []
        with self._lock:
            for color_pair, text in self._right_lines:
                wrapped = textwrap.wrap(text, width=cw) or [""]
                for wl in wrapped:
                    display_lines.append((color_pair, wl))

        total = len(display_lines)
        start = max(0, total - conv_height)
        visible = display_lines[start : start + conv_height]

        for i in range(conv_height):
            row = body_top + i
            if i < len(visible):
                cp, txt = visible[i]
                self._put(row, start_col, txt.ljust(cw)[:cw], curses.color_pair(cp))
            else:
                self._put(row, start_col, " " * cw, 0)

    def _draw_status(self, h: int, w: int, dim: int) -> None:
        """Bottom status line."""
        row = h - 1
        dot = "●" if self._vad_active else "·"
        dot_attr = curses.color_pair(6) | curses.A_BOLD if self._vad_active else dim

        self._put(row, 1, dot, dot_attr)
        self._put(row, 3, self._status_text, dim | curses.A_BOLD)

        info = f"{self._exchange_count} exchanges"
        self._put(row, w - len(info) - 1, info, dim)

    # ── Events ────────────────────────────────────────────────────────────────

    def _add_right(self, pair: int, text: str) -> None:
        with self._lock:
            self._right_lines.append((pair, text))

    def _handle_event(self, event: DisplayEvent) -> None:
        if event.type == "speak":
            with self._lock:
                self._speaking_text = event.content
                self._speaking_active = True
        elif event.type == "text":
            for line in event.content.splitlines():
                self._add_right(3, f"  {line}")
        elif event.type == "transcript":
            self._add_right(5, f"  you: {event.content}")
            self._exchange_count += 1
        elif event.type == "status":
            msg = event.content.strip()
            if msg.startswith("[") and msg.endswith("]"):
                self._add_right(5, f"  {msg}")
            if "Listening..." in event.content or "Processing..." in event.content:
                with self._lock:
                    self._speaking_active = False
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
                    self._status_text = val
                    break
        elif event.type == "error":
            self._add_right(4, f"  error: {event.content}")
        elif event.type == "vad":
            self._vad_active = event.content == "up"
        elif event.type == "boot":
            self._add_right(7, event.content)

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(
        self, event_queue: Queue[DisplayEvent], shutdown_event: threading.Event
    ) -> None:
        scr = self._init_curses()
        try:
            self._draw()
            while not shutdown_event.is_set():
                had_events = False
                try:
                    while True:
                        event = event_queue.get_nowait()
                        self._handle_event(event)
                        had_events = True
                except Empty:
                    pass

                try:
                    key = scr.getch()
                    if key == curses.KEY_RESIZE:
                        had_events = True
                    elif key == ord("q"):
                        shutdown_event.set()
                        break
                except curses.error:
                    pass

                if had_events:
                    self._draw()

                time.sleep(0.05)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._stdscr is not None:
            curses.nocbreak()
            self._stdscr.keypad(False)
            curses.echo()
            curses.endwin()
            self._stdscr = None
