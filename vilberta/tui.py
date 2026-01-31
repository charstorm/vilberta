"""Curses-based split-panel TUI for vilberta."""

from __future__ import annotations

import curses
import locale
import textwrap
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty

from vilberta.config import MODEL_NAME, TTS_VOICE, SAMPLE_RATE

# Ensure the locale supports UTF-8 box-drawing characters.
locale.setlocale(locale.LC_ALL, "")


@dataclass
class DisplayEvent:
    type: str  # "speak", "text", "transcript", "status", "error", "vad", "boot"
    content: str


# ── Color palette ────────────────────────────────────────────────────────────
# We define two palettes: one for 256-color terminals and a fallback for basic
# 16-color terminals.  Each palette maps a logical slot to (fg, bg).

_SLOT_BORDER = 1
_SLOT_SPEECH = 2
_SLOT_AI_TEXT = 3
_SLOT_ERROR = 4
_SLOT_USER = 5
_SLOT_GREEN = 6
_SLOT_ACCENT = 7
_SLOT_STATUSBAR = 8
_SLOT_YELLOW = 9
_SLOT_DIM = 10
_SLOT_AI_PREFIX = 11
_SLOT_USER_PREFIX = 12
_SLOT_SEPARATOR = 13

_PALETTE_256: dict[int, tuple[int, int]] = {
    _SLOT_BORDER:      (243, -1),
    _SLOT_SPEECH:      (255, -1),
    _SLOT_AI_TEXT:     (253, -1),
    _SLOT_ERROR:       (167, -1),
    _SLOT_USER:        (249, -1),
    _SLOT_GREEN:       (108, -1),
    _SLOT_ACCENT:      (110, -1),
    _SLOT_STATUSBAR:   (249, 236),
    _SLOT_YELLOW:      (179, -1),
    _SLOT_DIM:         (243, -1),
    _SLOT_AI_PREFIX:   (110, -1),
    _SLOT_USER_PREFIX: (179, -1),
    _SLOT_SEPARATOR:   (237, -1),
}

_PALETTE_16: dict[int, tuple[int, int]] = {
    _SLOT_BORDER:      (curses.COLOR_WHITE,   -1),
    _SLOT_SPEECH:      (curses.COLOR_WHITE,   -1),
    _SLOT_AI_TEXT:     (curses.COLOR_WHITE,   -1),
    _SLOT_ERROR:       (curses.COLOR_RED,     -1),
    _SLOT_USER:        (curses.COLOR_WHITE,   -1),
    _SLOT_GREEN:       (curses.COLOR_GREEN,   -1),
    _SLOT_ACCENT:      (curses.COLOR_CYAN,    -1),
    _SLOT_STATUSBAR:   (curses.COLOR_WHITE,   curses.COLOR_BLACK),
    _SLOT_YELLOW:      (curses.COLOR_YELLOW,  -1),
    _SLOT_DIM:         (curses.COLOR_WHITE,   -1),
    _SLOT_AI_PREFIX:   (curses.COLOR_CYAN,    -1),
    _SLOT_USER_PREFIX: (curses.COLOR_YELLOW,  -1),
    _SLOT_SEPARATOR:   (curses.COLOR_BLACK,   -1),
}


def _init_palette() -> None:
    palette = _PALETTE_256 if curses.COLORS >= 256 else _PALETTE_16
    for slot, (fg, bg) in palette.items():
        curses.init_pair(slot, fg, bg)


def _cp(slot: int) -> int:
    return curses.color_pair(slot)


class CursesTUI:
    def __init__(self) -> None:
        self._stdscr: curses.window | None = None
        self._lock = threading.Lock()

        # Left panel — current speech
        self._speaking_text = ""
        self._speaking_active = False

        # Right panel — conversation log
        # Each entry: (color_pair_slot, prefix_slot, prefix, text)
        self._right_lines: list[tuple[int, int, str, str]] = []

        # Status
        self._status_text = "INITIALIZING"
        self._vad_active = False
        self._exchange_count = 0

        # Track whether we're inside an AI text block
        self._in_ai_block = False

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
            _init_palette()

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
            scr.addnstr(row, col, text, w - col, attr)
        except curses.error:
            pass

    def _hline(self, row: int, col: int, length: int, attr: int) -> None:
        scr = self._stdscr
        assert scr is not None
        h, w = self._dims()
        if row < 0 or row >= h:
            return
        n = min(length, w - col)
        if n <= 0:
            return
        try:
            scr.hline(row, col, curses.ACS_HLINE | attr, n)
        except curses.error:
            pass

    def _vline(self, col: int, row_start: int, row_end: int, attr: int) -> None:
        scr = self._stdscr
        assert scr is not None
        h, w = self._dims()
        if col < 0 or col >= w:
            return
        for r in range(max(0, row_start), min(row_end, h)):
            try:
                scr.addch(r, col, curses.ACS_VLINE, attr)
            except curses.error:
                pass

    def _addch(self, row: int, col: int, ch: int, attr: int = 0) -> None:
        scr = self._stdscr
        assert scr is not None
        h, w = self._dims()
        if row < 0 or row >= h or col < 0 or col >= w:
            return
        try:
            scr.addch(row, col, ch, attr)
        except curses.error:
            pass

    # ── Layout ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        scr = self._stdscr
        assert scr is not None
        h, w = self._dims()
        if h < 6 or w < 40:
            return

        scr.erase()

        left_w = max(w * 2 // 5, 26)
        if left_w > w - 22:
            left_w = w // 2

        ba = _cp(_SLOT_BORDER)  # border attr

        # ── Outer border (ACS line-drawing — works on any terminal) ──
        self._addch(0, 0, curses.ACS_ULCORNER, ba)
        self._addch(0, w - 1, curses.ACS_URCORNER, ba)
        self._addch(h - 1, 0, curses.ACS_LLCORNER, ba)
        self._addch(h - 1, w - 1, curses.ACS_LRCORNER, ba)
        self._hline(0, 1, w - 2, ba)
        self._hline(h - 1, 1, w - 2, ba)
        self._vline(0, 1, h - 1, ba)
        self._vline(w - 1, 1, h - 1, ba)

        # ── Title in top border ──
        self._put(0, 3, " vilberta ", _cp(_SLOT_ACCENT) | curses.A_BOLD)

        # ── Vertical divider ──
        div = left_w
        self._addch(0, div, curses.ACS_TTEE, ba)
        self._vline(div, 1, h - 3, ba)

        # ── Status separator ──
        ss = h - 3
        self._addch(ss, 0, curses.ACS_LTEE, ba)
        self._hline(ss, 1, div - 1, ba)
        self._addch(ss, div, curses.ACS_BTEE, ba)
        self._hline(ss, div + 1, w - div - 2, ba)
        self._addch(ss, w - 1, curses.ACS_RTEE, ba)

        # ── Shared header separator at row 3 ──
        self._addch(3, 0, curses.ACS_LTEE, ba)
        self._hline(3, 1, div - 1, ba)
        self._addch(3, div, curses.ACS_PLUS, ba)
        self._hline(3, div + 1, w - div - 2, ba)
        self._addch(3, w - 1, curses.ACS_RTEE, ba)

        self._draw_left(left_w, h)
        self._draw_right(left_w, h, w)
        self._draw_status(h, w)
        scr.refresh()

    def _draw_left(self, left_w: int, h: int) -> None:
        x0 = 2
        cw = left_w - x0 - 1
        if cw < 10:
            return

        body_end = h - 4

        # ── Header: model + voice (2 rows) ──
        self._put(1, x0, MODEL_NAME, _cp(_SLOT_ACCENT))
        self._put(2, x0, f"{TTS_VOICE}  {SAMPLE_RATE // 1000}kHz", _cp(_SLOT_DIM))

        # Separator at row 3 drawn by _draw()

        # ── Speaking state ──
        row = 4
        if self._speaking_active:
            self._put(row, x0, "SPEAKING", _cp(_SLOT_YELLOW) | curses.A_BOLD)
        else:
            self._put(row, x0, "idle", _cp(_SLOT_DIM))
        row += 2  # blank line before speech text

        available = body_end - row + 1
        if available < 1:
            return

        with self._lock:
            text = self._speaking_text

        wrapped = textwrap.wrap(text, width=cw) if text else []

        attr = _cp(_SLOT_SPEECH) | curses.A_BOLD if self._speaking_active else _cp(_SLOT_DIM)
        for i in range(min(available, len(wrapped))):
            self._put(row + i, x0, wrapped[i], attr)

    def _draw_right(self, left_w: int, h: int, w: int) -> None:
        x0 = left_w + 2
        cw = w - x0 - 1
        if cw < 10:
            return

        # Header (2 rows to match left panel)
        self._put(1, x0, "CONVERSATION", _cp(_SLOT_ACCENT) | curses.A_BOLD)
        count_str = f"#{self._exchange_count}"
        self._put(2, x0, count_str, _cp(_SLOT_DIM))

        # Separator at row 3 drawn by _draw()

        body_top = 4
        body_end = h - 4
        conv_height = body_end - body_top + 1
        if conv_height < 1:
            return

        # Build display lines: (text_slot, prefix, prefix_slot, text)
        display_lines: list[tuple[int, str, int, str]] = []
        with self._lock:
            entries = list(self._right_lines)

        for text_slot, prefix_slot, prefix, text in entries:
            plen = len(prefix)
            text_w = cw - plen
            if text_w < 5:
                plen = 0
                prefix = ""
                text_w = cw

            lines = textwrap.wrap(text, width=text_w) or [""]
            # First line: colored prefix + text
            display_lines.append((text_slot, prefix, prefix_slot, lines[0]))
            # Continuation: indent with gutter
            cont_prefix = " " * (plen - 2) + "| " if plen >= 2 else " " * plen
            for ln in lines[1:]:
                display_lines.append((text_slot, cont_prefix, _SLOT_BORDER, ln))

        # Empty state
        if not display_lines:
            msg = "Waiting for conversation..."
            self._put(body_top + conv_height // 2, x0 + (cw - len(msg)) // 2, msg, _cp(_SLOT_DIM))
            return

        total = len(display_lines)
        start = max(0, total - conv_height)
        visible = display_lines[start : start + conv_height]

        for i, (text_slot, prefix, prefix_slot, txt) in enumerate(visible):
            r = body_top + i
            if prefix:
                self._put(r, x0, prefix, _cp(prefix_slot) if prefix_slot else _cp(text_slot))
            self._put(r, x0 + len(prefix), txt, _cp(text_slot))

        # Scrollbar
        if total > conv_height:
            sb_top = body_top + (start * conv_height) // total
            sb_len = max(1, (conv_height * conv_height) // total)
            sb_col = w - 1
            for r in range(body_top, body_top + conv_height):
                if sb_top <= r < sb_top + sb_len:
                    self._addch(r, sb_col, curses.ACS_BLOCK, _cp(_SLOT_DIM))

    def _draw_status(self, h: int, w: int) -> None:
        row = h - 2
        sba = _cp(_SLOT_STATUSBAR)

        # Fill bar
        self._put(row, 1, " " * (w - 2), sba)

        # VAD
        if self._vad_active:
            self._put(row, 2, " MIC ", _cp(_SLOT_GREEN) | curses.A_BOLD)
        else:
            self._put(row, 2, " MIC ", sba)

        # Status
        self._put(row, 8, self._status_text, sba | curses.A_BOLD)

        # Right side: quit hint + exchanges
        right = f" q:quit  {self._exchange_count} exchanges "
        self._put(row, w - len(right) - 1, right, sba)

    # ── Events ────────────────────────────────────────────────────────────────

    def _add_right(self, text_slot: int, prefix_slot: int, prefix: str, text: str) -> None:
        with self._lock:
            self._right_lines.append((text_slot, prefix_slot, prefix, text))

    def _add_separator(self) -> None:
        with self._lock:
            self._right_lines.append((_SLOT_SEPARATOR, 0, "", ""))

    def _handle_event(self, event: DisplayEvent) -> None:
        if event.type == "speak":
            with self._lock:
                self._speaking_text = event.content
                self._speaking_active = True
        elif event.type == "text":
            lines = event.content.splitlines()
            for line in lines:
                if not self._in_ai_block:
                    self._add_right(_SLOT_AI_TEXT, _SLOT_AI_PREFIX, "  ai > ", line)
                    self._in_ai_block = True
                else:
                    self._add_right(_SLOT_AI_TEXT, _SLOT_BORDER, "     | ", line)
        elif event.type == "transcript":
            self._in_ai_block = False
            self._add_separator()
            self._add_right(_SLOT_USER, _SLOT_USER_PREFIX, " you > ", event.content)
            self._exchange_count += 1
        elif event.type == "status":
            msg = event.content.strip()
            if msg.startswith("[") and msg.endswith("]"):
                self._add_right(_SLOT_DIM, 0, "       ", msg)
            if "Listening..." in event.content or "Processing..." in event.content:
                self._in_ai_block = False
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
            self._add_right(_SLOT_ERROR, _SLOT_ERROR, " err > ", event.content)
        elif event.type == "vad":
            self._vad_active = event.content == "up"
        elif event.type == "boot":
            self._add_right(_SLOT_ACCENT, _SLOT_DIM, "     : ", event.content)

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
