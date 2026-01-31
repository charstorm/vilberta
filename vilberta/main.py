import os
import sys
import signal
import threading
import time
from queue import Queue

from openai import OpenAI, AuthenticationError

from vilberta.audio_capture import record_speech, audio_to_base64_wav
from vilberta.config import API_BASE_URL, API_KEY_ENV, MODEL_NAME, SAMPLE_RATE
from vilberta.interrupt_monitor import InterruptMonitor
from vilberta.llm_service import LLMService
from vilberta.response_parser import SectionType
from vilberta.tts_engine import TTSEngine
from vilberta.display import (
    init_display,
    print_speak,
    print_text,
    print_transcript,
    print_status,
    print_error,
    print_stats,
)
from vilberta.sound_effects import (
    play_response_send,
    play_response_received,
    play_text_start,
    play_response_end,
    _SOUNDS_DIR,
)
from vilberta.tui import CursesTUI, DisplayEvent, RequestStats


_EXPECTED_SOUNDS = [
    "response_send.wav",
    "response_received.wav",
    "text_start.wav",
    "text_end.wav",
    "line_print.wav",
    "response_end.wav",
]


# ── Preflight (runs BEFORE curses, uses plain print via fallback) ────────────


def _run_preflight_checks() -> None:
    print("Running preflight checks...")

    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(f"ERROR: Environment variable {API_KEY_ENV} is not set.")
        sys.exit(1)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
        client.models.list()
        print("  ✓ API key valid")
    except AuthenticationError:
        print(f"ERROR: API key in {API_KEY_ENV} is invalid.")
        sys.exit(1)
    except Exception as e:
        print(f"  ⚠ Could not verify API key ({e}), continuing anyway")

    missing = [
        f for f in _EXPECTED_SOUNDS if not os.path.isfile(os.path.join(_SOUNDS_DIR, f))
    ]
    if missing:
        print("  Generating missing sound files...")
        from vilberta.generate_sounds import main as generate_sounds

        generate_sounds()
        print("  ✓ Sound files generated")
    else:
        print("  ✓ Sound files OK")

    print("Preflight checks passed.\n")


# ── Boot sequence (runs inside curses via queue) ─────────────────────────────

_BOOT_LINES = [
    ("[ OK ] Neural link", "ONLINE"),
    ("[ OK ] Voice matrix", "ONLINE"),
    ("[ OK ] Audio subsystem", "ONLINE"),
    (f"[ OK ] Model: {MODEL_NAME}", "LINKED"),
    ("[ OK ] System", "READY"),
]


def _play_boot_sequence(queue: Queue[DisplayEvent]) -> None:
    for label, status in _BOOT_LINES:
        dots = "." * (42 - len(label) - len(status))
        queue.put(DisplayEvent(type="boot", content=f"  {label} {dots} {status}"))
        time.sleep(0.25)
    queue.put(DisplayEvent(type="boot", content=""))
    queue.put(DisplayEvent(type="boot", content="  ═══ All systems nominal ═══"))
    queue.put(DisplayEvent(type="boot", content=""))
    time.sleep(0.5)


# ── Core voice loop (unchanged logic) ────────────────────────────────────────


def _speak_with_monitor(tts: TTSEngine, monitor: InterruptMonitor, text: str) -> bool:
    """Speak text while monitoring for interrupts. Returns True if completed."""
    stop_poll = threading.Event()

    def _poll() -> None:
        while not stop_poll.is_set():
            if monitor.triggered:
                tts.interrupt()
                return
            time.sleep(0.02)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    completed = tts.speak(text)
    stop_poll.set()
    poll_thread.join()
    return completed


def _process_response(
    llm: LLMService, tts: TTSEngine, monitor: InterruptMonitor,
    audio_b64: str, audio_duration_s: float,
) -> None:
    interrupted = False
    monitor.start()
    first_section = True

    play_response_send()
    t0 = time.monotonic()

    try:
        for section in llm.stream_response(audio_b64):
            if first_section:
                play_response_received()
                first_section = False

            if section.type == SectionType.SPEAK:
                print_speak(section.content)
                completed = _speak_with_monitor(tts, monitor, section.content)
                if not completed or monitor.triggered:
                    interrupted = True
                    break
            elif section.type == SectionType.TEXT:
                play_text_start()
                time.sleep(0.3)
                print_text(section.content)
                time.sleep(0.8)
            elif section.type == SectionType.TRANSCRIPT:
                print_transcript(section.content)
    finally:
        monitor.stop()

    total_latency = time.monotonic() - t0

    # Emit stats
    stats = RequestStats(
        audio_duration_s=audio_duration_s,
        input_tokens=llm.last_input_tokens,
        output_tokens=llm.last_output_tokens,
        cache_read_tokens=llm.last_cache_read_tokens,
        cache_write_tokens=llm.last_cache_write_tokens,
        ttft_s=llm.last_ttft,
        total_latency_s=total_latency,
    )
    print_stats(stats)

    if interrupted:
        llm.mark_interrupted()
        print_status("[interrupted]")
    else:
        play_response_end()


def _worker(queue: Queue[DisplayEvent], shutdown_event: threading.Event) -> None:
    """Worker thread: init services, run voice loop."""
    _play_boot_sequence(queue)

    print_status("Loading TTS model...")
    tts = TTSEngine()
    print_status("TTS ready.")

    print_status("Initializing LLM service...")
    llm = LLMService()
    print_status("Ready. Listening...")

    monitor = InterruptMonitor()

    while not shutdown_event.is_set():
        audio_data = record_speech()
        if audio_data is None:
            continue

        print_status("Processing...")
        audio_dur = len(audio_data) / SAMPLE_RATE
        audio_b64 = audio_to_base64_wav(audio_data)

        try:
            _process_response(llm, tts, monitor, audio_b64, audio_dur)
        except Exception as e:
            print_error(f"LLM error: {e}")

        prefix = monitor.buffered_audio if monitor.triggered else None
        if prefix is not None:
            print_status("Continuing recording...")
            audio_data = record_speech(prefix_audio=prefix)
            if audio_data is not None:
                print_status("Processing...")
                audio_dur = len(audio_data) / SAMPLE_RATE
                audio_b64 = audio_to_base64_wav(audio_data)
                try:
                    _process_response(llm, tts, monitor, audio_b64, audio_dur)
                except Exception as e:
                    print_error(f"LLM error: {e}")

        print_status("Listening...")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    _run_preflight_checks()

    event_queue: Queue[DisplayEvent] = Queue()
    init_display(event_queue)

    shutdown_event = threading.Event()

    def _sighandler(sig: int, frame: object) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    tui = CursesTUI()

    worker_thread = threading.Thread(
        target=_worker, args=(event_queue, shutdown_event), daemon=True
    )
    worker_thread.start()

    # TUI runs on main thread (curses requirement)
    tui.run(event_queue, shutdown_event)


if __name__ == "__main__":
    main()
