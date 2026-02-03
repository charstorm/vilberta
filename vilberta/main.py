import os
import sys
import signal
import threading
import time
import argparse
from queue import Queue

from openai import OpenAI, AuthenticationError

from vilberta.audio_capture import record_speech, audio_to_base64_wav
from vilberta.config import init_config, get_config
from vilberta.interrupt_monitor import InterruptMonitor
from vilberta.llm_service import BaseLLMService, BasicLLMService
from vilberta.mcp_llm_service import MCPAwareLLMService
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
    DisplayEvent,
    RequestStats,
)
from vilberta.sound_effects import (
    play_response_send,
    play_response_received,
    play_text_start,
    play_response_end,
    play_ready,
    _SOUNDS_DIR,
)
from vilberta.cli import SimpleCLI
from vilberta.tui import CursesTUI


_EXPECTED_SOUNDS = [
    "response_send.wav",
    "response_received.wav",
    "text_start.wav",
    "text_end.wav",
    "line_print.wav",
    "response_end.wav",
    "ready.wav",
]


# ── Preflight (runs BEFORE UI, uses plain print via fallback) ────────────────


def _run_preflight_checks() -> None:
    print("Running preflight checks...")

    cfg = get_config()
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        print(f"ERROR: Environment variable {cfg.api_key_env} is not set.")
        sys.exit(1)

    try:
        client = OpenAI(base_url=cfg.api_base_url, api_key=api_key)
        client.models.list()
        print("  ✓ API key valid")
    except AuthenticationError:
        print(f"ERROR: API key in {cfg.api_key_env} is invalid.")
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


# ── Boot sequence (runs inside UI via queue) ─────────────────────────────────


def _get_boot_lines() -> list[tuple[str, str]]:
    cfg = get_config()
    return [
        ("[ OK ] Neural link", "ONLINE"),
        ("[ OK ] Voice matrix", "ONLINE"),
        ("[ OK ] Audio subsystem", "ONLINE"),
        (f"[ OK ] Model: {cfg.model_name}", "LINKED"),
        ("[ OK ] System", "READY"),
    ]


def _play_boot_sequence(queue: Queue[DisplayEvent]) -> None:
    for label, status in _get_boot_lines():
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


def _create_llm_service() -> BaseLLMService:
    """Create appropriate LLM service based on config mode."""
    cfg = get_config()

    if cfg.mode == "mcp":
        if not cfg.mcp_server_url:
            print_error("MCP mode selected but no server URL configured")
            sys.exit(1)
        return MCPAwareLLMService()

    return BasicLLMService()


def _process_response(
    llm: BaseLLMService,
    tts: TTSEngine,
    monitor: InterruptMonitor,
    audio_b64: str,
    audio_duration_s: float,
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
    llm = _create_llm_service()

    # Connect to MCP server if in MCP mode
    cfg = get_config()
    if cfg.mode == "mcp" and isinstance(llm, MCPAwareLLMService):
        print_status("Connecting to MCP server...")
        try:
            llm.connect()
        except Exception as e:
            print_error(f"Failed to connect to MCP server: {e}")
            sys.exit(1)

    print_status("Ready. Listening...")
    play_ready()

    monitor = InterruptMonitor()
    cfg = get_config()

    try:
        while not shutdown_event.is_set():
            audio_data = record_speech()
            if audio_data is None:
                continue

            print_status("Processing...")
            audio_dur = len(audio_data) / cfg.sample_rate
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
                    audio_dur = len(audio_data) / cfg.sample_rate
                    audio_b64 = audio_to_base64_wav(audio_data)
                    try:
                        _process_response(llm, tts, monitor, audio_b64, audio_dur)
                    except Exception as e:
                        print_error(f"LLM error: {e}")

            print_status("Listening...")
    finally:
        # Cleanup MCP connection if applicable
        if isinstance(llm, MCPAwareLLMService):
            print_status("Disconnecting from MCP server...")
            llm.cleanup()


# ── Entry point ──────────────────────────────────────────────────────────────


def _get_ui(interface: str) -> SimpleCLI | CursesTUI:
    """Get the appropriate UI based on user preference."""
    if interface == "tui":
        return CursesTUI()
    return SimpleCLI()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vilberta - Voice Intelligence System")
    parser.add_argument(
        "-i",
        "--interface",
        choices=["cli", "tui"],
        default="cli",
        help="Interface to use: cli (default) or tui",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config.toml file",
    )
    args = parser.parse_args()

    init_config(args.config)
    _run_preflight_checks()

    event_queue: Queue[DisplayEvent] = Queue()
    init_display(event_queue)

    shutdown_event = threading.Event()

    def _sighandler(sig: int, frame: object) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    ui = _get_ui(args.interface)

    worker_thread = threading.Thread(
        target=_worker, args=(event_queue, shutdown_event), daemon=True
    )
    worker_thread.start()

    # UI runs on main thread
    ui.run(event_queue, shutdown_event)


if __name__ == "__main__":
    main()


# main.py
