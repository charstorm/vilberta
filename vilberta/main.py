import os
import sys
import signal
import threading
import time
import argparse
from queue import Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from openai import OpenAI, AuthenticationError

from vilberta.asr_service import ASRService
from vilberta.audio_capture import record_speech, audio_to_base64_wav
from vilberta.config import init_config, get_config, SectionType, Config
from vilberta.interrupt_monitor import InterruptMonitor
from vilberta.llm_service import BaseLLMService, BasicLLMService
from vilberta.mcp_llm_service import MCPAwareLLMService
from vilberta.tts_engine import TTSEngine
from vilberta.display import (
    init_display,
    print_speak,
    print_text,
    print_transcript,
    print_status,
    print_error,
    print_subsystem_ready,
    DisplayEvent,
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
from vilberta.logger import init_logger, get_logger


_EXPECTED_SOUNDS = [
    "response_send.wav",
    "response_received.wav",
    "text_start.wav",
    "text_end.wav",
    "line_print.wav",
    "response_end.wav",
    "ready.wav",
    "tool_call_start.wav",
    "tool_call_success.wav",
    "tool_call_error.wav",
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
    model_name = (
        cfg.basic_chat_llm_model_name
        if cfg.mode == "basic"
        else cfg.toolcall_chat_llm_model_name
    )
    return [
        ("[ OK ] Neural link", "ONLINE"),
        ("[ OK ] Voice matrix", "ONLINE"),
        ("[ OK ] Audio subsystem", "ONLINE"),
        (f"[ OK ] Model: {model_name}", "LINKED"),
        ("[ OK ] System", "READY"),
    ]


def _play_boot_sequence(queue: Queue[DisplayEvent]) -> None:
    for label, status in _get_boot_lines():
        dots = "." * (42 - len(label) - len(status))
        queue.put(DisplayEvent(type="boot", content=f"  {label} {dots} {status}"))
        time.sleep(0.25)
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
    transcript: str,
) -> bool:
    """Process transcript through LLM and TTS. Returns True if completed without interruption."""
    interrupted = False
    monitor.start()

    play_response_send()

    # Step 2: LLM - Process transcript
    print_status("Processing...")
    llm_start = time.monotonic()
    try:
        sections, _full_response = llm.get_response(transcript)
        llm_time = time.monotonic() - llm_start
        print_status(
            f"LLM: {llm_time:.2f}s, "
            f"tokens: {llm.last_input_tokens}/{llm.last_output_tokens}"
        )
    except Exception as e:
        monitor.stop()
        raise e

    play_response_received()

    try:
        for section in sections:
            if section.type == SectionType.SPEAK:
                print_speak(section.content)
                tts_start = time.monotonic()
                completed = _speak_with_monitor(tts, monitor, section.content)
                tts_time = time.monotonic() - tts_start
                print_status(f"TTS line: {tts_time:.2f}s")
                if not completed or monitor.triggered:
                    interrupted = True
                    break
            elif section.type == SectionType.TEXT:
                play_text_start()
                time.sleep(0.3)
                print_text(section.content)
                time.sleep(0.8)
    finally:
        monitor.stop()

    if interrupted:
        llm.mark_interrupted()
        print_status("[interrupted]")
    else:
        play_response_end()

    return not interrupted


def _process_turn(
    llm: BaseLLMService,
    tts: TTSEngine,
    monitor: InterruptMonitor,
    audio_data: NDArray[np.int16],
    asr: ASRService,
    cfg: Config,
    logger: Any,
) -> None:
    """Process a single turn: ASR -> LLM -> TTS."""
    # Prepare audio data
    audio_dur = len(audio_data) / cfg.sample_rate
    audio_b64 = audio_to_base64_wav(audio_data)

    # Step 1: ASR - Transcribe audio with context from conversation history
    print_status("Transcribing...")
    try:
        context_words = llm.get_unique_words(max_words=100)
        transcript, asr_stats = asr.transcribe(audio_b64, audio_dur, context_words)
    except Exception as e:
        logger.error(f"ASR error: {e}")
        print_error(f"ASR error: {e}")
        return

    # Display transcript immediately
    print_transcript(transcript)
    print_status(
        f"ASR: {asr_stats.processing_time_s:.2f}s, "
        f"tokens: {asr_stats.input_tokens}/{asr_stats.output_tokens}"
    )

    # Step 2 & 3: LLM -> TTS
    try:
        _process_response(llm, tts, monitor, transcript)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        print_error(f"LLM error: {e}")


class _SubsystemTracker:
    """Track subsystem initialization status."""

    def __init__(self) -> None:
        self._ready: set[str] = set()
        self._expected: set[str] = {"tts", "asr", "llm"}

    def add_expected(self, name: str) -> None:
        self._expected.add(name)

    def mark_ready(self, name: str) -> None:
        self._ready.add(name)
        print_subsystem_ready(name)
        if self._ready == self._expected:
            print_status("")
            print_status("  ═══ All systems nominal ═══")
            print_status("")

    def is_ready(self, name: str) -> bool:
        return name in self._ready


def _worker(queue: Queue[DisplayEvent], shutdown_event: threading.Event) -> None:
    """Worker thread: init services, run voice loop."""
    logger = get_logger("main")
    _play_boot_sequence(queue)

    tracker = _SubsystemTracker()
    cfg = get_config()
    if cfg.mode == "mcp":
        tracker.add_expected("mcp")

    print_status("Loading TTS model...")
    try:
        tts = TTSEngine()
        tracker.mark_ready("tts")
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        print_error(f"Failed to initialize TTS: {e}")
        sys.exit(1)

    print_status("Initializing ASR service...")
    try:
        asr = ASRService()
        tracker.mark_ready("asr")
    except Exception as e:
        logger.error(f"Failed to initialize ASR: {e}")
        print_error(f"Failed to initialize ASR: {e}")
        sys.exit(1)

    print_status("Initializing LLM service...")
    llm = _create_llm_service()
    tracker.mark_ready("llm")

    # Connect to MCP server if in MCP mode
    if cfg.mode == "mcp" and isinstance(llm, MCPAwareLLMService):
        print_status("Connecting to MCP server...")
        try:
            llm.connect()
            llm.set_tts_engine(tts)
            tracker.mark_ready("mcp")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
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

            _process_turn(llm, tts, monitor, audio_data, asr, cfg, logger)

            # Handle continuation after interruption
            prefix = monitor.buffered_audio if monitor.triggered else None
            if prefix is not None:
                print_status("Continuing recording...")
                audio_data = record_speech(prefix_audio=prefix)
                if audio_data is not None:
                    _process_turn(llm, tts, monitor, audio_data, asr, cfg, logger)

            print_status("Listening...")
    except Exception as e:
        logger.error(f"Worker thread error: {e}")
        raise
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
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=None,
        help="Enable logging to file (e.g., -l run.log)",
    )
    args = parser.parse_args()

    init_logger(args.log)

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
