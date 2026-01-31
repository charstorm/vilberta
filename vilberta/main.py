import sys
import signal
import threading
import time

from vilberta.audio_capture import record_speech, audio_to_base64_wav
from vilberta.interrupt_monitor import InterruptMonitor
from vilberta.llm_service import LLMService
from vilberta.response_parser import SectionType
from vilberta.tts_engine import TTSEngine
from vilberta.display import (
    print_speak,
    print_text,
    print_transcript,
    print_status,
    print_error,
)
from vilberta.sound_effects import (
    play_response_send,
    play_response_received,
    play_text_start,
    play_response_end,
)


def _setup_signal_handler() -> threading.Event:
    shutdown_event = threading.Event()

    def handler(sig: int, frame: object) -> None:
        print_status("\nShutting down...")
        shutdown_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    return shutdown_event


def _speak_with_monitor(tts: TTSEngine, monitor: InterruptMonitor, text: str) -> bool:
    """Speak text while monitoring for interrupts. Returns True if completed."""
    # Poll monitor in a background thread to trigger TTS interrupt
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
    llm: LLMService, tts: TTSEngine, monitor: InterruptMonitor, audio_b64: str
) -> None:
    interrupted = False
    monitor.start()
    first_section = True

    play_response_send()

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

    if interrupted:
        llm.mark_interrupted()
        print_status("[interrupted]")
    else:
        play_response_end()


def main() -> None:
    shutdown_event = _setup_signal_handler()

    print_status("Loading TTS model...")
    tts = TTSEngine()
    print_status("TTS ready.")

    print_status("Initializing LLM service...")
    llm = LLMService()
    print_status("Ready. Listening...\n")

    monitor = InterruptMonitor()

    while not shutdown_event.is_set():
        audio_data = record_speech()
        if audio_data is None:
            continue

        print_status("Processing...")
        audio_b64 = audio_to_base64_wav(audio_data)

        try:
            _process_response(llm, tts, monitor, audio_b64)
        except Exception as e:
            print_error(f"LLM error: {e}")

        # If monitor was triggered, the user started speaking —
        # continue recording with the buffered audio prefix
        prefix = monitor.buffered_audio if monitor.triggered else None
        if prefix is not None:
            print_status("Continuing recording...")
            audio_data = record_speech(prefix_audio=prefix)
            if audio_data is not None:
                print_status("Processing...")
                audio_b64 = audio_to_base64_wav(audio_data)
                try:
                    _process_response(llm, tts, monitor, audio_b64)
                except Exception as e:
                    print_error(f"LLM error: {e}")

        print()
        print_status("Listening...\n")


if __name__ == "__main__":
    main()
