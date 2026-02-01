"""Load and play sound effects for the vilberta."""

import os
import wave

import numpy as np
import sounddevice as sd

_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
_cache: dict[str, np.ndarray] = {}
_SAMPLE_RATE = 24000


def _load(name: str) -> np.ndarray:
    if name not in _cache:
        path = os.path.join(_SOUNDS_DIR, name)
        with wave.open(path, "r") as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        _cache[name] = audio
    return _cache[name]


def _play(name: str) -> None:
    try:
        audio = _load(name)
        sd.play(audio, samplerate=_SAMPLE_RATE)
    except Exception:
        pass  # Never let sound effects crash the app


def play_response_send() -> None:
    _play("response_send.wav")


def play_response_received() -> None:
    _play("response_received.wav")


def play_text_start() -> None:
    _play("text_start.wav")


def play_text_end() -> None:
    _play("text_end.wav")


def play_line_print() -> None:
    _play("line_print.wav")


def play_response_end() -> None:
    _play("response_end.wav")


def play_ready() -> None:
    _play("ready.wav")
