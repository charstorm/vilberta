from dataclasses import dataclass, field
from collections import deque

import numpy as np
from numpy.typing import NDArray


# --- API ---
API_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-2.5-flash"
# MODEL_NAME = "openai/gpt-audio-mini"
API_KEY_ENV = "OPENROUTER_API_KEY"

# --- Audio ---
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
DTYPE = np.int16

# --- VAD ---
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 300
MIN_SILENCE_DURATION_MS = 1200
SPEECH_PAD_MS = 300
MAX_SPEECH_DURATION_SEC = 300

# --- TTS ---
TTS_VOICE = "fantine"
TTS_SPEED_FACTOR = 1

# --- History ---
MAX_HIST_THRESHOLD_SIZE = 16
HIST_RESET_SIZE = 8

# --- Interruption ---
INTERRUPT_SPEECH_DURATION_MS = 300


@dataclass
class VADConfig:
    threshold: float
    min_speech_chunks: int
    min_silence_chunks: int
    max_speech_chunks: int
    speech_pad_chunks: int


@dataclass
class AudioState:
    ring_buffer: deque[NDArray[np.int16]] = field(default_factory=deque)
    speech_buffer: list[NDArray[np.int16]] = field(default_factory=list)
    is_speech_active: bool = False
    silence_chunks: int = 0
    speech_chunks: int = 0

    def reset_speech(self) -> None:
        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_chunks = 0
        self.speech_chunks = 0

    def start_speech(self) -> None:
        self.is_speech_active = True
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.speech_buffer = list(self.ring_buffer)


def create_vad_config() -> VADConfig:
    return VADConfig(
        threshold=VAD_THRESHOLD,
        min_speech_chunks=int(MIN_SPEECH_DURATION_MS * SAMPLE_RATE / 1000 / CHUNK_SIZE),
        min_silence_chunks=int(
            MIN_SILENCE_DURATION_MS * SAMPLE_RATE / 1000 / CHUNK_SIZE
        ),
        max_speech_chunks=int(MAX_SPEECH_DURATION_SEC * SAMPLE_RATE / CHUNK_SIZE),
        speech_pad_chunks=int(SPEECH_PAD_MS * SAMPLE_RATE / 1000 / CHUNK_SIZE),
    )


# config.py
