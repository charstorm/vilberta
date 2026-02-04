"""Monitors microphone during TTS playback for interrupt detection.

Runs VAD on incoming audio. If speech is detected continuously for
INTERRUPT_CONFIRM_MS, triggers an interrupt on the TTS engine and
keeps the buffered audio frames so they can be prepended to the next
recording.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from pysilero_vad import SileroVoiceActivityDetector

from vilberta.config import get_config
from vilberta.display import print_vad
from vilberta.logger import get_logger


class InterruptMonitor:
    def __init__(self) -> None:
        cfg = get_config()
        self._detector = SileroVoiceActivityDetector()
        self._confirm_chunks = int(
            cfg.interrupt_speech_duration_ms * cfg.sample_rate / 1000 / cfg.chunk_size
        )
        self._speech_chunks = 0
        self._buffered_frames: list[NDArray[np.int16]] = []
        self._triggered = False
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self.logger = get_logger("InterruptMonitor")

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def buffered_audio(self) -> NDArray[np.int16] | None:
        with self._lock:
            if not self._buffered_frames:
                return None
            return np.concatenate(self._buffered_frames)

    def _callback(
        self,
        indata: NDArray[np.int16],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        audio_chunk = indata[:, 0].copy()
        with self._lock:
            self._buffered_frames.append(audio_chunk)

        if self._triggered:
            return

        is_speech = self._is_speech(audio_chunk)
        if is_speech:
            if self._speech_chunks == 0:
                print_vad(up=True)
            self._speech_chunks += 1
            if self._speech_chunks >= self._confirm_chunks:
                self._triggered = True
                self.logger.info("Interrupt triggered by user speech")
        else:
            if self._speech_chunks > 0:
                print_vad(up=False)
            self._speech_chunks = 0

    def _is_speech(self, audio_chunk: NDArray[np.int16]) -> bool:
        cfg = get_config()
        if len(audio_chunk) != cfg.chunk_size:
            return False
        audio_bytes = audio_chunk.astype(np.int16).tobytes()
        if len(audio_bytes) != self._detector.chunk_bytes():
            return False
        return bool(self._detector(audio_bytes) >= cfg.vad_threshold)

    def start(self) -> None:
        cfg = get_config()
        self._triggered = False
        self._speech_chunks = 0
        self._buffered_frames = []
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="int16",
            blocksize=cfg.chunk_size,
            callback=self._callback,
        )
        self._stream.start()
        self.logger.debug("Interrupt monitoring started")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._speech_chunks > 0:
            print_vad(up=False)
        self.logger.debug(f"Interrupt monitoring stopped (triggered={self._triggered})")
