import io
import wave
import base64
from collections import deque

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from pysilero_vad import SileroVoiceActivityDetector

from vilberta.config import (
    VADConfig,
    AudioState,
    create_vad_config,
    get_config,
)
from vilberta.display import print_vad


class VADProcessor:
    def __init__(
        self, detector: SileroVoiceActivityDetector, config: VADConfig
    ) -> None:
        self.detector = detector
        self.config = config

    def is_speech(self, audio_chunk: NDArray[np.int16]) -> bool:
        cfg = get_config()
        if len(audio_chunk) != cfg.chunk_size:
            return False
        audio_bytes = audio_chunk.astype(np.int16).tobytes()
        if len(audio_bytes) != self.detector.chunk_bytes():
            return False
        return bool(self.detector(audio_bytes) >= self.config.threshold)


class AudioStreamHandler:
    def __init__(self, vad: VADProcessor, config: VADConfig) -> None:
        self.vad = vad
        self.config = config
        self.state = AudioState(ring_buffer=deque(maxlen=config.speech_pad_chunks))
        self.completed_audio: NDArray[np.int16] | None = None

    def callback(
        self, indata: NDArray[np.int16], frames: int, time_info: object, status: object
    ) -> None:
        audio_chunk = indata[:, 0].copy()
        is_speech = self.vad.is_speech(audio_chunk)

        if not self.state.is_speech_active:
            self._handle_no_speech(audio_chunk, is_speech)
        else:
            self._handle_active_speech(audio_chunk, is_speech)

    def _handle_no_speech(
        self, audio_chunk: NDArray[np.int16], is_speech: bool
    ) -> None:
        self.state.ring_buffer.append(audio_chunk)
        if is_speech:
            print_vad(up=True)
            self.state.start_speech()
            self.state.speech_buffer.append(audio_chunk)

    def _handle_active_speech(
        self, audio_chunk: NDArray[np.int16], is_speech: bool
    ) -> None:
        self.state.speech_buffer.append(audio_chunk)
        self.state.speech_chunks += 1

        if is_speech:
            self.state.silence_chunks = 0
        else:
            self.state.silence_chunks += 1

        if self._should_end_speech():
            print_vad(up=False)
            self.completed_audio = np.concatenate(self.state.speech_buffer)
            self.state.reset_speech()
            raise sd.CallbackStop

    def _should_end_speech(self) -> bool:
        sufficient_speech = self.state.speech_chunks >= self.config.min_speech_chunks
        sufficient_silence = self.state.silence_chunks >= self.config.min_silence_chunks
        too_long = self.state.speech_chunks >= self.config.max_speech_chunks
        return (sufficient_speech and sufficient_silence) or too_long


def audio_to_base64_wav(audio_data: NDArray[np.int16]) -> str:
    cfg = get_config()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(cfg.channels)
        wf.setsampwidth(2)
        wf.setframerate(cfg.sample_rate)
        wf.writeframes(audio_data.tobytes())
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def record_speech(
    prefix_audio: NDArray[np.int16] | None = None,
) -> NDArray[np.int16] | None:
    cfg = get_config()
    detector = SileroVoiceActivityDetector()
    config = create_vad_config()
    vad = VADProcessor(detector, config)
    handler = AudioStreamHandler(vad, config)

    if prefix_audio is not None:
        handler.state.start_speech()
        handler.state.speech_buffer.append(prefix_audio)
        handler.state.speech_chunks = len(prefix_audio) // cfg.chunk_size
        print_vad(up=True)

    stream = sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=cfg.channels,
        dtype=cfg.dtype,
        blocksize=cfg.chunk_size,
        callback=handler.callback,
    )

    with stream:
        while stream.active:
            sd.sleep(100)

    return handler.completed_audio
