import numpy as np
from scipy import signal as scipy_signal
import sounddevice as sd
import warnings


import torch

torch.backends.nnpack.enabled = False  # type: ignore[attr-defined]
warnings.filterwarnings("ignore", message="Could not initialize NNPACK")

from pocket_tts import TTSModel  # noqa: E402

from vilberta.config import get_config  # noqa: E402
from vilberta.logger import get_logger  # noqa: E402

FADE_SAMPLES = 64


def _apply_fade(audio: np.ndarray) -> np.ndarray:
    n = min(FADE_SAMPLES, len(audio) // 2)
    if n <= 0:
        return audio
    audio = audio.copy()
    audio[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return audio


class TTSEngine:
    def __init__(self) -> None:
        self.logger = get_logger("TTSEngine")
        cfg = get_config()
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(cfg.tts_voice)
        self._interrupted = False
        self.logger.info(f"TTS engine initialized with voice: {cfg.tts_voice}")

    def interrupt(self) -> None:
        self._interrupted = True
        self.logger.debug("TTS interrupted")

    def speak(self, text: str) -> bool:
        """Speak a single line of text. Returns True if completed, False if interrupted."""
        self.logger.info(f"Speaking text ({len(text)} chars)")
        cfg = get_config()
        self._interrupted = False
        original_rate = self.model.sample_rate
        playback_rate = int(original_rate / cfg.tts_speed_factor)

        with sd.OutputStream(
            samplerate=playback_rate, channels=1, dtype="float32"
        ) as stream:
            for chunk in self.model.generate_audio_stream(self.voice_state, text):
                if self._interrupted:
                    self.logger.info("TTS interrupted during playback")
                    return False
                audio_np = chunk.numpy().astype(np.float32)
                if cfg.tts_speed_factor != 1.0:
                    num_samples = int(len(audio_np) * playback_rate / original_rate)
                    audio_np = scipy_signal.resample(audio_np, num_samples)
                audio_np = _apply_fade(audio_np)
                stream.write(audio_np)

        self.logger.info("TTS completed successfully")
        return True
