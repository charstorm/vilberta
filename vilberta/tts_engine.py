import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal
from pocket_tts import TTSModel

from vilberta.config import TTS_VOICE, TTS_SPEED_FACTOR

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
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(TTS_VOICE)
        self._interrupted = False

    def interrupt(self) -> None:
        self._interrupted = True

    def speak(self, text: str) -> bool:
        """Speak a single line of text. Returns True if completed, False if interrupted."""
        self._interrupted = False
        original_rate = self.model.sample_rate
        playback_rate = int(original_rate / TTS_SPEED_FACTOR)

        with sd.OutputStream(
            samplerate=playback_rate, channels=1, dtype="float32"
        ) as stream:
            for chunk in self.model.generate_audio_stream(self.voice_state, text):
                if self._interrupted:
                    return False
                audio_np = chunk.numpy().astype(np.float32)
                if TTS_SPEED_FACTOR != 1.0:
                    num_samples = int(len(audio_np) * playback_rate / original_rate)
                    audio_np = scipy_signal.resample(audio_np, num_samples)
                audio_np = _apply_fade(audio_np)
                stream.write(audio_np)

        return True
