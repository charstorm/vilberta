"""Generate short sound effect WAV files for the vilberta."""

import os
import wave

import numpy as np

SAMPLE_RATE = 24000
SILENCE_PAD_S = 0.04
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sounds")


def _save_wav(filename: str, audio: np.ndarray) -> None:
    pad = np.zeros(int(SAMPLE_RATE * SILENCE_PAD_S), dtype=np.float32)
    audio = np.concatenate([pad, audio, pad])
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 8000).astype(np.int16)
    path = os.path.join(OUTPUT_DIR, filename)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    print(f"  {filename} ({len(pcm)} samples, {len(pcm) / SAMPLE_RATE:.3f}s)")


def _envelope(n: int, attack: int = 0, decay: int = 0) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if decay > 0:
        env[-decay:] = np.linspace(1, 0, decay)
    return env


def _sweep(duration_s: float, f0: float, f1: float, volume: float = 0.5) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    freq = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    audio = np.sin(phase).astype(np.float32) * volume
    env = _envelope(n, attack=n // 8, decay=n // 4)
    result: np.ndarray = audio * env
    return result


def generate_response_send() -> None:
    n = int(SAMPLE_RATE * 0.1)
    t = np.linspace(0, 0.1, n, dtype=np.float32)
    audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.3
    env = np.exp(-t * 40)
    _save_wav("response_send.wav", audio * env)


def generate_response_received() -> None:
    dur_each = 0.06
    n = int(SAMPLE_RATE * dur_each)
    t = np.linspace(0, dur_each, n, dtype=np.float32)
    tone1 = np.sin(2 * np.pi * 600 * t) * _envelope(n, attack=n // 8, decay=n // 4)
    tone2 = np.sin(2 * np.pi * 900 * t) * _envelope(n, attack=n // 8, decay=n // 4)
    audio = np.concatenate([tone1, tone2]).astype(np.float32) * 0.35
    _save_wav("response_received.wav", audio)


def generate_text_start() -> None:
    dur = 0.05
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = _envelope(n, attack=n // 8, decay=n // 3)
    t1 = np.sin(2 * np.pi * 500 * t) * env
    t2 = np.sin(2 * np.pi * 750 * t) * env
    audio = np.concatenate([t1, t2]).astype(np.float32) * 0.4
    _save_wav("text_start.wav", audio)


def generate_text_end() -> None:
    n = int(SAMPLE_RATE * 0.04)
    t = np.linspace(0, 0.04, n, dtype=np.float32)
    pulse = np.sin(2 * np.pi * 300 * t).astype(np.float32) * 0.25
    env = _envelope(n, attack=n // 8, decay=n // 3)
    pulse = pulse * env
    gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
    _save_wav("text_end.wav", np.concatenate([pulse, gap, pulse]))


def generate_line_print() -> None:
    n = int(SAMPLE_RATE * 0.03)
    t = np.linspace(0, 0.03, n, dtype=np.float32)
    audio = np.sin(2 * np.pi * 500 * t).astype(np.float32) * 0.12
    env = _envelope(n, attack=n // 10, decay=n // 3)
    _save_wav("line_print.wav", audio * env)


def generate_ready() -> None:
    dur = 0.08
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = _envelope(n, attack=n // 6, decay=n // 3)
    t1 = np.sin(2 * np.pi * 880 * t) * env
    t2 = np.sin(2 * np.pi * 1320 * t) * env
    gap = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32)
    audio = np.concatenate([t1, gap, t2]).astype(np.float32) * 0.35
    _save_wav("ready.wav", audio)


def generate_response_end() -> None:
    dur = 0.05
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = _envelope(n, attack=n // 8, decay=n // 3)
    t1 = np.sin(2 * np.pi * 800 * t) * env
    t2 = np.sin(2 * np.pi * 600 * t) * env
    t3 = np.sin(2 * np.pi * 400 * t) * env
    audio = np.concatenate([t1, t2, t3]).astype(np.float32) * 0.3
    _save_wav("response_end.wav", audio)


def generate_tool_call_start() -> None:
    # Short mechanical "processing" sound - rising beep
    dur = 0.04
    n = int(SAMPLE_RATE * dur)
    env = _envelope(n, attack=n // 6, decay=n // 3)
    freq = np.linspace(800, 1200, n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    audio = np.sin(phase).astype(np.float32) * 0.25 * env
    _save_wav("tool_call_start.wav", audio)


def generate_tool_call_success() -> None:
    # Short ascending two-tone success chime
    dur = 0.04
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = _envelope(n, attack=n // 8, decay=n // 3)
    t1 = np.sin(2 * np.pi * 880 * t) * env * 0.25
    t2 = np.sin(2 * np.pi * 1100 * t) * env * 0.25
    gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
    audio = np.concatenate([t1, gap, t2]).astype(np.float32)
    _save_wav("tool_call_success.wav", audio)


def generate_tool_call_error() -> None:
    # Short descending low tone for error
    dur = 0.06
    n = int(SAMPLE_RATE * dur)
    env = _envelope(n, attack=n // 8, decay=n // 2)
    freq = np.linspace(400, 200, n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    audio = np.sin(phase).astype(np.float32) * 0.3 * env
    _save_wav("tool_call_error.wav", audio)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating sound effects...")
    generate_response_send()
    generate_response_received()
    generate_text_start()
    generate_text_end()
    generate_line_print()
    generate_response_end()
    generate_ready()
    generate_tool_call_start()
    generate_tool_call_success()
    generate_tool_call_error()
    print("Done.")


if __name__ == "__main__":
    main()
