import os
import time
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAI

from vilberta.config import get_config
from vilberta.logger import get_logger


@dataclass
class ASRStats:
    """Stats for ASR transcription."""

    audio_duration_s: float
    processing_time_s: float
    input_tokens: int
    output_tokens: int


class ASRService:
    """ASR service for transcribing audio using a lightweight LLM."""

    def __init__(self) -> None:
        cfg = get_config()
        api_key = os.environ.get(cfg.api_key_env, "")
        self.client = OpenAI(base_url=cfg.api_base_url, api_key=api_key)
        self.logger = get_logger("ASRService")

    def transcribe(
        self, audio_b64: str, audio_duration_s: float
    ) -> tuple[str, ASRStats]:
        """Transcribe audio to text.

        Args:
            audio_b64: Base64-encoded audio data
            audio_duration_s: Duration of audio in seconds

        Returns:
            Tuple of (transcript text, ASR stats)
        """
        self.logger.debug("Starting transcription")
        cfg = get_config()

        messages = [
            {
                "role": "system",
                "content": "Transcribe the following speech to text.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": "Please transcribe this audio"},
                ],
            },
        ]

        t0 = time.monotonic()
        response = self.client.chat.completions.create(
            model=cfg.transcriber_llm_model_name,
            messages=cast(Any, messages),
            stream=False,
            user="vilberta",
            temperature=0.0,
        )
        processing_time = time.monotonic() - t0

        self.logger.info(f"Transcription completed in {processing_time:.3f}s")

        transcript = response.choices[0].message.content or ""
        transcript = transcript.strip()

        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        self.logger.debug(f"ASR tokens: input={input_tokens}, output={output_tokens}")

        stats = ASRStats(
            audio_duration_s=audio_duration_s,
            processing_time_s=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return transcript, stats


# asr_service.py
