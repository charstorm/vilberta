import os
import time
from typing import cast, Any
from pathlib import Path
from collections.abc import Generator

from openai import OpenAI

from vilberta.config import (
    API_BASE_URL,
    MODEL_NAME,
    API_KEY_ENV,
    MAX_HIST_THRESHOLD_SIZE,
    HIST_RESET_SIZE,
)
from vilberta.response_parser import StreamingParser, Section

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_audio_user_message(audio_b64: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
            {"type": "text", "text": "Respond to the user."},
        ],
    }


def _build_text_user_message(transcript: str) -> dict[str, str]:
    return {"role": "user", "content": transcript}


def _build_assistant_message(full_response: str) -> dict[str, str]:
    return {"role": "assistant", "content": full_response}


class ConversationHistory:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user_audio(self, audio_b64: str) -> None:
        self.messages.append(_build_audio_user_message(audio_b64))

    def add_assistant(self, full_response: str) -> None:
        self.messages.append(_build_assistant_message(full_response))

    def replace_last_user_audio_with_transcript(self, transcript: str) -> None:
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg["role"] == "user" and isinstance(msg["content"], list):
                self.messages[i] = _build_text_user_message(transcript)
                return

    def trim_if_needed(self) -> None:
        if len(self.messages) <= MAX_HIST_THRESHOLD_SIZE:
            return
        self.messages = self.messages[-HIST_RESET_SIZE:]

    def get_api_messages(self, system_prompt: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}] + self.messages


class LLMService:
    def __init__(self) -> None:
        api_key = os.environ.get(API_KEY_ENV, "")
        self.client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
        self.system_prompt = _load_system_prompt()
        self.history = ConversationHistory()

        # Metrics from last request
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_ttft = 0.0

    def stream_response(self, audio_b64: str) -> Generator[Section, None, str]:
        self.history.add_user_audio(audio_b64)
        messages = self.history.get_api_messages(self.system_prompt)

        t0 = time.monotonic()
        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=cast(Any, messages),
            stream=True,
            stream_options={"include_usage": True},
            user="vilberta",
            temperature=0.7,
        )

        parser = StreamingParser()
        full_response = ""
        first_token = True

        # Reset metrics
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_ttft = 0.0

        for chunk in stream:
            # Extract usage from the final chunk
            if hasattr(chunk, "usage") and chunk.usage is not None:
                self.last_input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                self.last_output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                # OpenRouter / some providers expose cache info
                detail = getattr(chunk.usage, "prompt_tokens_details", None)
                if detail:
                    self.last_cache_read_tokens = getattr(detail, "cached_tokens", 0) or 0
                    self.last_cache_write_tokens = getattr(detail, "audio_tokens", 0) or 0

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if first_token:
                    self.last_ttft = time.monotonic() - t0
                    first_token = False
                text = delta.content
                full_response += text
                yield from parser.feed(text)

        yield from parser.flush()

        self.history.add_assistant(full_response)

        transcript = self._extract_transcript(full_response)
        if transcript:
            self.history.replace_last_user_audio_with_transcript(transcript)

        self.history.trim_if_needed()

        return full_response

    def mark_interrupted(self) -> None:
        if not self.history.messages:
            return
        last = self.history.messages[-1]
        if last["role"] == "assistant" and isinstance(last["content"], str):
            last["content"] = last["content"] + "\n[interrupted by user]"

    @staticmethod
    def _extract_transcript(full_response: str) -> str | None:
        start = full_response.find("[transcript]")
        end = full_response.find("[/transcript]")
        if start == -1 or end == -1:
            return None
        return full_response[start + len("[transcript]") : end].strip()


# llm_service.py
