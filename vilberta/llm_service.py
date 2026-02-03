import os
import time
from abc import ABC, abstractmethod
from typing import cast, Any
from pathlib import Path

from openai import OpenAI

from vilberta.config import get_config, Section, SectionType
from vilberta.text_section_splitter import StreamSection, StreamTextSectionSplitter
from vilberta.logger import get_logger

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

_TAG_SECTIONS = [
    StreamSection("[speak]", "[/speak]", inner_split_on=["\n"]),
    StreamSection("[text]", "[/text]", inner_split_on=None),
    StreamSection("[transcript]", "[/transcript]", inner_split_on=None),
]

_TAG_OPEN = {
    "[speak]": SectionType.SPEAK,
    "[text]": SectionType.TEXT,
    "[transcript]": SectionType.TRANSCRIPT,
}

_TAG_STRINGS = {s.starting_tag for s in _TAG_SECTIONS} | {
    s.ending_tag for s in _TAG_SECTIONS if s.ending_tag
}


def _parse_response(text: str) -> list[Section]:
    splitter = StreamTextSectionSplitter(sections=_TAG_SECTIONS)
    parts = list(splitter.split(text))
    parts.extend(splitter.flush())

    sections: list[Section] = []
    for part in parts:
        if part.text in _TAG_STRINGS or part.section is None:
            continue
        section_type = _TAG_OPEN.get(part.section)
        if section_type is None:
            continue
        content = part.text.strip()
        if content:
            sections.append(Section(type=section_type, content=content))
    return sections


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
        cfg = get_config()
        if len(self.messages) <= cfg.max_hist_threshold_size:
            return
        self.messages = self.messages[-cfg.hist_reset_size :]

    def get_api_messages(self, system_prompt: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}] + self.messages


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""

    @abstractmethod
    def get_response(self, audio_b64: str) -> tuple[list[Section], str]:
        """Get response from LLM.

        Returns list of parsed sections and the full response string.
        """
        ...

    @abstractmethod
    def mark_interrupted(self) -> None:
        """Mark that the user interrupted the response."""
        ...

    @property
    @abstractmethod
    def last_input_tokens(self) -> int:
        """Input tokens from last request."""
        ...

    @property
    @abstractmethod
    def last_output_tokens(self) -> int:
        """Output tokens from last request."""
        ...

    @property
    @abstractmethod
    def last_cache_read_tokens(self) -> int:
        """Cache read tokens from last request."""
        ...

    @property
    @abstractmethod
    def last_cache_write_tokens(self) -> int:
        """Cache write tokens from last request."""
        ...

    @property
    @abstractmethod
    def last_latency_s(self) -> float:
        """Total latency of last request in seconds."""
        ...


class BasicLLMService(BaseLLMService):
    """Basic LLM service with non-streaming responses (no tool calling)."""

    def __init__(self) -> None:
        cfg = get_config()
        api_key = os.environ.get(cfg.api_key_env, "")
        self.client = OpenAI(base_url=cfg.api_base_url, api_key=api_key)
        self.system_prompt = _load_system_prompt()
        self.history = ConversationHistory()
        self.logger = get_logger("BasicLLMService")

        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_cache_read_tokens = 0
        self._last_cache_write_tokens = 0
        self._last_latency_s = 0.0

    @property
    def last_input_tokens(self) -> int:
        return self._last_input_tokens

    @property
    def last_output_tokens(self) -> int:
        return self._last_output_tokens

    @property
    def last_cache_read_tokens(self) -> int:
        return self._last_cache_read_tokens

    @property
    def last_cache_write_tokens(self) -> int:
        return self._last_cache_write_tokens

    @property
    def last_latency_s(self) -> float:
        return self._last_latency_s

    def get_response(self, audio_b64: str) -> tuple[list[Section], str]:
        self.logger.debug("Triggering LLM request")
        self.history.add_user_audio(audio_b64)
        messages = self.history.get_api_messages(self.system_prompt)

        self.logger.debug(f"Sending request with {len(messages)} messages")
        t0 = time.monotonic()
        cfg = get_config()
        response = self.client.chat.completions.create(
            model=cfg.model_name,
            messages=cast(Any, messages),
            stream=False,
            user="vilberta",
            temperature=0.7,
        )
        self._last_latency_s = time.monotonic() - t0

        self.logger.info(f"LLM response received in {self._last_latency_s:.3f}s")

        # Extract usage metrics
        self._last_input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        self._last_output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        detail = getattr(response.usage, "prompt_tokens_details", None)
        if detail:
            self._last_cache_read_tokens = getattr(detail, "cached_tokens", 0) or 0
            self._last_cache_write_tokens = getattr(detail, "audio_tokens", 0) or 0

        self.logger.debug(
            f"Tokens: input={self._last_input_tokens}, output={self._last_output_tokens}, "
            f"cache_read={self._last_cache_read_tokens}, cache_write={self._last_cache_write_tokens}"
        )

        full_response = response.choices[0].message.content or ""
        self.logger.debug(f"Raw response length: {len(full_response)} chars")

        sections = _parse_response(full_response)
        self.logger.info(f"Parsed {len(sections)} sections from response")

        self.history.add_assistant(full_response)

        transcript = self._extract_transcript(full_response)
        if transcript:
            self.logger.debug(f"Extracted transcript: {transcript[:100]}...")
            self.history.replace_last_user_audio_with_transcript(transcript)

        self.history.trim_if_needed()

        return sections, full_response

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


# Keep alias for backward compatibility
LLMService = BasicLLMService


# llm_service.py
