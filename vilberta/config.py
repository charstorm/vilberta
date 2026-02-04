from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from pathlib import Path
import tomllib

import numpy as np
from numpy.typing import NDArray


class SectionType(Enum):
    SPEAK = "speak"
    TEXT = "text"


@dataclass
class Section:
    type: SectionType
    content: str


@dataclass
class Config:
    # Mode: "basic" or "mcp"
    mode: str = "basic"

    # LLM API settings
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    transcriber_llm_model_name: str = "google/gemini-2.5-flash-lite"
    basic_chat_llm_model_name: str = "openai/gpt-4o-mini"
    toolcall_chat_llm_model_name: str = "openai/gpt-4o-mini"

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    dtype: type = np.int16

    # VAD settings
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 300
    min_silence_duration_ms: int = 1200
    speech_pad_ms: int = 300
    max_speech_duration_sec: int = 300

    # TTS settings
    tts_voice: str = "fantine"
    tts_speed_factor: float = 1.0

    # Chat/History settings
    max_hist_threshold_size: int = 16
    hist_reset_size: int = 8

    # Interruption settings
    interrupt_speech_duration_ms: int = 300

    # MCP settings
    mcp_server_url: str | None = None

    @classmethod
    def from_toml(cls, path: Path | str | None = None) -> "Config":
        if path is None:
            path = Path("config.toml")
        else:
            path = Path(path)

        if not path.exists():
            return cls()

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            return cls()

        general = data.get("GENERAL", {})
        llm_api = data.get("LLM_API", {})
        tts = data.get("TTS", {})
        chat = data.get("CHAT", {})
        mcp = data.get("MCP", {})

        return cls(
            mode=general.get("mode", cls.mode),
            api_base_url=llm_api.get("api_base_url", cls.api_base_url),
            api_key_env=llm_api.get("api_key_env", cls.api_key_env),
            transcriber_llm_model_name=llm_api.get(
                "transcriber_llm_model_name", cls.transcriber_llm_model_name
            ),
            basic_chat_llm_model_name=llm_api.get(
                "basic_chat_llm_model_name", cls.basic_chat_llm_model_name
            ),
            toolcall_chat_llm_model_name=llm_api.get(
                "toolcall_chat_llm_model_name", cls.toolcall_chat_llm_model_name
            ),
            tts_voice=tts.get("tts_voice", cls.tts_voice),
            max_hist_threshold_size=chat.get(
                "max_hist_threshold_size", cls.max_hist_threshold_size
            ),
            hist_reset_size=chat.get("hist_reset_size", cls.hist_reset_size),
            mcp_server_url=mcp.get("server_url", cls.mcp_server_url),
        )


# Global config instance - initialized with defaults
_config: Config | None = None


def init_config(path: Path | str | None = None) -> Config:
    global _config
    _config = Config.from_toml(path)
    return _config


def get_config() -> Config:
    if _config is None:
        return Config()
    return _config


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
    cfg = get_config()
    return VADConfig(
        threshold=cfg.vad_threshold,
        min_speech_chunks=int(
            cfg.min_speech_duration_ms * cfg.sample_rate / 1000 / cfg.chunk_size
        ),
        min_silence_chunks=int(
            cfg.min_silence_duration_ms * cfg.sample_rate / 1000 / cfg.chunk_size
        ),
        max_speech_chunks=int(
            cfg.max_speech_duration_sec * cfg.sample_rate / cfg.chunk_size
        ),
        speech_pad_chunks=int(
            cfg.speech_pad_ms * cfg.sample_rate / 1000 / cfg.chunk_size
        ),
    )


# config.py
