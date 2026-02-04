# AGENTS.md

## Project Overview
Vilberta is an interactive voice assistant that provides real-time voice interaction with LLMs. It features speech-to-text, text-to-speech, interruption handling, and multimodal output. The codebase follows Python best practices with a focus on type hints, modular architecture, and simple non-streaming LLM responses for lower latency.


## Code Style Guidelines

- use type annotation on function args and return
- avoid docstrings
- use helper functions
- keep code modular
- return early when possible
- reduce indented code when possible
- target python version 3.11 or above
- write code that is easy to read and explain
- use idiomatic code
- unless specially asked, don't add try/except blocks
- typing.List, typing.Dict etc are outdated. Use list, dict etc directly for type annotation
- suggest a filename as comment at the end of the code
- comments should handle the "why", not the "what"
- don't mix low level code with high level code
- produce code that is low in cognitive complexity


## File Structure

### Root Files
- `README.md` - Project documentation and usage instructions
- `LICENSE` - License file
- `requirements.txt` - Python dependencies (openai, textual, pocket-tts, etc.)
- `mypy.ini` - MyPy type checker configuration
- `.gitignore` - Git ignore rules

### vilberta/ - Main Package
Core application modules:

- `__init__.py` - Package initialization
- `__main__.py` - Entry point (`python -m vilberta`)
- `main.py` - Main application logic, voice loop, preflight checks, boot sequence
- `config.py` - Configuration constants (API, audio, VAD, TTS settings), dataclasses (Section, SectionType, VADConfig, AudioState, Config), and shared types. Loads from config.toml
- `logger.py` - Logger initialization using loguru with file rotation, formatting, and retention settings. Provides `get_logger()` function
- `asr_service.py` - ASR (Automatic Speech Recognition) service using a lightweight LLM to transcribe audio with optional conversation context words
- `audio_capture.py` - Audio recording with Silero VAD, audio-to-base64 conversion
- `llm_service.py` - OpenRouter LLM client, non-streaming responses, conversation history management. Defines BaseLLMService abstract class and BasicLLMService implementation
- `mcp_service.py` - MCP (Model Context Protocol) service for tool calling. Connects to MCP server, manages tool execution loop
- `mcp_llm_service.py` - Wrapper around MCPService providing sync interface for basic LLM service compatibility
- `text_section_splitter.py` - Generic streaming text section splitter with inner delimiters. Used by llm_service and mcp_service for parsing tagged responses
- `tts_engine.py` - Text-to-speech engine using pocket-tts with interruption support
- `interrupt_monitor.py` - Monitors microphone during TTS playback for user interruptions
- `sound_effects.py` - Sound effect playback (WAV files from sounds/)
- `generate_sounds.py` - Generates sound effect WAV files programmatically
- `display.py` - Display adapter that routes all output through queue to UI. Defines DisplayEvent and RequestStats dataclasses shared by all interfaces
- `cli.py` - Simple CLI interface using plain print statements (no curses/textual dependencies). Default interface
- `tui.py` - Textual-based terminal UI with three-panel layout (system, conversation, events). Optional, requires textual package

Subdirectories:
- `prompts/system.md` - System prompt defining response format for LLM (speak/text/transcript tags)
- `prompts/system_mcp.md` - System prompt for MCP mode with tool calling
- `sounds/` - Sound effect WAV files (ready.wav, response_send.wav, etc.)

### docs/
- `screenshot.png` - Screenshot for README

### tmp/
Temporary files (logs, chat history, todo lists). Ignore files here.


## Key Architecture Notes

### LLM Response Format
The assistant responds with tagged sections:
- `[speak]...[/speak]` - Spoken responses (text-to-speech)
- `[text]...[/text]` - Text-only responses (no TTS)

### Modes
- **Basic mode**: Direct LLM interaction without tools (uses BasicLLMService)
- **MCP mode**: LLM with tool calling via MCP server (uses MCPAwareLLMService)

### Non-Streaming Architecture
All LLM responses are fetched as complete responses (non-streaming) for simplicity and lower perceived latency. The `_parse_response()` function in both `llm_service.py` and `mcp_service.py` uses `text_section_splitter.py` to parse tagged sections.


## Virtual Environment
Check for venv or .venv for tools like ruff and mypy
