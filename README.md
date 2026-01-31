# VoiceBot

A Python-based voice assistant that provides real-time, bidirectional voice interaction with an LLM. Features Voice Activity Detection (VAD), streaming text-to-speech, interruption handling, and dual-mode output (spoken and visual).

## Features

- **Voice Activity Detection (VAD)**: Automatically detects when you start and stop speaking using Silero VAD
- **Streaming TTS**: Speaks responses in real-time as they arrive from the LLM
- **Interruption Support**: Detects when you speak during TTS playback and gracefully handles interruptions
- **Dual Output Modes**:
  - `[speak]`: Spoken responses for conversational content
  - `[text]`: On-screen text for detailed information, code, lists, etc.
  - `[transcript]`: Displays what you said for confirmation
- **Sound Effects**: Audio feedback for key events (recording, response received, etc.)
- **Conversation History**: Maintains context across multiple turns
- **Terminal UI**: Beautiful colored terminal output with status indicators

## Architecture

```
vilberta/
├── main.py                 # Main event loop and orchestration
├── audio_capture.py        # VAD-based speech recording
├── llm_service.py         # LLM API integration with conversation history
├── tts_engine.py          # Text-to-speech using pocket-tts
├── interrupt_monitor.py   # Detects user interruptions during TTS
├── display.py             # Terminal output with ANSI colors
├── sound_effects.py       # Audio feedback system
├── response_parser.py     # Parses LLM tagged responses
├── text_section_splitter.py  # Streaming text parser utility
├── config.py              # Configuration constants
├── generate_sounds.py     # Generates sound effect WAV files
└── prompts/system.md      # System prompt for the LLM
```

## Requirements

- Python 3.10+
- Microphone access
- OpenRouter API key (for LLM access)
- See `pyproject.toml` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vilberta
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Set up your API key:
```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

## Usage

Run the voice assistant:

```bash
python -m vilberta
```

Or use the provided entry point:

```bash
vilberta
```

The assistant will:
1. Load the TTS model
2. Initialize the LLM service
3. Start listening for your voice
4. Process your speech and respond

Press `Ctrl+C` to exit gracefully.

## How It Works

### Conversation Flow

1. **Recording**: VAD detects speech and records until you stop speaking
2. **Transcription**: Audio is sent to the LLM which transcribes it
3. **Response Streaming**: The LLM streams back a response with special tags:
   - `[transcript]What you said[/transcript]`
   - `[speak]Spoken response[/speak]`
   - `[text]On-screen content[/text]`
4. **TTS Playback**: Speak sections are played aloud while monitoring for interruptions
5. **Text Display**: Text sections are printed to the terminal

### Interruption Handling

If you speak during TTS playback:
- The TTS immediately stops
- The buffered audio is prepended to the next recording
- The conversation continues naturally

### Configuration

Key settings in `vilberta/config.py`:

- `API_BASE_URL`: LLM API endpoint (default: OpenRouter)
- `MODEL_NAME`: LLM model to use (default: google/gemini-2.5-flash)
- `SAMPLE_RATE`: Audio sample rate (16kHz)
- `VAD_THRESHOLD`: Speech detection sensitivity (0.5)
- `TTS_VOICE`: Voice for TTS (default: alba)
- `TTS_SPEED_FACTOR`: Speech speed multiplier (1.0)

## Dependencies

- `openai`: OpenAI-compatible API client
- `pocket-tts`: Fast local text-to-speech
- `pysilero-vad`: Voice activity detection
- `sounddevice`: Audio I/O
- `numpy`, `scipy`: Audio processing

## Acknowledgments

- Uses Google's Gemini model via OpenRouter
- TTS powered by pocket-tts
- VAD powered by Silero
