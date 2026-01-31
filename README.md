# Vilberta

An interactive voice assistant powered by LLMs, featuring real-time speech-to-text, streaming text-to-speech, intelligent interruption handling, and multimodal output modes.

## Features

- **Real-time Voice Interaction**: Bidirectional audio communication with low-latency processing
- **Voice Activity Detection (VAD)**: Uses Silero VAD for automatic speech detection
- **Streaming TTS**: Real-time speech synthesis with customizable voice and speed
- **Smart Interruption Handling**: Gracefully interrupts TTS when you speak and resumes context
- **Multimodal Output**:
  - `[speak]`: Audio responses for conversational interaction
  - `[text]`: Visual content for code, lists, and complex information
  - `[transcript]`: Confirmation of your input
- **Audio Feedback**: Sound effects for user events and system states
- **Persistent Conversation**: Maintains history across interactions
- **Rich Terminal Interface**: Colored output with status indicators and progress bars

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anomalyco/vilberta.git
cd vilberta
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your OpenRouter API key:
```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```
Alternatively, create a `.env` file in the project root with `OPENAI_API_KEY=your-key-here`.

## Usage

Start the voice assistant:

```bash
python -m vilberta
```

Or if installed as a package:

```bash
vilberta
```

The system will:
1. Initialize audio components and LLM service
2. Display a ready prompt
3. Begin listening for voice input

Speak naturally and the assistant will respond both audibly and visually. Press `Ctrl+C` to exit at any time.

### Command Line Options

- `--model MODEL`: Override default LLM model
- `--voice VOICE`: Set TTS voice (default: alba)
- `--speed FACTOR`: Adjust TTS speed (default: 1.0)
- `--help`: Show all options

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
- `MODEL_NAME`: LLM model to use (default: google/gemini-flash-1.5)
- `SAMPLE_RATE`: Audio sample rate (16kHz)
- `VAD_THRESHOLD`: Speech detection sensitivity (0.5 = medium)
- `TTS_VOICE`: Voice for TTS (default: alba)
- `TTS_SPEED_FACTOR`: Speech speed multiplier (1.0 = normal)

See `config.py` for all available settings.

## Dependencies

Main runtime dependencies:
- `torch`/`torchaudio`: Machine learning framework with CPU support
- `scipy`: Scientific computing for audio processing
- `openai`: API client for LLM services
- `pocket-tts`: Local text-to-speech engine
- `pysilero-vad`: Voice activity detection
- `sounddevice`: Audio input/output

See `requirements.txt` for specific versions and additional dev dependencies.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (including edge cases)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- **LLM**: Powered by Google's Gemini models via OpenRouter
- **TTS**: pocket-tts for fast, local speech synthesis
- **VAD**: Silero VAD for reliable voice detection
- **Audio**: SoundDevice and SciPy for audio processing
