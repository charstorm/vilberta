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
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

## Usage

Start the voice assistant:

```bash
python -m vilberta
```

The system will:
1. Initialize audio components and LLM service
2. Display a ready prompt
3. Begin listening for voice input

Speak naturally and the assistant will respond both audibly and visually. Press `Ctrl+C` to exit at any time.

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

## Acknowledgments

- **LLM**: Powered by Google's Gemini models via OpenRouter
- **TTS**: pocket-tts for fast, local speech synthesis
- **VAD**: Silero VAD for reliable voice detection
- **Audio**: SoundDevice and SciPy for audio processing
