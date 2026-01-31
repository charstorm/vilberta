# AGENTS.md

## Project Overview
Vilberta is an interactive voice assistant that provides real-time voice interaction with LLMs. It features speech-to-text, streaming text-to-speech, interruption handling, and multimodal output. The codebase follows Python best practices with a focus on async/await patterns, type hints, and modular architecture.

## Build, Test, and Lint Commands

### Linting and Formatting
```bash
# Lint the entire codebase
ruff check .

# Auto-fix linting issues where possible
ruff check --fix .

# Format code (assuming ruff handles formatting)
ruff format .

# Type checking with mypy (if configured)
mypy --strict vilberta/  # ignore issues with libraries
```

### Build and Development
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install from requirements
pip install -r requirements.txt

# Run the main application
python -m vilberta

# Run with specific options for debugging
python -m vilberta --voice alba --speed 1.0
```

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

