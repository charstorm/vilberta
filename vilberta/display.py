import time

from vilberta.sound_effects import play_line_print

RESET = "\033[0m"
CYAN = "\033[36m"
DIM = "\033[2m"
GRAY = "\033[90m"
YELLOW = "\033[33m"
RED = "\033[31m"


def print_speak(text: str) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            print(f"{CYAN}{stripped}{RESET}")


def print_text(text: str) -> None:
    print(flush=True)
    lines = text.splitlines()
    for line in lines:
        play_line_print()
        print(line)
        time.sleep(0.2)
    time.sleep(0.5)
    print(flush=True)


def print_transcript(text: str) -> None:
    print(f"{GRAY}» {text}{RESET}")


def print_status(message: str) -> None:
    print(f"{DIM}{message}{RESET}")


def print_error(message: str) -> None:
    print(f"{RED}ERROR: {message}{RESET}")


def print_vad(*, up: bool) -> None:
    if up:
        print(f"{YELLOW}▲ VAD{RESET}", flush=True)
    else:
        print(f"{YELLOW}▼ VAD{RESET}", flush=True)
