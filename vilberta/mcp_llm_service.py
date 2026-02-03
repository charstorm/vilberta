import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

from vilberta.config import get_config, Section
from vilberta.llm_service import BaseLLMService
from vilberta.mcp_service import MCPService, ToolCallEvent, ToolResultEvent
from vilberta.display import print_status
from vilberta.sound_effects import (
    play_tool_call_start,
    play_tool_call_success,
    play_tool_call_error,
)


@dataclass
class MCPProcessResult:
    """Result from processing a message through MCP."""

    sections: list[Section]
    events: list[ToolCallEvent | ToolResultEvent]
    full_response: str


class MCPAwareLLMService(BaseLLMService):
    """MCP-aware LLM service that wraps MCPService for sync usage.

    Uses a single background event loop for all async operations to avoid
    context manager issues when entering/exiting across different loops.
    """

    def __init__(self) -> None:
        cfg = get_config()
        if not cfg.mcp_server_url:
            raise ValueError("MCP server URL not configured")

        self.mcp_service = MCPService(cfg.mcp_server_url)

        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_cache_read_tokens = 0
        self._last_cache_write_tokens = 0
        self._last_latency_s = 0.0

        # Background event loop for all async operations
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()

    def _run_loop(self) -> None:
        """Run the event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._loop_ready.set()
        self._loop.run_forever()

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine in the background loop and return the result."""
        if self._loop is None:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

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

    def connect(self) -> None:
        """Connect to MCP server using a background event loop."""
        # Start the background event loop
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait(timeout=5.0)

        if self._loop is None:
            raise RuntimeError("Failed to start event loop")

        # Connect to MCP server using the background loop
        self._run_async(self.mcp_service.connect())
        print_status("Connected to MCP server")

    def cleanup(self) -> None:
        """Cleanup MCP connection using the same event loop."""
        if self._loop is not None:
            # Run cleanup in the same loop
            self._run_async(self.mcp_service.cleanup())

            # Stop the loop
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

    def get_response(self, audio_b64: str) -> tuple[list[Section], str]:
        """Process message through MCP and return sections."""
        # Track active tool calls for busy indicator
        active_tool: str | None = None
        stop_spinner = threading.Event()

        def _spinner_worker() -> None:
            """Show busy indicator while tool is active."""
            symbols = ["◐", "◓", "◑", "◒"]
            idx = 0
            while not stop_spinner.is_set():
                if active_tool:
                    symbol = symbols[idx % len(symbols)]
                    print_status(f"{symbol} Running {active_tool}...")
                    idx += 1
                time.sleep(0.2)

        def _event_callback(event: ToolCallEvent | ToolResultEvent) -> None:
            """Handle tool events in real-time with status updates and sounds."""
            nonlocal active_tool

            if isinstance(event, ToolCallEvent):
                active_tool = event.tool_name
                play_tool_call_start()
                args_str = self._format_tool_args(event.arguments)
                print_status(f"⚡ Calling {event.tool_name}({args_str})")
            elif isinstance(event, ToolResultEvent):
                active_tool = None
                if event.success:
                    play_tool_call_success()
                    result = (
                        event.result[:24] + "..."
                        if len(event.result) > 24
                        else event.result
                    )
                    print_status(f"✓ {event.tool_name} returned: {result}")
                else:
                    play_tool_call_error()
                    error = (
                        event.result[:24] + "..."
                        if len(event.result) > 24
                        else event.result
                    )
                    print_status(f"✗ {event.tool_name} failed: {error}")

        # Start spinner thread
        spinner_thread = threading.Thread(target=_spinner_worker, daemon=True)
        spinner_thread.start()

        try:
            sections, events = self._run_async(
                self.mcp_service.process_message(audio_b64, _event_callback)
            )
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=0.5)

        # Copy metrics
        self._last_input_tokens = self.mcp_service.last_input_tokens
        self._last_output_tokens = self.mcp_service.last_output_tokens
        self._last_cache_read_tokens = self.mcp_service.last_cache_read_tokens
        self._last_cache_write_tokens = self.mcp_service.last_cache_write_tokens
        self._last_latency_s = 0.0  # Not measured for MCP

        # Reconstruct full response
        full_response_parts = []
        for section in sections:
            full_response_parts.append(
                f"[{section.type.value}]{section.content}[/{section.type.value}]"
            )
        full_response = "\n".join(full_response_parts)

        return sections, full_response

    def _format_tool_args(self, args: dict[str, Any]) -> str:
        """Format tool arguments, truncating to max 24 chars total."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        if len(args_str) > 24:
            return args_str[:21] + "..."
        return args_str

    def mark_interrupted(self) -> None:
        """Mark that user interrupted."""
        self.mcp_service.mark_interrupted()


# mcp_llm_service.py
