import asyncio
from collections.abc import Generator
from dataclasses import dataclass

from vilberta.config import get_config
from vilberta.llm_service import BaseLLMService
from vilberta.mcp_service import MCPService, ToolCallEvent, ToolResultEvent
from vilberta.response_parser import Section
from vilberta.display import print_status


@dataclass
class MCPProcessResult:
    """Result from processing a message through MCP."""

    sections: list[Section]
    events: list[ToolCallEvent | ToolResultEvent]
    full_response: str


class MCPAwareLLMService(BaseLLMService):
    """MCP-aware LLM service that wraps MCPService for sync usage.

    Converts async MCP operations to sync generator interface.
    Non-streaming: yields all sections at once after complete processing.
    """

    def __init__(self) -> None:
        cfg = get_config()
        if not cfg.mcp_server_url:
            raise ValueError("MCP server URL not configured")

        self.mcp_service = MCPService(cfg.mcp_server_url)
        self._loop: asyncio.AbstractEventLoop | None = None

        # Metrics from last request
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_cache_read_tokens = 0
        self._last_cache_write_tokens = 0
        self._last_ttft = 0.0

        # Track full response for return
        self._last_full_response = ""

        # Track if currently processing
        self._processing = False

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
    def last_ttft(self) -> float:
        return self._last_ttft

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing event loop or create new one."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def connect(self) -> None:
        """Connect to MCP server (sync wrapper)."""
        loop = self._get_or_create_loop()
        loop.run_until_complete(self.mcp_service.connect())
        print_status("Connected to MCP server")

    def cleanup(self) -> None:
        """Cleanup MCP connection (sync wrapper)."""
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self.mcp_service.cleanup())
            self._loop.close()

    def _process_sync(self, audio_b64: str) -> MCPProcessResult:
        """Process message through MCP service (async in sync context)."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._process_async(audio_b64))

    async def _process_async(self, audio_b64: str) -> MCPProcessResult:
        """Async processing wrapper."""
        sections, events = await self.mcp_service.process_message(audio_b64)

        # Reconstruct full response from sections
        full_response_parts = []
        for section in sections:
            if section.type.value == "speak":
                full_response_parts.append(f"[speak]{section.content}[/speak]")
            elif section.type.value == "text":
                full_response_parts.append(f"[text]{section.content}[/text]")
            elif section.type.value == "transcript":
                full_response_parts.append(
                    f"[transcript]{section.content}[/transcript]"
                )

        full_response = "\n".join(full_response_parts)

        return MCPProcessResult(
            sections=sections,
            events=events,
            full_response=full_response,
        )

    def _display_tool_events(
        self, events: list[ToolCallEvent | ToolResultEvent]
    ) -> None:
        """Display tool calls and results as UI events."""
        for event in events:
            if isinstance(event, ToolCallEvent):
                args_str = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
                print_status(f"[Tool Call] {event.tool_name}({args_str})")
            elif isinstance(event, ToolResultEvent):
                status = "✓" if event.success else "✗"
                # Truncate long results for display
                result = (
                    event.result[:100] + "..."
                    if len(event.result) > 100
                    else event.result
                )
                print_status(f"[Tool Result] {status} {event.tool_name}: {result}")

    def stream_response(self, audio_b64: str) -> Generator[Section, None, str]:
        """Process message and yield sections.

        For MCP mode, this is non-streaming: all sections yielded at once.
        """
        self._processing = True

        # Reset metrics
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_cache_read_tokens = 0
        self._last_cache_write_tokens = 0
        self._last_ttft = 0.0

        result = self._process_sync(audio_b64)

        if not self._processing:
            # Was interrupted
            self.mcp_service.mark_interrupted()
            return ""

        # Copy metrics from MCP service
        self._last_input_tokens = self.mcp_service.last_input_tokens
        self._last_output_tokens = self.mcp_service.last_output_tokens
        self._last_cache_read_tokens = self.mcp_service.last_cache_read_tokens
        self._last_cache_write_tokens = self.mcp_service.last_cache_write_tokens
        # MCP is non-streaming so TTFT equals total time, but we set it to 0
        # to indicate no streaming measurement
        self._last_ttft = 0.0

        # Display tool events
        self._display_tool_events(result.events)

        # Yield all sections (non-streaming)
        for section in result.sections:
            yield section

        self._last_full_response = result.full_response
        self._processing = False

        return result.full_response

    def mark_interrupted(self) -> None:
        """Mark that user interrupted."""
        self._processing = False
        self.mcp_service.mark_interrupted()


# mcp_llm_service.py
