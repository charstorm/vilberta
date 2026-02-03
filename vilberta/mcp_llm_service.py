from dataclasses import dataclass

from vilberta.config import get_config, Section
from vilberta.llm_service import BaseLLMService
from vilberta.mcp_service import MCPService, ToolCallEvent, ToolResultEvent
from vilberta.display import print_status


@dataclass
class MCPProcessResult:
    """Result from processing a message through MCP."""

    sections: list[Section]
    events: list[ToolCallEvent | ToolResultEvent]
    full_response: str


class MCPAwareLLMService(BaseLLMService):
    """MCP-aware LLM service that wraps MCPService for sync usage."""

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
        """Connect to MCP server using asyncio run."""
        import asyncio

        asyncio.run(self.mcp_service.connect())
        print_status("Connected to MCP server")

    def cleanup(self) -> None:
        """Cleanup MCP connection using asyncio run."""
        import asyncio

        asyncio.run(self.mcp_service.cleanup())

    def get_response(self, audio_b64: str) -> tuple[list[Section], str]:
        """Process message through MCP and return sections."""
        import asyncio

        sections, events = asyncio.run(self.mcp_service.process_message(audio_b64))

        # Copy metrics
        self._last_input_tokens = self.mcp_service.last_input_tokens
        self._last_output_tokens = self.mcp_service.last_output_tokens
        self._last_cache_read_tokens = self.mcp_service.last_cache_read_tokens
        self._last_cache_write_tokens = self.mcp_service.last_cache_write_tokens
        self._last_latency_s = 0.0  # Not measured for MCP

        # Display tool events
        self._display_tool_events(events)

        # Reconstruct full response
        full_response_parts = []
        for section in sections:
            full_response_parts.append(f"[{section.type.value}]{section.content}[/{section.type.value}]")
        full_response = "\n".join(full_response_parts)

        return sections, full_response

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
                result = event.result[:100] + "..." if len(event.result) > 100 else event.result
                print_status(f"[Tool Result] {status} {event.tool_name}: {result}")

    def mark_interrupted(self) -> None:
        """Mark that user interrupted."""
        self.mcp_service.mark_interrupted()


# mcp_llm_service.py
