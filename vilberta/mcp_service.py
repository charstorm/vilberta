import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageFunctionToolCall,
)

from vilberta.config import get_config
from vilberta.response_parser import Section

PROMPT_PATH = Path(__file__).parent / "prompts" / "system_mcp.md"


@dataclass
class ToolCallEvent:
    """Event emitted when a tool is called"""

    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultEvent:
    """Event emitted when a tool call completes"""

    tool_name: str
    success: bool
    result: str


def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "You are a helpful voice assistant with access to tools."


def _build_audio_user_message(audio_b64: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
            {"type": "text", "text": "Respond to the user."},
        ],
    }


def _convert_mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", []),
            },
        },
    }


class MCPService:
    """MCP-based LLM service with tool calling support.

    Non-streaming: buffers entire response including tool results.
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.messages: list[ChatCompletionMessageParam] = []
        self.available_tools: list[dict[str, Any]] = []
        self.system_prompt = _load_system_prompt()
        self._interrupted = False

        cfg = get_config()
        api_key = os.environ.get(cfg.api_key_env, "")
        self.openai_client = OpenAI(base_url=cfg.api_base_url, api_key=api_key)
        self.model = cfg.model_name

        # Metrics from last request
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_ttft = 0.0

    async def connect(self) -> None:
        """Connect to MCP server and discover tools."""
        transport = await self.exit_stack.enter_async_context(
            streamable_http_client(self.server_url)
        )
        read_stream, write_stream, session_id = transport

        session = ClientSession(read_stream, write_stream)
        await self.exit_stack.enter_async_context(session)
        self.session = session

        await self.session.initialize()

        response = await self.session.list_tools()
        self.available_tools = [
            _convert_mcp_tool_to_openai(tool) for tool in response.tools
        ]

    async def process_message(
        self, audio_b64: str
    ) -> tuple[list[Section], list[ToolCallEvent | ToolResultEvent]]:
        """Process user audio message through MCP tool calling loop.

        Returns:
            Tuple of (response sections, tool events)
        """
        if self.session is None:
            raise RuntimeError("Not connected to MCP server")

        self._interrupted = False
        self.messages.append(
            cast(ChatCompletionMessageParam, _build_audio_user_message(audio_b64))
        )

        events: list[ToolCallEvent | ToolResultEvent] = []
        sections: list[Section] = []

        while not self._interrupted:
            # Get LLM response (non-streaming)
            completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=cast(Any, [self._build_system_message()] + self.messages),
                tools=self.available_tools,  # type: ignore
            )

            assistant_message = completion.choices[0].message
            self.messages.append(
                cast(ChatCompletionMessageParam, assistant_message.model_dump())
            )

            # Update metrics
            if completion.usage:
                self.last_input_tokens = completion.usage.prompt_tokens or 0
                self.last_output_tokens = completion.usage.completion_tokens or 0
                detail = getattr(completion.usage, "prompt_tokens_details", None)
                if detail:
                    self.last_cache_read_tokens = (
                        getattr(detail, "cached_tokens", 0) or 0
                    )
                    self.last_cache_write_tokens = (
                        getattr(detail, "audio_tokens", 0) or 0
                    )

            # Check for tool calls
            if not assistant_message.tool_calls:
                # Final response - parse into sections
                content = assistant_message.content or ""
                sections = self._parse_response(content)
                break

            # Execute tool calls
            for tool_call in assistant_message.tool_calls:
                if self._interrupted:
                    break

                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue

                tool_name = tool_call.function.name
                tool_args = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )

                events.append(ToolCallEvent(tool_name=tool_name, arguments=tool_args))

                result_str = await self._call_tool(tool_name, tool_args)
                success = not result_str.startswith("Error")

                events.append(
                    ToolResultEvent(
                        tool_name=tool_name, success=success, result=result_str
                    )
                )

                self.messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result_str,
                        },
                    )
                )

        self._trim_history_if_needed()

        return sections, events

    async def _call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Call a single tool and return result as string."""
        if self.session is None:
            return "Error: Not connected to MCP server"

        try:
            result = await self.session.call_tool(tool_name, tool_args)
            tool_result_content = [
                str(getattr(content_item, "text", ""))
                for content_item in result.content
                if hasattr(content_item, "text")
            ]
            return "\n".join(tool_result_content)
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    def _build_system_message(self) -> dict[str, str]:
        """Build system message with tool descriptions."""
        tool_descriptions = "\n".join(
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in self.available_tools
        )
        content = f"{self.system_prompt}\n\nAvailable tools:\n{tool_descriptions}"
        return {"role": "system", "content": content}

    def _parse_response(self, content: str) -> list[Section]:
        """Parse response into sections."""
        from vilberta.response_parser import StreamingParser

        parser = StreamingParser()
        sections = []

        for char in content:
            for section in parser.feed(char):
                sections.append(section)

        for section in parser.flush():
            sections.append(section)

        return sections

    def _trim_history_if_needed(self) -> None:
        """Trim conversation history if it exceeds threshold."""
        cfg = get_config()

        # Count non-system messages
        non_system = [m for m in self.messages if m.get("role") != "system"]

        if len(non_system) <= cfg.max_hist_threshold_size:
            return

        # Keep most recent messages
        keep_count = cfg.hist_reset_size
        self.messages = self.messages[-keep_count:]

    def mark_interrupted(self) -> None:
        """Mark that the user interrupted."""
        self._interrupted = True
        if not self.messages:
            return
        last = self.messages[-1]
        if last.get("role") == "assistant":
            content = last.get("content")
            if isinstance(content, str):
                last["content"] = content + "\n[interrupted by user]"

    async def cleanup(self) -> None:
        """Close MCP connection."""
        await self.exit_stack.aclose()
        self.session = None


# mcp_service.py
