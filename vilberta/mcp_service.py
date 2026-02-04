import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageFunctionToolCall,
)

from vilberta.config import get_config, Section, SectionType
from vilberta.text_section_splitter import StreamSection, StreamTextSectionSplitter
from vilberta.logger import get_logger
from vilberta.history_pruner import prune_turns

PROMPT_PATH = Path(__file__).parent / "prompts" / "system_mcp.md"

_INFORM_TOOL = {
    "type": "function",
    "function": {
        "name": "inform_user_about_toolcall",
        "description": "Inform the user that tool calls are being executed. When you need to use tools, you MUST call this tool FIRST, in parallel with other tool calls. Keep the message short (max 12 words) and conversational.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Short message to speak to the user (max 12 words)",
                    "maxLength": 100,
                }
            },
            "required": ["message"],
        },
    },
}

_TAG_SECTIONS = [
    StreamSection("[speak]", "[/speak]", inner_split_on=["\n"]),
    StreamSection("[text]", "[/text]", inner_split_on=None),
]

_TAG_OPEN = {
    "[speak]": SectionType.SPEAK,
    "[text]": SectionType.TEXT,
}

_TAG_STRINGS = {s.starting_tag for s in _TAG_SECTIONS} | {
    s.ending_tag for s in _TAG_SECTIONS if s.ending_tag
}


def _parse_response(text: str) -> list[Section]:
    splitter = StreamTextSectionSplitter(sections=_TAG_SECTIONS)
    parts = list(splitter.split(text))
    parts.extend(splitter.flush())

    sections: list[Section] = []
    for part in parts:
        if part.text in _TAG_STRINGS or part.section is None:
            continue
        section_type = _TAG_OPEN.get(part.section)
        if section_type is None:
            continue
        content = part.text.strip()
        if content:
            sections.append(Section(type=section_type, content=content))
    return sections


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


@dataclass
class InformUserEvent:
    """Event emitted when inform_user_about_toolcall is triggered"""

    message: str


@dataclass
class PruningEvent:
    """Event emitted when conversation history is pruned"""

    turns_pruned: int
    tools_redacted: int
    original_turn_count: int
    final_turn_count: int


ToolEventCallback = Callable[
    [ToolCallEvent | ToolResultEvent | InformUserEvent | PruningEvent], None
]


def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "You are a helpful voice assistant with access to tools."


def _build_text_user_message(transcript: str) -> dict[str, str]:
    return {"role": "user", "content": transcript}


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
        self.logger = get_logger("MCPService")

        cfg = get_config()
        api_key = os.environ.get(cfg.api_key_env, "")
        self.openai_client = AsyncOpenAI(base_url=cfg.api_base_url, api_key=api_key)
        self.model = cfg.toolcall_chat_llm_model_name

        # Metrics from last request
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_ttft = 0.0

    async def connect(self) -> None:
        """Connect to MCP server and discover tools."""
        self.logger.info(f"Connecting to MCP server: {self.server_url}")
        transport = await self.exit_stack.enter_async_context(
            streamable_http_client(self.server_url)
        )
        read_stream, write_stream, session_id = transport
        self.logger.debug(f"MCP session established: {session_id}")

        session = ClientSession(read_stream, write_stream)
        await self.exit_stack.enter_async_context(session)
        self.session = session

        await self.session.initialize()
        self.logger.info("MCP session initialized")

        response = await self.session.list_tools()
        self.available_tools = [
            _convert_mcp_tool_to_openai(tool) for tool in response.tools
        ]
        self.logger.info(f"Discovered {len(self.available_tools)} tools")
        for tool in self.available_tools:
            self.logger.debug(f"  - {tool['function']['name']}")

    async def process_message(
        self,
        transcript: str,
        event_callback: ToolEventCallback | None = None,
    ) -> tuple[list[Section], list[ToolCallEvent | ToolResultEvent]]:
        """Process user transcript through MCP tool calling loop.

        Args:
            transcript: User's transcribed speech
            event_callback: Optional callback for real-time tool event notifications

        Returns:
            Tuple of (response sections, tool events)
        """
        if self.session is None:
            raise RuntimeError("Not connected to MCP server")

        self.logger.debug("Triggering MCP LLM request")
        self._interrupted = False
        self.messages.append(
            cast(ChatCompletionMessageParam, _build_text_user_message(transcript))
        )

        events: list[ToolCallEvent | ToolResultEvent] = []
        sections: list[Section] = []
        turn_count = 0

        while not self._interrupted:
            turn_count += 1
            self.logger.debug(f"MCP turn {turn_count}: Requesting LLM response")

            # Get LLM response (non-streaming, async)
            completion = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=cast(Any, [self._build_system_message()] + self.messages),
                user="vilberta",
                tools=self.available_tools + [_INFORM_TOOL],  # type: ignore
                tool_choice="auto",
                parallel_tool_calls=True,
            )

            self.logger.info(f"LLM response received (turn {turn_count})")

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
                self.logger.debug(
                    f"Tokens: input={self.last_input_tokens}, output={self.last_output_tokens}, "
                    f"cache_read={self.last_cache_read_tokens}, cache_write={self.last_cache_write_tokens}"
                )

            # Check for tool calls
            if not assistant_message.tool_calls:
                # Final response - parse into sections
                content = assistant_message.content or ""
                sections = self._parse_response(content)
                self.logger.info(f"Final response: {len(sections)} sections parsed")
                break

            # Execute tool calls
            tool_call_count = len(assistant_message.tool_calls)
            self.logger.info(f"LLM requested {tool_call_count} tool calls")

            if self._interrupted:
                self.logger.debug("Tool execution interrupted before start")
                break

            # Separate inform tool from other tools
            inform_call = None
            other_calls = []
            for tool_call in assistant_message.tool_calls:
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue
                if tool_call.function.name == "inform_user_about_toolcall":
                    inform_call = tool_call
                else:
                    other_calls.append(tool_call)

            # Handle inform tool first - emit event for TTS
            if inform_call:
                tool_args = (
                    json.loads(inform_call.function.arguments)
                    if inform_call.function.arguments
                    else {}
                )
                message = tool_args.get("message", "Working on that for you.")
                self.logger.info(f"Inform user: {message}")
                inform_event = InformUserEvent(message=message)
                events.append(inform_event)  # type: ignore
                if event_callback:
                    event_callback(inform_event)

            # Execute other tool calls in parallel
            async def _execute_single_tool(
                tool_call: ChatCompletionMessageFunctionToolCall,
            ) -> tuple[ChatCompletionMessageFunctionToolCall, str, bool]:
                tool_name = tool_call.function.name
                tool_args = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )

                self.logger.info(f"Calling tool: {tool_name}({tool_args})")
                call_event = ToolCallEvent(tool_name=tool_name, arguments=tool_args)
                events.append(call_event)
                if event_callback:
                    event_callback(call_event)

                result_str = await self._call_tool(tool_name, tool_args)
                success = not result_str.startswith("Error")

                self.logger.info(f"Tool response: {tool_name} success={success}")
                self.logger.debug(f"Tool result: {result_str[:200]}...")

                result_event = ToolResultEvent(
                    tool_name=tool_name, success=success, result=result_str
                )
                events.append(result_event)
                if event_callback:
                    event_callback(result_event)

                return tool_call, result_str, success

            # Run all tool calls concurrently
            if other_calls:
                tool_tasks = [_execute_single_tool(tc) for tc in other_calls]
                results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                # Add results to messages in order
                for result in results:
                    if isinstance(result, BaseException):
                        error_msg = f"Tool execution failed: {result}"
                        self.logger.error(error_msg)
                        # Create a tool response for the error to maintain 1:1 correspondence
                        # Find the corresponding tool call from other_calls
                        tool_call_index = results.index(result)
                        if tool_call_index < len(other_calls):
                            failed_tool_call = other_calls[tool_call_index]
                            self.messages.append(
                                cast(
                                    ChatCompletionMessageParam,
                                    {
                                        "role": "tool",
                                        "tool_call_id": failed_tool_call.id,
                                        "name": failed_tool_call.function.name,
                                        "content": f"Error: {error_msg}",
                                    },
                                )
                            )
                        continue
                    # result is a tuple[ChatCompletionMessageFunctionToolCall, str, bool]
                    result_tuple = result
                    self.messages.append(
                        cast(
                            ChatCompletionMessageParam,
                            {
                                "role": "tool",
                                "tool_call_id": result_tuple[0].id,
                                "name": result_tuple[0].function.name,
                                "content": result_tuple[1],
                            },
                        )
                    )

        self._prune_history_if_needed(event_callback)
        self.logger.debug(f"MCP processing complete after {turn_count} turns")

        return sections, events

    async def _call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Call a single tool and return result as string.

        Uses a 30-second timeout to prevent hanging on unresponsive MCP servers.
        """
        if self.session is None:
            return "Error: Not connected to MCP server"

        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, tool_args),
                timeout=30.0,
            )
            tool_result_content = [
                str(getattr(content_item, "text", ""))
                for content_item in result.content
                if hasattr(content_item, "text")
            ]
            return "\n".join(tool_result_content)
        except asyncio.TimeoutError:
            return f"Error calling tool {tool_name}: timeout after 30 seconds"
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    def _build_system_message(self) -> dict[str, str]:
        """Build system message with tool descriptions."""
        tool_descriptions = "\n".join(
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in self.available_tools
        )
        content = (
            f"{self.system_prompt}\n\n"
            f"IMPORTANT: You have access to external tools listed below. "
            f"When the user asks a question that can be answered by a tool, "
            f"you MUST use the tool calling API - do NOT provide the answer yourself. "
            f"Only respond directly without tools for greetings, opinions, or conversation.\n\n"
            f"Available tools:\n{tool_descriptions}"
        )
        return {"role": "system", "content": content}

    def _parse_response(self, content: str) -> list[Section]:
        """Parse response into sections."""
        return _parse_response(content)

    def _prune_history_if_needed(
        self, event_callback: ToolEventCallback | None = None
    ) -> None:
        """Turn-based pruning with tool output redaction."""
        result = prune_turns(self.messages)

        if result.turns_pruned > 0:
            self.messages = result.messages
            self.logger.info(
                f"History pruned: {result.turns_pruned} turns removed, "
                f"{result.tools_redacted} tools redacted, "
                f"turns {result.original_turn_count} -> {result.final_turn_count}"
            )

            if event_callback:
                event = PruningEvent(
                    turns_pruned=result.turns_pruned,
                    tools_redacted=result.tools_redacted,
                    original_turn_count=result.original_turn_count,
                    final_turn_count=result.final_turn_count,
                )
                event_callback(event)

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

    def get_unique_words(self, max_words: int = 100) -> list[str]:
        """Extract unique words from user and assistant messages for ASR context.

        Only considers messages with role 'user' or 'assistant' (excludes tool calls).
        Returns sorted list of unique words, limited to max_words.
        """
        import re

        words: set[str] = set()

        for msg in self.messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Extract words (alphanumeric, 2+ chars)
            found = re.findall(r"\b[a-zA-Z]{2,}\b", content.lower())
            words.update(found)

        # Sort alphabetically and limit
        sorted_words = sorted(words)
        return sorted_words[:max_words]

    async def cleanup(self) -> None:
        """Close MCP connection."""
        self.logger.info("Closing MCP connection")
        await self.exit_stack.aclose()
        self.session = None
        self.logger.info("MCP connection closed")


# mcp_service.py
