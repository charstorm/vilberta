from dataclasses import dataclass, field
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from vilberta.config import get_config

REDACTED_TOOL_PLACEHOLDER = "Tool output redacted due to size constraints"


@dataclass
class Turn:
    """Represents a conversation turn."""

    user_message: ChatCompletionMessageParam
    assistant_messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    tool_messages: list[ChatCompletionMessageParam] = field(default_factory=list)

    def to_messages(self) -> list[ChatCompletionMessageParam]:
        """Convert turn back to flat message list."""
        result: list[ChatCompletionMessageParam] = [self.user_message]
        result.extend(self.assistant_messages)
        result.extend(self.tool_messages)
        return result


@dataclass
class PruningResult:
    """Result of pruning operation with stats."""

    messages: list[ChatCompletionMessageParam]
    turns_pruned: int
    tools_redacted: int
    original_turn_count: int
    final_turn_count: int


def _get_role(msg: ChatCompletionMessageParam) -> str:
    """Extract role from message param safely."""
    role = msg.get("role", "")
    return role if isinstance(role, str) else ""


def _get_content(msg: ChatCompletionMessageParam) -> Any:
    """Extract content from message param safely."""
    return msg.get("content", "")


def _set_content(
    msg: ChatCompletionMessageParam, content: str
) -> ChatCompletionMessageParam:
    """Create a copy of message with new content."""
    new_msg = dict(msg)
    new_msg["content"] = content
    return cast(ChatCompletionMessageParam, new_msg)


def identify_turns(messages: list[ChatCompletionMessageParam]) -> list[Turn]:
    """Parse flat message list into turns.

    A turn starts with a user message and includes:
    - Assistant messages with tool_calls
    - Corresponding tool messages
    - Final assistant response without tool_calls
    """
    turns: list[Turn] = []
    current_turn: Turn | None = None

    for msg in messages:
        role = _get_role(msg)

        if role == "user":
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = Turn(user_message=msg)
        elif role == "assistant":
            if current_turn is None:
                continue
            current_turn.assistant_messages.append(msg)
        elif role == "tool":
            if current_turn is None:
                continue
            current_turn.tool_messages.append(msg)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


def redact_large_tool_outputs(
    turns: list[Turn],
    redact_threshold: int,
    protected_turn_count: int,
) -> tuple[list[Turn], int]:
    """Replace oversized tool messages with placeholder.

    Only redacts tool messages in turns older than protected_turn_count.
    Returns (redacted_turns, redaction_count).
    """
    redaction_count = 0

    if len(turns) <= protected_turn_count:
        return turns, redaction_count

    turns_to_redact = turns[:-protected_turn_count]
    protected_turns = turns[-protected_turn_count:]

    redacted_turns: list[Turn] = []
    for turn in turns_to_redact:
        redacted_tool_messages: list[ChatCompletionMessageParam] = []
        for tool_msg in turn.tool_messages:
            content = _get_content(tool_msg)
            if isinstance(content, str) and len(content) > redact_threshold:
                redacted_msg = _set_content(tool_msg, REDACTED_TOOL_PLACEHOLDER)
                redacted_tool_messages.append(redacted_msg)
                redaction_count += 1
            else:
                redacted_tool_messages.append(tool_msg)

        redacted_turn = Turn(
            user_message=turn.user_message,
            assistant_messages=turn.assistant_messages.copy(),
            tool_messages=redacted_tool_messages,
        )
        redacted_turns.append(redacted_turn)

    return redacted_turns + protected_turns, redaction_count


def prune_turns(
    messages: list[ChatCompletionMessageParam],
) -> PruningResult:
    """Main entry point for turn-based pruning with tool output redaction.

    Preserves system prompt at position 0.
    Only prunes when max_turns exceeded.
    Only redacts during actual pruning events.
    """
    cfg = get_config()

    if not messages:
        return PruningResult(
            messages=messages,
            turns_pruned=0,
            tools_redacted=0,
            original_turn_count=0,
            final_turn_count=0,
        )

    system_messages: list[ChatCompletionMessageParam] = []
    conversation_messages: list[ChatCompletionMessageParam] = []

    for msg in messages:
        if _get_role(msg) == "system":
            system_messages.append(msg)
        else:
            conversation_messages.append(msg)

    turns = identify_turns(conversation_messages)
    original_turn_count = len(turns)

    if len(turns) <= cfg.mcp_max_turns:
        return PruningResult(
            messages=messages,
            turns_pruned=0,
            tools_redacted=0,
            original_turn_count=original_turn_count,
            final_turn_count=original_turn_count,
        )

    turns_to_remove = len(turns) - cfg.mcp_pruned_turns
    pruned_turns = turns[turns_to_remove:]
    turns_pruned = turns_to_remove

    redacted_turns, tools_redacted = redact_large_tool_outputs(
        pruned_turns,
        cfg.mcp_tool_redact_threshold_chars,
        cfg.mcp_tool_redact_window,
    )

    result_messages: list[ChatCompletionMessageParam] = []
    result_messages.extend(system_messages)
    for turn in redacted_turns:
        result_messages.extend(turn.to_messages())

    return PruningResult(
        messages=result_messages,
        turns_pruned=turns_pruned,
        tools_redacted=tools_redacted,
        original_turn_count=original_turn_count,
        final_turn_count=len(redacted_turns),
    )


# history_pruner.py
