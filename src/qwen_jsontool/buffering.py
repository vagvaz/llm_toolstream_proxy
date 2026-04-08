"""Tool call buffering and reassembly for streaming SSE responses.

Handles the common failure modes of Qwen/MiniMax/vllm/ollama backends:
1. Tool call name arrives as "" or "null" or None in the first chunk
2. Tool call id arrives in a later chunk than the name
3. Arguments arrive before name/id are available
4. Multiple tool calls interleaved with incomplete metadata

Design:
- Accumulates tool call deltas by index until both `id` and `function.name`
  are present and non-empty.
- Emits a well-formed initial chunk (with id, type, name) only when complete.
- Arguments stream through immediately once the call has been "started".
- On stream end ([DONE]), flushes any buffered tool calls that are now complete,
  discards incomplete ones.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _sanitize_name(value: object) -> str | None:
    """Normalize various malformed name values to Optional[str].

    Qwen/vllm backends sometimes send:
    - null / None        -> None (not yet available)
    - "" (empty string)  -> None (continuation chunk, no name)
    - "null" (string)    -> None (litellm serializes Python None as "null")
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == "" or value == "null":
            return None
        return value
    return None


def _sanitize_id(value: object) -> str | None:
    """Normalize various malformed id values to Optional[str]."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            return None
        return value
    return None


def _is_valid_json(text: str) -> bool:
    """Check if text is parseable as JSON."""
    if not text:
        return True
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _try_repair_json(text: str) -> str | None:
    """Attempt basic JSON repair for truncated arguments.

    Common truncation patterns:
    - Missing closing brace: '{"key": "val"'
    - Missing closing bracket for arrays
    - Trailing comma: '{"key": "val",}'
    """
    if not text:
        return text
    if _is_valid_json(text):
        return text

    repaired = text.rstrip()
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    if open_braces > 0:
        repaired += "}" * open_braces
    if open_brackets > 0:
        repaired += "]" * open_brackets

    if repaired.endswith(","):
        repaired = repaired[:-1]

    if _is_valid_json(repaired):
        return repaired

    return None


@dataclass
class BufferedToolCall:
    """Accumulates a single tool call across multiple streaming chunks."""

    id: str | None = None
    name: str | None = None
    type: str = "function"
    arguments: str = ""
    started: bool = False

    @property
    def is_complete(self) -> bool:
        """Whether we have enough metadata to emit the initial chunk."""
        return bool(self.id) and bool(self.name)


class ToolCallBuffer:
    """Stateful buffer that reassembles fragmented streaming tool calls.

    Usage::

        buf = ToolCallBuffer()
        for chunk in sse_stream:
            events = buf.process_chunk(chunk)
            for event in events:
                yield format_sse(event)
        flush_events = buf.flush()
        for event in flush_events:
            yield format_sse(event)
    """

    def __init__(self, *, validate_json: bool = True) -> None:
        self.calls: dict[int, BufferedToolCall] = {}
        self.validate_json = validate_json

    def _get_or_create(self, index: int) -> BufferedToolCall:
        if index not in self.calls:
            self.calls[index] = BufferedToolCall()
        return self.calls[index]

    def process_delta(self, delta_tc: dict) -> list[dict]:
        """Process a single tool_call delta from a streaming chunk.

        Args:
            delta_tc: A single entry from the ``choices[i].delta.tool_calls`` array.
                Example: ``{"index": 0, "id": "call_abc", "type": "function",
                           "function": {"name": "read_file", "arguments": ""}}``

        Returns:
            List of tool_call deltas that are safe to emit now. May be empty
            if we're still buffering (name/id not yet available).
        """
        index: int = delta_tc.get("index", 0)
        func = delta_tc.get("function") or {}
        raw_id = delta_tc.get("id")
        raw_type = delta_tc.get("type") or "function"
        raw_name = func.get("name")
        raw_args = func.get("arguments", "")

        name = _sanitize_name(raw_name)
        tc_id = _sanitize_id(raw_id)
        args = raw_args if raw_args is not None else ""

        call = self._get_or_create(index)

        if tc_id and not call.id:
            call.id = tc_id
        if name and not call.name:
            call.name = name
        call.type = raw_type or call.type

        if args:
            call.arguments += args

        events: list[dict] = []

        if call.started:
            if args:
                events.append(
                    {
                        "index": index,
                        "function": {"arguments": args},
                    }
                )
        elif call.is_complete:
            call.started = True
            events.append(
                {
                    "index": index,
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.name,
                        "arguments": "",
                    },
                }
            )
            if call.arguments:
                accumulated = call.arguments
                events.append(
                    {
                        "index": index,
                        "function": {"arguments": accumulated},
                    }
                )
            else:
                pass
        else:
            logger.debug(
                "Buffering incomplete tool call at index %d "
                "(id=%r, name=%r, args_so_far=%d chars)",
                index,
                call.id,
                call.name,
                len(call.arguments),
            )

        return events

    def flush(self) -> list[dict]:
        """Flush any remaining buffered tool calls on stream end.

        Called when the SSE stream sends ``[DONE]``. For tool calls that have
        been started (initial chunk already emitted), nothing to do—the client
        will handle the rest. For tool calls that are still buffered (complete
        metadata arrived but we never got a chance to emit the start), emit
        them as a single complete chunk.

        Tool calls missing id or name are discarded with a warning.

        Returns:
            List of tool_call deltas to emit before the ``[DONE]`` sentinel.
        """
        events: list[dict] = []

        for index in sorted(self.calls.keys()):
            call = self.calls[index]

            if call.started:
                continue

            if not call.is_complete:
                logger.warning(
                    "Discarding incomplete tool call at index %d: "
                    "id=%r, name=%r, args_so_far=%d chars",
                    index,
                    call.id,
                    call.name,
                    len(call.arguments),
                )
                continue

            arguments = call.arguments
            if self.validate_json and arguments and not _is_valid_json(arguments):
                repaired = _try_repair_json(arguments)
                if repaired is not None:
                    arguments = repaired
                    logger.info(
                        "Repaired JSON arguments for tool call %r at index %d",
                        call.name,
                        index,
                    )
                else:
                    logger.warning(
                        "Tool call %r at index %d has invalid JSON arguments; "
                        "emitting as-is",
                        call.name,
                        index,
                    )

            events.append(
                {
                    "index": index,
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.name,
                        "arguments": arguments,
                    },
                }
            )

        return events

    def reset(self) -> None:
        """Reset the buffer for a new request."""
        self.calls.clear()
