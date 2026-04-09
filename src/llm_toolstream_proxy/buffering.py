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

Argument handling:
  Arguments are string fragments per the OpenAI streaming spec. Each delta's
  ``function.arguments`` is a substring of the final JSON string. We concatenate
  them with ``+=`` which is the correct merging policy — they are NOT separate
  JSON objects to be merged, they are text fragments to be appended.

  For calls that were **buffered** (name/id not yet available when args arrived),
  we replay all accumulated arguments as a single delta when the call starts.
  For calls that started normally, individual deltas pass through as-is.

  At finish time, we validate accumulated arguments and attempt JSON repair
  for buffered-only calls. For started calls (arguments already streamed to
  the client), we cannot repair because the client has already received the
  raw fragments — we can only validate and log a warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger


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
    - Trailing comma before closing brace: '{"key": "val",}'
    """
    if not text:
        return text
    if _is_valid_json(text):
        return text

    repaired = text.rstrip()

    if repaired.endswith(","):
        repaired = repaired[:-1]
    elif repaired.endswith(",}"):
        repaired = repaired[:-2] + "}"
    elif repaired.endswith(",]"):
        repaired = repaired[:-2] + "]"

    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    repaired += "]" * open_brackets
    repaired += "}" * open_braces

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
    finished: bool = False

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

        logger.debug(
            "Processing tool call delta: index=%d, raw_id=%r, sanitized_name=%r, "
            "args_len=%d, raw_name=%r",
            index,
            raw_id,
            name,
            len(args),
            raw_name,
        )

        call = self._get_or_create(index)

        if tc_id and not call.id:
            logger.debug("Tool call index %d: received id=%r", index, tc_id)
            call.id = tc_id
        if name and not call.name:
            logger.info(
                "Tool call index %d: received name=%r (was %r)",
                index,
                name,
                call.name,
            )
            call.name = name
        call.type = raw_type or call.type

        if args:
            call.arguments += args

        events: list[dict] = []

        if call.started:
            if args:
                logger.debug(
                    "Tool call %r index %d: streaming argument delta (%d bytes)",
                    call.name,
                    index,
                    len(args),
                )
                events.append(
                    {
                        "index": index,
                        "function": {"arguments": args},
                    }
                )
        elif call.is_complete:
            call.started = True
            logger.info(
                "Tool call %r index %d: emitting start (id=%r, buffered_args=%d bytes)",
                call.name,
                index,
                call.id,
                len(call.arguments),
            )
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
                logger.debug(
                    "Tool call %r index %d: emitting accumulated arguments (%d bytes)",
                    call.name,
                    index,
                    len(accumulated),
                )
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

    def finish_call(self, index: int) -> None:
        """Mark a tool call as finished and validate its accumulated arguments.

        Called when the stream sends a finish_reason of 'tool_calls' or 'stop'
        for a choice. Validates accumulated arguments and logs warnings for
        invalid JSON.

        For calls that were **started** (argument deltas already streamed to
        the client), we cannot repair the arguments because the client has
        already received the raw fragments. We can only validate and log.

        For calls that were **never started** (still buffered), repair will
        happen in ``flush()`` which emits the complete call.
        """
        call = self._get_or_create(index)
        call.finished = True

        if not call.arguments:
            logger.debug(
                "Tool call %r index %d: finished with empty arguments",
                call.name,
                index,
            )
            return

        if not self.validate_json:
            return

        if _is_valid_json(call.arguments):
            logger.debug(
                "Tool call %r index %d: finished, arguments valid JSON (%d bytes)",
                call.name,
                index,
                len(call.arguments),
            )
            return

        if call.started:
            logger.warning(
                "Tool call %r index %d: finished with INVALID JSON arguments "
                "(%d bytes). Arguments were already streamed to client — "
                "cannot repair. Accumulated: %s",
                call.name,
                index,
                len(call.arguments),
                call.arguments[:200],
            )
        else:
            logger.warning(
                "Tool call %r index %d: finished with invalid JSON arguments "
                "(%d bytes). Will attempt repair in flush().",
                call.name,
                index,
                len(call.arguments),
            )

    def flush(self) -> list[dict]:
        """Flush any remaining buffered tool calls on stream end.

        Called when the SSE stream sends ``[DONE]``. Handles three cases:

        1. **Started but not finished**: Validate arguments. Cannot repair
           (already streamed to client). Log warning.
        2. **Buffered and complete**: Emit the full tool call with repaired
           arguments if needed.
        3. **Still incomplete (missing id or name)**: Discard with a warning.

        Returns:
            List of tool_call deltas to emit before the ``[DONE]`` sentinel.
        """
        events: list[dict] = []

        for index in sorted(self.calls.keys()):
            call = self.calls[index]

            if call.started:
                if not call.finished:
                    call.finished = True
                    if (
                        call.arguments
                        and self.validate_json
                        and not _is_valid_json(call.arguments)
                    ):
                        logger.warning(
                            "Flush: tool call %r index %d has invalid JSON "
                            "arguments (%d bytes) but they were already streamed. "
                            "Cannot repair.",
                            call.name,
                            index,
                            len(call.arguments),
                        )
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

            call.finished = True
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
