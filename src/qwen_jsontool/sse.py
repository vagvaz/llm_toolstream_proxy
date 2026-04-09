"""SSE stream parsing, buffering, and re-emission.

This module provides the core streaming transformation that sits between
a litellm upstream and an opencode downstream client. It:

1. Parses OpenAI-format SSE chunks from the upstream.
2. Routes tool_call deltas through a ToolCallBuffer for reassembly.
3. Re-encodes buffered and passthrough events into proper SSE format.
4. Forwards non-tool-call chunks (content, reasoning, role, finish) unchanged.

The key challenge: a single SSE event may contain BOTH non-tool delta fields
(role, content, reasoning_content) AND tool_calls. We must split such events
so that the non-tool delta fields pass through immediately while the tool_calls
are buffered until their metadata is complete.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import AsyncIterator

from .buffering import ToolCallBuffer

logger = logging.getLogger(__name__)

DONE_SENTINEL = "[DONE]"


def parse_sse_line(line: str) -> dict | None:
    """Parse a single ``data: ...`` SSE line into a dict.

    Returns None for:
    - Comment lines (starting with ``:``)
    - Empty lines
    - The ``[DONE]`` sentinel (returns a special marker)
    - Lines that don't start with ``data:``
    - Invalid JSON
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None

    payload = stripped[len("data:") :].strip()
    if payload == DONE_SENTINEL:
        return {"__done__": True}

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Failed to parse SSE JSON: %s", payload[:200])
        return None


def encode_sse_event(data: dict) -> str:
    """Encode a dict back into an SSE ``data:`` line with trailing newlines."""
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def encode_sse_done() -> str:
    """Encode the [DONE] sentinel as an SSE event."""
    return f"data: {DONE_SENTINEL}\n\n"


def _extract_tool_calls(chunk: dict) -> tuple[list[dict], dict]:
    """Extract tool_calls from a streaming chunk and return a cleaned chunk
    without tool_calls.

    Returns:
        (tool_call_deltas, chunk_without_tool_calls)
        The chunk_without_tool_calls has tool_calls removed from all choices'
        deltas, but preserves all other fields (role, content, reasoning, etc.)
    """
    cleaned = deepcopy(chunk)
    tool_call_deltas: list[dict] = []

    choices = cleaned.get("choices", [])
    for choice in choices:
        delta = choice.get("delta", {})
        if delta and "tool_calls" in delta:
            tcs = delta.pop("tool_calls")
            if tcs:
                tool_call_deltas.extend(tcs)

    return tool_call_deltas, cleaned


def _inject_tool_call_events(
    chunk: dict,
    choice_index: int,
    tool_call_deltas: list[dict],
) -> list[dict]:
    """Create properly formed SSE chunk dicts with tool_call deltas
    injected into a specific choice.

    Each delta becomes its own chunk so that downstream consumers
    process them one at a time, matching the original streaming behavior.
    """
    events: list[dict] = []
    for tc_delta in tool_call_deltas:
        new_chunk = deepcopy(chunk)
        choices = new_chunk.get("choices", [])
        if not choices:
            new_chunk["choices"] = [
                {
                    "index": choice_index,
                    "delta": {"tool_calls": [tc_delta]},
                    "finish_reason": None,
                }
            ]
        else:
            for ch in choices:
                if ch.get("index") == choice_index or len(choices) == 1:
                    ch["delta"] = ch.get("delta", {})
                    ch["delta"]["tool_calls"] = [tc_delta]
                    if "finish_reason" not in ch:
                        ch["finish_reason"] = None
                    break
        events.append(new_chunk)
    return events


class SSETransformer:
    """Stateful transformer that processes an SSE stream and reassembles
    fragmented tool calls.

    Usage::

        transformer = SSETransformer()
        async for raw_chunk in upstream_sse_stream():
            for sse_line in transformer.process_raw(raw_chunk):
                yield sse_line
        for sse_line in transformer.flush():
            yield sse_line
    """

    def __init__(self, *, validate_json: bool = True) -> None:
        self.buffer = ToolCallBuffer(validate_json=validate_json)
        self._done = False

    def process_raw(self, raw_line: str) -> list[str]:
        """Process a raw SSE line from upstream.

        Args:
            raw_line: A single line from the SSE stream (may include
                the ``data:`` prefix and trailing newlines).

        Returns:
            List of SSE-formatted strings (``data: ...\\n\\n``) to emit
            to the downstream client. May be empty if tool calls are still
            being buffered.
        """
        if self._done:
            return []

        parsed = parse_sse_line(raw_line)
        if parsed is None:
            return []

        if parsed.get("__done__"):
            self._done = True
            logger.info("SSE stream ended, flushing remaining buffered tool calls")
            return self._flush_and_done()

        tool_call_deltas, cleaned = _extract_tool_calls(parsed)
        output_lines: list[str] = []

        non_empty_delta = False
        choices = parsed.get("choices", [])
        for ch in choices:
            delta = ch.get("delta", {})
            if delta:
                remaining = {k: v for k, v in delta.items() if k != "tool_calls"}
                if any(
                    v is not None and v != "" and v != [] for k, v in remaining.items()
                ):
                    non_empty_delta = True

        if tool_call_deltas:
            logger.debug(
                "SSE chunk has %d tool_call deltas, non_empty_delta=%s",
                len(tool_call_deltas),
                non_empty_delta,
            )
            for choice in choices:
                choice_index = choice.get("index", 0)
                buffered_events: list[dict] = []
                for tc_delta in tool_call_deltas:
                    buffered_events.extend(self.buffer.process_delta(tc_delta))

                if non_empty_delta:
                    output_lines.append(encode_sse_event(cleaned))

                if buffered_events:
                    logger.debug(
                        "Emitting %d buffered tool call events for choice %d",
                        len(buffered_events),
                        choice_index,
                    )
                    for event_chunk in _inject_tool_call_events(
                        parsed, choice_index, buffered_events
                    ):
                        output_lines.append(encode_sse_event(event_chunk))
        else:
            output_lines.append(encode_sse_event(parsed))

        return output_lines

    def flush(self) -> list[str]:
        """Public flush: emit any remaining buffered tool calls and [DONE].

        Called when the upstream SSE stream sends [DONE] or when the
        connection closes. Also accessible for testing.
        """
        return self._flush_and_done()

    def _flush_and_done(self) -> list[str]:
        """Emit any remaining buffered tool calls followed by the [DONE] sentinel."""
        output_lines: list[str] = []

        flush_events = self.buffer.flush()
        if flush_events:
            logger.info(
                "Flushing %d remaining tool call(s) on stream end",
                len(flush_events),
            )
            base_chunk: dict = {
                "id": "chatcmpl-tool-buffer",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            }
            for tc_delta in flush_events:
                idx = tc_delta.pop("index", 0)
                chunk = deepcopy(base_chunk)
                chunk["choices"] = [
                    {
                        "index": idx,
                        "delta": {"tool_calls": [tc_delta]},
                        "finish_reason": None,
                    }
                ]
                output_lines.append(encode_sse_event(chunk))

        output_lines.append(encode_sse_done())
        return output_lines

    def reset(self) -> None:
        """Reset the transformer for a new request."""
        self.buffer.reset()
        self._done = False
