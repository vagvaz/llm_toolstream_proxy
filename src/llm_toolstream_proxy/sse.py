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
from typing import Any

from loguru import logger

from .buffering import ToolCallBuffer

DONE_SENTINEL = "[DONE]"


def parse_sse_line(line: str) -> dict[str, Any] | None:
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
        result: Any = json.loads(payload)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        logger.warning("Failed to parse SSE JSON: {}", payload[:200])
        return None


def encode_sse_event(data: dict[str, Any]) -> str:
    """Encode a dict back into an SSE ``data:`` line with trailing newlines."""
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def encode_sse_done() -> str:
    """Encode the [DONE] sentinel as an SSE event."""
    return f"data: {DONE_SENTINEL}\n\n"


def _extract_tool_calls(
    chunk: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Extract tool_calls from a streaming chunk and return a cleaned chunk
    without tool_calls.

    Returns:
        (tool_call_deltas, chunk_without_tool_calls or None)
        chunk_without_tool_calls is None if the chunk had no non-tool delta
        fields worth emitting (i.e. the delta only contained tool_calls).
        The chunk has tool_calls removed from all choices' deltas, but
        preserves all other fields (role, content, reasoning, etc.).
    """
    tool_call_deltas: list[dict[str, Any]] = []
    has_tool_calls = False

    choices = chunk.get("choices", [])
    for choice in choices:
        delta = choice.get("delta", {})
        if delta and "tool_calls" in delta:
            has_tool_calls = True
            tcs = delta.pop("tool_calls")
            if tcs:
                tool_call_deltas.extend(tcs)

    if not has_tool_calls:
        return [], None

    # Check if the cleaned chunk has any non-tool content worth emitting
    has_content = False
    for choice in choices:
        delta = choice.get("delta", {})
        if delta:
            remaining = {k: v for k, v in delta.items() if k != "tool_calls"}
            if any(v is not None and v != "" and v != [] for v in remaining.values()):
                has_content = True
                break

    # If no non-tool content, don't emit the cleaned chunk
    if not has_content:
        return tool_call_deltas, None

    return tool_call_deltas, chunk


def _inject_tool_call_events(
    chunk: dict[str, Any],
    choice_index: int,
    tool_call_deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create properly formed SSE chunk dicts with tool_call deltas
    injected into a specific choice.

    Each delta becomes its own chunk so that downstream consumers
    process them one at a time, matching the original streaming behavior.
    """
    events: list[dict[str, Any]] = []
    for tc_delta in tool_call_deltas:
        # Construct a minimal chunk with just the tool call delta
        new_chunk: dict[str, Any] = {
            "id": chunk.get("id", "chatcmpl-tool-buffer"),
            "object": chunk.get("object", "chat.completion.chunk"),
            "created": chunk.get("created", 0),
            "model": chunk.get("model", ""),
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"tool_calls": [tc_delta]},
                    "finish_reason": None,
                }
            ],
        }
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

    def __init__(
        self,
        *,
        validate_json: bool = True,
        max_arguments_size: int = 1024 * 1024,
        max_tool_calls: int = 32,
    ) -> None:
        self.buffer = ToolCallBuffer(
            validate_json=validate_json,
            max_arguments_size=max_arguments_size,
            max_tool_calls=max_tool_calls,
        )
        self._done = False
        self._line_buffer: list[str] = []

    def process_raw(self, raw_line: str) -> list[str]:
        """Process a raw SSE line from upstream.

        Handles multi-line SSE events by buffering lines until a
        complete event (blank line) is received. Multiple ``data:``
        lines within a single event are concatenated per the SSE spec.

        Args:
            raw_line: A single line from the SSE stream (may include
                the ``data:`` prefix and trailing newlines).

        Returns:
            List of SSE-formatted strings (``data: ...\\n\\n``) to emit
            to the downstream client. May be empty if tool calls are still
            being buffered or if the event is incomplete.
        """
        if self._done:
            return []

        # Split the raw input into individual lines and buffer them
        # until we see a blank line (event boundary)
        output_lines: list[str] = []

        # Split on newlines to handle cases where upstream sends multiple
        # lines in a single read
        lines = raw_line.split("\n")

        for line in lines:
            stripped = line.strip()

            if stripped == "":
                # Blank line = end of event, process buffered data
                if self._line_buffer:
                    data_lines = [
                        line for line in self._line_buffer if line.startswith("data:")
                    ]
                    self._line_buffer = []

                    if not data_lines:
                        continue

                    # Join multi-line data per SSE spec
                    # For our JSON use case, concatenate without newlines
                    payloads = [line[len("data:") :].strip() for line in data_lines]

                    if len(payloads) == 1:
                        payload = payloads[0]
                    else:
                        # Multi-line data: join with \n per SSE spec
                        payload = "\n".join(payloads)

                    # Process the complete event
                    event_output = self._process_complete_event(payload)
                    output_lines.extend(event_output)
                continue

            # Accumulate non-empty lines
            self._line_buffer.append(line)

        return output_lines

    def _process_complete_event(self, payload: str) -> list[str]:
        """Process a complete SSE event payload (after line joining)."""
        if payload == DONE_SENTINEL:
            self._done = True
            logger.info("SSE stream ended, flushing remaining buffered tool calls")
            return self._flush_and_done()

        try:
            parsed: Any = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SSE JSON: {}", payload[:200])
            return []

        if not isinstance(parsed, dict):
            return []

        # Fast path: if the chunk has no tool_calls, pass through as-is
        # without re-encoding. This avoids JSON parse → re-encode
        # for the common case of content-only chunks.
        has_tool_calls = False
        choices = parsed.get("choices", [])
        for ch in choices:
            delta = ch.get("delta", {})
            if delta and "tool_calls" in delta:
                has_tool_calls = True
                break

        if not has_tool_calls:
            # Check for finish_reason to mark tool calls as finished
            for ch in choices:
                finish_reason = ch.get("finish_reason")
                if finish_reason == "tool_calls" or finish_reason == "stop":
                    choice_index = ch.get("index", 0)
                    finish_indices = list(self.buffer.calls.keys())
                    for idx in finish_indices:
                        if self.buffer.calls[idx].name is not None:
                            logger.debug(
                                "Finish reason {} for choice {}: "
                                "marking tool call {} finished",
                                finish_reason,
                                choice_index,
                                idx,
                            )
                            self.buffer.finish_call(idx)

            # Re-encode the parsed event for passthrough
            return [f"data: {json.dumps(parsed, separators=(',', ':'))}\n\n"]

        tool_call_deltas, cleaned = _extract_tool_calls(parsed)

        output_lines: list[str] = []

        for ch in choices:
            finish_reason = ch.get("finish_reason")
            choice_index = ch.get("index", 0)
            if finish_reason == "tool_calls" or finish_reason == "stop":
                finish_indices = list(self.buffer.calls.keys())
                for idx in finish_indices:
                    if self.buffer.calls[idx].name is not None:
                        logger.debug(
                            "Finish reason {} for choice {}: "
                            "marking tool call {} finished",
                            finish_reason,
                            choice_index,
                            idx,
                        )
                        self.buffer.finish_call(idx)

        if tool_call_deltas:
            logger.debug(
                "SSE chunk has {} tool_call deltas",
                len(tool_call_deltas),
            )
            for choice in choices:
                choice_index = choice.get("index", 0)
                buffered_events: list[dict[str, Any]] = []
                for tc_delta in tool_call_deltas:
                    buffered_events.extend(self.buffer.process_delta(tc_delta))

                # Emit the cleaned chunk (non-tool content) if it has content
                if cleaned is not None:
                    output_lines.append(encode_sse_event(cleaned))

                if buffered_events:
                    logger.debug(
                        "Emitting {} buffered tool call events for choice {}",
                        len(buffered_events),
                        choice_index,
                    )
                    for event_chunk in _inject_tool_call_events(
                        parsed, choice_index, buffered_events
                    ):
                        output_lines.append(encode_sse_event(event_chunk))

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
                "Flushing {} remaining tool call(s) on stream end",
                len(flush_events),
            )
            for tc_delta in flush_events:
                idx = tc_delta.pop("index", 0)
                chunk: dict[str, Any] = {
                    "id": "chatcmpl-tool-buffer",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [
                        {
                            "index": idx,
                            "delta": {"tool_calls": [tc_delta]},
                            "finish_reason": None,
                        }
                    ],
                }
                output_lines.append(encode_sse_event(chunk))

        output_lines.append(encode_sse_done())
        return output_lines

    def reset(self) -> None:
        """Reset the transformer for a new request."""
        self.buffer.reset()
        self._done = False
        self._line_buffer = []
