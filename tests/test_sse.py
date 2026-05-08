"""Tests for SSE parsing and transformation."""

import json

from llm_toolstream_proxy.buffering import ToolCallArgsDelta, ToolCallStartEvent
from llm_toolstream_proxy.sse import (
    SSELineBuffer,
    SSETransformer,
    _build_tool_call_chunk,
    encode_sse_done,
    encode_sse_event,
    parse_sse_line,
)


def _make_chunk(
    tool_calls: list[dict] | None = None,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    model: str = "qwen-3.5-122b",
    choice_index: int = 0,
) -> dict:
    delta: dict = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    choice: dict = {"index": choice_index, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    else:
        choice["finish_reason"] = None

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": model,
        "choices": [choice],
    }


def _encode_chunk(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


def _parse_sse_output(lines: list[str]) -> list[dict]:
    """Parse SSE output lines back into dicts."""
    result = []
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            result.append(json.loads(line[len("data: ") :]))
    return result


class TestParseSseLine:
    def test_valid_json(self):
        line = 'data: {"id":"chatcmpl-1"}\n\n'
        result = parse_sse_line(line)
        assert result is not None
        assert result["id"] == "chatcmpl-1"

    def test_done_sentinel(self):
        line = "data: [DONE]\n\n"
        result = parse_sse_line(line)
        assert result is not None
        assert result.get("__done__") is True

    def test_comment_line(self):
        line = ": this is a comment\n"
        result = parse_sse_line(line)
        assert result is None

    def test_empty_line(self):
        result = parse_sse_line("\n")
        assert result is None

    def test_non_data_prefix(self):
        line = "event: ping\n"
        result = parse_sse_line(line)
        assert result is None

    def test_invalid_json(self):
        line = "data: {not json}\n\n"
        result = parse_sse_line(line)
        assert result is None


class TestEncodeSseEvent:
    def test_round_trip(self):
        chunk = _make_chunk(content="hello")
        encoded = encode_sse_event(chunk)
        assert encoded.startswith("data: ")
        assert encoded.endswith("\n\n")
        parsed = json.loads(encoded[len("data: ") :])
        assert parsed["choices"][0]["delta"]["content"] == "hello"

    def test_done_sentinel(self):
        result = encode_sse_done()
        assert result == "data: [DONE]\n\n"


class TestSSETransformerHappyPath:
    """Test the full SSE pipeline with well-formed tool calls."""

    def test_passthrough_content(self):
        transformer = SSETransformer()
        chunk = _make_chunk(content="Hello, world!")
        lines = transformer.process_raw(_encode_chunk(chunk))
        assert len(lines) == 1
        parsed = json.loads(lines[0][len("data: ") :])
        assert parsed["choices"][0]["delta"]["content"] == "Hello, world!"

    def test_passthrough_role(self):
        transformer = SSETransformer()
        chunk = _make_chunk(role="assistant")
        lines = transformer.process_raw(_encode_chunk(chunk))
        assert len(lines) == 1

    def test_streaming_tool_call(self):
        transformer = SSETransformer()
        lines_all: list[str] = []

        chunk1 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": ""},
                }
            ]
        )
        lines_all.extend(transformer.process_raw(_encode_chunk(chunk1)))

        chunk2 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "function": {"arguments": '{"command":'},
                }
            ]
        )
        lines_all.extend(transformer.process_raw(_encode_chunk(chunk2)))

        chunk3 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "function": {"arguments": ' "ls"}'},
                }
            ]
        )
        lines_all.extend(transformer.process_raw(_encode_chunk(chunk3)))

        flush_lines = transformer.flush()
        lines_all.extend(flush_lines)

        events = _parse_sse_output(lines_all)
        tool_names = [
            tc["function"]["name"]
            for e in events
            for ch in e.get("choices", [])
            for tc in ch.get("delta", {}).get("tool_calls", [])
            if "name" in tc.get("function", {})
        ]
        assert "bash" in tool_names


class TestSSETransformerBuffering:
    """Test the SSE pipeline with malformed tool calls that need buffering."""

    def test_empty_name_buffered(self):
        transformer = SSETransformer()
        chunk1 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            ]
        )
        lines1 = transformer.process_raw(_encode_chunk(chunk1))

        for line in lines1:
            if line.startswith("data: ") and "[DONE]" not in line:
                data = json.loads(line[len("data: ") :])
                for ch in data.get("choices", []):
                    assert "tool_calls" not in ch.get("delta", {})

        chunk2 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "function": {"name": "read_file", "arguments": ""},
                }
            ]
        )
        lines2 = transformer.process_raw(_encode_chunk(chunk2))

        tool_found = False
        for line in lines2:
            if line.startswith("data: ") and "[DONE]" not in line:
                data = json.loads(line[len("data: ") :])
                for ch in data.get("choices", []):
                    for tc in ch.get("delta", {}).get("tool_calls", []):
                        if tc.get("function", {}).get("name") == "read_file":
                            tool_found = True
        assert tool_found

    def test_content_and_tools_split(self):
        transformer = SSETransformer()
        chunk = _make_chunk(
            content="",
            role="assistant",
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": ""},
                }
            ],
        )
        lines = transformer.process_raw(_encode_chunk(chunk))
        tool_found = False
        for line in lines:
            if line.startswith("data: ") and "[DONE]" not in line:
                data = json.loads(line[len("data: ") :])
                for ch in data.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("tool_calls"):
                        tool_found = True
        assert tool_found

    def test_done_sentinel_flushes(self):
        transformer = SSETransformer()
        chunk = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            ]
        )
        transformer.process_raw(_encode_chunk(chunk))

        lines = transformer.process_raw("data: [DONE]\n\n")
        has_done = any("[DONE]" in line for line in lines)
        assert has_done

    def test_null_name_string(self):
        transformer = SSETransformer()
        chunk1 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "null", "arguments": ""},
                }
            ]
        )
        lines1 = transformer.process_raw(_encode_chunk(chunk1))

        for line in lines1:
            if line.startswith("data: ") and "[DONE]" not in line:
                data = json.loads(line[len("data: ") :])
                for ch in data.get("choices", []):
                    for tc in ch.get("delta", {}).get("tool_calls", []):
                        assert False, (
                            "No tool_calls should be emitted when name is 'null'"
                        )

        chunk2 = _make_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "function": {"name": "read_file", "arguments": ""},
                }
            ]
        )
        lines2 = transformer.process_raw(_encode_chunk(chunk2))
        assert len(lines2) >= 1


# ---------------------------------------------------------------------------
# Tests for extracted SSELineBuffer (Candidate 1)
# ---------------------------------------------------------------------------


class TestSSELineBuffer:
    """Test the SSE line buffer in isolation — no tool call state involved."""

    def test_single_line_event(self):
        buf = SSELineBuffer()
        payloads = buf.feed('data: {"key": "val"}\n\n')
        assert len(payloads) == 1
        assert payloads[0] == '{"key": "val"}'

    def test_multi_line_event(self):
        buf = SSELineBuffer()
        # Feed first data line without trailing blank line (incomplete)
        payloads = buf.feed("data: first")
        assert len(payloads) == 0  # incomplete — no blank line yet

        # Feed second line with blank line to complete the event
        payloads = buf.feed("data: second\n\n")
        assert len(payloads) == 1
        assert "first" in payloads[0] and "second" in payloads[0]

    def test_event_with_comment_lines(self):
        buf = SSELineBuffer()
        payloads = buf.feed(": heartbeat\n\n")
        assert len(payloads) == 0  # comment-only event yields no data payload

    def test_multiple_events_in_one_feed(self):
        buf = SSELineBuffer()
        payloads = buf.feed(
            'data: {"a": 1}\n\n'
            'data: {"b": 2}\n\n'
        )
        assert len(payloads) == 2
        assert payloads[0] == '{"a": 1}'
        assert payloads[1] == '{"b": 2}'

    def test_partial_line_across_feeds(self):
        buf = SSELineBuffer()
        # Feed line without trailing newline — incomplete event
        payloads = buf.feed("data: first")
        assert len(payloads) == 0

        # Trailing blank line completes the event
        payloads = buf.feed("\n")
        assert len(payloads) == 1
        assert payloads[0] == "first"

    def test_reset_clears_buffered_lines(self):
        buf = SSELineBuffer()
        buf.feed("data: incomplete\n")
        buf.reset()
        payloads = buf.feed("\n")
        assert len(payloads) == 0

    def test_done_sentinel(self):
        buf = SSELineBuffer()
        payloads = buf.feed("data: [DONE]\n\n")
        assert len(payloads) == 1
        assert payloads[0] == "[DONE]"

    def test_non_data_prefix_lines_ignored(self):
        buf = SSELineBuffer()
        payloads = buf.feed("event: ping\ndata: hello\n\n")
        assert len(payloads) == 1
        assert payloads[0] == "hello"


# ---------------------------------------------------------------------------
# Tests for _build_tool_call_chunk (Candidate 6)
# ---------------------------------------------------------------------------


class TestBuildToolCallChunk:
    """Test the shared chunk construction function."""

    def test_start_event_chunk(self):
        delta: ToolCallStartEvent = {
            "index": 0,
            "id": "call_abc",
            "type": "function",
            "function": {"name": "bash", "arguments": ""},
        }
        chunk = _build_tool_call_chunk(delta, index=0)

        assert chunk["choices"][0]["index"] == 0
        assert chunk["choices"][0]["delta"]["tool_calls"][0] is delta
        assert chunk["id"] == "chatcmpl-tool-buffer"
        assert chunk["object"] == "chat.completion.chunk"

    def test_args_delta_chunk(self):
        delta: ToolCallArgsDelta = {
            "index": 0,
            "function": {"arguments": '{"cmd":' },
        }
        chunk = _build_tool_call_chunk(delta, index=3)

        assert chunk["choices"][0]["index"] == 3
        assert chunk["choices"][0]["delta"]["tool_calls"][0] is delta

    def test_custom_defaults(self):
        delta: ToolCallStartEvent = {
            "index": 0,
            "id": "x",
            "type": "function",
            "function": {"name": "n", "arguments": ""},
        }
        chunk = _build_tool_call_chunk(
            delta,
            index=0,
            id="custom-id",
            object="custom.object",
            created=42,
            model="custom-model",
        )
        assert chunk["id"] == "custom-id"
        assert chunk["object"] == "custom.object"
        assert chunk["created"] == 42
        assert chunk["model"] == "custom-model"

    def test_derived_from_parent_chunk(self):
        parent: dict[str, object] = {
            "id": "chatcmpl-parent",
            "object": "chat.completion.chunk",
            "created": 999,
            "model": "gpt-4",
        }
        delta: ToolCallStartEvent = {
            "index": 0,
            "id": "call_x",
            "type": "function",
            "function": {"name": "echo", "arguments": ""},
        }
        chunk = _build_tool_call_chunk(
            delta,
            index=0,
            id=parent.get("id", "chatcmpl-tool-buffer"),  # type: ignore[arg-type]
            object=parent.get("object", "chat.completion.chunk"),  # type: ignore[arg-type]
            created=parent.get("created", 0),  # type: ignore[arg-type]
            model=parent.get("model", ""),  # type: ignore[arg-type]
        )
        assert chunk["id"] == "chatcmpl-parent"
        assert chunk["model"] == "gpt-4"
        assert chunk["created"] == 999

    def test_finish_reason_is_none(self):
        delta: ToolCallStartEvent = {
            "index": 0,
            "id": "call_x",
            "type": "function",
            "function": {"name": "n", "arguments": ""},
        }
        chunk = _build_tool_call_chunk(delta, index=0)
        assert chunk["choices"][0]["finish_reason"] is None
