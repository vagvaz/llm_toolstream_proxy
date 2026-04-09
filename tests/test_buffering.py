"""Tests for the ToolCallBuffer core algorithm."""


from llm_toolstream_proxy.buffering import (
    BufferedToolCall,
    ToolCallBuffer,
    _is_valid_json,
    _sanitize_id,
    _sanitize_name,
    _try_repair_json,
)


class TestSanitizeName:
    def test_none(self):
        assert _sanitize_name(None) is None

    def test_empty_string(self):
        assert _sanitize_name("") is None

    def test_string_null(self):
        assert _sanitize_name("null") is None

    def test_valid_name(self):
        assert _sanitize_name("read_file") == "read_file"

    def test_arbitrary_string(self):
        assert _sanitize_name("bash") == "bash"


class TestSanitizeId:
    def test_none(self):
        assert _sanitize_id(None) is None

    def test_empty_string(self):
        assert _sanitize_id("") is None

    def test_valid_id(self):
        assert _sanitize_id("call_abc123") == "call_abc123"


class TestIsValidJson:
    def test_empty_string(self):
        assert _is_valid_json("") is True

    def test_valid_object(self):
        assert _is_valid_json('{"key": "value"}') is True

    def test_valid_array(self):
        assert _is_valid_json("[1, 2, 3]") is True

    def test_invalid_json(self):
        assert _is_valid_json('{"key":') is False

    def test_number(self):
        assert _is_valid_json("42") is True


class TestTryRepairJson:
    def test_valid_json_unchanged(self):
        assert _try_repair_json('{"key": "value"}') == '{"key": "value"}'

    def test_missing_closing_brace(self):
        assert _try_repair_json('{"key": "value"') == '{"key": "value"}'

    def test_missing_closing_bracket(self):
        assert _try_repair_json('{"items": [1, 2') == '{"items": [1, 2]}'

    def test_trailing_comma(self):
        assert _try_repair_json('{"key": "value",}') == '{"key": "value"}'

    def test_unrepairable(self):
        result = _try_repair_json("}{garbage")
        assert result is None

    def test_empty_string(self):
        assert _try_repair_json("") == ""


class TestBufferedToolCall:
    def test_is_complete_with_both(self):
        tc = BufferedToolCall(id="call_abc", name="read_file")
        assert tc.is_complete is True

    def test_is_complete_missing_id(self):
        tc = BufferedToolCall(id=None, name="read_file")
        assert tc.is_complete is False

    def test_is_complete_missing_name(self):
        tc = BufferedToolCall(id="call_abc", name=None)
        assert tc.is_complete is False

    def test_is_complete_both_missing(self):
        tc = BufferedToolCall(id=None, name=None)
        assert tc.is_complete is False


class TestToolCallBufferHappyPath:
    """Test the standard OpenAI-compliant streaming pattern."""

    def test_single_tool_call_streamed(self):
        buf = ToolCallBuffer()
        chunk1 = buf.process_delta(
            {
                "index": 0,
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        assert len(chunk1) == 1
        assert chunk1[0]["function"]["name"] == "read_file"
        assert chunk1[0]["id"] == "call_abc123"
        assert chunk1[0]["function"]["arguments"] == ""

        chunk2 = buf.process_delta(
            {
                "index": 0,
                "function": {"arguments": '{"filePath":'},
            }
        )
        assert len(chunk2) == 1
        assert chunk2[0]["function"]["arguments"] == '{"filePath":'

        chunk3 = buf.process_delta(
            {
                "index": 0,
                "function": {"arguments": ' "/tmp/test"}'},
            }
        )
        assert len(chunk3) == 1
        assert chunk3[0]["function"]["arguments"] == ' "/tmp/test"}'

        flush = buf.flush()
        assert len(flush) == 0

    def test_multiple_tool_calls(self):
        buf = ToolCallBuffer()
        for index, name in enumerate(["read_file", "bash", "glob"]):
            events = buf.process_delta(
                {
                    "index": index,
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            )
            assert len(events) >= 1
            assert events[0]["function"]["name"] == name


class TestBufferingEmptyName:
    """Test Qwen/vllm pattern: empty name in first chunk."""

    def test_empty_name_in_first_chunk(self):
        buf = ToolCallBuffer()

        chunk1 = buf.process_delta(
            {
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
        assert len(chunk1) == 0

        chunk2 = buf.process_delta(
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        assert len(chunk2) == 1
        assert chunk2[0]["id"] == "call_abc"
        assert chunk2[0]["function"]["name"] == "read_file"
        assert chunk2[0]["function"]["arguments"] == ""

    def test_name_null_string(self):
        buf = ToolCallBuffer()

        chunk1 = buf.process_delta(
            {
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "null", "arguments": ""},
            }
        )
        assert len(chunk1) == 0

        chunk2 = buf.process_delta(
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        assert len(chunk2) == 1
        assert chunk2[0]["function"]["name"] == "read_file"

    def test_arguments_before_name(self):
        buf = ToolCallBuffer()

        chunk1 = buf.process_delta(
            {
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
        assert len(chunk1) == 0

        chunk2 = buf.process_delta(
            {
                "index": 0,
                "function": {"name": "", "arguments": '{"file'},
            }
        )
        assert len(chunk2) == 0

        chunk3 = buf.process_delta(
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        assert len(chunk3) == 2
        assert chunk3[0]["function"]["name"] == "read_file"
        assert chunk3[1]["function"]["arguments"] == '{"file'

        chunk4 = buf.process_delta(
            {
                "index": 0,
                "function": {"arguments": 'Path": "/tmp"}'},
            }
        )
        assert len(chunk4) == 1
        assert chunk4[0]["function"]["arguments"] == 'Path": "/tmp"}'


class TestBufferingMissingId:
    """Test pattern where ID arrives in a later chunk."""

    def test_id_arrives_after_name(self):
        buf = ToolCallBuffer()

        chunk1 = buf.process_delta(
            {
                "index": 0,
                "type": "function",
                "function": {"name": "bash", "arguments": ""},
            }
        )
        assert len(chunk1) == 0

        chunk2 = buf.process_delta(
            {
                "index": 0,
                "id": "call_xyz",
                "function": {"arguments": '{"command": "ls"}'},
            }
        )
        assert len(chunk2) == 2
        assert chunk2[0]["id"] == "call_xyz"
        assert chunk2[0]["function"]["name"] == "bash"
        assert chunk2[1]["function"]["arguments"] == '{"command": "ls"}'


class TestBufferingFlush:
    """Test flush behavior when stream ends before all chunks arrived."""

    def test_flush_discards_incomplete(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
        flush = buf.flush()
        assert len(flush) == 0

    def test_flush_includes_complete_unstarted(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'},
            }
        )
        flush = buf.flush()
        assert len(flush) == 0

    def test_flush_buffered_then_revealed(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
        buf.process_delta(
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        flush = buf.flush()
        assert len(flush) == 0


class TestBufferingNameNotOverwritten:
    """Test that a valid name is never overwritten by later empty names."""

    def test_name_stays_after_empty_continuation(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        buf.process_delta(
            {
                "index": 0,
                "function": {"name": "", "arguments": '{"path":'},
            }
        )
        assert buf.calls[0].name == "read_file"

    def test_name_not_overwritten_by_null_string(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": ""},
            }
        )
        buf.process_delta(
            {
                "index": 0,
                "function": {"name": "null", "arguments": '{"cmd": "ls"}'},
            }
        )
        assert buf.calls[0].name == "bash"


class TestBufferingMultipleCalls:
    """Test buffering with multiple interleaved tool calls."""

    def test_two_calls_interleaved(self):
        buf = ToolCallBuffer()
        ev1 = buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        assert len(ev1) == 1

        ev2 = buf.process_delta(
            {
                "index": 1,
                "id": "call_2",
                "type": "function",
                "function": {"name": "bash", "arguments": ""},
            }
        )
        assert len(ev2) == 1

        ev3 = buf.process_delta(
            {
                "index": 0,
                "function": {"arguments": '{"path": "/tmp"}'},
            }
        )
        assert len(ev3) == 1

        ev4 = buf.process_delta(
            {
                "index": 1,
                "function": {"arguments": '{"command": "ls"}'},
            }
        )
        assert len(ev4) == 1

    def test_first_call_buffered_second_complete(self):
        buf = ToolCallBuffer()
        ev1 = buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
        assert len(ev1) == 0

        ev2 = buf.process_delta(
            {
                "index": 1,
                "id": "call_2",
                "type": "function",
                "function": {"name": "bash", "arguments": ""},
            }
        )
        assert len(ev2) == 1
        assert ev2[0]["function"]["name"] == "bash"

        ev3 = buf.process_delta(
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": ""},
            }
        )
        assert len(ev3) == 1
        assert ev3[0]["function"]["name"] == "read_file"

    def test_first_buffered_second_buffered_then_both_resolve(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {"index": 0, "id": "call_1", "function": {"name": "", "arguments": ""}}
        )
        buf.process_delta(
            {"index": 1, "id": "call_2", "function": {"name": "", "arguments": ""}}
        )

        ev = buf.process_delta(
            {"index": 0, "function": {"name": "read_file", "arguments": ""}}
        )
        assert len(ev) == 1
        assert ev[0]["function"]["name"] == "read_file"

        ev = buf.process_delta(
            {"index": 1, "function": {"name": "bash", "arguments": ""}}
        )
        assert len(ev) == 1
        assert ev[0]["function"]["name"] == "bash"

    def test_arguments_accumulated_while_buffering(self):
        buf = ToolCallBuffer()
        buf.process_delta(
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "", "arguments": '{"pa'},
            }
        )
        buf.process_delta(
            {
                "index": 0,
                "function": {"name": "", "arguments": 'th":'},
            }
        )
        ev = buf.process_delta(
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": ' "/tmp"}'},
            }
        )
        assert len(ev) == 2
        assert ev[0]["function"]["name"] == "read_file"
        assert ev[1]["function"]["arguments"] == '{"path": "/tmp"}'
        assert buf.calls[0].arguments == '{"path": "/tmp"}'
