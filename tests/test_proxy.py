"""Integration tests for the llm-toolstream-proxy.

These tests use aiohttp.test_utils.TestClient to test the full proxy
functionality including health/metrics endpoints, non-streaming passthrough,
streaming with tool call buffering, and resource limits.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llm_toolstream_proxy import config
from llm_toolstream_proxy.main import create_app

# -----------------------------------------------------------------------------
# Helper functions for creating SSE chunks
# -----------------------------------------------------------------------------


def make_sse_chunk(
    tool_calls: list[dict] | None = None,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    model: str = "qwen-3.5-122b",
    choice_index: int = 0,
    chunk_id: str = "chatcmpl-test",
) -> dict[str, Any]:
    """Create an OpenAI-style SSE chunk."""
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    choice: dict[str, Any] = {"index": choice_index, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    else:
        choice["finish_reason"] = None

    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": model,
        "choices": [choice],
    }


def encode_sse_event(data: dict[str, Any]) -> str:
    """Encode a dict as an SSE data line."""
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def encode_sse_done() -> str:
    """Encode the [DONE] sentinel."""
    return "data: [DONE]\n\n"


# -----------------------------------------------------------------------------
# Mock upstream server for testing
# -----------------------------------------------------------------------------


class MockUpstreamServer:
    """A mock upstream (litellm) server for testing proxy behavior."""

    def __init__(self) -> None:
        self.app = web.Application()
        # Store handlers as instance variables so they can be replaced
        self._chat_completions_handler = self.handle_chat_completions
        self._health_handler = self.handle_health
        self.app.router.add_post("/v1/chat/completions", self._route_chat_completions)
        self.app.router.add_get("/health", self._route_health)
        self.response_delay: float = 0.0
        self.sse_chunks: list[str] = []
        self.non_streaming_response: dict[str, Any] = {}
        self._custom_handler: Any = None

    async def _route_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse | web.Response:
        """Route to the current chat completions handler."""
        if self._custom_handler is not None:
            return await self._custom_handler(request)
        return await self._chat_completions_handler(request)

    async def _route_health(self, request: web.Request) -> web.Response:
        """Route to the current health handler."""
        return await self._health_handler(request)

    def set_custom_handler(self, handler: Any) -> None:
        """Set a custom handler for chat completions testing."""
        self._custom_handler = handler

    def clear_custom_handler(self) -> None:
        """Clear the custom handler to use default behavior."""
        self._custom_handler = None

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "upstream": "mock"})

    async def handle_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse | web.Response:
        """Handle chat completion requests - streaming or non-streaming."""
        body = await request.json()
        is_streaming = body.get("stream", False)

        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        if is_streaming:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)

            for chunk in self.sse_chunks:
                await response.write(chunk.encode("utf-8"))

            await response.write_eof()
            return response
        else:
            return web.json_response(self.non_streaming_response)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def mock_upstream() -> AsyncIterator[MockUpstreamServer]:
    """Create a mock upstream server."""
    server = MockUpstreamServer()
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)  # Random port
    await site.start()

    # Get the actual port
    port = site._server.sockets[0].getsockname()[1]  # type: ignore
    server.base_url = f"http://127.0.0.1:{port}"

    yield server

    await runner.cleanup()


@pytest.fixture
async def client(mock_upstream: MockUpstreamServer) -> AsyncIterator[TestClient]:
    """Create a test client for the proxy app with mocked upstream."""
    # Patch config to point to mock upstream
    with patch.object(config, "LITELLM_URL", mock_upstream.base_url):
        with patch.object(config, "MAX_CONCURRENT_STREAMS", 2):
            with patch.object(config, "MAX_REQUEST_BODY_SIZE", 1024 * 1024):  # 1MB
                app = create_app()
                client = TestClient(TestServer(app))
                await client.start_server()
                yield client
                await client.close()


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client: TestClient) -> None:
    """Test GET /health returns 200 with expected JSON fields."""
    resp = await client.request("GET", "/health")
    assert resp.status == 200

    data = await resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "upstream" in data
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_metrics_endpoint(client: TestClient) -> None:
    """Test GET /metrics returns 200 with expected JSON fields."""
    resp = await client.request("GET", "/metrics")
    assert resp.status == 200

    data = await resp.json()
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "total_streaming_requests" in data
    assert "total_non_streaming_requests" in data
    assert "active_streams" in data
    assert "total_tool_calls_buffered" in data
    assert "total_tool_calls_repaired" in data
    assert "total_bytes_transferred" in data
    assert "total_errors" in data


@pytest.mark.asyncio
async def test_non_streaming_passthrough(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test non-streaming request is forwarded to upstream and response returned."""
    # Set up mock response
    mock_upstream.non_streaming_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, world!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 200
    data = await resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello, world!"
    assert data["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_streaming_with_tool_call_buffering(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test streaming request with malformed tool call (empty name first, then name).

    The proxy should buffer the incomplete tool call and emit a well-formed
    tool call start chunk when the name arrives.
    """
    # Set up SSE chunks with empty name first, then name arrives
    mock_upstream.sse_chunks = [
        encode_sse_event(
            make_sse_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                ]
            )
        ),
        encode_sse_event(
            make_sse_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": "bash", "arguments": ""},
                    }
                ]
            )
        ),
        encode_sse_event(
            make_sse_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '{"command": "ls"}'},
                    }
                ]
            )
        ),
        encode_sse_done(),
    ]

    request_body = {
        "model": "qwen-3.5-122b",
        "messages": [{"role": "user", "content": "List files"}],
        "stream": True,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 200
    assert resp.content_type == "text/event-stream"

    # Read and parse the SSE response
    body = await resp.read()
    lines = body.decode("utf-8").strip().split("\n")

    # Find the tool call start event (should have name="bash")
    tool_call_events = []
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            data = json.loads(line[len("data: ") :])
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if "tool_calls" in delta:
                    tool_call_events.extend(delta["tool_calls"])

    # Verify we got a well-formed tool call with name
    assert len(tool_call_events) >= 1
    # The first emitted tool call should have the name
    first_tool = tool_call_events[0]
    assert first_tool.get("function", {}).get("name") == "bash"
    assert first_tool.get("id") == "call_abc123"


@pytest.mark.asyncio
async def test_malformed_tool_call_buffering(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test buffering when upstream sends name="" then name="bash".

    Verify the proxy buffers and emits correctly assembled tool call.
    """
    # Set up SSE chunks: empty name, then name arrives with arguments
    mock_upstream.sse_chunks = [
        # First chunk: empty name (should be buffered)
        encode_sse_event(
            make_sse_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                ]
            )
        ),
        # Second chunk: name arrives, should emit start event
        encode_sse_event(
            make_sse_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": "read_file", "arguments": ""},
                    }
                ]
            )
        ),
        # Third chunk: arguments
        encode_sse_event(
            make_sse_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '{"path": "/tmp"}'},
                    }
                ]
            )
        ),
        encode_sse_done(),
    ]

    request_body = {
        "model": "qwen-3.5-122b",
        "messages": [{"role": "user", "content": "Read file"}],
        "stream": True,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 200

    # Read and parse the SSE response
    body = await resp.read()
    lines = body.decode("utf-8").strip().split("\n")

    # Collect all tool call events
    tool_calls_found = []
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            data = json.loads(line[len("data: ") :])
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        tool_calls_found.append(tc)

    # Verify the tool call was correctly assembled
    assert len(tool_calls_found) >= 1

    # Find the tool call with the name (start event)
    start_events = [
        tc for tc in tool_calls_found if tc.get("function", {}).get("name") is not None
    ]
    assert len(start_events) == 1
    assert start_events[0]["function"]["name"] == "read_file"
    assert start_events[0]["id"] == "call_test"

    # Find the tool call with arguments
    arg_events = [
        tc
        for tc in tool_calls_found
        if tc.get("function", {}).get("arguments") is not None
        and tc.get("function", {}).get("name") is None
    ]
    assert len(arg_events) >= 1
    # Arguments should be accumulated
    all_args = "".join(e["function"]["arguments"] for e in arg_events)
    assert all_args == '{"path": "/tmp"}'


@pytest.mark.asyncio
async def test_large_body_rejected(client: TestClient) -> None:
    """Test that requests with body size > MAX_REQUEST_BODY_SIZE get 413."""
    # Create a request body larger than MAX_REQUEST_BODY_SIZE (1MB in tests)
    large_content = "x" * (2 * 1024 * 1024)  # 2MB
    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": large_content}],
        "stream": False,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 413
    # Response may be text/plain or JSON depending on aiohttp version
    text = await resp.text()
    assert (
        "maximum request body size" in text.lower()
        or "too large" in text.lower()
        or "error" in text.lower()
    )


@pytest.mark.asyncio
async def test_concurrent_stream_limit(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test that proxy returns 503 when MAX_CONCURRENT_STREAMS is reached.

    We set MAX_CONCURRENT_STREAMS=2 for tests. Fill both slots with slow
    streaming requests, then verify the third request gets 503.
    """
    # Set up slow SSE chunks that will keep connections open
    mock_upstream.response_delay = 0.1  # Small delay to ensure streams are active
    mock_upstream.sse_chunks = (
        [
            encode_sse_event(make_sse_chunk(content="thinking...")),
        ]
        + [encode_sse_event(make_sse_chunk(content="...")) for _ in range(100)]
        + [
            encode_sse_done(),
        ]
    )

    request_body = {
        "model": "qwen-3.5-122b",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Start two streaming requests to fill the semaphore
    async def make_streaming_request() -> web.StreamResponse | web.Response:
        return await client.request(
            "POST",
            "/v1/chat/completions",
            data=json.dumps(request_body),
            headers={"Content-Type": "application/json"},
        )

    # Create first two requests (these should succeed)
    resp1_task = asyncio.create_task(make_streaming_request())
    resp2_task = asyncio.create_task(make_streaming_request())

    # Wait a bit for the first two to acquire the semaphore
    await asyncio.sleep(0.05)

    # Third request should get 503
    resp3 = await make_streaming_request()

    # Wait for the first two to complete
    resp1 = await resp1_task
    resp2 = await resp2_task

    # First two should succeed
    assert resp1.status == 200
    assert resp2.status == 200

    # Third should get 503
    assert resp3.status == 503
    data = await resp3.json()
    assert "error" in data
    assert (
        "concurrent" in data.get("error", "").lower()
        or "overloaded" in data.get("error", "").lower()
    )


@pytest.mark.asyncio
async def test_streaming_content_passthrough(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test that non-tool content is passed through correctly in streaming mode."""
    mock_upstream.sse_chunks = [
        encode_sse_event(make_sse_chunk(role="assistant")),
        encode_sse_event(make_sse_chunk(content="Hello")),
        encode_sse_event(make_sse_chunk(content=", ")),
        encode_sse_event(make_sse_chunk(content="world!")),
        encode_sse_done(),
    ]

    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": True,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 200

    # Read and parse the SSE response
    body = await resp.read()
    lines = body.decode("utf-8").strip().split("\n")

    # Collect content pieces
    contents = []
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            data = json.loads(line[len("data: ") :])
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if "content" in delta:
                    contents.append(delta["content"])

    # Verify content was passed through
    assert "Hello" in contents
    assert ", " in contents
    assert "world!" in contents


@pytest.mark.asyncio
async def test_proxy_forwards_headers(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test that the proxy forwards relevant headers to upstream."""
    received_headers: dict[str, str] = {}

    async def capture_headers(request: web.Request) -> web.Response:
        received_headers.update(dict(request.headers))
        return web.json_response({"status": "ok"})

    # Use custom handler to capture headers (router is frozen after startup)
    mock_upstream.set_custom_handler(capture_headers)

    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
            "X-Custom-Header": "custom-value",
        },
    )

    # Clear custom handler
    mock_upstream.clear_custom_handler()

    # Verify headers were forwarded
    assert "Content-Type" in received_headers
    assert received_headers.get("Authorization") == "Bearer test-token"
    assert received_headers.get("X-Custom-Header") == "custom-value"


@pytest.mark.asyncio
async def test_non_streaming_upstream_error(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test that upstream errors are handled gracefully for non-streaming requests."""

    async def return_error(request: web.Request) -> web.Response:
        return web.Response(
            status=502,
            content_type="application/json",
            text=json.dumps({"error": "upstream failure"}),
        )

    # Use custom handler (router is frozen after startup)
    mock_upstream.set_custom_handler(return_error)

    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    # Clear custom handler
    mock_upstream.clear_custom_handler()

    # Proxy should return the upstream error status
    assert resp.status == 502


@pytest.mark.asyncio
async def test_streaming_with_finish_reason(
    client: TestClient, mock_upstream: MockUpstreamServer
) -> None:
    """Test that finish_reason triggers proper cleanup in streaming mode."""
    mock_upstream.sse_chunks = [
        encode_sse_event(make_sse_chunk(content="Done")),
        encode_sse_event(make_sse_chunk(finish_reason="stop")),
        encode_sse_done(),
    ]

    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    resp = await client.request(
        "POST",
        "/v1/chat/completions",
        data=json.dumps(request_body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 200

    # Read the response
    body = await resp.read()
    lines = body.decode("utf-8").strip().split("\n")

    # Verify we got the finish_reason
    finish_reasons = []
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            data = json.loads(line[len("data: ") :])
            for choice in data.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reasons.append(choice["finish_reason"])

    assert "stop" in finish_reasons
