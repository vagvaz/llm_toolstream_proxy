"""aiohttp-based reverse proxy that intercepts and transforms streaming SSE responses.

Architecture::

    opencode  -->  localhost:8787 (this proxy)  -->  litellm:4000

Non-streaming requests are forwarded transparently with no modification.
Streaming requests with ``stream: true`` are intercepted: the SSE response
is parsed line-by-line, tool_call deltas are buffered and reassembled, and
the cleaned SSE stream is returned to opencode.

Resource management:
    - ``STREAM_MAX_DURATION`` caps total stream time even if data keeps trickling
      in (each chunk resets the sock_read timer, so sock_read alone is insufficient).
    - ``KEEPALIVE_TIMEOUT=30s`` closes idle pooled connections quickly.
    - Client disconnections are detected via ``request.transport.is_closing()``
      and abort the upstream stream immediately.
    - ``STREAM_TIMEOUT=120s`` aborts if upstream stops sending data.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import aiohttp
from aiohttp import web
from loguru import logger

from . import config
from .sse import SSETransformer

STREAMING_ROUTES = {
    "/v1/chat/completions",
    "/chat/completions",
}

NON_STREAMING_FORWARD_HEADERS = {
    "content-type",
    "content-length",
    "content-encoding",
    "cache-control",
    "x-request-id",
    "x-ratelimit-remaining",
    "x-ratelimit-limit",
    "x-ratelimit-reset",
}


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "version": config.LITELLM_URL})


async def _is_streaming_request(body: dict[str, Any]) -> bool:
    return body.get("stream", False) is True


async def _read_request_body(request: web.Request) -> dict[str, Any]:
    """Read and parse the JSON request body."""
    raw = await request.read()
    if not raw:
        return {}
    try:
        result: Any = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _forward_headers(request: web.Request) -> dict[str, str]:
    """Build headers dict for forwarding the request to litellm."""
    headers = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in ("host", "content-length", "transfer-encoding"):
            continue
        headers[key] = value
    headers["Host"] = config.LITELLM_URL.split("//", 1)[-1].split("/")[0].split(":")[0]
    return headers


async def _stream_response(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    *,
    buffer_tool_calls: bool,
    validate_json: bool,
    cancel_event: asyncio.Event,
) -> AsyncIterator[bytes]:
    """Stream the response from litellm, applying tool call buffering if needed.

    Args:
        session: Shared aiohttp ClientSession for connection reuse.
        cancel_event: Set when the downstream client disconnects, causing
            the upstream request to be cancelled immediately.
    """
    transformer = (
        SSETransformer(
            validate_json=validate_json,
            max_arguments_size=config.MAX_ARGUMENTS_SIZE,
            max_tool_calls=config.MAX_TOOL_CALLS,
        )
        if buffer_tool_calls
        else None
    )

    timeout = aiohttp.ClientTimeout(
        total=None,  # no total limit — controlled by STREAM_MAX_DURATION wrapper
        connect=config.CONNECT_TIMEOUT,
        sock_read=config.STREAM_TIMEOUT,
    )

    try:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=timeout,
        ) as resp:
            async for line in resp.content:
                # Check if the downstream client has disconnected
                if cancel_event.is_set():
                    logger.info("Client disconnected, aborting upstream stream")
                    return

                decoded = line.decode("utf-8", errors="replace")

                if transformer is not None:
                    for output_line in transformer.process_raw(decoded):
                        yield output_line.encode("utf-8")
                else:
                    yield line

            if transformer is not None:
                for output_line in transformer.flush():
                    if cancel_event.is_set():
                        return
                    yield output_line.encode("utf-8")
    except asyncio.CancelledError:
        logger.info("Upstream stream cancelled (client disconnect or shutdown)")
        raise
    except aiohttp.ClientError as exc:
        logger.error("Upstream connection error: %s", exc)
        raise


async def handle_proxy(request: web.Request) -> web.StreamResponse | web.Response:
    """Main proxy handler for all requests."""
    path = request.path
    method = request.method
    litellm_url = config.LITELLM_URL.rstrip("/")
    url = f"{litellm_url}{path}"

    headers = _forward_headers(request)
    body = await request.read()
    body_json = await _read_request_body(request)
    is_streaming = await _is_streaming_request(body_json)
    model = body_json.get("model", "unknown")

    logger.info(
        "%s %s model=%s streaming=%s buffered=%s",
        method,
        path,
        model,
        is_streaming,
        is_streaming and config.BUFFER_TOOL_CALLS and path in STREAMING_ROUTES,
    )

    if is_streaming and config.BUFFER_TOOL_CALLS and path in STREAMING_ROUTES:
        return await _handle_streaming(request, url, headers, body)
    else:
        return await _handle_non_streaming(request, url, headers, body, method)


async def _handle_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> web.StreamResponse:
    """Handle a streaming request with tool call buffering.

    Streaming requests use ``Connection: close`` to ensure the upstream
    TCP connection is closed after the stream ends. This prevents stale
    connections from accumulating with unacked data in their Send-Q.
    """
    headers["Accept"] = "text/event-stream"
    headers["Cache-Control"] = "no-cache"

    logger.info("Starting streaming proxy to %s", url)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    # Event set when the downstream client disconnects
    cancel_event = asyncio.Event()

    # Monitor for client disconnection
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request, cancel_event))

    session: aiohttp.ClientSession = request.app["client_session"]

    try:
        # Hard ceiling on total stream duration. sock_read resets on each
        # chunk, so a slow trickle can keep a stream alive indefinitely.
        # STREAM_MAX_DURATION kills the stream regardless.
        stream_task = asyncio.ensure_future(
            _stream_to_response(
                session, request, url, headers, body, cancel_event, response
            )
        )
        try:
            await asyncio.wait_for(stream_task, timeout=config.STREAM_MAX_DURATION)
        except asyncio.TimeoutError:
            logger.warning(
                "Streaming request exceeded max duration (%ds), closing",
                config.STREAM_MAX_DURATION,
            )
            stream_task.cancel()
    except (aiohttp.ClientError, asyncio.CancelledError):
        logger.warning(
            "Streaming proxy request failed (upstream error or cancellation)"
        )
    finally:
        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass

    logger.info("Streaming proxy request completed")
    await response.write_eof()
    return response


async def _stream_to_response(
    session: aiohttp.ClientSession,
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    cancel_event: asyncio.Event,
    response: web.StreamResponse,
) -> None:
    """Stream upstream response chunks to the downstream client.

    This is the core streaming loop, extracted so it can be wrapped in
    an ``asyncio.wait_for`` for a hard total-duration timeout.
    """
    async for chunk in _stream_response(
        session,
        method=request.method,
        url=url,
        headers=headers,
        body=body,
        buffer_tool_calls=True,
        validate_json=config.VALIDATE_JSON_ARGS,
        cancel_event=cancel_event,
    ):
        await response.write(chunk)


async def _wait_for_disconnect(
    request: web.Request, cancel_event: asyncio.Event
) -> None:
    """Wait for the downstream client to disconnect and set the cancel event.

    This allows the upstream stream to be aborted immediately when the
    client goes away, rather than continuing to read from litellm into the void.
    """
    try:
        # Poll the transport — aiohttp doesn't expose a clean disconnect awaitable
        while not cancel_event.is_set():
            transport = request.transport
            if transport is None or transport.is_closing():
                logger.info("Client transport closing, setting cancel event")
                cancel_event.set()
                return
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass


async def _handle_non_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    method: str,
) -> web.Response:
    """Handle a non-streaming request with transparent forwarding."""
    session: aiohttp.ClientSession = request.app["client_session"]

    timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)

    try:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=timeout,
        ) as resp:
            resp_body = await resp.read()
            response = web.Response(
                status=resp.status,
                body=resp_body,
            )
            for key, value in resp.headers.items():
                lower = key.lower()
                if lower in NON_STREAMING_FORWARD_HEADERS:
                    response.headers[key] = value
            return response
    except aiohttp.ClientError as exc:
        logger.error("Non-streaming upstream error: %s", exc)
        return web.Response(
            status=502,
            content_type="application/json",
            text=json.dumps({"error": f"upstream error: {exc}"}),
        )
    except asyncio.TimeoutError:
        logger.error("Non-streaming upstream timeout (%ds)", config.REQUEST_TIMEOUT)
        return web.Response(
            status=504,
            content_type="application/json",
            text=json.dumps(
                {"error": f"upstream timeout after {config.REQUEST_TIMEOUT}s"}
            ),
        )


# ---------------------------------------------------------------------------
# Application lifecycle: shared ClientSession
# ---------------------------------------------------------------------------


async def on_startup(app: web.Application) -> None:
    """Create a shared aiohttp.ClientSession for all upstream requests.

    Uses connection pooling with keepalive for efficiency. Idle connections
    are closed after KEEPALIVE_TIMEOUT seconds.
    """
    connector = aiohttp.TCPConnector(
        limit=config.MAX_UPSTREAM_CONNECTIONS,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=config.KEEPALIVE_TIMEOUT,
    )
    timeout = aiohttp.ClientTimeout(
        connect=config.CONNECT_TIMEOUT,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
    )
    app["client_session"] = session
    logger.info(
        "Created shared ClientSession (limit=%d, connect_timeout=%ds, keepalive=%ds)",
        config.MAX_UPSTREAM_CONNECTIONS,
        config.CONNECT_TIMEOUT,
        config.KEEPALIVE_TIMEOUT,
    )


async def on_cleanup(app: web.Application) -> None:
    """Close the shared ClientSession on shutdown."""
    session: aiohttp.ClientSession = app["client_session"]
    await session.close()
    logger.info("Closed shared ClientSession")
