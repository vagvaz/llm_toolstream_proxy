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
    - ``MAX_CONCURRENT_STREAMS`` limits concurrent streaming requests; excess
      requests receive 503 Service Unavailable.
    - ``response.drain()`` applies backpressure when the downstream client is slow,
      preventing Send-Q accumulation on the proxy-to-client connection.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import aiohttp
from aiohttp import web
from loguru import logger

from . import config
from .metrics import RequestMetrics
from .sse import SSETransformer

_START_TIME = time.monotonic()

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
    """Health check endpoint returning proxy status and uptime."""
    from .metrics import collector

    return web.json_response(
        {
            "status": "ok",
            "version": "0.1.0",
            "upstream": config.LITELLM_URL,
            "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
            "active_streams": collector.active_streams,
        }
    )


async def handle_metrics(request: web.Request) -> web.Response:
    """Metrics endpoint returning aggregate request statistics."""
    from .metrics import collector

    return web.json_response(collector.get_metrics())


async def _is_streaming_request(body: dict[str, Any]) -> bool:
    return body.get("stream", False) is True


async def _read_request_body(request: web.Request) -> dict[str, Any]:
    """Read and parse the JSON request body."""
    raw = await request.read()
    if len(raw) > config.MAX_REQUEST_BODY_SIZE:
        raise web.HTTPRequestEntityTooLarge(
            max_size=config.MAX_REQUEST_BODY_SIZE,
            actual_size=len(raw),
            content_type="application/json",
            text=json.dumps(
                {
                    "error": "request body too large",
                    "detail": f"max body size is {config.MAX_REQUEST_BODY_SIZE} bytes",
                }
            ),
        )
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
    req_metrics: RequestMetrics | None = None,
) -> AsyncIterator[bytes]:
    """Stream the response from litellm, applying tool call buffering if needed.

    Args:
        session: Shared aiohttp ClientSession for connection reuse.
        cancel_event: Set when the downstream client disconnects, causing
            the upstream request to be cancelled immediately.
        req_metrics: Optional RequestMetrics to track TTFB and bytes.
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

    ttfb_recorded = False

    try:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=timeout,
        ) as resp:
            async for line in resp.content:
                # Record TTFB on first chunk
                if not ttfb_recorded and req_metrics is not None:
                    req_metrics.ttfb = time.monotonic() - req_metrics.start_time
                    ttfb_recorded = True

                # Check if the downstream client has disconnected
                if cancel_event.is_set():
                    logger.info("Client disconnected, aborting upstream stream")
                    return

                decoded = line.decode("utf-8", errors="replace")

                if transformer is not None:
                    for output_line in transformer.process_raw(decoded):
                        chunk = output_line.encode("utf-8")
                        if req_metrics is not None:
                            req_metrics.bytes_transferred += len(chunk)
                            req_metrics.chunks_processed += 1
                        yield chunk
                else:
                    if req_metrics is not None:
                        req_metrics.bytes_transferred += len(line)
                        req_metrics.chunks_processed += 1
                    yield line

            if transformer is not None:
                for output_line in transformer.flush():
                    if cancel_event.is_set():
                        return
                    chunk = output_line.encode("utf-8")
                    if req_metrics is not None:
                        req_metrics.bytes_transferred += len(chunk)
                        req_metrics.chunks_processed += 1
                    yield chunk
    except asyncio.CancelledError:
        logger.info("Upstream stream cancelled (client disconnect or shutdown)")
        raise
    except aiohttp.ClientError as exc:
        logger.error("Upstream connection error: {}", exc)
        raise


async def handle_proxy(request: web.Request) -> web.StreamResponse | web.Response:
    """Main proxy handler for all requests."""
    from .metrics import collector

    path = request.path
    method = request.method
    litellm_url = config.LITELLM_URL.rstrip("/")
    url = f"{litellm_url}{path}"

    headers = _forward_headers(request)
    body = await request.read()
    body_json = await _read_request_body(request)
    is_streaming = await _is_streaming_request(body_json)
    model = body_json.get("model", "unknown")

    req_metrics = collector.new_request(method, path, model, is_streaming)

    logger.info(
        "[{}] {} {} model={} streaming={} buffered={}",
        req_metrics.request_id,
        method,
        path,
        model,
        is_streaming,
        is_streaming and config.BUFFER_TOOL_CALLS and path in STREAMING_ROUTES,
    )

    try:
        if is_streaming and config.BUFFER_TOOL_CALLS and path in STREAMING_ROUTES:
            result = await _handle_streaming(request, url, headers, body, req_metrics)
            return result
        else:
            result = await _handle_non_streaming(
                request, url, headers, body, method, req_metrics
            )
            return result
    except Exception:
        req_metrics.error = "unhandled_exception"
        collector.record_error("unhandled_exception")
        raise
    finally:
        collector.finish_request(req_metrics)


async def _handle_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    req_metrics: RequestMetrics,
) -> web.StreamResponse:
    """Handle a streaming request with tool call buffering.

    Applies backpressure via ``response.drain()`` to prevent Send-Q
    accumulation when the downstream client is slow. Limits concurrent
    streams via a semaphore; returns 503 when the limit is exceeded or
    the proxy is shutting down.
    """
    from .metrics import collector

    app = request.app

    # Reject new requests if the proxy is shutting down
    shutdown_event: asyncio.Event = app["shutdown_event"]
    if shutdown_event.is_set():
        collector.record_error("proxy_shutting_down")
        req_metrics.error = "proxy_shutting_down"
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {"error": "proxy shutting down", "detail": "try again shortly"}
            ),
        )

    # Check concurrent stream limit
    semaphore: asyncio.Semaphore = app["stream_semaphore"]
    if semaphore.locked():
        # All slots are taken — reject with 503
        logger.warning(
            "Rejecting streaming request: MAX_CONCURRENT_STREAMS ({}) reached",
            config.MAX_CONCURRENT_STREAMS,
        )
        collector.record_error("concurrent_stream_limit")
        req_metrics.error = "concurrent_stream_limit"
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {
                    "error": "proxy overloaded",
                    "detail": (
                        f"max concurrent streams "
                        f"({config.MAX_CONCURRENT_STREAMS}) reached"
                    ),
                }
            ),
        )

    async with semaphore:
        # Track active streams for graceful shutdown
        app["active_streams"] = app["active_streams"] + 1
        try:
            headers["Accept"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"

            logger.info(
                "[{}] Starting streaming proxy to {}", req_metrics.request_id, url
            )

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
            disconnect_task = asyncio.create_task(
                _wait_for_disconnect(request, cancel_event)
            )

            session: aiohttp.ClientSession = request.app["client_session"]

            try:
                # Hard ceiling on total stream duration. sock_read resets on each
                # chunk, so a slow trickle can keep a stream alive indefinitely.
                # STREAM_MAX_DURATION kills the stream regardless.
                stream_task = asyncio.ensure_future(
                    _stream_to_response(
                        session,
                        request,
                        url,
                        headers,
                        body,
                        cancel_event,
                        response,
                        req_metrics,
                    )
                )
                try:
                    await asyncio.wait_for(
                        stream_task, timeout=config.STREAM_MAX_DURATION
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[{}] Streaming request exceeded max duration ({}s), closing",
                        req_metrics.request_id,
                        config.STREAM_MAX_DURATION,
                    )
                    req_metrics.error = "stream_max_duration"
                    stream_task.cancel()
            except (aiohttp.ClientError, asyncio.CancelledError):
                logger.warning(
                    "[{}] Streaming proxy request failed "
                    "(upstream error or cancellation)",
                    req_metrics.request_id,
                )
                if not req_metrics.error:
                    req_metrics.error = "upstream_error"
            finally:
                disconnect_task.cancel()
                try:
                    await disconnect_task
                except asyncio.CancelledError:
                    pass

            logger.info(
                "[{}] Streaming proxy request completed", req_metrics.request_id
            )
            await response.write_eof()
        finally:
            # Decrement active streams counter — must happen even on exception
            app["active_streams"] = app["active_streams"] - 1

        return response


async def _stream_to_response(
    session: aiohttp.ClientSession,
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    cancel_event: asyncio.Event,
    response: web.StreamResponse,
    req_metrics: RequestMetrics | None = None,
) -> None:
    """Stream upstream response chunks to the downstream client.

    This is the core streaming loop, extracted so it can be wrapped in
    an ``asyncio.wait_for`` for a hard total-duration timeout.
    Applies backpressure via ``response.drain()`` after each write.
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
        req_metrics=req_metrics,
    ):
        await response.write(chunk)
        await response.drain()


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
    req_metrics: RequestMetrics,
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
            if req_metrics is not None:
                req_metrics.bytes_transferred = len(resp_body)
                req_metrics.ttfb = time.monotonic() - req_metrics.start_time
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
        logger.error("Non-streaming upstream error: {}", exc)
        req_metrics.error = "upstream_error"
        return web.Response(
            status=502,
            content_type="application/json",
            text=json.dumps({"error": f"upstream error: {exc}"}),
        )
    except asyncio.TimeoutError:
        logger.error("Non-streaming upstream timeout ({}s)", config.REQUEST_TIMEOUT)
        req_metrics.error = "upstream_timeout"
        return web.Response(
            status=504,
            content_type="application/json",
            text=json.dumps(
                {"error": f"upstream timeout after {config.REQUEST_TIMEOUT}s"}
            ),
        )


# ---------------------------------------------------------------------------
# Application lifecycle: shared ClientSession, stream semaphore
# ---------------------------------------------------------------------------


async def on_startup(app: web.Application) -> None:
    """Create a shared aiohttp.ClientSession and stream semaphore.

    Uses connection pooling with keepalive for efficiency. Idle connections
    are closed after KEEPALIVE_TIMEOUT seconds. The semaphore limits
    concurrent streaming requests to MAX_CONCURRENT_STREAMS.
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
    app["stream_semaphore"] = asyncio.Semaphore(config.MAX_CONCURRENT_STREAMS)
    app["shutdown_event"] = asyncio.Event()
    app["active_streams"] = 0
    logger.info(
        "Created shared ClientSession (limit={}, connect_timeout={}s, keepalive={}s) "
        "max_concurrent_streams={}",
        config.MAX_UPSTREAM_CONNECTIONS,
        config.CONNECT_TIMEOUT,
        config.KEEPALIVE_TIMEOUT,
        config.MAX_CONCURRENT_STREAMS,
    )


async def on_cleanup(app: web.Application) -> None:
    """Gracefully shut down the proxy.

    1. Signal shutdown — reject new streaming requests with 503.
    2. Wait up to 55s for active streams to drain (systemd sends SIGKILL
       after TimeoutStopSec=60s; we leave 5s margin).
    3. Close the shared ClientSession, cancelling any remaining streams.
    """
    logger.info("Shutdown initiated — signaling stop to new streams")
    shutdown_event: asyncio.Event = app["shutdown_event"]
    shutdown_event.set()

    # Wait for active streams to drain
    active = app["active_streams"]
    if active > 0:
        logger.info("Waiting for {} active stream(s) to finish...", active)
        for _ in range(55):  # poll every second, up to 55s
            await asyncio.sleep(1)
            remaining = app["active_streams"]
            if remaining == 0:
                logger.info("All active streams finished")
                break
            logger.info("Still waiting — {} active stream(s) remaining", remaining)
        else:
            logger.warning(
                "Timeout waiting for active streams — {} still running, forcing close",
                app["active_streams"],
            )
    else:
        logger.info("No active streams — shutting down immediately")

    session: aiohttp.ClientSession = app["client_session"]
    await session.close()
    logger.info("Closed shared ClientSession")
