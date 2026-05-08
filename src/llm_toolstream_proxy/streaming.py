"""Streaming request handler with tool call buffering, concurrency limits,
and graceful shutdown support.

Extracted from ``proxy.py`` to provide a testable seam for the streaming
orchestration: semaphore acquisition, shutdown gating, client disconnect
detection, and STREAM_MAX_DURATION timeout wrapping.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import aiohttp
from aiohttp import web
from loguru import logger

from . import config as cfg_module
from .config import Config
from .metrics import MetricsCollector, RequestMetrics
from .sse import SSETransformer

# --- aiohttp AppKey constants for typed application state ---

_client_session = web.AppKey("client_session", aiohttp.ClientSession)
_stream_semaphore = web.AppKey("stream_semaphore", asyncio.Semaphore)
_shutdown_event = web.AppKey("shutdown_event", asyncio.Event)
_active_streams = web.AppKey("active_streams", list)
_metrics_collector = web.AppKey("metrics_collector", MetricsCollector)


async def _stream_response(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    *,
    cfg: Config,
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
            max_arguments_size=cfg.MAX_ARGUMENTS_SIZE,
            max_tool_calls=cfg.MAX_TOOL_CALLS,
        )
        if buffer_tool_calls
        else None
    )

    timeout = aiohttp.ClientTimeout(
        total=None,  # no total limit — controlled by STREAM_MAX_DURATION wrapper
        connect=cfg.CONNECT_TIMEOUT,
        sock_read=cfg.STREAM_TIMEOUT,
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


async def _stream_to_response(
    session: aiohttp.ClientSession,
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    cancel_event: asyncio.Event,
    response: web.StreamResponse,
    *,
    cfg: Config,
    req_metrics: RequestMetrics | None = None,
) -> None:
    """Stream upstream response chunks to the downstream client.

    This is the core streaming loop, extracted so it can be wrapped in
    an ``asyncio.wait_for`` for a hard total-duration timeout.
    Applies backpressure via ``response.write()`` after each chunk.
    """
    async for chunk in _stream_response(
        session,
        method=request.method,
        url=url,
        headers=headers,
        body=body,
        cfg=cfg,
        buffer_tool_calls=True,
        validate_json=cfg.VALIDATE_JSON_ARGS,
        cancel_event=cancel_event,
        req_metrics=req_metrics,
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


async def handle_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    req_metrics: RequestMetrics,
    *,
    cfg: Config | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> web.StreamResponse:
    """Handle a streaming request with tool call buffering.

    Applies backpressure via ``response.write()`` to prevent Send-Q
    accumulation when the downstream client is slow. Limits concurrent
    streams via a semaphore; returns 503 when the limit is exceeded or
    the proxy is shutting down.
    """
    if cfg is None:
        cfg = cfg_module.config
    if metrics_collector is None:
        from .metrics import collector as _default_collector

        metrics_collector = _default_collector

    app = request.app

    # Reject new requests if the proxy is shutting down
    shutdown_event: asyncio.Event = app[_shutdown_event]
    if shutdown_event.is_set():
        metrics_collector.record_error("proxy_shutting_down")
        req_metrics.error = "proxy_shutting_down"
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {"error": "proxy shutting down", "detail": "try again shortly"}
            ),
        )

    # Check concurrent stream limit
    semaphore: asyncio.Semaphore = app[_stream_semaphore]
    if semaphore.locked():
        # All slots are taken — reject with 503
        logger.warning(
            "Rejecting streaming request: MAX_CONCURRENT_STREAMS ({}) reached",
            cfg.MAX_CONCURRENT_STREAMS,
        )
        metrics_collector.record_error("concurrent_stream_limit")
        req_metrics.error = "concurrent_stream_limit"
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {
                    "error": "proxy overloaded",
                    "detail": (
                        f"max concurrent streams "
                        f"({cfg.MAX_CONCURRENT_STREAMS}) reached"
                    ),
                }
            ),
        )

    async with semaphore:
        # Track active streams for graceful shutdown (mutable list avoids
        # the "Changing state of started application" deprecation).
        counts = app[_active_streams]
        counts[0] += 1
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

            session: aiohttp.ClientSession = request.app[_client_session]

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
                        cfg=cfg,
                        req_metrics=req_metrics,
                    )
                )
                try:
                    await asyncio.wait_for(
                        stream_task, timeout=cfg.STREAM_MAX_DURATION
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[{}] Streaming request exceeded max duration ({}s), closing",
                        req_metrics.request_id,
                        cfg.STREAM_MAX_DURATION,
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
            counts[0] -= 1

        return response
