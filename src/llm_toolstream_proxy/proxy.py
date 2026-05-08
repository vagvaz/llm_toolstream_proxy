"""aiohttp-based reverse proxy that intercepts and transforms streaming SSE responses.

Architecture::

    opencode  -->  localhost:8787 (this proxy)  -->  litellm:4000

Non-streaming requests are forwarded transparently with no modification.
Streaming requests with ``stream: true`` are intercepted: the SSE response
is parsed line-by-line, tool_call deltas are buffered and reassembled, and
the cleaned SSE stream is returned to opencode.

Streaming orchestration is delegated to ``streaming.py``, which exposes a
testable seam for semaphore acquisition, shutdown gating, client disconnect
detection, and STREAM_MAX_DURATION timeout wrapping.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
from aiohttp import web
from loguru import logger

from . import config
from .metrics import RequestMetrics
from .streaming import (
    _active_streams,
    _client_session,
    _shutdown_event,
    _stream_semaphore,
    handle_streaming,
)

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
    hosts = config.LITELLM_URL.split("//", 1)[-1].split("/")[0].split(":")[0]
    headers["Host"] = hosts
    return headers


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
            result = await handle_streaming(request, url, headers, body, req_metrics)
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


async def _handle_non_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    method: str,
    req_metrics: RequestMetrics,
) -> web.Response:
    """Handle a non-streaming request with transparent forwarding."""
    session: aiohttp.ClientSession = request.app[_client_session]

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
    """Create a shared aiohttp ClientSession and stream semaphore.

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
    app[_client_session] = session
    app[_stream_semaphore] = asyncio.Semaphore(config.MAX_CONCURRENT_STREAMS)
    app[_shutdown_event] = asyncio.Event()
    app[_active_streams] = [0]
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
    shutdown_event: asyncio.Event = app[_shutdown_event]
    shutdown_event.set()

    # Wait for active streams to drain
    active = app[_active_streams][0]
    if active > 0:
        logger.info("Waiting for {} active stream(s) to finish...", active)
        for _ in range(55):  # poll every second, up to 55s
            await asyncio.sleep(1)
            remaining = app[_active_streams][0]
            if remaining == 0:
                logger.info("All active streams finished")
                break
            logger.info("Still waiting — {} active stream(s) remaining", remaining)
        else:
            logger.warning(
                "Timeout waiting for active streams — {} still running, forcing close",
                app[_active_streams][0],
            )
    else:
        logger.info("No active streams — shutting down immediately")

    session: aiohttp.ClientSession = app[_client_session]
    await session.close()
    logger.info("Closed shared ClientSession")
