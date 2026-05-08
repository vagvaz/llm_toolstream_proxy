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

from . import config as cfg_module
from .config import Config
from .metrics import MetricsCollector, RequestMetrics
from .streaming import (
    _client_session,
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


async def _read_request_body(request: web.Request, cfg: Config) -> dict[str, Any]:
    """Read and parse the JSON request body."""
    raw = await request.read()
    if len(raw) > cfg.MAX_REQUEST_BODY_SIZE:
        raise web.HTTPRequestEntityTooLarge(
            max_size=cfg.MAX_REQUEST_BODY_SIZE,
            actual_size=len(raw),
            content_type="application/json",
            text=json.dumps(
                {
                    "error": "request body too large",
                    "detail": f"max body size is {cfg.MAX_REQUEST_BODY_SIZE} bytes",
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


def _forward_headers(request: web.Request, cfg: Config) -> dict[str, str]:
    """Build headers dict for forwarding the request to litellm."""
    headers = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in ("host", "content-length", "transfer-encoding"):
            continue
        headers[key] = value
    hosts = cfg.LITELLM_URL.split("//", 1)[-1].split("/")[0].split(":")[0]
    headers["Host"] = hosts
    return headers


async def handle_proxy(
    request: web.Request,
    *,
    cfg: Config | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> web.StreamResponse | web.Response:
    """Main proxy handler for all requests."""
    if cfg is None:
        cfg = cfg_module.config
    if metrics_collector is None:
        from .metrics import collector

        metrics_collector = collector

    path = request.path
    method = request.method
    litellm_url = cfg.LITELLM_URL.rstrip("/")
    url = f"{litellm_url}{path}"

    headers = _forward_headers(request, cfg)
    body = await request.read()
    body_json = await _read_request_body(request, cfg)
    is_streaming = await _is_streaming_request(body_json)
    model = body_json.get("model", "unknown")

    req_metrics = metrics_collector.new_request(method, path, model, is_streaming)

    logger.info(
        "[{}] {} {} model={} streaming={} buffered={}",
        req_metrics.request_id,
        method,
        path,
        model,
        is_streaming,
        is_streaming and cfg.BUFFER_TOOL_CALLS and path in STREAMING_ROUTES,
    )

    try:
        if is_streaming and cfg.BUFFER_TOOL_CALLS and path in STREAMING_ROUTES:
            result = await handle_streaming(
                request, url, headers, body, req_metrics,
                cfg=cfg, metrics_collector=metrics_collector,
            )
            return result
        else:
            result = await _handle_non_streaming(
                request, url, headers, body, method, req_metrics,
                cfg=cfg,
            )
            return result
    except Exception:
        req_metrics.error = "unhandled_exception"
        metrics_collector.record_error("unhandled_exception")
        raise
    finally:
        metrics_collector.finish_request(req_metrics)


async def _handle_non_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    method: str,
    req_metrics: RequestMetrics,
    *,
    cfg: Config,
) -> web.Response:
    """Handle a non-streaming request with transparent forwarding."""
    session: aiohttp.ClientSession = request.app[_client_session]

    timeout = aiohttp.ClientTimeout(total=cfg.REQUEST_TIMEOUT)

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
        logger.error("Non-streaming upstream timeout ({}s)", cfg.REQUEST_TIMEOUT)
        req_metrics.error = "upstream_timeout"
        return web.Response(
            status=504,
            content_type="application/json",
            text=json.dumps(
                {"error": f"upstream timeout after {cfg.REQUEST_TIMEOUT}s"}
            ),
        )
