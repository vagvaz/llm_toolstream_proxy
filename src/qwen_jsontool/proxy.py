"""aiohttp-based reverse proxy that intercepts and transforms streaming SSE responses.

Architecture::

    opencode  -->  localhost:8787 (this proxy)  -->  litellm:4000

Non-streaming requests are forwarded transparently with no modification.
Streaming requests with ``stream: true`` are intercepted: the SSE response
is parsed line-by-line, tool_call deltas are buffered and reassembled, and
the cleaned SSE stream is returned to opencode.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

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


async def _is_streaming_request(body: dict) -> bool:
    return body.get("stream", False) is True


async def _read_request_body(request: web.Request) -> dict:
    """Read and parse the JSON request body."""
    raw = await request.read()
    if not raw:
        return {}
    try:
        return json.loads(raw)
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
) -> AsyncIterator[bytes]:
    """Stream the response from litellm, applying tool call buffering if needed."""
    transformer = (
        SSETransformer(validate_json=validate_json) if buffer_tool_calls else None
    )

    async with session.request(
        method=method,
        url=url,
        headers=headers,
        data=body,
        timeout=aiohttp.ClientTimeout(sock_read=config.STREAM_TIMEOUT),
    ) as resp:
        async for line in resp.content:
            decoded = line.decode("utf-8", errors="replace")

            if transformer is not None:
                for output_line in transformer.process_raw(decoded):
                    yield output_line.encode("utf-8")
            else:
                yield line

        if transformer is not None:
            for output_line in transformer.flush():
                yield output_line.encode("utf-8")


async def handle_proxy(request: web.Request) -> web.StreamResponse:
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
    """Handle a streaming request with tool call buffering."""
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

    async with aiohttp.ClientSession() as session:
        async for chunk in _stream_response(
            session,
            method=request.method,
            url=url,
            headers=headers,
            body=body,
            buffer_tool_calls=True,
            validate_json=config.VALIDATE_JSON_ARGS,
        ):
            await response.write(chunk)

    logger.info("Streaming proxy request completed")
    await response.write_eof()
    return response


async def _handle_non_streaming(
    request: web.Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    method: str,
) -> web.Response:
    """Handle a non-streaming request with transparent forwarding."""
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=aiohttp.ClientTimeout(total=300),
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
