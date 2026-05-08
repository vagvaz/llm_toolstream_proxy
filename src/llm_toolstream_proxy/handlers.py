"""HTTP handlers for health check and metrics endpoints.

Extracted from ``proxy.py`` — these handlers have no proxy-specific
dependencies; they only depend on ``config`` and ``metrics``.
"""

from __future__ import annotations

import time

from aiohttp import web

from . import config

_START_TIME = time.monotonic()


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint returning proxy status and uptime."""
    return web.json_response(
        {
            "status": "ok",
            "version": "0.1.0",
            "upstream": config.LITELLM_URL,
            "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
        }
    )


async def handle_metrics(request: web.Request) -> web.Response:
    """Metrics endpoint returning aggregate request statistics."""
    from .metrics import collector

    return web.json_response(collector.get_metrics())
