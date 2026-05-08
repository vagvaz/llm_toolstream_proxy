"""HTTP handlers for health check and metrics endpoints.

Extracted from ``proxy.py`` — these handlers have no proxy-specific
dependencies; they only depend on ``config`` and ``metrics``.
"""

from __future__ import annotations

import time

from aiohttp import web

from .config import Config
from .metrics import MetricsCollector

_START_TIME = time.monotonic()


async def handle_health(
    request: web.Request,
    *,
    cfg: Config,
) -> web.Response:
    """Health check endpoint returning proxy status and uptime.

    Args:
        cfg: Explicitly required — no hidden global fallback. Caller must
            provide a Config instance (typically via ``create_app``).
    """
    return web.json_response(
        {
            "status": "ok",
            "version": "0.1.0",
            "upstream": cfg.LITELLM_URL,
            "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
        }
    )


async def handle_metrics(
    request: web.Request,
    *,
    metrics_collector: MetricsCollector,
) -> web.Response:
    """Metrics endpoint returning aggregate request statistics.

    Args:
        metrics_collector: Explicitly required — no hidden global fallback.
            Caller must provide a MetricsCollector instance (typically via
            ``create_app``).
    """
    return web.json_response(metrics_collector.get_metrics())
