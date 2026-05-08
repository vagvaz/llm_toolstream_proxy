"""Entry point for llm-toolstream-proxy server."""

from __future__ import annotations

import asyncio
import sys

import aiohttp
from aiohttp import web
from loguru import logger

from . import config as cfg_module
from .config import Config
from .handlers import handle_health, handle_metrics
from .metrics import MetricsCollector
from .proxy import handle_proxy
from .streaming import (
    _active_streams,
    _client_session,
    _metrics_collector,
    _shutdown_event,
    _stream_semaphore,
)


def setup_logging(cfg: Config) -> None:
    """Configure loguru with console + file output."""
    logger.remove()

    console_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} "
        "| <level>{level:<8}</level> "
        "| {name}:{function}:{line} "
        "| <level>{message}</level>"
    )
    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} "
        "| {level:<8} "
        "| {name}:{function}:{line} "
        "| {message}"
    )

    logger.add(
        sys.stderr,
        level=cfg.LOG_LEVEL,
        format=console_fmt,
        colorize=True,
    )

    logger.add(
        cfg.LOG_FILE,
        level=cfg.LOG_LEVEL,
        format=file_fmt,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info(
        "llm-toolstream-proxy logging initialized | level={} | file={}",
        cfg.LOG_LEVEL,
        cfg.LOG_FILE,
    )


async def on_startup(app: web.Application, cfg: Config) -> None:
    """Create a shared aiohttp ClientSession and stream semaphore.

    Uses connection pooling with keepalive for efficiency. Idle connections
    are closed after KEEPALIVE_TIMEOUT seconds. The semaphore limits
    concurrent streaming requests to MAX_CONCURRENT_STREAMS.
    """
    connector = aiohttp.TCPConnector(
        limit=cfg.MAX_UPSTREAM_CONNECTIONS,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=cfg.KEEPALIVE_TIMEOUT,
    )
    timeout = aiohttp.ClientTimeout(
        connect=cfg.CONNECT_TIMEOUT,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
    )
    app[_client_session] = session
    app[_stream_semaphore] = asyncio.Semaphore(cfg.MAX_CONCURRENT_STREAMS)
    app[_shutdown_event] = asyncio.Event()
    app[_active_streams] = [0]
    logger.info(
        "Created shared ClientSession (limit={}, connect_timeout={}s, keepalive={}s) "
        "max_concurrent_streams={}",
        cfg.MAX_UPSTREAM_CONNECTIONS,
        cfg.CONNECT_TIMEOUT,
        cfg.KEEPALIVE_TIMEOUT,
        cfg.MAX_CONCURRENT_STREAMS,
    )


async def on_cleanup(app: web.Application, cfg: Config) -> None:
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


def create_app(
    cfg: Config | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> web.Application:
    """Create the aiohttp application with all routes and lifecycle hooks."""
    if cfg is None:
        cfg = cfg_module.config
    if metrics_collector is None:
        metrics_collector = MetricsCollector()

    app = web.Application()

    # Store metrics_collector in app state for handlers
    app[_metrics_collector] = metrics_collector

    # Lifecycle hooks for shared ClientSession
    app.on_startup.append(lambda app: on_startup(app, cfg))
    app.on_cleanup.append(lambda app: on_cleanup(app, cfg))

    # Route helpers that inject config and metrics_collector
    async def _health(request: web.Request) -> web.Response:
        return await handle_health(request, cfg=cfg)

    async def _metrics(request: web.Request) -> web.Response:
        return await handle_metrics(request, metrics_collector=metrics_collector)

    async def _proxy(request: web.Request) -> web.StreamResponse | web.Response:
        return await handle_proxy(
            request, cfg=cfg, metrics_collector=metrics_collector
        )

    app.router.add_get("/health", _health)
    app.router.add_get("/metrics", _metrics)
    app.router.add_route("*", "/v1/{path:.*}", _proxy)
    app.router.add_route("*", "/{path:.*}", _proxy)

    return app


def main() -> None:
    """Run the proxy server.

    Uses uvloop for better async performance if available.
    Falls back to the default asyncio event loop if uvloop is not installed.

    IMPORTANT: Do NOT deploy with gunicorn. Gunicorn's worker timeout model
    kills workers mid-stream, leaving orphaned TCP connections with large
    Send-Q buffers that stall the machine. Use this entry point directly.
    """
    cfg = Config.from_env()
    setup_logging(cfg)
    cfg.validate()

    # Try to use uvloop for better async performance
    try:
        import uvloop

        uvloop.install()
        logger.info("Using uvloop event loop")
    except ImportError:
        logger.info("uvloop not available, using default asyncio event loop")

    app = create_app(cfg=cfg)
    logger.info(
        "llm-toolstream-proxy starting on {}:{} -> {} "
        "(buffer_tool_calls={}, validate_json={})",
        cfg.PROXY_HOST,
        cfg.PROXY_PORT,
        cfg.LITELLM_URL,
        cfg.BUFFER_TOOL_CALLS,
        cfg.VALIDATE_JSON_ARGS,
    )
    web.run_app(app, host=cfg.PROXY_HOST, port=cfg.PROXY_PORT)


if __name__ == "__main__":
    main()
