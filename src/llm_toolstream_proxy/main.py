"""Entry point for llm-toolstream-proxy server."""

from __future__ import annotations

import sys

from aiohttp import web
from loguru import logger

from . import config
from .proxy import handle_health, handle_metrics, handle_proxy, on_cleanup, on_startup


def setup_logging() -> None:
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
        level=config.LOG_LEVEL,
        format=console_fmt,
        colorize=True,
    )

    logger.add(
        config.LOG_FILE,
        level=config.LOG_LEVEL,
        format=file_fmt,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info(
        "llm-toolstream-proxy logging initialized | level={} | file={}",
        config.LOG_LEVEL,
        config.LOG_FILE,
    )


def create_app() -> web.Application:
    """Create the aiohttp application with all routes and lifecycle hooks."""
    app = web.Application()

    # Lifecycle hooks for shared ClientSession
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_route("*", "/v1/{path:.*}", handle_proxy)
    app.router.add_route("*", "/{path:.*}", handle_proxy)

    return app


def main() -> None:
    """Run the proxy server.

    Uses uvloop for better async performance if available.
    Falls back to the default asyncio event loop if uvloop is not installed.

    IMPORTANT: Do NOT deploy with gunicorn. Gunicorn's worker timeout model
    kills workers mid-stream, leaving orphaned TCP connections with large
    Send-Q buffers that stall the machine. Use this entry point directly.
    """
    setup_logging()
    config.validate_config()

    # Try to use uvloop for better async performance
    try:
        import uvloop

        uvloop.install()
        logger.info("Using uvloop event loop")
    except ImportError:
        logger.info("uvloop not available, using default asyncio event loop")

    app = create_app()
    logger.info(
        "llm-toolstream-proxy starting on {}:{} -> {} "
        "(buffer_tool_calls={}, validate_json={})",
        config.PROXY_HOST,
        config.PROXY_PORT,
        config.LITELLM_URL,
        config.BUFFER_TOOL_CALLS,
        config.VALIDATE_JSON_ARGS,
    )
    web.run_app(app, host=config.PROXY_HOST, port=config.PROXY_PORT)


if __name__ == "__main__":
    main()
