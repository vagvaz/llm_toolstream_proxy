"""Entry point for qwen-jsontool proxy server."""

from __future__ import annotations

import sys

from aiohttp import web
from loguru import logger

from . import config
from .proxy import handle_health, handle_proxy


def setup_logging() -> None:
    """Configure loguru with console + file output."""
    logger.remove()

    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level:<8}</level> | {name}:{function}:{line} | <level>{message}</level>",
        colorize=True,
    )

    logger.add(
        config.LOG_FILE,
        level=config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info(
        "qwen-jsontool logging initialized | level={} | file={}",
        config.LOG_LEVEL,
        config.LOG_FILE,
    )


def create_app() -> web.Application:
    """Create the aiohttp application with all routes."""
    app = web.Application()

    app.router.add_get("/health", handle_health)
    app.router.add_route("*", "/v1/{path:.*}", handle_proxy)
    app.router.add_route("*", "/{path:.*}", handle_proxy)

    return app


def main() -> None:
    """Run the proxy server."""
    setup_logging()

    app = create_app()
    logger.info(
        "qwen-jsontool proxy starting on {}:{} -> {} (buffer_tool_calls={}, validate_json={})",
        config.PROXY_HOST,
        config.PROXY_PORT,
        config.LITELLM_URL,
        config.BUFFER_TOOL_CALLS,
        config.VALIDATE_JSON_ARGS,
    )
    web.run_app(app, host=config.PROXY_HOST, port=config.PROXY_PORT)


if __name__ == "__main__":
    main()
