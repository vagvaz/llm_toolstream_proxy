"""Entry point for qwen-jsontool proxy server."""

from __future__ import annotations

import logging

from aiohttp import web

from . import config
from .proxy import handle_health, handle_proxy

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    app = create_app()
    logger = logging.getLogger(__name__)
    logger.info(
        "qwen-jsontool proxy starting on %s:%d -> %s (buffer_tool_calls=%s)",
        config.PROXY_HOST,
        config.PROXY_PORT,
        config.LITELLM_URL,
        config.BUFFER_TOOL_CALLS,
    )
    web.run_app(app, host=config.PROXY_HOST, port=config.PROXY_PORT)


if __name__ == "__main__":
    main()
