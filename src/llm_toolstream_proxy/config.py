"""Configuration for llm-toolstream-proxy via environment variables."""

from __future__ import annotations

import os

from loguru import logger

LITELLM_URL: str = os.getenv("LITELLM_URL", "http://localhost:4000")
PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8787"))
LOG_LEVEL: str = os.getenv("PROXY_LOG_LEVEL", "INFO").upper()
LOG_FILE: str = os.getenv("PROXY_LOG_FILE", "llm_proxy.log")
BUFFER_TOOL_CALLS: bool = os.getenv("PROXY_BUFFER_TOOLS", "true").lower() in (
    "true",
    "1",
    "yes",
)
VALIDATE_JSON_ARGS: bool = os.getenv("PROXY_VALIDATE_JSON", "true").lower() in (
    "true",
    "1",
    "yes",
)

# --- Timeouts ---
# Maximum time to wait between SSE chunks from upstream (seconds).
# If upstream stops sending data for this long, the stream is aborted.
# Lower values prevent stuck connections from accumulating.
STREAM_TIMEOUT: float = float(os.getenv("PROXY_STREAM_TIMEOUT", "120"))
# Maximum total time for a non-streaming request (seconds).
REQUEST_TIMEOUT: float = float(os.getenv("PROXY_REQUEST_TIMEOUT", "300"))
# Maximum time to wait for the initial TCP connection to upstream (seconds).
CONNECT_TIMEOUT: float = float(os.getenv("PROXY_CONNECT_TIMEOUT", "15"))
# Hard ceiling on total stream duration (seconds).
# Even if data keeps trickling in (resetting sock_read), this caps the
# total time a single streaming request can run. Prevents zombie streams.
STREAM_MAX_DURATION: float = float(os.getenv("PROXY_STREAM_MAX_DURATION", "600"))

# --- Connection & resource limits ---
# Maximum concurrent connections to the upstream (litellm).
MAX_UPSTREAM_CONNECTIONS: int = int(os.getenv("PROXY_MAX_UPSTREAM_CONNECTIONS", "100"))
# How long to keep idle connections in the pool (seconds).
# Short values prevent stale connections from accumulating.
KEEPALIVE_TIMEOUT: float = float(os.getenv("PROXY_KEEPALIVE_TIMEOUT", "30"))
# Maximum accumulated arguments per tool call (bytes). Prevents unbounded memory growth.
MAX_ARGUMENTS_SIZE: int = int(os.getenv("PROXY_MAX_ARGS_SIZE", str(1024 * 1024)))  # 1MB
# Maximum number of tool calls per request. Prevents unbounded buffer growth.
MAX_TOOL_CALLS: int = int(os.getenv("PROXY_MAX_TOOL_CALLS", "32"))
# Maximum concurrent streaming requests. When exceeded, new requests get 503.
# Prevents the proxy from being overwhelmed when the upstream is slow.
MAX_CONCURRENT_STREAMS: int = int(os.getenv("PROXY_MAX_CONCURRENT_STREAMS", "50"))

# Proxy settings for validation
PROXY_STREAM_TIMEOUT: float = STREAM_TIMEOUT
PROXY_STREAM_MAX_DURATION: float = STREAM_MAX_DURATION
PROXY_CONNECT_TIMEOUT: float = CONNECT_TIMEOUT
PROXY_REQUEST_TIMEOUT: float = REQUEST_TIMEOUT
PROXY_KEEPALIVE_TIMEOUT: float = KEEPALIVE_TIMEOUT
PROXY_MAX_UPSTREAM_CONNECTIONS: int = MAX_UPSTREAM_CONNECTIONS
PROXY_MAX_ARGS_SIZE: int = MAX_ARGUMENTS_SIZE


def validate_config() -> None:
    """Validate configuration values and exit with error if invalid."""
    errors = []

    # Validate LITELLM_URL
    if not (LITELLM_URL.startswith("http://") or LITELLM_URL.startswith("https://")):
        errors.append(
            f"LITELLM_URL must start with http:// or https://, got: {LITELLM_URL}"
        )

    # Validate PROXY_PORT
    if not (1 <= PROXY_PORT <= 65535):
        errors.append(f"PROXY_PORT must be between 1 and 65535, got: {PROXY_PORT}")

    # Validate timeout and limit values are positive
    if PROXY_STREAM_TIMEOUT <= 0:
        errors.append(f"PROXY_STREAM_TIMEOUT must be > 0, got: {PROXY_STREAM_TIMEOUT}")
    if PROXY_STREAM_MAX_DURATION <= 0:
        errors.append(
            f"PROXY_STREAM_MAX_DURATION must be > 0, got: {PROXY_STREAM_MAX_DURATION}"
        )
    if PROXY_CONNECT_TIMEOUT <= 0:
        errors.append(
            f"PROXY_CONNECT_TIMEOUT must be > 0, got: {PROXY_CONNECT_TIMEOUT}"
        )
    if PROXY_REQUEST_TIMEOUT <= 0:
        errors.append(
            f"PROXY_REQUEST_TIMEOUT must be > 0, got: {PROXY_REQUEST_TIMEOUT}"
        )
    if PROXY_KEEPALIVE_TIMEOUT <= 0:
        errors.append(
            f"PROXY_KEEPALIVE_TIMEOUT must be > 0, got: {PROXY_KEEPALIVE_TIMEOUT}"
        )
    if PROXY_MAX_UPSTREAM_CONNECTIONS <= 0:
        errors.append(
            f"PROXY_MAX_UPSTREAM_CONNECTIONS must be > 0, "
            f"got: {PROXY_MAX_UPSTREAM_CONNECTIONS}"
        )
    if PROXY_MAX_ARGS_SIZE <= 0:
        errors.append(f"PROXY_MAX_ARGS_SIZE must be > 0, got: {PROXY_MAX_ARGS_SIZE}")
    if MAX_TOOL_CALLS <= 0:
        errors.append(f"MAX_TOOL_CALLS must be > 0, got: {MAX_TOOL_CALLS}")
    if MAX_CONCURRENT_STREAMS <= 0:
        errors.append(
            f"PROXY_MAX_CONCURRENT_STREAMS must be > 0, got: {MAX_CONCURRENT_STREAMS}"
        )

    # Warn if STREAM_MAX_DURATION is not greater than STREAM_TIMEOUT
    if STREAM_MAX_DURATION <= STREAM_TIMEOUT:
        logger.warning(
            "STREAM_MAX_DURATION ({}) should be greater than STREAM_TIMEOUT ({})",
            STREAM_MAX_DURATION,
            STREAM_TIMEOUT,
        )

    if errors:
        for error in errors:
            logger.error(error)
        raise SystemExit(1)
