"""Configuration for llm-toolstream-proxy via environment variables."""

from __future__ import annotations

import os

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
