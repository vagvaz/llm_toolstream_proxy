"""Configuration for qwen-jsontool proxy via environment variables."""

from __future__ import annotations

import os

LITELLM_URL: str = os.getenv("LITELLM_URL", "http://localhost:4000")
PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8787"))
LOG_LEVEL: str = os.getenv("PROXY_LOG_LEVEL", "INFO").upper()
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
STREAM_TIMEOUT: float = float(os.getenv("PROXY_STREAM_TIMEOUT", "300"))
