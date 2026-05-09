"""Configuration for llm-toolstream-proxy via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class Config:
    """Immutable configuration loaded from environment variables."""

    # --- Upstream ---
    LITELLM_URL: str = os.getenv("LITELLM_URL", "http://localhost:4000")
    PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8787"))

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("PROXY_LOG_LEVEL", "INFO").upper()
    LOG_FILE: str = os.getenv("PROXY_LOG_FILE", "llm_proxy.log")

    # --- Feature flags ---
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
    STREAM_TIMEOUT: float = float(os.getenv("PROXY_STREAM_TIMEOUT", "120"))
    REQUEST_TIMEOUT: float = float(os.getenv("PROXY_REQUEST_TIMEOUT", "300"))
    CONNECT_TIMEOUT: float = float(os.getenv("PROXY_CONNECT_TIMEOUT", "15"))
    STREAM_MAX_DURATION: float = float(os.getenv("PROXY_STREAM_MAX_DURATION", "600"))

    # --- Connection & resource limits ---
    MAX_UPSTREAM_CONNECTIONS: int = int(
        os.getenv("PROXY_MAX_UPSTREAM_CONNECTIONS", "100")
    )
    KEEPALIVE_TIMEOUT: float = float(os.getenv("PROXY_KEEPALIVE_TIMEOUT", "30"))
    MAX_ARGUMENTS_SIZE: int = int(os.getenv("PROXY_MAX_ARGS_SIZE", str(1024 * 1024)))
    MAX_TOOL_CALLS: int = int(os.getenv("PROXY_MAX_TOOL_CALLS", "32"))
    MAX_CONCURRENT_STREAMS: int = int(os.getenv("PROXY_MAX_CONCURRENT_STREAMS", "50"))
    MAX_REQUEST_BODY_SIZE: int = int(
        os.getenv("PROXY_MAX_REQUEST_BODY_SIZE", str(1024 * 1024))
    )

    @classmethod
    def from_env(cls) -> Config:
        """Construct a Config instance from environment variables."""
        return cls()

    def validate(self) -> None:
        """Validate configuration values and exit with error if invalid."""
        errors = []

        if not (
            self.LITELLM_URL.startswith("http://")
            or self.LITELLM_URL.startswith("https://")
        ):
            errors.append(
                f"LITELLM_URL must start with http:// or https://, "
                f"got: {self.LITELLM_URL}"
            )

        if not (1 <= self.PROXY_PORT <= 65535):
            errors.append(
                f"PROXY_PORT must be between 1 and 65535, got: {self.PROXY_PORT}"
            )

        if self.STREAM_TIMEOUT <= 0:
            errors.append(f"STREAM_TIMEOUT must be > 0, got: {self.STREAM_TIMEOUT}")
        if self.STREAM_MAX_DURATION <= 0:
            errors.append(
                f"STREAM_MAX_DURATION must be > 0, got: {self.STREAM_MAX_DURATION}"
            )
        if self.CONNECT_TIMEOUT <= 0:
            errors.append(f"CONNECT_TIMEOUT must be > 0, got: {self.CONNECT_TIMEOUT}")
        if self.REQUEST_TIMEOUT <= 0:
            errors.append(f"REQUEST_TIMEOUT must be > 0, got: {self.REQUEST_TIMEOUT}")
        if self.KEEPALIVE_TIMEOUT <= 0:
            errors.append(
                f"KEEPALIVE_TIMEOUT must be > 0, got: {self.KEEPALIVE_TIMEOUT}"
            )
        if self.MAX_UPSTREAM_CONNECTIONS <= 0:
            errors.append(
                f"MAX_UPSTREAM_CONNECTIONS must be > 0, "
                f"got: {self.MAX_UPSTREAM_CONNECTIONS}"
            )
        if self.MAX_ARGUMENTS_SIZE <= 0:
            errors.append(
                f"MAX_ARGUMENTS_SIZE must be > 0, got: {self.MAX_ARGUMENTS_SIZE}"
            )
        if self.MAX_TOOL_CALLS <= 0:
            errors.append(f"MAX_TOOL_CALLS must be > 0, got: {self.MAX_TOOL_CALLS}")
        if self.MAX_CONCURRENT_STREAMS <= 0:
            errors.append(
                f"MAX_CONCURRENT_STREAMS must be > 0, "
                f"got: {self.MAX_CONCURRENT_STREAMS}"
            )
        if self.MAX_REQUEST_BODY_SIZE <= 0:
            errors.append(
                f"MAX_REQUEST_BODY_SIZE must be > 0, got: {self.MAX_REQUEST_BODY_SIZE}"
            )

        if self.STREAM_MAX_DURATION <= self.STREAM_TIMEOUT:
            logger.warning(
                "STREAM_MAX_DURATION ({}) should be greater than STREAM_TIMEOUT ({})",
                self.STREAM_MAX_DURATION,
                self.STREAM_TIMEOUT,
            )

        if errors:
            for error in errors:
                logger.error(error)
            raise SystemExit(1)


# Module-level singleton for backward compatibility during transition.
# New code should create a Config explicitly via Config.from_env().
config = Config.from_env()
