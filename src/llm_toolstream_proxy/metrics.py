"""Request-level metrics for llm-toolstream-proxy."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RequestMetrics:
    """Per-request metrics."""

    request_id: str
    method: str
    path: str
    model: str
    is_streaming: bool
    start_time: float = field(default_factory=time.monotonic)
    ttfb: float | None = None  # Time to first byte from upstream
    end_time: float | None = None
    chunks_processed: int = 0
    bytes_transferred: int = 0
    tool_calls_buffered: int = 0
    tool_calls_repaired: int = 0
    error: str | None = None


class MetricsCollector:
    """Collects metrics across all requests."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.total_streaming_requests: int = 0
        self.total_non_streaming_requests: int = 0
        self.active_streams: int = 0
        self.total_errors: dict[str, int] = {}
        self.total_tool_calls_buffered: int = 0
        self.total_tool_calls_repaired: int = 0
        self.total_bytes_transferred: int = 0
        self._start_time: float = time.monotonic()

    def new_request(
        self, method: str, path: str, model: str, is_streaming: bool
    ) -> RequestMetrics:
        """Create a new RequestMetrics with a unique ID."""
        self.total_requests += 1
        if is_streaming:
            self.total_streaming_requests += 1
            self.active_streams += 1
        else:
            self.total_non_streaming_requests += 1
        return RequestMetrics(
            request_id=str(uuid.uuid4())[:8],
            method=method,
            path=path,
            model=model,
            is_streaming=is_streaming,
        )

    def finish_request(self, req: RequestMetrics) -> None:
        """Mark a request as finished and update aggregate stats."""
        req.end_time = time.monotonic()
        if req.is_streaming:
            self.active_streams -= 1
        self.total_tool_calls_buffered += req.tool_calls_buffered
        self.total_tool_calls_repaired += req.tool_calls_repaired
        self.total_bytes_transferred += req.bytes_transferred
        if req.error:
            self.total_errors[req.error] = self.total_errors.get(req.error, 0) + 1
        duration = req.end_time - req.start_time
        logger.info(
            "Request {} {} model={} streaming={} "
            "duration={:.2f}s chunks={} bytes={} ttfb={}",
            req.request_id,
            req.method,
            req.model,
            req.is_streaming,
            duration,
            req.chunks_processed,
            req.bytes_transferred,
            f"{req.ttfb:.3f}s" if req.ttfb else "N/A",
        )

    def record_error(self, error_type: str) -> None:
        self.total_errors[error_type] = self.total_errors.get(error_type, 0) + 1

    def get_metrics(self) -> dict[str, Any]:
        uptime = time.monotonic() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "total_streaming_requests": self.total_streaming_requests,
            "total_non_streaming_requests": self.total_non_streaming_requests,
            "active_streams": self.active_streams,
            "total_tool_calls_buffered": self.total_tool_calls_buffered,
            "total_tool_calls_repaired": self.total_tool_calls_repaired,
            "total_bytes_transferred": self.total_bytes_transferred,
            "total_errors": dict(self.total_errors),
        }

    def get_summary(self) -> str:
        m = self.get_metrics()
        return (
            f"requests={m['total_requests']} "
            f"streaming={m['total_streaming_requests']} "
            f"active={m['active_streams']} "
            f"buffered={m['total_tool_calls_buffered']} "
            f"repaired={m['total_tool_calls_repaired']} "
            f"bytes={m['total_bytes_transferred']} "
            f"errors={m['total_errors']}"
        )


# Global metrics collector
collector = MetricsCollector()


# aiohttp web handler for /metrics endpoint
async def metrics_handler(request: Any) -> Any:
    """Return metrics as JSON for the /metrics endpoint."""
    from aiohttp import web

    return web.json_response(collector.get_metrics())
