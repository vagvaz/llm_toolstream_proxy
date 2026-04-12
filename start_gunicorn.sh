#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${LITELLM_URL:=http://localhost:4000}"
: "${PROXY_HOST:=0.0.0.0}"
: "${PROXY_PORT:=8787}"
: "${PROXY_LOG_LEVEL:=INFO}"
: "${PROXY_LOG_FILE:=llm_proxy.log}"
: "${PROXY_BUFFER_TOOLS:=true}"
: "${PROXY_VALIDATE_JSON:=true}"
: "${PROXY_STREAM_TIMEOUT:=120}"
: "${PROXY_STREAM_MAX_DURATION:=600}"
: "${PROXY_MAX_UPSTREAM_CONNECTIONS:=100}"
: "${PROXY_REQUEST_TIMEOUT:=300}"
: "${PROXY_CONNECT_TIMEOUT:=15}"
: "${PROXY_KEEPALIVE_TIMEOUT:=30}"
: "${PROXY_MAX_ARGS_SIZE:=1048576}"
: "${PROXY_MAX_TOOL_CALLS:=32}"
: "${PROXY_MAX_CONCURRENT_STREAMS:=50}"
: "${PROXY_WORKERS:=4}"
: "${PROXY_MAX_REQUESTS:=10000}"
: "${PROXY_MAX_REQUESTS_JITTER:=1000}"

export LITELLM_URL
export PROXY_HOST
export PROXY_PORT
export PROXY_LOG_LEVEL
export PROXY_LOG_FILE
export PROXY_BUFFER_TOOLS
export PROXY_VALIDATE_JSON
export PROXY_STREAM_TIMEOUT
export PROXY_STREAM_MAX_DURATION
export PROXY_MAX_UPSTREAM_CONNECTIONS
export PROXY_REQUEST_TIMEOUT
export PROXY_CONNECT_TIMEOUT
export PROXY_KEEPALIVE_TIMEOUT
export PROXY_MAX_ARGS_SIZE
export PROXY_MAX_TOOL_CALLS
export PROXY_MAX_CONCURRENT_STREAMS

mkdir -p "$(dirname "$PROXY_LOG_FILE")" 2>/dev/null || true

echo "Starting llm-toolstream-proxy with gunicorn"
echo "  LITELLM_URL                    = $LITELLM_URL"
echo "  PROXY_HOST                      = $PROXY_HOST"
echo "  PROXY_PORT                      = $PROXY_PORT"
echo "  PROXY_WORKERS                   = $PROXY_WORKERS"
echo "  PROXY_LOG_LEVEL                 = $PROXY_LOG_LEVEL"
echo "  PROXY_LOG_FILE                  = $PROXY_LOG_FILE"
echo "  PROXY_MAX_REQUESTS              = $PROXY_MAX_REQUESTS"
echo "  PROXY_MAX_REQUESTS_JITTER       = $PROXY_MAX_REQUESTS_JITTER"
echo ""

# NOTE: Gunicorn is NOT recommended for this streaming proxy.
# Gunicorn's --timeout kills workers mid-stream for SSE responses that take
# minutes, leaving orphaned TCP connections with large Send-Q that can stall
# the machine. Use start.sh (direct python) instead when possible.
#
# If you need multi-worker process management, use --timeout >= 1200 (2x
# STREAM_MAX_DURATION=600) and monitor for zombie connections.
exec gunicorn \
    llm_toolstream_proxy.main:create_app() \
    --bind "${PROXY_HOST}:${PROXY_PORT}" \
    --workers "$PROXY_WORKERS" \
    --worker-class aiohttp.worker.GunicornWebWorker \
    --log-level "$PROXY_LOG_LEVEL" \
    --timeout 1200 \
    --graceful-timeout 30 \
    --max-requests "$PROXY_MAX_REQUESTS" \
    --max-requests-jitter "$PROXY_MAX_REQUESTS_JITTER"
