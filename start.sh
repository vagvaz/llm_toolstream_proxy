#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${LITELLM_URL:=http://localhost:4000}"
: "${PROXY_HOST:=0.0.0.0}"
: "${PROXY_PORT:=8787}"
: "${PROXY_WORKERS:=4}"
: "${PROXY_LOG_LEVEL:=INFO}"
: "${PROXY_LOG_FILE:=llm_proxy.log}"
: "${PROXY_BUFFER_TOOLS:=true}"
: "${PROXY_VALIDATE_JSON:=true}"
: "${PROXY_STREAM_TIMEOUT:=300}"

export LITELLM_URL
export PROXY_HOST
export PROXY_PORT
export PROXY_LOG_LEVEL
export PROXY_LOG_FILE
export PROXY_BUFFER_TOOLS
export PROXY_VALIDATE_JSON
export PROXY_STREAM_TIMEOUT

mkdir -p "$(dirname "$PROXY_LOG_FILE")" 2>/dev/null || true

echo "Starting llm-toolstream-proxy"
echo "  LITELLM_URL        = $LITELLM_URL"
echo "  PROXY_HOST          = $PROXY_HOST"
echo "  PROXY_PORT          = $PROXY_PORT"
echo "  PROXY_WORKERS       = $PROXY_WORKERS"
echo "  PROXY_LOG_LEVEL     = $PROXY_LOG_LEVEL"
echo "  PROXY_LOG_FILE      = $PROXY_LOG_FILE"
echo "  PROXY_BUFFER_TOOLS  = $PROXY_BUFFER_TOOLS"
echo "  PROXY_VALIDATE_JSON = $PROXY_VALIDATE_JSON"
echo "  PROXY_STREAM_TIMEOUT= $PROXY_STREAM_TIMEOUT"
echo ""

exec gunicorn \
    llm_toolstream_proxy.main:create_app() \
    --bind "${PROXY_HOST}:${PROXY_PORT}" \
    --workers "$PROXY_WORKERS" \
    --worker-class aiohttp.worker.GunicornWebWorker \
    --log-level "$PROXY_LOG_LEVEL" \
    --timeout "$PROXY_STREAM_TIMEOUT" \
    --graceful-timeout 30