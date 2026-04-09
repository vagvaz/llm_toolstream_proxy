#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${LITELLM_URL:=http://localhost:4000}"
: "${PROXY_HOST:=0.0.0.0}"
: "${PROXY_PORT:=8787}"
: "${PROXY_WORKERS:=4}"
: "${PROXY_LOG_LEVEL:=info}"
: "${PROXY_BUFFER_TOOLS:=true}"
: "${PROXY_VALIDATE_JSON:=true}"
: "${PROXY_STREAM_TIMEOUT:=300}"
: "${PROXY_ACCESS_LOG:=/var/log/qwen-jsontool/access.log}"
: "${PROXY_ERROR_LOG:=/var/log/qwen-jsontool/error.log}"

export LITELLM_URL
export PROXY_HOST
export PROXY_PORT
export PROXY_LOG_LEVEL
export PROXY_BUFFER_TOOLS
export PROXY_VALIDATE_JSON
export PROXY_STREAM_TIMEOUT

mkdir -p "$(dirname "$PROXY_ACCESS_LOG")" 2>/dev/null || true
mkdir -p "$(dirname "$PROXY_ERROR_LOG")" 2>/dev/null || true

echo "Starting qwen-jsontool proxy"
echo "  LITELLM_URL       = $LITELLM_URL"
echo "  PROXY_HOST         = $PROXY_HOST"
echo "  PROXY_PORT         = $PROXY_PORT"
echo "  PROXY_WORKERS      = $PROXY_WORKERS"
echo "  PROXY_LOG_LEVEL    = $PROXY_LOG_LEVEL"
echo "  PROXY_BUFFER_TOOLS = $PROXY_BUFFER_TOOLS"
echo "  PROXY_VALIDATE_JSON= $PROXY_VALIDATE_JSON"
echo ""

exec gunicorn \
    qwen_jsontool.main:create_app() \
    --bind "${PROXY_HOST}:${PROXY_PORT}" \
    --workers "$PROXY_WORKERS" \
    --worker-class aiohttp.worker.GunicornWebWorker \
    --access-logfile "$PROXY_ACCESS_LOG" \
    --error-logfile "$PROXY_ERROR_LOG" \
    --log-level "$PROXY_LOG_LEVEL" \
    --timeout "$PROXY_STREAM_TIMEOUT" \
    --graceful-timeout 30 \
    --access-logformat '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'