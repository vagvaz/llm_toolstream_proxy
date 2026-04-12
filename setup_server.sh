#!/usr/bin/env bash
# =============================================================================
# llm-toolstream-proxy server setup script
# =============================================================================
# Applies OS-level TCP tuning, creates a dedicated service user, installs
# the systemd unit, and optionally starts the proxy.
#
# Usage:
#   sudo ./setup_server.sh          # Full setup + start
#   sudo ./setup_server.sh --dry-run  # Show what would be done
#   sudo ./setup_server.sh --tuning-only  # Only apply TCP/sysctl tuning
#   sudo ./setup_server.sh --uninstall  # Remove service and TCP tuning
#
# Tested on: Ubuntu 22.04+, Debian 12+, Rocky Linux 9+

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSCTL_FILE="/etc/sysctl.d/99-llm-proxy.conf"
SERVICE_FILE="/etc/systemd/system/llm-toolstream-proxy.service"
PROXY_USER="${PROXY_USER:-llm-proxy}"
PROXY_GROUP="${PROXY_GROUP:-llm-proxy}"
PROXY_USER_HOME="/var/lib/llm-proxy"
PROXY_INSTALL_DIR="/opt/llm-toolstream-proxy"
PROXY_PORT="${PROXY_PORT:-8787}"

# =============================================================================
# Helper functions
# =============================================================================

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
fatal() { echo "[FATAL] $*" >&2; exit 1; }

need_root() {
    if [[ $EUID -ne 0 ]]; then
        fatal "This script must be run as root (use sudo)"
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

confirm() {
    local prompt="${1:-Continue?}"
    local reply
    read -rp "$prompt [y/N] " reply || reply="n"
    [[ "${reply,,}" == "y" ]] || [[ "${reply,,}" == "yes" ]]
}

# =============================================================================
# TCP/sysctl tuning
# =============================================================================

apply_tcp_tuning() {
    info "Applying TCP tuning to ${SYSCTL_FILE}"

    cat > "${SYSCTL_FILE}" << 'SYSCTL_EOF'
# =============================================================================
# TCP tuning for llm-toolstream-proxy
# Applied by /opt/llm-toolstream-proxy/setup_server.sh
#
# WHY THESE SETTINGS MATTER:
# The proxy sits between opencode and litellm handling long-lived SSE streams
# (2-5 minutes). The 85k+ Send-Q issue occurs because the proxy's per-chunk
# processing is slower than litellm's send rate, causing data to accumulate
# in the kernel's TCP send buffer. These tunables help in three ways:
#
#   1. Detect dead connections faster  → free up resources sooner
#   2. Reduce retransmit wait times    → stuck connections fail fast
#   3. Increase socket buffers         → absorb bursts without dropping
# =============================================================================

# --- Connection lifecycle: detect dead connections faster ---
#
# Problem: Default TCP keepalive waits 2 HOURS (7200s) before detecting a dead
# connection. If litellm crashes mid-stream, the proxy holds 100 connections
# for 2 hours waiting for the kernel to declare them dead.
#
# Fix: Send keepalive probes after 60s of inactivity, every 10s, give up after 6.
# This detects a dead upstream in ~70s instead of 2h, freeing connections.
#
# Default: net.ipv4.tcp_keepalive_time=7200, tcp_keepalive_intvl=75,
#          tcp_keepalive_probes=9  (total ~2h before giving up)
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6

# --- Retransmission: fail fast on stuck connections ---
#
# Problem: TCP retries a failed packet up to 15 times with exponential backoff.
# In some network conditions (flaky WAN links to cloud litellm), a single packet
# loss can cause 15 retransmit attempts over ~13 minutes, holding the socket.
#
# Fix: tcp_retries1=3 (abort after ~6s on local network issues)
#      tcp_retries2=5 (abort after ~100s on path problems, vs default ~15min)
#
# Default: net.ipv4.tcp_retries1=3 (already optimal)
#          net.ipv4.tcp_retries2=15 (~13 minutes of retries)
net.ipv4.tcp_retries1 = 3
net.ipv4.tcp_retries2 = 5

# --- Socket reuse: prevent port exhaustion under load ---
#
# Problem: When proxy closes a connection, it enters TIME_WAIT state for 60s.
# Under heavy load (100 concurrent streams, each lasting minutes), new connections
# may be delayed waiting for available ports.
#
# Fix: Allow reuse of sockets in TIME_WAIT for new connections immediately.
# This is safe for a proxy because we initiate connections (client role).
#
# Default: net.ipv4.tcp_tw_reuse=0
net.ipv4.tcp_tw_reuse = 1

# --- Buffer sizes: absorb Send-Q bursts ---
#
# Problem: When proxy processes chunks slower than litellm sends, data accumulates
# in the kernel's send buffer (proxy→client) and receive buffer (proxy from litellm).
# Default buffer caps of ~128KB are too small for 2-5 minute streams at high throughput.
#
# Fix: Raise per-socket buffer caps to 16MB. The proxy also enforces
# application-level limits (MAX_ARGUMENTS_SIZE=1MB, MAX_CONCURRENT_STREAMS=50).
# These are a safety net; aiohttp's flow control (response.drain()) handles the rest.
#
# tcp_rmem / tcp_wmem: (min, default, max) per socket
# Default: net.core.rmem_max=131071, net.ipv4.tcp_rmem="4096 87380 6291456"
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# --- Connection tracking: prevent table overflow ---
#
# Only needed if the proxy machine also runs a firewall (nftables/iptables).
# If you see "nf_conntrack: table full" in dmesg, increase these.
# The proxy itself does NOT need connection tracking.
#
# Default: net.netfilter.nf_conntrack_max=262144 (can fill fast with 100 conns)
net.netfilter.nf_conntrack_max = 1048576
net.nf_conntrack_max = 1048576

# --- File descriptors: support many concurrent connections ---
#
# Problem: Each upstream connection uses 1 FD. With 100 concurrent upstream
# connections + server sockets + logs, default fs.file-max=2097152 is fine but
# per-process limit may be hit. systemd's default is 512-1024.
#
# Fix: Raise system-wide file-max and let systemd set LimitNOFILE=1048576
# in the service unit (see llm-toolstream-proxy.service).
#
# Default: fs.file-max=2097152 (already generous)
fs.file-max = 2097152

# --- TCP congestion: better throughput on high-BDP links ---
#
# The proxy→litellm path may traverse WAN links with high bandwidth-delay product.
# CUBIC (default on most Linux) is good for this; this just makes it explicit.
# Leave as cubic unless you have measured BBR performs better on your network.
#
# Default: net.ipv4.tcp_congestion_control=cubic (set by kernel)
net.ipv4.tcp_congestion_control = cubic
SYSCTL_EOF

    # Apply immediately (without reboot)
    if command_exists sysctl; then
        sysctl -p "${SYSCTL_FILE}" >/dev/null 2>&1 || \
            warn "sysctl -p failed (may already be applied or not applicable in container)"
        info "TCP tuning applied. Run 'sysctl -p ${SYSCTL_FILE}' after reboot."
    else
        warn "sysctl command not found — TCP tuning not applied"
    fi
}

remove_tcp_tuning() {
    info "Removing TCP tuning from ${SYSCTL_FILE}"
    rm -f "${SYSCTL_FILE}"
    sysctl -q -w net.ipv4.tcp_keepalive_time=7200 \
              net.ipv4.tcp_keepalive_intvl=75 \
              net.ipv4.tcp_keepalive_probes=9 \
              net.ipv4.tcp_retries1=3 \
              net.ipv4.tcp_retries2=15 \
              net.ipv4.tcp_tw_reuse=0 \
              net.core.rmem_max=131071 \
              net.core.wmem_max=131071 \
              net.ipv4.tcp_rmem="4096 87380 6291456" \
              net.ipv4.tcp_wmem="4096 16384 4194304" \
              2>/dev/null || true
    info "TCP tuning removed"
}

# =============================================================================
# Service user / group
# =============================================================================

create_service_user() {
    if id "${PROXY_USER}" >/dev/null 2>&1; then
        info "User '${PROXY_USER}' already exists — skipping user creation"
    else
        info "Creating service user '${PROXY_USER}'"
        useradd --system --gid "${PROXY_GROUP}" \
            --home-dir "${PROXY_USER_HOME}" \
            --shell /usr/sbin/nologin \
            --comment "llm-toolstream-proxy service account" \
            "${PROXY_USER}" 2>/dev/null || \
        useradd --system --create-home --home-dir "${PROXY_USER_HOME}" \
            --shell /usr/sbin/nologin \
            --comment "llm-toolstream-proxy service account" \
            "${PROXY_USER}" 2>/dev/null || \
        warn "Could not create user '${PROXY_USER}' — will run as root"
    fi

    if getent group "${PROXY_GROUP}" >/dev/null 2>&1; then
        info "Group '${PROXY_GROUP}' already exists — skipping group creation"
    else
        info "Creating service group '${PROXY_GROUP}'"
        groupadd --system "${PROXY_GROUP}" 2>/dev/null || \
            warn "Could not create group '${PROXY_GROUP}'"
    fi
}

# =============================================================================
# Systemd unit
# =============================================================================

install_systemd_unit() {
    need_root

    info "Installing systemd unit to ${SERVICE_FILE}"

    cat > "${SERVICE_FILE}" << SERVICE_EOF
# =============================================================================
# systemd unit for llm-toolstream-proxy
# Installed by setup_server.sh
# =============================================================================
# Place in /etc/systemd/system/ then run:
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now llm-toolstream-proxy
# =============================================================================

[Unit]
Description=llm-toolstream-proxy (SSE reverse proxy for LLM tool calls)
Documentation=https://github.com/anomalyco/qwen_jsontool
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Run as dedicated service user (set PROXY_USER/PROXY_GROUP env vars to change)
# User=${PROXY_USER}
# Group=${PROXY_GROUP}
WorkingDirectory=${PROXY_INSTALL_DIR}
EnvironmentFile=-${PROXY_INSTALL_DIR}/.env
ExecStart=${PROXY_INSTALL_DIR}/start.sh
# Restart on failure, with exponential back-off
Restart=on-failure
RestartSec=5s
# Hardening: prevent privilege escalation
NoNewPrivileges=true
# Hardening: restrict filesystem access
ProtectSystem=strict
ReadWritePaths=${PROXY_INSTALL_DIR} /var/log
ProtectHome=true
# Hardening: prevent raw IP access
RestrictAddressFamilies=AF_INET AF_INET6
# Hardening: limit number of processes
LimitNOFILE=1048576
# Logging to journal
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llm-proxy

# Graceful shutdown: give active streams up to 60s to finish
TimeoutStopSec=60s

[Install]
WantedBy=multi-user.target
SERVICE_EOF

    systemctl daemon-reload
    info "systemd unit installed. Run 'systemctl enable --now llm-toolstream-proxy' to start."
}

uninstall_systemd_unit() {
    need_root
    info "Removing systemd unit from ${SERVICE_FILE}"
    systemctl stop llm-toolstream-proxy 2>/dev/null || true
    systemctl disable llm-toolstream-proxy 2>/dev/null || true
    rm -f "${SERVICE_FILE}"
    systemctl daemon-reload
    info "systemd unit removed"
}

# =============================================================================
# Installation
# =============================================================================

install_proxy() {
    need_root

    if [[ ! -d "${SCRIPT_DIR}/src" ]]; then
        fatal "Cannot find src/ directory in ${SCRIPT_DIR} — run from the project root"
    fi

    info "Installing llm-toolstream-proxy to ${PROXY_INSTALL_DIR}"

    # Create install dir
    mkdir -p "${PROXY_INSTALL_DIR}"
    cp -r "${SCRIPT_DIR}/src" "${PROXY_INSTALL_DIR}/"
    cp -r "${SCRIPT_DIR}/start.sh" "${PROXY_INSTALL_DIR}/"
    cp -r "${SCRIPT_DIR}/start_gunicorn.sh" "${PROXY_INSTALL_DIR}/"
    cp -r "${SCRIPT_DIR}/pyproject.toml" "${PROXY_INSTALL_DIR}/"
    cp "${SCRIPT_DIR}/README.md" "${PROXY_INSTALL_DIR}/" 2>/dev/null || true
    cp "${SCRIPT_DIR}/DESIGN.md" "${PROXY_INSTALL_DIR}/" 2>/dev/null || true

    # Create log dir
    mkdir -p /var/log/llm-proxy
    touch /var/log/llm-proxy/llm_proxy.log
    chmod 755 /var/log/llm-proxy
    chmod 644 /var/log/llm-proxy/llm_proxy.log

    # Create .env template
    cat > "${PROXY_INSTALL_DIR}/.env" << ENV_EOF
# Environment for llm-toolstream-proxy
# Copy this to /etc/llm-proxy/env or set variables in systemd Environment=
LITELLM_URL=http://litellm:4000
PROXY_HOST=0.0.0.0
PROXY_PORT=${PROXY_PORT}
PROXY_LOG_LEVEL=INFO
PROXY_LOG_FILE=/var/log/llm-proxy/llm_proxy.log
PROXY_BUFFER_TOOLS=true
PROXY_VALIDATE_JSON=true
PROXY_STREAM_TIMEOUT=120
PROXY_STREAM_MAX_DURATION=600
PROXY_MAX_UPSTREAM_CONNECTIONS=100
PROXY_REQUEST_TIMEOUT=300
PROXY_CONNECT_TIMEOUT=15
PROXY_KEEPALIVE_TIMEOUT=30
PROXY_MAX_ARGS_SIZE=1048576
PROXY_MAX_TOOL_CALLS=32
PROXY_MAX_CONCURRENT_STREAMS=50
ENV_EOF
    chmod 640 "${PROXY_INSTALL_DIR}/.env"

    info "Proxy installed to ${PROXY_INSTALL_DIR}"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    cat << USAGE_EOF
Usage: sudo ${0} [OPTIONS]

Options:
  --dry-run         Show what would be done without making changes
  --tuning-only     Only apply TCP/sysctl tuning (no service/user setup)
  --uninstall       Remove service, TCP tuning, and installed files
  --help            Show this help message

Examples:
  # Full production setup (run as root)
  sudo ./setup_server.sh

  # Preview actions without applying
  sudo ./setup_server.sh --dry-run

  # Only apply OS-level TCP tuning (useful for containers)
  sudo ./setup_server.sh --tuning-only

  # Tear down
  sudo ./setup_server.sh --uninstall
USAGE_EOF
}

main() {
    local mode="full"
    local dry_run=false

    # Parse arguments
    for arg in "$@"; do
        case "$arg" in
            --dry-run)      dry_run=true; mode="full" ;;
            --tuning-only)  mode="tuning-only" ;;
            --uninstall)    mode="uninstall" ;;
            --help|-h)      usage; exit 0 ;;
            *)              usage; exit 1 ;;
        esac
    done

    if [[ "$mode" == "uninstall" ]]; then
        info "Uninstall mode — will remove everything"
        confirm "This will stop the service, remove the systemd unit, and delete ${PROXY_INSTALL_DIR}" || exit 0
        if [[ "$dry_run" == "true" ]]; then
            info "[DRY RUN] Would: remove_tcp_tuning, uninstall_systemd_unit, rm -rf ${PROXY_INSTALL_DIR}"
        else
            need_root
            remove_tcp_tuning
            uninstall_systemd_unit
            rm -rf "${PROXY_INSTALL_DIR}"
            info "Uninstall complete"
        fi
        return
    fi

    if [[ "$mode" == "tuning-only" ]]; then
        info "TCP tuning only mode"
        if [[ "$dry_run" == "true" ]]; then
            info "[DRY RUN] Would: apply_tcp_tuning"
        else
            need_root
            apply_tcp_tuning
            info "Done — TCP tuning applied to ${SYSCTL_FILE}"
        fi
        return
    fi

    # Full setup
    if [[ "$dry_run" == "true" ]]; then
        info "[DRY RUN] Would: apply_tcp_tuning, create_service_user, install_proxy, install_systemd_unit"
        return
    fi

    need_root

    info "=== llm-toolstream-proxy server setup ==="

    apply_tcp_tuning
    create_service_user
    install_proxy
    install_systemd_unit

    echo ""
    info "=== Setup complete ==="
    echo ""
    echo "  Proxy installed to: ${PROXY_INSTALL_DIR}"
    echo "  Config template:   ${PROXY_INSTALL_DIR}/.env"
    echo "  Log file:          /var/log/llm-proxy/llm_proxy.log"
    echo "  Systemd unit:       ${SERVICE_FILE}"
    echo ""
    echo "Next steps:"
    echo "  1. Edit ${PROXY_INSTALL_DIR}/.env with your LITELLM_URL"
    echo "  2. sudo systemctl daemon-reload"
    echo "  3. sudo systemctl enable --now llm-toolstream-proxy"
    echo "  4. sudo systemctl status llm-toolstream-proxy"
    echo ""
}

main "$@"
