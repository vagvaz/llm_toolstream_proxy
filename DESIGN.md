# Design: llm-toolstream-proxy

## Overview

llm-toolstream-proxy is a reverse proxy that sits between an LLM client (e.g. opencode) and a litellm upstream, fixing malformed streaming tool call chunks emitted by Qwen, MiniMax, vllm, ollama, and other backends that violate the OpenAI SSE specification.

## Problem Statement

The OpenAI streaming protocol for tool calls expects each `tool_calls[i]` delta to include at minimum the `id` and `function.name` on first appearance, with `function.arguments` arriving as string fragments in subsequent chunks. Multiple LLM backends violate this:

| Bug | Source | Example |
|-----|--------|---------|
| `function.name` is `""` in first chunk | Qwen, vllm | `{"function": {"name": "", "arguments": ""}}` |
| `function.name` is `"null"` (string) | litellm serialization | `{"function": {"name": "null"}}` |
| `id` absent from first chunk | vllm | `{index: 0, type: "function", function: {name: "bash"}}` |
| Arguments arrive before name/id | Qwen 3.5 | `{"function": {"name": "", "arguments": '{"file'}}` |
| Empty tool call objects in parallel streams | litellm #17425 | `{index: 1, id: null, function: {name: null}}` |
| Truncated JSON arguments | vllm | `{"arguments": '{"path": "/tm'}`  (stream cut off) |

This causes consumers like opencode to crash with `Expected 'function.name' to be a string`.

Related issues: [agno#6757](https://github.com/agno-agi/agno/issues/6757), [litellm#12513](https://github.com/BerriAI/litellm/issues/12513), [litellm#17425](https://github.com/BerriAI/litellm/issues/17425), [vllm#17614](https://github.com/vllm-project/vllm/issues/17614), [spring-ai#4790](https://github.com/spring-projects/spring-ai/issues/4790), [opencode#10855](https://github.com/anomalyco/opencode/issues/10855).

## Architecture

```
┌──────────────┐       ┌──────────────────────────────────────┐       ┌──────────┐
│              │       │         llm-toolstream-proxy          │       │          │
│   opencode   │──────▶│                                      │──────▶│  litellm │
│   (client)   │◀──────│  ┌─────────┐  ┌──────────┐  ┌─────┐ │◀──────│  (upstream)│
│              │       │  │ proxy   │─▶│   SSE    │─▶│ buf │ │       │          │
└──────────────┘       │  │  (aiohttp)│  │transform│  │ er  │ │       └──────────┘
                       │  └─────────┘  └──────────┘  └─────┘ │
                       │       │              │           │     │
                       │       │         ┌─────┘      ┌────┘  │
                       │       │         ▼            ▼       │
                       │       │   ┌──────────┐ ┌─────────┐  │
                       │       │   │ parse +  │ │Buffered │  │
                       │       │   │ extract  │ │ToolCall │  │
                       │       │   └──────────┘ └─────────┘  │
                       └──────────────────────────────────────┘
```

### Module map

```
src/llm_toolstream_proxy/
├── main.py          # Entry point, aiohttp app, loguru setup
├── config.py        # Environment variable configuration
├── proxy.py         # aiohttp reverse proxy handler
├── sse.py           # SSE parsing, extraction, re-encoding, transformer
└── buffering.py     # ToolCallBuffer, BufferedToolCall, JSON repair
```

### Data flow

```
                          REQUEST PATH
                    ─────────────────────▶

  opencode ──POST /v1/chat/completions──▶ proxy ──same body──▶ litellm
              {stream: true, model:...}       (unchanged)


                          RESPONSE PATH
                    ◀──────────────────────

  litellm ──SSE line──▶ proxy ───────────────────────────▶ opencode
                           │
                    ┌──────▼──────┐
                    │ parse_sse   │   "data: {json}\n\n"
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────┐
                    │ _extract_tool_calls │  Split into:
                    │                     │    tool_call_deltas[]  ──▶ buffer
                    │                     │    cleaned chunk       ──▶ passthrough
                    └──────┬──────────────┘
                           │
              ┌────────────▼─────────────────┐
              │     ToolCallBuffer           │
              │  ┌─────────────────────────┐ │
              │  │ For each tool_call delta │ │
              │  │                          │ │
              │  │  ┌─── sanitize ───┐     │ │
              │  │  │ name: "" → None│     │ │
              │  │  │ name:"null"→None│     │ │
              │  │  │ id:   "" → None│     │ │
              │  │  └────────────────┘     │ │
              │  │                          │ │
              │  │  ┌─── accumulate ───┐   │ │
              │  │  │ call.id    ◀─────│   │ │
              │  │  │ call.name  ◀─────│   │ │
              │  │  │ call.args  ◀ +=─ │   │ │
              │  │  └────────────────┘     │ │
              │  │                          │ │
              │  │  ┌─── decision ─────┐    │ │
              │  │  │ started?        │    │ │
              │  │  │   YES → stream  │    │ │──▶ emit arg delta
              │  │  │ is_complete?    │    │ │
              │  │  │   YES → emit    │    │ │──▶ emit start chunk + buffered args
              │  │  │   NO  → buffer  │    │ │──▶ emit nothing, keep accumulating
              │  │  └────────────────┘    │ │
              │  └─────────────────────────┘ │
              └──────────────────────────────┘
```

## The Three Paths Through the Buffer

### Path 1: Buffering (name or id not yet available)

```
  Time ──────────────────────────────────────────────────▶

  Upstream:   ┌──────────────────┐   ┌──────────────────┐
              │ id:"call_1"      │   │ name:"bash"      │
              │ name:""  ◀──BUG │   │ arguments:""     │
              │ arguments:""     │   └──────┬───────────┘
              └───────┬─────────┘          │
                      │                    │
               Buffer accumulates     Buffer has both
               id, empty name          id AND name
               args silently           → is_complete = True
                      │                    │
                      ▼                    ▼
  Downstream:   (nothing emitted)    ┌──────────────────────────┐
                                  │ emit start chunk:         │
                                  │   {id, name, args:""}     │
                                  │ + emit buffered args:     │
                                  │   {args: all accumulated} │
                                  └──────────────────────────┘
```

### Path 2: Already started (normal streaming)

```
  Upstream:   ┌──────────────────────┐
              │ arguments: '{"cmd":' │  (call already started)
              └──────────┬───────────┘
                         │
                         ▼
  Downstream: ┌──────────────────────┐
              │ {arguments: '{"cmd":'}│  (passthrough as-is)
              └──────────────────────┘
```

### Path 3: Flush on stream end

```
  [DONE] arrives from upstream
          │
          ▼
  For each buffered call:
    ├── started=True     → validate args, log warning if invalid (cannot repair,
    │                      already streamed to client)
    ├── started=False, complete → emit full call with repaired args if needed
    └── started=False, incomplete → DISCARD with warning log
```

## Argument Handling: Why Concatenation Is Correct

The OpenAI streaming spec defines `function.arguments` as **string fragments** that concatenate to form the final JSON. They are NOT separate JSON objects to be merged. This is how all consumers (opencode, the OpenAI SDK, etc.) reconstruct arguments:

```
  Chunk 1: arguments = '{"file'          → client accumulates: '{"file'
  Chunk 2: arguments = 'Path":'          → client accumulates: '{"filePath":'
  Chunk 3: arguments = ' "/tmp"}'        → client accumulates: '{"filePath": "/tmp"}'
```

Thus `call.arguments += args` is the correct merging policy. However, this creates an important asymmetry for **validation and repair**:

| Call state | When args arrive | Can we repair? | Why |
|-----------|-----------------|----------------|-----|
| **Buffered** (not started) | Accumulated in `call.arguments` | **YES** | Nothing has been sent to the client yet. We can repair the full string before emitting. |
| **Started** (emitting deltas) | Streamed to client immediately | **NO** | Client has already concatenated the raw fragments. Sending a repair delta would cause duplication. |

This is why `finish_call()` for started calls only **validates and logs warnings** — it cannot un-send data that the client has already received.

## Sanitization Rules

The proxy normalizes malformed values coming from upstream backends:

```
  ┌───────────────────────────────────────────────────┐
  │              _sanitize_name(value)                 │
  ├──────────────────┬───────────────────────────────┤
  │  Input           │  Output                        │
  ├──────────────────┼───────────────────────────────┤
  │  None            │  None (not yet available)      │
  │  ""   (empty)    │  None (continuation chunk)     │
  │  "null" (string) │  None (litellm None bug)      │
  │  "bash" (real)   │  "bash"                       │
  └──────────────────┴───────────────────────────────┘

  ┌───────────────────────────────────────────────────┐
  │              _sanitize_id(value)                   │
  ├──────────────────┬───────────────────────────────┤
  │  None            │  None                          │
  │  ""   (empty)    │  None                          │
  │  "call_abc"      │  "call_abc"                    │
  └──────────────────┴───────────────────────────────┘

  A BufferedToolCall is "complete" when both name and id are non-None.
  Only then does is_complete return True and the call transitions to "started".
```

## JSON Repair

When `PROXY_VALIDATE_JSON=true` (default), the proxy attempts repair on buffered calls' arguments before emission:

```
  _try_repair_json(text):
    1. If already valid JSON → return as-is
    2. Strip trailing comma: '{"key": "val",' → '{"key": "val"}'
    3. Fix comma before closure: '{"key": "val",}' → '{"key": "val"}'
    4. Fix comma before bracket: '{"items": [1,]' → '{"items": [1]}'
    5. Append missing closing brackets/braces
    6. If still invalid → return None (cannot repair)

  Applied in two places:
    - flush() for buffered-but-never-started calls (can replace args)
    - finish_call() for started calls (can only LOG, cannot replace)
```

## SSE Event Splitting

A single upstream SSE chunk may contain both non-tool content and tool_calls:

```json
{
  "choices": [{
    "delta": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{"index": 0, "function": {"name": "bash"}}]
    }
  }]
}
```

The proxy splits this into two downstream events:

1. **Content event** (passthrough): `{"delta": {"role": "assistant", "content": ""}}`
2. **Tool call event** (buffered/reassembled): `{"delta": {"tool_calls": [...]}}`

This ensures text content reaches the client immediately while tool calls go through the buffer.

## Finish Reason Handling

The `finish_reason` signals the end of a generation. It arrives in a chunk that typically has **no** tool_calls:

```json
{"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
```

The proxy detects `finish_reason: "tool_calls"` or `"stop"` in any chunk and marks all known tool calls as finished, triggering argument validation. This is critical because:

- The finish chunk often has no `delta.tool_calls` at all
- This is the last chance to detect invalid accumulated arguments
- Previously this was only checked in the `[DONE]` flush, which was too late for meaningful logging

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `LITELLM_URL` | `http://localhost:4000` | Upstream litellm URL |
| `PROXY_HOST` | `0.0.0.0` | Host to bind to |
| `PROXY_PORT` | `8787` | Port to listen on |
| `PROXY_LOG_LEVEL` | `INFO` | Log level |
| `PROXY_LOG_FILE` | `llm_proxy.log` | Log file path (10MB rotation, 7-day retention) |
| `PROXY_BUFFER_TOOLS` | `true` | Enable/disable tool call buffering |
| `PROXY_VALIDATE_JSON` | `true` | Attempt JSON repair on buffered tool call arguments |
| `PROXY_STREAM_TIMEOUT` | `120` | Max seconds to wait between SSE chunks from upstream |
| `PROXY_STREAM_MAX_DURATION` | `600` | Hard ceiling on total stream duration (seconds) |
| `PROXY_MAX_UPSTREAM_CONNECTIONS` | `100` | Max concurrent connections to upstream |
| `PROXY_REQUEST_TIMEOUT` | `300` | Total timeout for non-streaming requests (seconds) |
| `PROXY_CONNECT_TIMEOUT` | `15` | Timeout for connecting to upstream (seconds) |
| `PROXY_KEEPALIVE_TIMEOUT` | `30` | Seconds to keep idle connections in pool |
| `PROXY_MAX_ARGS_SIZE` | `1048576` | Max accumulated arguments per tool call (bytes) |
| `PROXY_MAX_TOOL_CALLS` | `32` | Max tool calls per request |

## Resource Management & Reliability

### The 85k+ Send-Q issue

The proxy runs on a separate machine from litellm. Observed symptom: `ss -tn`
showed **85k+ Send-Q** on connections to the upstream (litellm). The Send-Q
builds up during streaming responses and eventually drains — responses flow
back to the client most of the time, but the accumulation causes latency and
can make the machine temporarily unresponsive.

**Root cause**: the proxy's per-chunk processing overhead (JSON parse →
tool_call extraction → buffer processing → JSON re-encode) is slower than
the upstream's send rate. When litellm sends SSE chunks faster than the
proxy can process and forward them, data accumulates in the kernel's TCP
send buffer. This is a throughput bottleneck, not a connection leak.

The proxy mitigates this with:

1. **Fast-path passthrough** — chunks without tool_calls are forwarded as-is
   without JSON parse/re-encode, eliminating the most common per-chunk overhead.
2. **No `deepcopy`** — tool call extraction and injection use manual dict
   construction instead of `deepcopy`, which is 10-50x faster for our
   simple dict structures.
3. **`STREAM_MAX_DURATION=600s`** — hard ceiling on total stream time. Even
   if data trickles in (resetting sock_read), the stream is killed after
   10 minutes.
4. **`STREAM_TIMEOUT=120s`** — if upstream stops sending for 2 minutes, the
   stream is aborted.
5. **Client disconnect detection** — `_wait_for_disconnect()` polls
   `request.transport.is_closing()` and aborts the upstream stream immediately
   when the client goes away.
6. **`CONNECT_TIMEOUT=15s`** — don't hang on unreachable upstreams.
7. **Buffer size limits** — `MAX_ARGUMENTS_SIZE=1MB` and `MAX_TOOL_CALLS=32`
   prevent unbounded memory growth.

### Gunicorn warning

**Do not use gunicorn with this proxy.** Gunicorn's `--timeout` SIGKILLs
workers mid-stream, leaving orphaned TCP connections. For streaming SSE
responses that take 2-5 minutes, this is catastrophic. Use `python -m
llm_toolstream_proxy.main` directly (with optional uvloop for performance).

### Fixes applied

| Problem | Fix |
|---------|-----|
| Per-chunk processing slower than upstream send rate | Fast-path passthrough for non-tool-call chunks (no JSON parse/re-encode). Manual dict construction instead of `deepcopy`. |
| Zombie streams with no hard timeout | `STREAM_MAX_DURATION=600s` caps total stream time via `asyncio.wait_for`. Even if data trickles in (resetting sock_read), the stream is killed after 10 minutes. |
| No client disconnect detection | `_wait_for_disconnect()` polls `request.transport.is_closing()` and sets a cancel event that aborts the upstream stream immediately. |
| sock_read timeout too long | `STREAM_TIMEOUT` reduced from 300s to 120s. If upstream stops sending for 2 minutes, the stream is aborted. |
| No connect timeout | `CONNECT_TIMEOUT=15s` prevents hanging on unreachable upstreams. |
| No request timeout for non-streaming | `REQUEST_TIMEOUT=300s` caps total time for non-streaming requests. |
| Unbounded argument accumulation | `MAX_ARGUMENTS_SIZE=1MB` truncates further deltas when exceeded. |
| Unbounded tool call count | `MAX_TOOL_CALLS=32` rejects new tool calls beyond the limit. |
| No graceful shutdown | `on_cleanup` hook closes the shared `ClientSession` properly. |
| Non-streaming upstream errors crash the handler | `_handle_non_streaming` now catches `ClientError` (502) and `TimeoutError` (504). |

### Recommended OS-level TCP tuning

On the proxy machine, run `sudo ./setup_server.sh --tuning-only` or set these
manually in `/etc/sysctl.d/99-llm-proxy.conf`:

```ini
# TCP keepalive: detect dead upstream in ~70s instead of 2h
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6

# Retransmit: fail fast instead of 13-min retry loops
net.ipv4.tcp_retries1 = 3
net.ipv4.tcp_retries2 = 5

# Socket reuse: prevent port exhaustion under load
net.ipv4.tcp_tw_reuse = 1

# Larger socket buffers: absorb Send-Q bursts (16MB, proxy also enforces
# application-level limits)
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Connection tracking: prevent table overflow if running a firewall
net.netfilter.nf_conntrack_max = 1048576
net.nf_conntrack_max = 1048576

# File descriptors: support many concurrent connections
fs.file-max = 2097152
```

## Request Flow Diagrams

### Non-streaming request (transparent)

```
  opencode                              litellm
      │                                    │
      │  POST /v1/chat/completions         │
      │  {stream: false, ...}              │
      │───────────────────────────────────▶│
      │                                    │
      │  proxy forwards unchanged          │
      │───────────────────────────────────▶│
      │                                    │
      │  litellm responds (full JSON)      │
      │◀───────────────────────────────────│
      │                                    │
      │  proxy forwards unchanged          │
      │◀───────────────────────────────────│
      │                                    │
```

### Streaming request (intercepted)

```
  opencode                    proxy                     litellm
      │                        │                          │
      │ POST {stream:true}     │                          │
      │───────────────────────▶│ forward unchanged        │
      │                        │─────────────────────────▶│
      │                        │                          │
      │                        │       SSE line:          │
      │                        │◀─────────────────────────│
      │                        │                          │
      │                        │ ┌──────────────────┐     │
      │                        │ │ parse_sse_line   │     │
      │                        │ └────────┬─────────┘     │
      │                        │          │                │
      │                        │ ┌────────▼─────────┐     │
      │                        │ │_extract_tool_calls│    │
      │                        │ │  split into:      │    │
      │                        │ │  tool_deltas[] ──┐│    │
      │                        │ │  cleaned chunk   ││    │
      │                        │ └──────────────────┘│    │
      │                        │          │          │    │
      │                        │          │    ┌─────▼──┐ │
      │  ◀── content/reasoning │          │    │ buffer │ │
      │      passthrough      │◀─ passthru │  │  .pro- │ │
      │                        │          │    │ cess() │ │
      │                        │          │    └───┬──┬─┘ │
      │                        │          │        │  │   │
      │                        │    emit? │  │   │
      │                        │  NO → buffer, return []   │
      │                        │  YES → encode_sse_event───▶│
      │  ◀── well-formed       │                          │
      │      tool call chunk   │                          │
      │                        │       ... more lines ... │
      │                        │◀─────────────────────────│
      │                        │                          │
      │                        │       SSE: [DONE]        │
      │                        │◀─────────────────────────│
      │                        │                          │
      │                        │ ┌──────────────────┐     │
      │                        │ │ flush()          │     │
      │                        │ │ discard incompl  │     │
      │                        │ │ repair+emit buf  │     │
      │                        │ └────────┬─────────┘     │
      │                        │          │                │
      │  ◀── final tool calls  │          │                │
      │  ◀── [DONE]            │◀─────────┘                │
      │                        │                          │
```

## Logging

The proxy uses loguru with dual output:

| Sink | Format | Details |
|------|--------|---------|
| **stderr** | Colored, with level highlighting | Current session debugging |
| **`llm_proxy.log`** | Plain text, structured | 10MB rotation, 7-day retention, gzip compression |

Key log events:

| Level | Event | Module |
|-------|-------|--------|
| `INFO` | Tool call name received (with previous value) | `buffering.py` |
| `INFO` | Tool call start emitted | `buffering.py` |
| `INFO` | Request: method, path, model, streaming, buffered | `proxy.py` |
| `INFO` | Stream start/end | `proxy.py` |
| `INFO` | Flushed remaining buffered calls | `sse.py` |
| `INFO` | JSON arguments repaired | `buffering.py` |
| `DEBUG` | Each tool call delta processed | `buffering.py` |
| `DEBUG` | Argument delta streamed | `buffering.py` |
| `DEBUG` | SSE chunk tool_call count | `sse.py` |
| `WARNING` | Incomplete tool call discarded | `buffering.py` |
| `WARNING` | Invalid JSON args on started call (cannot repair) | `buffering.py` |
| `WARNING` | SSE JSON parse failure | `sse.py` |
| `ERROR` | JSON repair failed, emitting as-is | `buffering.py` |