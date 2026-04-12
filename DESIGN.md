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

### Why the proxy stalled the host machine (85k+ Send-Q)

The proxy runs on a separate machine from litellm. Observed symptom: `ss -tn`
showed **85k+ Send-Q** on connections to the upstream (litellm). Send-Q is the
kernel's TCP send buffer — data the proxy has written but litellm hasn't
acknowledged. When these buffers accumulate across many stuck connections, the
kernel's network stack spends all its CPU on retransmissions and eventually
can't accept new connections (including SSH).

**Root cause**: connections to litellm that never close properly. The proxy
opened a TCP connection, sent the request body, and then waited for the SSE
response. If litellm stopped responding (or the response trickled in slowly),
the connection stayed open with unacked data in the Send-Q. With no hard
timeout on total stream duration, these zombie connections accumulated
indefinitely.

Specific issues in the original code:

1. **`total=None` on streaming requests** — no maximum duration. A stream could
   run forever if data trickled in slowly (each chunk resets the `sock_read`
   timer). Zombie streams with 85k+ Send-Q accumulated.
2. **Connection pooling with `force_close=False`** — after a long streaming
   request, the connection could be in a bad state. Reusing it meant the next
   request also got stuck.
3. **No `Connection: close` on streaming requests** — the upstream didn't know
   to close the connection after the stream ended, so stale connections
   lingered.
4. **`STREAM_TIMEOUT=300`** (5 min sock-read) — way too long. If litellm
   stopped sending data, the proxy waited 5 minutes per chunk before giving up.
5. **No client disconnect detection** — if opencode disconnected mid-stream,
   the proxy kept reading from litellm into the void, holding the upstream
   connection open.

### Fixes applied

| Problem | Fix |
|---------|-----|
| Zombie streams with no hard timeout | `STREAM_MAX_DURATION=600s` caps total stream time via `asyncio.wait_for`. Even if data trickles in (resetting sock_read), the stream is killed after 10 minutes. |
| Stale connections with unacked Send-Q | `force_close=True` on TCPConnector — every request gets a fresh connection that is closed after use. No connection reuse means no stale connections. |
| Upstream not closing connections | `Connection: close` header on streaming requests tells litellm to close the connection after the response. |
| No client disconnect detection | `_wait_for_disconnect()` polls `request.transport.is_closing()` and sets a cancel event that aborts the upstream stream immediately. |
| sock_read timeout too long | `STREAM_TIMEOUT` reduced from 300s to 120s. If upstream stops sending for 2 minutes, the stream is aborted. |
| No connect timeout | `CONNECT_TIMEOUT=15s` prevents hanging on unreachable upstreams. |
| No request timeout for non-streaming | `REQUEST_TIMEOUT=300s` caps total time for non-streaming requests. |
| Unbounded argument accumulation | `MAX_ARGUMENTS_SIZE=1MB` truncates further deltas when exceeded. |
| Unbounded tool call count | `MAX_TOOL_CALLS=32` rejects new tool calls beyond the limit. |
| No worker recycling | `--max-requests 10000 --max-requests-jitter 1000` recycles gunicorn workers to prevent memory/FD leaks. |
| No graceful shutdown | `on_cleanup` hook closes the shared `ClientSession` properly. |
| Non-streaming upstream errors crash the handler | `_handle_non_streaming` now catches `ClientError` (502) and `TimeoutError` (504). |

### Why gunicorn causes the 85k+ Send-Q stall

**Gunicorn's worker timeout kills workers mid-stream.** The `--timeout` flag
tells gunicorn to SIGKILL any worker that hasn't completed a request within N
seconds. For streaming SSE responses that routinely take 2-5 minutes, this
means gunicorn kills workers while they're still reading from litellm and
writing to opencode.

When a worker is killed mid-stream:
1. The worker process is SIGKILL'd — no cleanup, no graceful shutdown
2. All its TCP connections are left half-open with data in their Send-Q
3. The kernel keeps retransmitting that data (85k+ per connection)
4. These orphaned connections accumulate until the network stack chokes
5. SSH can't establish new connections because the kernel is busy retransmitting

This was confirmed by the user: the same proxy code works fine when run
directly via `python -m llm_toolstream_proxy.main` (no gunicorn), but stalls
when deployed via gunicorn. Another aiohttp-based proxy had the same issue
with gunicorn but not with direct deployment.

**Solution: don't use gunicorn.** The proxy now runs directly via
`web.run_app()` with optional uvloop for performance. Gunicorn's process
management model is designed for sync WSGI workers, not long-lived SSE streams.

### Recommended OS-level TCP tuning

On the proxy machine, set these in `/etc/sysctl.d/99-llm-proxy.conf`:

```ini
# Detect dead connections faster (default: 7200s)
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6

# Reduce TCP retry timeout (default: 15 min)
net.ipv4.tcp_retries1 = 3
net.ipv4.tcp_retries2 = 5

# Allow reuse of sockets in TIME_WAIT state
net.ipv4.tcp_tw_reuse = 1
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