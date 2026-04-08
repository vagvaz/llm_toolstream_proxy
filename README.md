# qwen-jsontool

Streaming SSE proxy that buffers and reassembles incomplete tool calls from Qwen/MiniMax models via litellm.

## Problem

Qwen 3.5, MiniMax 2.5, and other models served through litellm/vllm/ollama backends emit streaming tool call chunks that violate the OpenAI SSE spec:

- `function.name` arrives as `""`, `null`, or the string `"null"` in continuation chunks
- `id` may be absent in the first chunk and arrive later
- Arguments arrive before name/id are available
- Empty tool call objects appear in parallel call scenarios

This causes opencode (and other consumers) to crash with `Expected 'function.name' to be a string`.

Related issues: [agno#6757](https://github.com/agno-agi/agno/issues/6757), [litellm#12513](https://github.com/BerriAI/litellm/issues/12513), [litellm#17425](https://github.com/BerriAI/litellm/issues/17425), [vllm#17614](https://github.com/vllm-project/vllm/issues/17614), [spring-ai#4790](https://github.com/spring-projects/spring-ai/issues/4790), [opencode#10855](https://github.com/anomalyco/opencode/issues/10855)

## Architecture

```
opencode  -->  localhost:8787 (this proxy)  -->  litellm:4000
```

The proxy intercepts streaming (`stream: true`) responses and:
1. Parses each SSE `data:` line from litellm
2. Routes tool_call deltas through a `ToolCallBuffer` for reassembly
3. Buffers incomplete tool calls (missing name/id) until metadata is complete
4. Re-emits well-formed tool call chunks to opencode
5. Passes non-tool content/reasoning through immediately (zero latency)

Non-streaming requests are forwarded transparently with no modification.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Set your litellm upstream URL
export LITELLM_URL=http://your-litellm-host:4000

# Start the proxy
qwen-jsontool
```

Then point opencode at the proxy:

```bash
# In opencode config or environment
export OPENCODE_API_URL=http://localhost:8787
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `LITELLM_URL` | `http://localhost:4000` | Upstream litellm URL |
| `PROXY_HOST` | `0.0.0.0` | Host to bind the proxy to |
| `PROXY_PORT` | `8787` | Port to listen on |
| `PROXY_LOG_LEVEL` | `INFO` | Logging level |
| `PROXY_BUFFER_TOOLS` | `true` | Enable/disable tool call buffering |
| `PROXY_VALIDATE_JSON` | `true` | Validate/repair JSON arguments on flush |
| `PROXY_STREAM_TIMEOUT` | `300` | Stream read timeout in seconds |

## How It Works

### Buffering Algorithm

For each streaming chunk containing `tool_calls`:

1. **Sanitize**: Convert `name: ""`, `name: "null"`, `name: null` → `None` (not yet available)
2. **Accumulate**: Store id, name, arguments by tool call index. First non-empty value wins.
3. **Emit when ready**: Only emit a tool call start chunk when both `id` and `name` are present.
4. **Stream arguments**: Once started, forward argument deltas immediately.
5. **Flush on stream end**: Emit any fully-buffered tool calls. Discard incomplete ones with a warning.

### JSON Repair

When `PROXY_VALIDATE_JSON=true`, the proxy attempts basic JSON repair on tool call arguments before emission:
- Missing closing braces: `{"key": "val"` → `{"key": "val"}`
- Missing closing brackets: `{"items": [1, 2` → `{"items": [1, 2]}`  
- Trailing commas: `{"key": "val",}` → `{"key": "val"}`

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Complementary: Opencode Patch

For maximum resilience, also patch opencode's streaming handler at `openai-compatible-chat-language-model.ts:527-530` to be resilient to empty names (like agno's fix in [#7132](https://github.com/agno-agi/agno/pull/7132)):

```typescript
// Instead of throwing:
if (toolCallDelta.function?.name == null) {
    throw new InvalidResponseDataError({...})
}
// Buffer gracefully:
if (toolCallDelta.function?.name == null || toolCallDelta.function.name === "") {
    // Accumulate into toolCalls[index] and wait for more chunks
    continue
}
```

## Complementary: Prompt Hardening

Add to your project's `AGENTS.md`:

```markdown
When calling tools, always include the complete function name and well-formed JSON arguments
in the first output chunk. Do not emit partial tool calls where the function name or arguments
are incomplete across multiple chunks.
```