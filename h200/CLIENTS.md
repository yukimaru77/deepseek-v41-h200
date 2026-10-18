# H200 API client examples

The engine listens on the host's loopback interface, port 18891. Use a local client or an SSH tunnel; an internet-facing deployment needs a separate authenticated TLS reverse proxy. No private endpoint or working credential is included in this repository. The inherited CUDA HTTP `_api_key()` currently has no return value, so `TF_API_KEY` is not an authentication boundary in this source snapshot.

## DeepSeek Harness

Tested with the official source build 0.2.1-rc.2, commit `687f282e8625c6b8d11ff58211a90524be2e4317`.

For a client on the inference host, or after forwarding remote port 18891 to local port 18891:

```bash
# A placeholder satisfies the client's key requirement on the local-only route.
# It does not enable authentication on the engine.
H200_API_KEY=local-only dsh --profile web --patch h200/harness.patch.yml
```

For your own authenticated gateway, set `H200_ANTHROPIC_BASE_URL` to its API root and supply its actual `H200_API_KEY` through your local credential manager/environment. Do not append `/v1` to the Messages base URL. The model ID is `DeepSeek-V4.1-Flash-1017-H200`.

In the Custom model API form, choose **Anthropic Messages**, enter the API root and key, and add this model ID. Declare a context window of 1048576 and output cap of 32768. The supplied patch additionally configures the compatibility flags needed for reasoning levels and history preservation:

- **`forceAdaptiveThinking: true`** sends `thinking.type=adaptive` and `output_config.effort` for Low / High / Max. Without it, the generic adapter may use Claude-style token budgets instead of DeepSeek effort.
- `allowEmptySignature: true` preserves the returned thinking blocks when their signature is empty, including after tool results.
- The explicit `max: max` mapping keeps Max from falling back to High. Off sends `thinking.type=disabled`.

Effort values correspond to Low=50, High=75, Max=100 in the model prompt. They are strengths, not fixed token budgets or accuracy guarantees. A simple request may produce no thinking text even with Max. Thinking and final text share the 32768-token client output cap.

Verified behavior: Max thinking → real `read` tool → thinking/tool-result history replay → exact file-content answer; Off without thinking; model discovery with `?limit=1000`.

## Claude Code

Tested with Claude Code 2.1.302. Keep this invocation in its own configuration directory:

```bash
CLAUDE_CONFIG_DIR="$HOME/.config/claude-h200" \
ANTHROPIC_BASE_URL="${H200_ANTHROPIC_BASE_URL:-http://127.0.0.1:18891}" \
ANTHROPIC_AUTH_TOKEN="${H200_API_KEY:-local-only}" \
ANTHROPIC_MODEL=DeepSeek-V4.1-Flash-1017-H200 \
ANTHROPIC_DEFAULT_OPUS_MODEL=DeepSeek-V4.1-Flash-1017-H200 \
ANTHROPIC_DEFAULT_SONNET_MODEL=DeepSeek-V4.1-Flash-1017-H200 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=DeepSeek-V4.1-Flash-1017-H200 \
CLAUDE_CODE_SUBAGENT_MODEL=DeepSeek-V4.1-Flash-1017-H200 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
ENABLE_TOOL_SEARCH=false claude
```

A real `Read` tool roundtrip was verified through an authenticated gateway. This provides ordinary Messages compatibility, not complete Claude-native feature equivalence. Forced `tool_choice=any` / named-tool selection, tool search, some context edits, and encrypted/redacted thinking are unsupported.

## OpenAI Chat Completions

Use `http://127.0.0.1:18891/v1` as the base URL and the same model ID. To enable reasoning explicitly:

```json
{
  "model": "DeepSeek-V4.1-Flash-1017-H200",
  "messages": [{"role": "user", "content": "What is 17 times 19?"}],
  "max_tokens": 1024,
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "max"}
}
```

Keep assistant `reasoning_content` when replaying tool calls. For non-thinking requests, set `enable_thinking` to false. The server's output cap and per-request context are separate limits; clients should budget input and output within the total 1048576-token context.

## Images and screenshots

Start the server with vision enabled (the H200 launcher's default). In DeepSeek Harness, the model's `input` list must include both `text` and `image`; the example patch includes this declaration. Otherwise `read_image` rejects the selected model before sending an API request. Changing the client declaration alone does not enable the server's vision tower.

Both user images and images returned in tool results use the model's native vision path. Send base64 image content, not the client's local filesystem path, to the API. The H200 launcher uses `--vision-max-images 0`, removing the fixed count limit across the full request history, including earlier tool results. Images still share the model context and available memory. The example Harness provider sets `maxRequestImageBytes: 134217728`, a 128 MiB limit on the **base64-encoded** image payload; Harness's per-image defaults remain a 1 MiB encoded-file target and a 2048 × 2048 pixel budget. This differs from the server's 128 MiB limit on decoded image-file bytes. When using a reverse proxy, allow at least the launcher's 192 MiB HTTP body limit (for Nginx, `client_max_body_size 192m;`). These byte budgets remain in effect even without an image-count limit.

The current server profile preserves the 1,048,576-token per-request context and 63 active slots, while reducing the shared KV pool to 36,771,840 tokens (35 full-window extents by calculation) to leave room for image processing. One-image API and tool roundtrips were verified. A direct loopback Anthropic request with the same 1440 × 900 screenshot repeated in 65 tool results also returned HTTP 200 and a correct Japanese description: 28.54 seconds, 55,097 input tokens, 46 output tokens, and no prefix-cache reuse. This is a repeated-image history check, not 65 distinct images or a latency measurement through an external gateway. See [sanitized results](../benchmarks/h200/image-history-validation.json). Removing the count limit does not establish that every image history fitting the token and byte budgets will fit in VRAM; an image-filled 1M context remains untested.

Resending the same 151,108,217-byte request through the public gateway also returned HTTP 200, in 11.171 seconds, with the same 46-token answer and `end_turn`. This repeat reused 53,248 input tokens and processed 1,849 uncached input tokens. Its latency includes the external route and cache reuse; it is not an uncached 65-image performance measurement.
