# Publication validation

Validation date: 2026-10-18 JST. The original running two-H200 model was left running throughout initial publication. The subsequent image-input fix did restart the service; its validation is recorded separately below.

## Source and protocol checks

The publication tree was reconstructed from the pinned TensorFold commit, the H200 capacity patch, and the 1017 encoder patch. Before publication notices were added, its implementation files matched the deployed 1017 source byte-for-byte, excluding generated package metadata and bytecode. Modified-file comments were then added to preserve attribution; the comments do not alter execution. The models-list query-string fix is included in both the on-disk deployment source and this snapshot; it takes effect in the deployed process on its next startup.

The focused CPU suite passed in the pinned Linux image without GPU access:

```text
117 passed, 1 skipped, 19 subtests passed
```

Files exercised: `test_dsv41_1017_chat.py`, `test_dsv41_capacity_limits.py`, `test_dsv41_snapshot_budget.py`, `test_cuda_markers.py`, `test_cuda_thinking_controls.py`, and `test_h200_model_listing.py`. These verify encoder/tool history, reasoning options, context admission, snapshot accounting, and query-bearing model discovery. They are not GPU speed measurements.

## Portable packaging checks

- **The published Dockerfile builds** from the pinned image and checked-in source.
- The built image imports TensorFold and the official encoder, runs the CLI help, and prints the pinned offline download plan.
- Runtime libraries resolve to PyTorch `2.11.0+cu130`, Triton `3.6.0`, NCCL `2.28.9`.
- Starting without model files fails before rank launch, with a useful missing-file error.
- Shell syntax and help paths pass; mocked rank execution confirms that one rank's failure terminates its peer and preserves the failure status.
- The downloader's SHA-256 / Git-blob SHA-1 checks, corruption rejection, shard counts, argument handling, and offline plan were checked with small fixtures. The actual model shards had already been verified in the original deployment.

The new generic launcher was **not used to cold-start another full model** during publication: the two GPUs were occupied by the existing service. The kernel, capacity, quality and client results in [H200-BENCHMARKS.md](H200-BENCHMARKS.md) and [benchmarks/h200](../benchmarks/h200/) are measurements of the original deployed implementation, not a fresh timing run of this packaging.

The optional long-context reproduction scripts were generalized from the campaign scripts and checked for syntax/help. The fixture generator was also executed from the built image against the running tokenizer endpoint only, without generation: its 1,046,916 raw-token fixture reproduced the campaign SHA-256 `bde79cc15b8d7f07802fb84ed6e3cd0753e9be9d4d0fd94a25eff3eb79c82fae` exactly. The chat wrapper and follow-up question bring the measured long request to 1,046,955 tokens. The full GPU workload was not repeated during publication. Generated local outputs contain synthetic prompts and replies; the published result extracts omit those texts.

## Re-run the focused tests

The pinned base image includes pytest and the CPU-importable dependencies:

```bash
docker run --rm --network none -v "$PWD:/app:ro" -w /app \
  --entrypoint bash \
  lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155 \
  -lc 'PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
    tests/test_dsv41_1017_chat.py tests/test_dsv41_capacity_limits.py \
    tests/test_dsv41_snapshot_budget.py tests/test_cuda_markers.py \
    tests/test_cuda_thinking_controls.py tests/test_h200_model_listing.py'
```

No GPU devices or model weights are needed for this command. The included upstream tests for other model families were not comprehensively requalified by this H200 port.

## Image-input follow-up

The deployed weights already contained the vision tower, but the original launch command omitted `--vision` and the DSH model declared only text input. Enabling vision exposed an incompatibility between the older image-placeholder path and the 1017 official encoder. The fix connects `DsVision` to the shared image loader and tokenizer, preserves image order when the official encoder reorders parallel tool results, and removes the obsolete family-specific preparation override. The official encoder itself is unchanged.

The focused CPU suite passed **272 tests, with 7 skipped**: `test_dsv41_vision_adapter.py`, `test_dsv41_1017_chat.py`, `test_vision_tool_results.py`, `test_vision_server.py`, `test_vision_images.py`, `test_vision_limits.py`, `test_cuda_thinking_controls.py`, and `test_h200_model_listing.py`. It checks the official encoder's token sequence, image/pixel ordering, EXIF orientation, expanded context and visual-token budgets, and text-only behavior. The updated Docker image built successfully and imported Pillow 12.3.0 and the adapter without GPU access.

After deployment, a screenshot was recognized through OpenAI user-image, Anthropic user-image, and Anthropic tool-result-image requests. All returned HTTP 200; the 200-token cap truncated the tool-result description, while the other two stopped normally. These are functional checks of one image, not a broad vision quality evaluation. The image and its descriptions are not distributed. See [vision-validation.json](../benchmarks/h200/vision-validation.json) for the sanitized results.

DeepSeek Harness's live web settings were verified to declare `input: [text, image]`. A separate headless session then invoked `read_image` once, completed the tool successfully, and described the image's main object and colors without image-content hints in the prompt (exit 0, 22.833 seconds for the entire session). The web service did not need a restart.

At that stage, both ranks passed the existing capacity preflight with vision enabled, retaining 1,048,576 tokens per request, 63 active slots, and 37,822,464 shared KV slots (36 full-window extents). The rank-0 free memory after checks was about 1.66 GiB; with an additional 1 GiB workspace held, about 0.65 GiB remained. These results precede the pool reduction described below. The earlier full near-1M workload and decode benchmark were not repeated. The portable image was rebuilt, while live GPU checks used the updated source in the existing deployment container.

## Image-history follow-up

The H200 profile now sets `--vision-max-images 0`, removing the fixed count limit across messages and tool results. DeepSeek's visual-token budget follows the configured context window; expanded image tokens, text and output still share that window. Qwen's separate 65,536 visual-token option bound is unchanged. The loader retains byte, pixel and time budgets. The HTTP body setting is shared by Chat Completions, Responses, Messages and token-count routes, including the Anthropic path that previously retained an independent 32 MiB constant.

The image-count, image-budget and HTTP-body suite passed **279 tests**. It exercises 65-image preparation, invalid limits, expanded-context rejection and configured body boundaries through all four routes on both handlers. A subsequent image-sharing and capacity suite passed **46 tests and 19 subtests**, including chunk boundaries, rank ordering, exact returned values, retained storage size and prefix reuse that starts partway through an image.

The first 65-image GPU trial exposed a CUDA allocation failure during prefill. The full all-gather retained both ranks' image buffers even though only rank 0's image values were used. Image sharing now uses chunks of at most 2,048 rows and retains one copy per rank. Image assembly also fills one allocation incrementally, avoiding concatenation's duplicate full tensor. The 37,822,464-slot pool still left insufficient VRAM for the 65-image workload after these changes.

The current default pool is therefore 36,771,840 tokens, or 35 rounded full-window extents, releasing approximately 0.879 GiB per GPU for image workspace. The per-request 1,048,576-token context and 63 active slots are unchanged. Both ranks passed startup preflight with this pool; device-free memory after checks was 2,721,579,008 bytes on rank 0 and 3,791,126,528 bytes on rank 1 (approximately 2.535 and 3.531 GiB). These are startup readings, not image-workload peak-memory measurements.

The revised GPU trial succeeded through the direct loopback Anthropic endpoint. Its tool-result history contained **65 repetitions of the same 1440 × 900 screenshot**, not 65 distinct images. The POST body was 151,108,217 bytes, including 113,314,760 bytes of compressed image-file payload before base64 encoding. With no prefix-cache reuse, the request returned HTTP 200 in **28.54 seconds**, reported 55,097 input tokens and 46 output tokens, and ended with `end_turn`. The Japanese description matched the image. This is one first-pass loopback functional check; it is not an external-gateway latency measurement or a repeated performance benchmark. See [image-history-validation.json](../benchmarks/h200/image-history-validation.json).

The same 151,108,217-byte request was then resent through the public API gateway and returned HTTP 200 in **11.171 seconds**. It reused 53,248 input tokens and reported 1,849 uncached input tokens, totaling the same 55,097-token input. The 46-token answer matched the first response and ended with `end_turn`. This establishes successful transport of this request through the public route; the timing is a **cached repeat**, not an uncached 65-image speed result.

Historical text benchmarks and the earlier vision preflight remain results of the 36-extent configuration. An image-filled 1M context, 35 concurrent full-1M requests, and general multi-image recognition quality remain untested. Neither the count-limit removal nor the smaller pool guarantees that an arbitrary image history will fit in VRAM.
