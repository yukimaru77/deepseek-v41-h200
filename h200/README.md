# Run the H200 build

These commands target a Linux x86_64 host with two H200 GPUs, Docker, and NVIDIA Container Toolkit. The measured host had 1.1 TiB RAM and no detected NVLink connection. Model tensors alone occupy about 385.4 GiB; reserve additional storage for rank caches, compiler caches and the container. The minimum stable RAM configuration has not been measured.

## Build and acquire the pinned weights

From the repository root:

```bash
docker build -f h200/Dockerfile -t deepseek-v41-h200:local .
export DATA_DIR=/srv/deepseek-h200
mkdir -p "$DATA_DIR"

# Offline plan: prints the two model revisions and expected size, downloads nothing.
docker run --rm --entrypoint python3 deepseek-v41-h200:local \
  /app/h200/download_models.py --plan

# Explicitly download the EXL3 model and only the two official Engram shards.
# --verify reads every selected file and checks the pinned HF hashes afterward.
docker run --rm -v "$DATA_DIR:/data" --entrypoint python3 \
  deepseek-v41-h200:local /app/h200/download_models.py \
  --data-dir /data --download --verify
```

The revisions and shard counts are in [pins.json](pins.json). Both model sources are public; if your environment requires a Hugging Face token, supply `HF_TOKEN` to the download container using your own credential management. Do not put credentials in the image or repository. Downloads and SHA checks can take substantial time. Starting the server never implicitly downloads weights.

The base image is pinned to `lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155`. Its image ID on the measured host is `sha256:1cc022f1ddc29b84449b588a872c0e9cab2fc1268a36ef5449fe6956ef774f2a`; these identify the manifest and image configuration respectively. The image provides the CUDA/PyTorch toolchain. **TensorFold, not SGLang, runs inference.** Lightweight Python dependency versions are pinned in [requirements.txt](requirements.txt).

## Start both ranks

```bash
DATA_DIR=/srv/deepseek-h200 bash h200/run.sh
```

The launcher runs in the foreground. Ctrl-C stops it; if either rank exits, the other is terminated. It does not stop or replace an already running server. Initial startup loads/prefaults Engram, builds kernels, writes per-rank caches, warms graphs, and runs the capacity preflight. Wait for readiness before sending work. On subsequent launches the persistent caches can be reused.

Useful overrides:

```bash
DATA_DIR=/srv/deepseek-h200 H200_GPU_IDS=2,3 \
  H200_API_PORT=18891 H200_MASTER_PORT=29571 bash h200/run.sh
```

Both ranks use the same revisions, cache paths and settings. The container exposes only the two selected GPUs; the processes address them as local devices 0 and 1. The API binds to `127.0.0.1`. Port conflicts are not resolved by killing another process.

The tested round-02 optimization is opt-in, preserving the existing default profile:

```bash
DATA_DIR=/srv/deepseek-h200 H200_KERNEL_PROFILE=round02 bash h200/run.sh
```

This selects the reviewed `cand-r3` settings in [round02.env.sh](round02.env.sh): expert/grouped-linear Tensor Core operand layout, asynchronous loads, small same-host P2P gathers, and host scheduling. It requires peer access between the two H200 GPUs. Restart both ranks when changing the profile. Use `H200_KERNEL_PROFILE=baseline` for the default paths; an exact old-binary comparison requires the earlier source/image. [Measured results and A/H reproduction](../docs/H200-OPTIMIZATION.md).

The default configuration retains the recorded 1017 per-request context and active-slot limits, enables the checkpoint's image tower, and reduces the shared pool by one full-window extent to provide image workspace:

| Setting | Value |
|---|---:|
| Per-request input + output context | 1,048,576 tokens |
| Shared KV token slots | 36,771,840 (35 full-window extents; the earlier text profile used 36) |
| Active request slots | 63 |
| Full-window extents that fit in the pool | 35, calculated; not 35 concurrently executed full-1M requests |
| Prefill chunk | 2,048 tokens |
| DSpark draft cap | 5; reduced with active row count, disabled at 9+ rows |
| Retained CUDA graphs | 64 |
| Retained completed prefixes | Up to 64 entries, up to 10 boundaries each |
| Decoder prefill | `TF_DS_REPLAY=1`, bounded replay |
| Image input | Enabled; rank 0 holds the vision tower, largest image grid warmed at startup |

Paths under `DATA_DIR`: `models/`, revision-specific `cache/`, and `results/capacity/`. Engram is read-only mmap plus prefault; it is backed by shared host file pages. The two process RSS values double-count shared pages, so use PSS for combined process residency.

The environment settings are in [env.sh](env.sh). The original text benchmarks used 37,822,464 pool slots (36 extents); those historical results are unchanged. The current 36,771,840-slot pool releases approximately 0.879 GiB per GPU for image workspace. Both ranks passed startup preflight, and a loopback request containing the same screenshot in 65 tool results completed successfully; this did not test 65 distinct images or an image-filled 1M context. See [image-history validation](../benchmarks/h200/image-history-validation.json). `TF_DS_CAPACITY_GRAPH_STRESS=1` exercises the exact 63-slot/64-graph shape. Changing dimensions requires deliberate memory sizing and adjusting or disabling that specific stress check. Do not interpret a maximum-shape check as 63 fully populated 1M contexts.

## Image input

The pinned EXL3 checkpoint includes the BF16 vision tower and aligner, image delimiter embeddings, and image-routing biases. No additional model download is needed. `H200_VISION=1` passes `--vision` on both ranks. The earlier decode and long-context results used `H200_VISION=0` and `TF_DS_POOL_TOKENS=37822464`; both overrides are needed to restore that historical profile.

Send image bytes as base64: OpenAI Chat Completions uses an `image_url` content block containing a `data:image/png;base64,...` URL; Anthropic Messages uses an `image` block with a base64 `source`. Images inside tool results are supported, including screenshots returned by an agent's image-reading tool. HTTPS image fetching remains disabled by default.

The H200 profile has **no fixed image-count limit**, including images in earlier messages and tool results (`H200_VISION_MAX_IMAGES=0`). Visual tokens share the model's 1,048,576-token context with text; `H200_VISION_IMAGE_TOKENS` defaults to `H200_CONTEXT`. Each image is resized to the model's per-image visual-token budget. DeepSeek Harness must also declare `input: [text, image]` for this model, as in [harness.patch.yml](harness.patch.yml); a text-only declaration prevents `read_image` from sending the request.

The profile allows 128 MiB of compressed image data and 128 Mi pixels per request, and a 192 MiB HTTP body to accommodate base64 expansion. Individual-image limits remain 10 MiB and 16 Mi pixels. Removing the count limit does not remove these byte/pixel budgets, the context window, or available CPU/GPU memory limits. A context filled with visual tokens has not been validated. The environment settings are `H200_VISION_MAX_IMAGES`, `H200_VISION_IMAGE_TOKENS`, `TF_VISION_MAX_TOTAL_ENCODED_BYTES`, `TF_VISION_MAX_TOTAL_PIXELS`, and `TF_REQUEST_BODY_LIMIT_BYTES`. A reverse proxy must allow the same 192 MiB request body (for nginx, `client_max_body_size 192m;`).

## Use and measure

```bash
curl http://127.0.0.1:18891/v1/models
curl 'http://127.0.0.1:18891/v1/models?limit=1000'
curl http://127.0.0.1:18891/health

# Reproduce the published short-input decode protocol: warm-up, 512 tokens,
# 3 trials per prompt, one active stream. This actually generates output.
mkdir -p benchmark-runs
python3 tools/dsv41/kit_bench.py \
  --base http://127.0.0.1:18891 --model DeepSeek-V4.1-Flash-1017-H200 \
  decode --set a --tokens 512 --reps 3 --out benchmark-runs/decode.json
```

`kit_bench.py` uses the Python standard library for this API benchmark. To measure an authenticated gateway, add `--key-file /path/to/your/key` before the `decode` subcommand. For direct comparison, measure on the inference host and keep other inference jobs idle. The short-input speed protocol is separate from the long-context capacity validation. Details and the optional costly validation commands are in [H200-BENCHMARKS.md](../docs/H200-BENCHMARKS.md).

OpenAI and Anthropic client examples: [CLIENTS.md](CLIENTS.md). The server's inherited native API-key function currently does not enforce the intended key, so external access requires a separate authentication layer. No deployment-specific public URL, key, tunnel or account configuration is shipped.

## Validation scope

The original results describe the running deployment; the round-02 results describe the separately validated candidate. The checked-in source includes that candidate, while `H200_KERNEL_PROFILE=round02` selects its optimized paths. The portable packaging is checked separately: see [H200-VALIDATION.md](../docs/H200-VALIDATION.md). Building this package does not itself reload the running model or repeat the multi-minute 1M workload.
