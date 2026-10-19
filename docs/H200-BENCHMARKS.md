# H200 measurements and their limits

For the 2026-10-19 optimization A/B (285.8 → 338.1 tok/s on the fixed code suite), see [H200-OPTIMIZATION.md](H200-OPTIMIZATION.md) and the [per-request extracts](../benchmarks/h200/optimization-round02.json). The sections below retain the original publication measurements and conditions.

The **DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw** deployment produced **144–294 output tokens/s** on three short-input, single-stream workloads with speculative decoding. It also passed a 63-stream isolation test and two near-1M-token input tests. These are different experiments; their rates and capacity claims should not be combined.

The measurements below were recorded on 2026-10-17 UTC (2026-10-18 JST) with two NVIDIA H200 NVL GPUs, tensor parallelism 2, PyTorch 2.11.0+cu130 and CUDA 13.0. The host had an AMD EPYC 9115, 32 logical CPU threads and approximately 1.1 TiB RAM. See [runtime instructions](../h200/README.md) and [pinned assets](../h200/pins.json).

**The measured configuration enables bounded decoder replay (`TF_DS_REPLAY=1`).** Decoder layers run over the last 128 prompt tokens with their replay window truncated there; the relevant KV source still covers earlier input. This changes prefill work relative to the exact full-prefill path. The saved retrieval and quality checks do **not** establish equivalence to exact full prefill. All current-model numbers on this page use this setting.

Machine-readable results:

- [Current 1017 model](../benchmarks/h200/current-model.json): decode, near-1M inputs, capacity, GPU memory and quality.
- [Historical component experiments](../benchmarks/h200/historical-microbenchmarks.json): expert kernels, NCCL and Engram I/O. Model-dependent measurements here used **pre-1017 weights**.
- [Live RAM snapshot](../benchmarks/h200/live-memory-snapshot.json): a later read-only snapshot, separate from the benchmark peaks.
- [Client compatibility](../benchmarks/h200/client-compatibility.json): separate functional checks, not speed measurements.

The JSON files contain selected metrics, settings, relative original source names and SHA-256 hashes. They exclude credentials, deployment addresses, complete prompts, generations and evaluation datasets. Original server logs contain subsequent deployment activity and are not distributed. Hashes identify the archived evidence used for these extracts; the commands below permit independent reruns rather than reconstructing excluded logs.

## Short-input, single-stream decode

Each workload ran one 32-token warm-up, then three sequential 512-token trials, temperature 0, thinking disabled, with MTP enabled and at most five draft tokens. All nine trials generated the full 512 tokens. Output text was identical across the three repetitions of each workload.

| Workload | Median output tokens/s | Three trials, tokens/s | Median TTFT |
|---|---:|---|---:|
| Code generation | 256.53 | 256.907 / 256.526 / 256.455 | 69.45 ms |
| Prose generation | 144.38 | 144.435 / 144.235 / 144.380 | 71.45 ms |
| Structured JSON generation | 293.58 | 293.594 / 293.563 / 293.585 | 114.54 ms |

These are client-side SSE measurements over the local API. The benchmark defines decode rate as `(completion_tokens - 1) / (last_content_chunk_time - first_content_chunk_time)`. TTFT runs from request start to the first nonempty content chunk. Speculative decoding can emit several tokens per chunk, so this is a chunk-timed estimate, not an individual-token GPU timing. The warm-up and repeated prompts can exercise caching; cache-hit counts were not retained in this particular result file. Do not label these values cold-start latency or a workload-independent speed guarantee.

Evidence: `experiments/09-h200-1017/results/decode.json`; method: `tools/dsv41/kit_bench.py`, `Client.stream` and `cmd_decode`. Exact values and source hashes are in `single_stream_decode` of the current-model JSON.

## Near-1M inputs and prefix reuse

Two requests with **1,046,955 input tokens each** were submitted together, with distinct prefixes and zero reported cached tokens. Both returned all three planted passphrases. The input was deterministic random English-word filler with passphrases and Japanese fact records at 10%, 50% and 90% of **word positions**. This is a limited synthetic retrieval test, not a general long-document benchmark.

| Request | Cached input tokens | Server prefill | Server TTFT | Client total latency |
|---|---:|---:|---:|---:|
| First completed near-1M request | 0 | 265.35 s | 265.88 s | 266.47 s |
| Second completed near-1M request | 0 | 263.92 s | 530.23 s | 530.70 s |
| Cached repeat A | 1,046,528 | 0.42 s | 0.61 s | 0.886 s |
| Cached repeat B | 1,046,528 | 0.40 s | 0.57 s | 0.833 s |

The first two server-reported prefill times correspond to approximately **3,946 and 3,967 uncached input tokens/s**. These are derived from one pair of requests, with the model already loaded and warmed, 2,048-token prefill chunks and bounded replay. They are not an independent repeated prefill benchmark. The second request's TTFT includes queueing behind the first prefill. Each recall reply was only 24 output tokens, so its short decode segment is unsuitable for a general long-context decode-speed claim.

The cached repeats process only 427 uncached input tokens. Their subsecond latency must not be presented as cold 1M prefill. A subsequent Japanese fact-summary request used 1,046,981 input tokens, reused 1,046,528, generated 201 output tokens and completed in 1.649 s. Saved-response review confirmed all named facts, the budget total and deadline ordering. All long replies stopped normally.

Evidence: `results-validation/validation.json`, plus controlled-request timing lines 1012, 1526, 1530, 1534 and 1539 of the original rank-0 server log, within experiment 09. The public JSON records the source log hash and line numbers but omits request IDs and text. Review: `results/validation-review.json`.

## Capacity and GPU memory

The profile uses a 1,048,576-token **input-plus-output** limit per request, 37,822,464 shared pool token slots, 63 active request slots, CPU-pinned prefix snapshots and an LRU limit of 64 retained round graphs.

| Claim | Evidence |
|---|---|
| **63 simultaneous decoding streams observed** | 63 distinct 90-token inputs, 256 output tokens each; every result exactly matched its own draft-disabled solo baseline and retained its independent marker. |
| Two near-1M inputs completed | 2,093,910 total input tokens, zero cache hits on their first passes; six of six passphrases retrieved. |
| 36 full-window extents fit in the pool | Arithmetic: `36 × 1,050,624 = 37,822,464`; each extent includes internal headroom and 2,048-token alignment. **36 full-1M requests were not executed concurrently.** |
| Maximum graph/address checks passed | Both ranks exercised the 63-row × 1M-position shape, 64 retained graphs with 44 evictions, far-pool addressing and exact CPU-snapshot restoration. These checks do not fill 63 real 1M contexts. |

The 63-stream burst produced 16,128 output tokens in 24.059 s, or a **derived 670.36 output tokens/s end to end**. This includes prefill and queueing and comes from a correctness/capacity workload with `ignore_eos=True`; it is not a dedicated saturated decode-throughput benchmark or a per-user rate.

During the capacity and quality suite, GPU memory was sampled every two seconds:

| GPU | Observed peak used | Minimum reported free |
|---|---:|---:|
| Rank 0 | 141,941 MiB (138.61 GiB) | 1,216 MiB |
| Rank 1 | 141,849 MiB (138.52 GiB) | 1,308 MiB |

These are sampled observations, not instantaneous upper bounds. NVIDIA reported 143,771 MiB total per card; CUDA reported 150,110,273,536 usable bytes (143,156.3125 MiB), so nominal total minus used does not equal allocatable headroom. The pool itself accounts for 898 bytes per token per GPU, or 31.632 GiB per GPU at this size.

At startup, after the maximum-shape and full graph-cache checks, each rank had approximately 132.746 GiB allocated in PyTorch, 134.287 GiB reserved and 2.652 GiB device-free memory. A separate startup probe retained an additional 1 GiB workspace and still had 1.650 GiB free. This is a tightly sized profile validated on these GPUs, not a guarantee for every future input mix or allocator state.

Evidence: `results-capacity/startup-rank{0,1}.json`, `results/api-memory.json`, `results-validation/validation.json`, `results-validation/gpu-memory.csv`, and the capacity arithmetic in `results/startup-review.json` of experiment 09.

## Host RAM

The two Engram files contain **203,073,077,576 bytes (189.127 GiB)** and use read-only shared mmap with startup prefault. Both ranks can map the same physical file-backed pages; summing their RSS double-counts those pages.

A later read-only procfs snapshot, at 2026-10-17 20:15 UTC, measured combined proportional set size (PSS) of **198.935 GiB** across the two live model processes. This is a point-in-time resident footprint after the benchmark campaign, not a peak, whole-machine RAM consumption, or a minimum-RAM recommendation. Neither process reported swap use. The tested machine had 1,133.48 GiB of OS-reported RAM.

Prefault does not mlock the Engram file pages. A machine under memory pressure can lose page residency and behave differently. A minimum-RAM configuration and disk-cold startup were not measured.

## Small fixed quality subsets

The same saved dataset ordering and selected indices were used for both quantized checkpoint versions. Temperature was 0, thinking disabled, concurrency 63; MMLU used a 16-token output cap and GSM8K a 2,048-token cap. MMLU selected 200 test items with seed 20261004; GSM8K selected 100 with seed 20261005. Subset digest: `09c2e6cc43e275af`.

| Fixed subset | Pre-1017 checkpoint | 1017 checkpoint | Change |
|---|---:|---:|---:|
| MMLU, 200 items | 175/200 (87.5%) | 171/200 (85.5%) | −2.0 percentage points |
| GSM8K, 100 items | 97/100 (97%) | 96/100 (96%) | −1.0 percentage point |

No API errors, unparsed answers or output-limit cuts were recorded. Saved-answer review found ordinary answer differences, so the score reductions are retained. These small fixed subsets use the repository's custom HTTP evaluator, not a full standard benchmark harness. They do not establish broad model quality, parity with an unquantized reference, or a causal effect of the H200 kernel changes. The public JSON includes indices and dataset-file hashes, but no questions or answers.

Evidence: experiment 09 `results-validation/quality.json` and `summary.json`, compared with experiment 08 `results-capacity/final/quality.json`.

## Historical H200 component experiments

These measurements motivated the implementation. The expert and Engram experiments used **pre-1017 weights**; the NCCL test used synthetic tensors and no model weights. They must not be advertised as 1017 end-to-end speedups.

The expert benchmark used actual layer-10, TP-rank-0 weights, synthetic activations/picks, cold L2 before each CUDA-graph replay and interleaved variants. The table shows the recorded median milliseconds:

| Rows / expert selection | Original grouped path | `cp` path | Fused path with rounding fix |
|---|---:|---:|---:|
| 1 / clustered | 0.0981 | **0.0834** | 0.1018 |
| 1 / random | 0.1003 | **0.0851** | 0.1016 |
| 6 / clustered | 0.1623 | 0.1358 | **0.1239** |
| 6 / random | 0.3246 | 0.2810 | **0.2598** |

The explicit rounded activation-scale multiply prevents an FMA contraction at the Hadamard boundary. Tested actual-weight outputs matched bit for bit; 10 sampled windows / 65 rows had no row-invariance differences. This supports the selected R1 `cp` / multirow fused dispatch, within those tests. An optional two-stage pipeline experiment showed roughly 0.5–1.2% R6 improvements on three tested layers, but **the deployed 1017 kernel retains three stages**.

The selected sm90 CB2 gate/up kernel's static disassembly changed from 75 to 73 plain FFMA instructions, 39 to 40 FMUL, and 446 to 448 FADD after the rounding fix. These are static instruction counts for one kernel, not runtime instruction counts or a speedup estimate. [Counts and original disassembly hashes](../benchmarks/h200/sm90-static-instructions.json).

The TP2 NCCL all-gather test used float32 tensors, ten warm-up calls, 50 calls per CUDA graph and seven timed graph replays. Reported times are medians per gather:

| Bytes sent per rank | Forced Simple, minimum 4 channels | NCCL automatic |
|---|---:|---:|
| 20,480 | 23.276 µs | **12.571 µs** |
| 122,880 | 25.283 µs | **18.675 µs** |
| 1,551,360 | 67.646 µs | 68.215 µs |
| 10,485,760 | 343.625 µs | 343.860 µs |

Engram CPU gather used fresh random rows, both Engram layers summed, TP-rank-0's 12 columns, and 80 samples per shape. All measured paths matched independent `pread` byte references:

| Read path | R1 median | R6 median | R512 median |
|---|---:|---:|---:|
| **mmap + prefault** | **4.385 µs** | **13.105 µs** | **1,236.830 µs** |
| pread, 1-thread setting | 27.515 µs | 261.397 µs | 19,447.823 µs |
| pread, 8-thread setting | 37.326 µs | 96.406 µs | 9,715.601 µs |
| pread, 128-thread setting | 309.017 µs | 563.975 µs | 6,345.635 µs |
| O_DIRECT AIO | 296.378 µs | 595.944 µs | 41,955.490 µs |
| mmap without prefault | 162.631 µs | 828.846 µs | 25,762.108 µs |

The files were already completely resident in the OS page cache. The 13.678 s prefault measurement populated process page tables; it was **not disk-cold loading**. No shared cache was forcibly evicted, and other model loading/warm-up work shared the host. The R512 AIO result is diagnostic, chunked to a 1,024-read ring; production prompt handling normally used pread. Thirty GPU staging/decoding cases also matched the independent pread reference exactly. Their cold-allocation wall times are omitted from the public speed table.

## Reproduce the protocols

Run these explicitly against your own idle deployment. They generate model output and the long test can take several minutes. Use the API base without `/v1`; an authenticated gateway can use `--key-file /path/to/key`. Results and generated fixtures should stay outside the source checkout.

```bash
export H200_BENCH_BASE=http://127.0.0.1:18891
export H200_BENCH_MODEL=DeepSeek-V4.1-Flash-1017-H200
export H200_BENCH_DIR=/tmp/h200-benchmark-runs
mkdir -p "$H200_BENCH_DIR"

# Single active stream, three warmed trials for each of the three workloads.
python3 tools/dsv41/kit_bench.py --base "$H200_BENCH_BASE" --model "$H200_BENCH_MODEL" \
  decode --set a --tokens 512 --reps 3 --out "$H200_BENCH_DIR/decode.json"

# Explicitly generate the deterministic synthetic near-1M fixture using /tokenize.
# This reconstructs the original RNG history and reports whether its digest matches.
python3 h200/prepare_validation.py --base "$H200_BENCH_BASE" --model "$H200_BENCH_MODEL" \
  --out "$H200_BENCH_DIR/fixture"

# Run on the inference host with nvidia-smi available: 63 solo references,
# a 63-stream burst, two near-1M inputs, cached repeats and Japanese fact checks.
python3 h200/validate_capacity.py --base "$H200_BENCH_BASE" --model "$H200_BENCH_MODEL" \
  --streams 63 --tokens 256 --stage all --fixture-dir "$H200_BENCH_DIR/fixture" \
  --out "$H200_BENCH_DIR/capacity"

# Supply your own MMLU-all and GSM8K test JSONL files in the expected schema/order.
# No evaluation dataset is downloaded or included by this repository.
python3 tools/dsv41/quality_eval.py --base "$H200_BENCH_BASE" --model "$H200_BENCH_MODEL" \
  --mmlu /path/to/mmlu_all_test.jsonl --gsm8k /path/to/gsm8k_test.jsonl \
  --mmlu-n 200 --gsm8k-n 100 --modes off --parallel 63 \
  --out "$H200_BENCH_DIR/quality.json"
```

The portable scripts passed Python compilation and `--help`, including inside the built image. Fixture generation was also rerun using only the existing API tokenizer: raw lengths 32,703 / 131,019 / 1,046,916 tokens reproduced the original near-1M SHA-256 `bde79cc15b8d7f07802fb84ed6e3cd0753e9be9d4d0fd94a25eff3eb79c82fae`. No expensive inference campaign was repeated during publication. The scripts preserve the recorded validation logic while replacing deployment-specific paths. Compare fixture and subset digests before interpreting a rerun as the same input set. The automatic Japanese checks cover factual fields and the total; deadline ordering requires review of the saved response, as in the original campaign.

The historical JSON also identifies each original component harness and source hash. Those diagnostic timings are preserved evidence; the portable package does not automatically rerun old-weight or pipeline-candidate experiments.
