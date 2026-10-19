# H200 optimization: rounds 01–02

**The selected `cand-r3` implementation reaches 338.1 tok/s on the fixed five-prompt code suite, up from 285.8 tok/s (+18.3%) in a matched A/B.** The fastest individual code prompt is 369.5 tok/s. The 400 tok/s milestone is not reached.

These are measurements of the experimental service on 2026-10-19 UTC, before packaging. The original service was restored after validation; the optimized candidate was not deployed. This repository includes the selected source and an explicit `H200_KERNEL_PROFILE=round02` profile. The default launcher profile remains `baseline`.

## What changed

| Change | Mechanism |
|---|---|
| Expert and grouped linear Tensor Core operands | For supported windows of up to eight rows, put decoded weights in the MMA A fragment and input rows in B. This uses one `m16n8k16` instead of two per tile and halves accumulator storage. Larger windows retain the original path. |
| Grouped linear load/layout | Four-stage `cp.async` ring; split output columns across warps without changing the explicit K reduction. `auto4` selects a four-way split for small multi-row attention-input launches. |
| Same-host TP communication | A CUDA IPC receive ring and system-scope release/acquire flags implement small FP32 all-gathers; larger/other payloads use NCCL. Both ranks must have working CUDA peer access and run their collectives in the same order. |
| Host scheduling and launch dependencies | Defer Engram hashing until the first graph is queued; skip no-op mmap touch work. Launch `decode_prep` with a programmatic dependency, waiting before any input access. |
| Weight prefetch policy | Disable the inherited GB10-oriented L2 prefetch, which slowed this H200 workload. |
| Failed collective handling | After sampling's device-to-host completion, check lane health before committing or emitting any token, including the first token. A watchdog-aborted gather cannot emit undefined output. Both ranks still need a restart after a peer failure. |

Nsight Compute on real routed windows found that expert decode was constrained by integer instruction issue, rather than DRAM throughput. At R=6, gate/up issue activity rose from 71.0% to 75.6%; the ALU pipe rose from about 60% to 66%. The accepted expert kernel shortened representative layer calls by 6–9%. These component measurements are separate from the unprofiled API A/B below.

The source is the exact selected **cand-r3**, retaining its disabled APF branch and off-by-default diagnostic hooks. Optional cand-r4 cleanup and later research are not included. Changing the selected source or enabling another experimental switch creates a different configuration.

## Matched API results

Hardware: 2 × H200 NVL, 143,771 MiB each, same-host PCIe (`NODE`), no detected NVLink. Observed SM/memory clocks: 1785/3201 MHz; no clock changes. Toolchain: CUDA 13.0.88, PyTorch 2.11.0+cu130. The pinned model revisions are unchanged in [pins.json](../h200/pins.json).

Profile: vision enabled; context 1,048,576; 63 active slots; shared KV pool 36,771,840; 64 retained round graphs; Engram mmap/prefault; DSpark cap 5 and depth `5,5,3,3,2,1,1,1,0`; bounded replay prefill. These are **single-request short-input decode rates**, not aggregate concurrent throughput or cold 1M prefill rates.

| Fixed code prompt, 1024 output tokens | Original source | Round 01 | Selected round 02 |
|---|---:|---:|---:|
| Rust rate limiter | 285.769 | 313.708 | 338.071 |
| TypeScript pagination | 304.234 | 334.954 | 361.259 |
| Python CSV → SQLite | 276.522 | 304.180 | 327.745 |
| Go todo API | 310.838 | 342.823 | 369.510 |
| C++ SPSC queue | 224.255 | 245.327 | 264.374 |
| **Median, tok/s** | **285.769** | **313.708** | **338.071** |

The four H prose prompts have median 147.784 → 174.268 tok/s. Historical set A, at 512 output tokens, changes from 256.176 → 302.398 for code, 144.074 → 169.872 for prose, and 293.112 → 347.587 for structured output. The first-round candidate was remeasured in this same run; its 313.708 figure need not equal the earlier round's 315.074.

Protocol: fresh server startup for each pass, one full unrecorded prewarm pass, then three repetitions per prompt. Pass order: original, round 01, intermediate r2, selected r3, r3, r2, round 01, original. Temperature 0, thinking off; every measured response reached its exact 512/1024-token budget and finished with `length`; no foreign requests. Compute `(completion_tokens - 1) / (last content chunk time - first content chunk time)`, take the median of six rates per prompt, then the median of the five code or four prose prompt medians. The engine's decode-seconds rate agrees within 0.011%. Chunk counts are not token counts.

All 288 measured requests have matching per-prompt generated-text SHA-256 values and server token-ID digest prefixes across variants. Accepted tokens per round did not change. The H prompts were fixed before round 01 and subsequently used during tuning, so they are **not a fresh held-out evaluation**. A separate six-prompt sanity set was checked without replacing the headline suite.

[Per-request timing/hash extracts and summaries](../benchmarks/h200/optimization-round02.json) contain all 288 measured requests, pass order, clock snapshots, and hashes of the original measurement files. They omit generated texts, private paths, process environment and logs. [Source/build provenance](../benchmarks/h200/optimization-round02-provenance.json) records the frozen source identity and the corrected post-start binary inventory; the earlier inventory's two stale lazy-build hashes are retained as historical metadata.

## Reproduce the selected profile

Build/acquire weights as in [h200/README.md](../h200/README.md), then start on an idle two-H200 host:

```bash
DATA_DIR=/srv/deepseek-h200 H200_KERNEL_PROFILE=round02 bash h200/run.sh
```

The profile is [round02.env.sh](../h200/round02.env.sh). It leaves the model, API, image limits, context and capacity settings in `env.sh` unchanged. Extensions use separate names from the original kernels. After changing profiles, restart **both** ranks; the switches are read once at import/build/launch initialization.

From a second terminal on that host, after readiness:

```bash
mkdir -p benchmark-runs
python3 h200/bench_decode.py --label prewarm --reps 1 --out benchmark-runs/prewarm.json
python3 h200/bench_decode.py --label round02 --reps 3 --out benchmark-runs/round02-pass1.json
```

The script uses only the Python standard library and the loopback API. It runs the exact A/H prompts and records output, timing, health counters and GPU snapshots. `--log /path/to/rank0.log` additionally records token-ID digests from the rank-0 completion lines. Benchmark outputs are local runtime artifacts; review before sharing.

For a matched original-source comparison, build commit `a74d1026dc3110af47772f99b07e7b0889e45c16` in a separate checkout/image, use the same model/capacity settings, prewarm each fresh start, and alternate baseline/candidate/candidate/baseline with three repetitions per pass. Never run competing services on these GPUs. `H200_KERNEL_PROFILE=baseline` disables opt-in paths in the new source; it does **not** recreate the original compiled binaries. Do not compare a cold baseline against a warmed candidate.

## Validation and limits

The exact selected source/environment passed OpenAI protocol checks (8), vision cases (6), Anthropic cases (4), 8- and 16-stream output comparisons, 118,078- and 1,039,614-token recall/cache-reuse cases, six new greedy prompts, and two temperature-0.8 sampling cases. The 16-stream check exercises the non-TN fallback. The near-1M check matches the saved original-source output from round 01; it is not a same-hour original/candidate prefill A/B.

Expert component checks cover 216 real routed windows over three layers, with zero differing rows among 1,775 row-invariance checks. Grouped linear checks cover 222 outputs at each of six row counts. Fault injection demonstrated a wrong emitted suffix/first token before the abort fix, and only the correct prefix followed by an error and `[DONE]` after it.

Bit equality is established for these H200 builds, shapes and requests. Tensor Core internal accumulation order is not an architecture-independent bit-equality promise; other toolchains/GPUs require revalidation. MMLU/GSM8K were not rerun for these kernel changes. This work did not retrain or requantize the checkpoint. No general model-quality improvement or universal 338 tok/s rate is claimed.

Portable integration checks and their scope are in [H200-VALIDATION.md](H200-VALIDATION.md). The newly packaged tree has not been cold-started or timed on GPUs during integration, because the optimization campaign owns those GPUs.
