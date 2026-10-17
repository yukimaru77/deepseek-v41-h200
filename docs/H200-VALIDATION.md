# Publication validation

Validation date: 2026-10-18 JST. The original running two-H200 model was left running throughout preparation of this repository.

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
