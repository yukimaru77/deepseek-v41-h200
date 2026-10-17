# H200 fork: provenance and modifications

This distribution is based on [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/8e0966e2b09250abe2f6d9cc51edc20630e6eb1a), commit `8e0966e2b09250abe2f6d9cc51edc20630e6eb1a`, the Python DeepSeek V4.1 implementation. Its upstream is [ashhart/TensorFold](https://github.com/ashhart/TensorFold). TensorFold contributors and Bertholomus AI retain their existing attribution in [NOTICE](../NOTICE). This is not the later Zig implementation.

The root [Apache-2.0 LICENSE](../LICENSE), [LICENSES](../LICENSES), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), vendored license files, and original source notices are retained. Earlier TensorFold MIT notices and separately licensed components continue to apply; the root license does not replace them. See also the inherited [DeepSeek attribution](../tools/dsv41/ATTRIBUTION.md).

## H200 modifications

The following files were modified or added relative to the pinned Python base. Paths are relative to `src/tensorfold/`. This inventory describes the H200 implementation and API work, not ownership of the underlying upstream implementation.

| File | Change |
| --- | --- |
| `cuda/exl3/experts.py` | H200 expert dispatch and extension configuration. |
| `cuda/exl3/experts_grouped.cuh` | Explicit multiplication rounding in fused gate/up arithmetic, checked against the existing expert paths. |
| `cuda/chat_template.py` | DeepSeek V4.1-1017 fallback chat template using the official encoder. |
| `cuda/http.py` | Query-aware models routing for Messages clients. The Messages protocol implementation itself is inherited. |
| `cuda/health.py` | Serving health/compatibility reporting. |
| `families/deepseek_v41/cuda/engine.py` | H200 serving and capacity integration. |
| `families/deepseek_v41/cuda/engram_io.cpp` | Read-only host page-cache mapping and Engram row gathering. |
| `families/deepseek_v41/cuda/model.py` | H200 model/Engram integration. |
| `families/deepseek_v41/cuda/multi.py` | Shared KV capacity, admission, warmup, bounded snapshots and optional CPU snapshot storage. |
| `families/deepseek_v41/cuda/rounds.py` | H200 decode row/capacity support. |
| `families/deepseek_v41/cuda/capacity_check.py` | Capacity and numerical preflight checks. |
| `families/deepseek_v41/vendor/__init__.py` | Package for the official encoder. |
| `families/deepseek_v41/vendor/encoding.py` | Unmodified official DeepSeek V4.1-1017 encoder; see below. |

Deployment scripts, validation scripts, tests and H200 documentation supplied with this fork supplement this source inventory. Model weights, generated kernels, caches and runtime data are not part of the source distribution.

## Official 1017 encoder

`src/tensorfold/families/deepseek_v41/vendor/encoding.py` is copied from `encoding/encoding.py` in [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

It is MIT licensed, **Copyright (c) 2023 DeepSeek**. The complete license is retained beside it in `vendor/LICENSE`. Its SHA-256 is `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`, identical to the downloaded official file. This new vendored encoder is separate from the upstream independently implemented model math and from the older DeepSeek V4 encoder already documented in the inherited notices.

## Model sources (download separately)

| Asset | Repository | Pinned revision |
| --- | --- | --- |
| EXL3 2.9 bpw checkpoint | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | `afc7aa0aab98ee735c0f54e0a122019af1ac08f1` |
| Original model, Engram and encoder | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e1088993b9b82de08f59de5be5a34e42170` |

Both downloaded repositories include MIT `LICENSE` files with Copyright (c) 2023 DeepSeek. The quantization card identifies the official model as its base, ExLlamaV3 as the quantization tool, and MUL1/2.9 bpw trunk/6 bpw head settings. Attribution for the quantized release belongs to its named distributor as well as the original model authors. Follow the repository download instructions with these revisions; the Engram assets are `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`. Obtain configuration, tokenizer and license files alongside the weights. No weights are embedded here.

## Related implementations and recipes reviewed

| Reference | Inspected revision | Licensing evidence and relationship |
| --- | --- | --- |
| [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | `621ace2c2e021af1c284499f4722f65397d6f487` | Root Apache-2.0; `kit/KIT-LICENSES.md` separately inventories compiled binaries, model-derived assets and dependency licenses. Recipe/reference for the TensorFold path; its binary kit is not bundled. |
| [coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) | `1d8ac64af01c6fec87f39eb1dd526ff183615c73` | README License section and `CREDITS.md` identify integration and MiaAI-derived runtime code as **AGPL-3.0-only**, crediting MiaAI Lab / Wesley Young and contributors. A separately reviewed implementation, not the source base of this fork. |
| [shi3z/deepseekv4.1-A100-custom](https://github.com/shi3z/deepseekv4.1-A100-custom) | `580568edd6f84537ff042ded4ee842b892bf3ff4` | No project-level LICENSE found in this checkout; the license of a bundled agent skill does not license the project. A research reference, not a source dependency. |

Do not treat these repositories as interchangeable licensing sources. No code or binary assets from the latter two are intentionally bundled by this H200 port. Their source was reviewed during development; the inherited upstream clean-room statements describe the upstream authors' work, not a new clean-room claim for this fork. Any future copying requires separate provenance and license review.

PyTorch, Triton, CUDA/NCCL, compiler/toolchain and container images are external runtime/build dependencies, with their own licenses. This repository does not redistribute the NVIDIA container or recipe cubins. Existing vendored components (including Z Lab DFlash, DeepSeek V4 and MiaAI's MIT Flash Next vision integration) retain the notices already supplied by TensorFold. Names and links identify sources and do not imply endorsement.
