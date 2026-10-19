#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'HELP'
Usage: DATA_DIR=/absolute/data/path bash h200/run.sh

Run the built image in the foreground with two H200 GPUs. Does not build or
download anything. Stop with Ctrl-C; if either rank exits, the peer is stopped.

Environment:
  DATA_DIR          models/, cache/, results/ parent (default: ./h200-data)
  H200_IMAGE        built image name (default: deepseek-v41-h200:local)
  H200_GPU_IDS      two host GPU indices (default: 0,1)
  H200_API_PORT     loopback API port (default: 18891)
  H200_MASTER_PORT  local rank rendezvous port (default: 29571)
  H200_VISION       enable image input (default: 1; 0 for text-only)
  H200_VISION_MAX_IMAGES  images across full history (default: 0 = no count limit)
  H200_KERNEL_PROFILE  baseline (default) or round02 (measured H200 optimization)

Defaults: 1,048,576 tokens/request, 36,771,840 shared KV tokens, 63 slots.
The engine is loopback-only. Use a separately authenticated reverse proxy for
remote access; native API-key enforcement is not relied upon by this launcher.
HELP
}
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --inside-container) shift; inside=1 ;;
  "") inside=0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# != 0 ]]; then usage >&2; exit 2; fi

if [[ $inside == 0 ]]; then
  command -v docker >/dev/null || { printf '%s\n' 'Docker is required.' >&2; exit 1; }
  data_dir="${DATA_DIR:-$PWD/h200-data}"
  mkdir -p "$data_dir"
  data_dir="$(cd "$data_dir" && pwd)"
  gpu_ids="${H200_GPU_IDS:-0,1}"
  if [[ ! $gpu_ids =~ ^[0-9]+,[0-9]+$ || ${gpu_ids%,*} == "${gpu_ids#*,}" ]]; then
    printf '%s\n' 'H200_GPU_IDS must contain two distinct GPU indices, for example 0,1.' >&2
    exit 2
  fi
  image="${H200_IMAGE:-deepseek-v41-h200:local}"
  args=(run --rm --init --gpus "\"device=$gpu_ids\"" --ipc=host --network=host
        --ulimit memlock=-1 --volume "$data_dir:/data" --env DATA_DIR=/data)
  # Forward only documented tuning settings, never the host's complete environment.
  for name in H200_CONTEXT H200_PARALLEL H200_MTP_DRAFTS H200_VISION H200_API_PORT H200_MASTER_PORT H200_MODEL_NAME H200_KERNEL_PROFILE \
      TF_EXL3_DECODE OMP_NUM_THREADS MKL_NUM_THREADS MAX_JOBS TF_DS_RANK_CACHE_READERS \
      TF_DS_ENGRAM_MMAP TF_DS_ENGRAM_PREFAULT TF_DS_PREFILL_CHUNK TF_DS_WARM_LENGTHS \
      TF_DS_REPLAY TF_DS_CHUNK_LOG TF_DS_POOL_TOKENS TF_DS_DECODE_ROWS TF_DS_PARALLEL_DEPTH \
      TF_DS_KEEP_ENTRIES TF_DS_KEEP_MARKS TF_DS_SNAPSHOT_CPU TF_DS_WARM_ROWS \
      TF_DS_CAPACITY_CHECK TF_DS_ROUND_GRAPH_LIMIT TF_DS_CAPACITY_GRAPH_STRESS TF_DS_VISION_WARM \
      H200_VISION_MAX_IMAGES H200_VISION_IMAGE_TOKENS TF_VISION_MAX_TOTAL_ENCODED_BYTES \
      TF_VISION_MAX_TOTAL_PIXELS TF_REQUEST_BODY_LIMIT_BYTES; do
    if [[ -n ${!name:-} ]]; then args+=(--env "$name"); fi
  done
  exec docker "${args[@]}" "$image"
fi

source "$here/env.sh"
for required in "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json" \
    "$TF_DS_ENGRAM/model-00047-of-00048.safetensors" \
    "$TF_DS_ENGRAM/model-00048-of-00048.safetensors"; do
  if [[ ! -f $required ]]; then
    printf 'Required model file is missing: %s\nRun h200/download_models.py explicitly before starting.\n' "$required" >&2
    exit 1
  fi
done
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TF_DS_RANK_CACHE" \
  "$(dirname "$TF_DS_TOKEN_MAP")" "$TF_DS_CAPACITY_REPORT"
pids=()
cleanup() {
  trap - EXIT INT TERM
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for rank in 0 1; do
  bash "$here/serve-rank.sh" "$rank" &
  pids+=("$!")
done
# The pinned Linux image supplies Bash >=4.3 (wait -n); the host only runs Docker.
if wait -n "${pids[@]}"; then status=1; else status=$?; fi
printf 'A tensor-parallel rank exited (status %s); stopping the peer.\n' "$status" >&2
exit "$status"
