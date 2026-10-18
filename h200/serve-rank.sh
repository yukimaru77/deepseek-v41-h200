#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' 'Usage: serve-rank.sh 0|1 (inside the two-GPU container; normally started by run.sh)'
  exit 0
fi
if [[ $# != 1 || ( "$1" != 0 && "$1" != 1 ) ]]; then
  printf '%s\n' 'Expected exactly one rank: 0 or 1.' >&2
  exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
rank=$1
vision_args=()
case "$H200_VISION" in
  1) vision_args=(--vision --vision-max-images "$H200_VISION_MAX_IMAGES"
                 --vision-image-tokens "$H200_VISION_IMAGE_TOKENS") ;;
  0) ;;
  *) printf '%s\n' 'H200_VISION must be 0 or 1.' >&2; exit 2 ;;
esac
# Docker exposes exactly the selected two host GPUs; CUDA numbers them 0 and 1.
export CUDA_VISIBLE_DEVICES="$rank"
exec python3 -m tensorfold serve "$MODEL_DIR" \
  --backend cuda --tp 2 --rank "$rank" \
  --master 127.0.0.1 --master-port "$H200_MASTER_PORT" \
  --host 127.0.0.1 --port "$H200_API_PORT" \
  --context "$H200_CONTEXT" --parallel "$H200_PARALLEL" \
  --mtp-drafts "$H200_MTP_DRAFTS" "${vision_args[@]}" --name "$H200_MODEL_NAME" --no-update-check
