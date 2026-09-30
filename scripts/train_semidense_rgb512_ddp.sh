#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUTPUT_DIR [SOURCE_CHECKPOINT] [TRAIN_ARGS...] (auto-resumes latest.pt)" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
shift
SOURCE_ARGS=()
if [[ $# -gt 0 && "$1" != --* ]]; then
  SOURCE_ARGS=(--checkpoint "$1")
  shift
fi
RESUME_ARGS=()
CONFIG_ARGS=()
if [[ -f "$OUTPUT_DIR/latest.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/latest.pt")
else
  if [[ ${#SOURCE_ARGS[@]} -eq 0 ]]; then
    SOURCE_ARGS=(--checkpoint "$REPO_ROOT/outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt")
  fi
  CONFIG_ARGS=(--config "$REPO_ROOT/configs/semidense_rgb512_synthetic.json")
fi
if [[ -n "${TRAIN_CONFIG:-}" ]]; then CONFIG_ARGS=(--config "$TRAIN_CONFIG"); fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec /root/miniconda3/envs/loma-repro/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "$REPO_ROOT/scripts/train_semidense_next_tier.py" \
  "${CONFIG_ARGS[@]}" "${SOURCE_ARGS[@]}" \
  --output "$OUTPUT_DIR" --input-size 512 "${RESUME_ARGS[@]}" "$@"
