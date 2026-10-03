#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/outputs/semidense_jl1flight_512_lr8e6_5ep_seed42"
ARGS=()
if [[ -f "$OUTPUT_DIR/latest.pt" ]]; then
  ARGS=(--resume "$OUTPUT_DIR/latest.pt")
else
  ARGS=(--config "$REPO_ROOT/configs/semidense_jl1flight_512_finetune_5epoch.json"
    --checkpoint "$REPO_ROOT/outputs/semidense_rgb512_synth_seed42_5ep/latest.pt")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /root/miniconda3/envs/loma-repro/bin/python -u \
  "$REPO_ROOT/scripts/train_semidense_next_tier.py" \
  --output "$OUTPUT_DIR" --input-size 512 "${ARGS[@]}" "$@"
