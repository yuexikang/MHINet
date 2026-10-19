#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUTPUT_DIR [TRAIN_ARGS...] (auto-resumes latest.pt)" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
shift
SOURCE_ARGS=()
if [[ -f "$OUTPUT_DIR/latest.pt" ]]; then
  SOURCE_ARGS=(--resume "$OUTPUT_DIR/latest.pt")
else
  SOURCE_ARGS=(--config "$REPO_ROOT/configs/semidense_rgb512_h0_joint_5epoch.json"
    --checkpoint "$REPO_ROOT/weight/semidense_stable_v2_tier3_lr_e_seed0.pt")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec /root/miniconda3/envs/loma-repro/bin/python -u \
  "$REPO_ROOT/scripts/train_semidense_next_tier.py" \
  "${SOURCE_ARGS[@]}" --output "$OUTPUT_DIR" --input-size 512 --device cuda:0 "$@"
