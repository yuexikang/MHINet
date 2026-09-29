#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUTPUT_DIR [SOURCE_CHECKPOINT]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
SOURCE_CHECKPOINT="${2:-$REPO_ROOT/outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec /root/miniconda3/envs/loma-repro/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "$REPO_ROOT/scripts/train_semidense_next_tier.py" \
  --config "$REPO_ROOT/configs/semidense_rgb512_synthetic.json" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --output "$OUTPUT_DIR" --input-size 512
