#!/usr/bin/env bash
set -euo pipefail
# Validation-only default. For held-out testing pass an explicit test manifest.
cd /home/disk1/MHINet
export CUDA_VISIBLE_DEVICES="${GPU_ID:-3}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli evaluate-dense \
  --checkpoint "${MHINET_CHECKPOINT:-outputs/shared_descriptor_frozen_tier3_seed0/latest.pt}" \
  --manifest "${MHINET_MANIFEST:-/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2/val/pairs.jsonl}" \
  --tier "${TIER:-1}" --result-id "${RESULT_ID:-C3-P3}" \
  --output "${MHINET_OUTPUT_DIR:-outputs/dense_loma_stable_v2_tier1_p3}" "$@"
