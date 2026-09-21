#!/usr/bin/env bash
set -euo pipefail
cd /home/disk1/MHINet
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
# Registered 2000-step screening cap; original one-epoch cosine horizon unchanged.
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.pretraining.train \
  --config configs/shared_descriptor_stable_v2_tier1_adapt.json \
  --output "${MHINET_OUTPUT_DIR:-/home/disk1/MHINet/outputs/shared_stable_v2_tier1_adapt_seed0}" \
  --max-steps 2000 "$@"
