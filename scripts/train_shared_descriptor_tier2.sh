#!/usr/bin/env bash
set -euo pipefail
cd /home/disk1/MHINet
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.pretraining.train \
  --config configs/shared_descriptor_tier2.json \
  --output "${MHINET_OUTPUT_DIR:-/home/disk1/MHINet/outputs/shared_descriptor_frozen_tier2_seed0}" "$@"
