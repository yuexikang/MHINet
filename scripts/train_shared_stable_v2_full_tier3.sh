#!/usr/bin/env bash
set -euo pipefail
cd /home/disk1/MHINet
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec /root/miniconda3/envs/loma-repro/bin/python -u scripts/run_shared_stable_full.py --tier 3 "$@"
