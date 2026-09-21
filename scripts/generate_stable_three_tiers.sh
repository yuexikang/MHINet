#!/usr/bin/env bash
set -euo pipefail
cd /home/disk1/MHINet
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.dataio.generate_temporal \
  --three-tiers --stable-geometry \
  --output-dir /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2 "$@"
