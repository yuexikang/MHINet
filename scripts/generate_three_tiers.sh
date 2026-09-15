#!/usr/bin/env bash
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$MHINET_ROOT"
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.dataio.generate_temporal \
    --three-tiers --output-dir /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1 "$@"
