#!/usr/bin/env bash
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MHINET_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.dataio.generate_test_tier3 "$@"
