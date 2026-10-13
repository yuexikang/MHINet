#!/usr/bin/env bash
set -euo pipefail
cd /home/disk1/MHINet
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR
exec /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.dataio.generate_rgb512 "$@"
