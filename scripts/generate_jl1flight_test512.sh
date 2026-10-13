#!/usr/bin/env bash
set -euo pipefail

cd /home/disk1/MHINet
export PYTHONPATH=/home/disk1/MHINet

python -m mhinet.dataio.generate_jl1flight_test512 \
  --input-dir /home/disk1/Data/datasets/jl1flight/test/affine_pairs_test \
  --output-dir /home/disk1/Data/datasets/Test_Data/JL1Flight \
  "$@"
