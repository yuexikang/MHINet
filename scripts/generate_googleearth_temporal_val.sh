#!/usr/bin/env bash
set -euo pipefail

cd /home/disk1/MHINet
export PYTHONPATH=/home/disk1/MHINet

python -m mhinet.dataio.generate_googleearth_temporal_val \
  --input-dir /home/disk1/Data/datasets/GoogleEarth/evaluation_data \
  --output-dir /home/disk1/Data/datasets/Test_Data/GoogleEarth \
  --allow-empty-output-dir \
  "$@"
