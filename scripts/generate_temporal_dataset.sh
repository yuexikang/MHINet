#!/usr/bin/env bash
# 默认仅打印计划；显式 --generate 才生成；旧数据不覆盖。
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MHINET_PYTHON="${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}"
MHINET_DATA_OUTPUT="${MHINET_DATA_OUTPUT:-/home/disk1/Data/datasets/GoogleEarth_temporal4_v1}"
cd "$MHINET_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$MHINET_PYTHON" -u -m mhinet.dataio.generate_temporal --output-dir "$MHINET_DATA_OUTPUT" "$@"
