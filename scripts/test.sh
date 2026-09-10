#!/usr/bin/env bash
# 工程单元测试入口：不访问真实影像、不启动正式训练、不运行 test 数据集评估。
set -euo pipefail

MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MHINET_PYTHON="${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}"
if [[ ! -x "$MHINET_PYTHON" ]]; then
    echo "找不到 Python：$MHINET_PYTHON；请设置 MHINET_PYTHON。" >&2
    exit 2
fi
cd -- "$MHINET_ROOT"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$MHINET_PYTHON" -m unittest discover -s tests -v "$@"
