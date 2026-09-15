#!/usr/bin/env bash
# 原始GoogleEarth母图对：仅视觉测试，无真值精度指标。
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
    echo '用法: GPU_ID=0 bash scripts/test.sh CHECKPOINT [--max-pairs N] [--output-dir DIR]；默认原始500对，仅可视化'
    exit 0
fi
[[ -f "$1" ]] || { echo 'checkpoint不存在' >&2; exit 2; }
[[ "${GPU_ID:-1}" =~ ^[0-9]+$ ]] || exit 2
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$MHINET_ROOT"
command=("${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}" -u -m mhinet.engine.visual_test
    --output-dir "$MHINET_ROOT/outputs/visual_test_$(date -u +%Y%m%dT%H%M%SZ)" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
