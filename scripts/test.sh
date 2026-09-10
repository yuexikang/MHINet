#!/usr/bin/env bash
# 真实影像精度评估，必须指定checkpoint；默认val，不使用保留的test集调参。
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MHINET_PYTHON="${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}"
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
    echo '用法: bash scripts/test.sh CHECKPOINT [--split val|test] [--max-pairs N] [--output-dir DIR]'
    echo '默认GPU_ID=1，全量val，输出七阶段指标及首个pair的7张图。单元测试：bash scripts/unit_tests.sh'
    exit 0
fi
MHINET_CHECKPOINT="$1"
shift
[[ -f "$MHINET_CHECKPOINT" ]] || { echo "checkpoint不存在：$MHINET_CHECKPOINT" >&2; exit 2; }
MHINET_CHECKPOINT="$(cd -- "$(dirname -- "$MHINET_CHECKPOINT")" && pwd)/$(basename -- "$MHINET_CHECKPOINT")"
GPU_ID="${GPU_ID:-1}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'GPU_ID必须是单个物理卡编号' >&2; exit 2; }
[[ -x "$MHINET_PYTHON" ]] || { echo "Python不可执行：$MHINET_PYTHON" >&2; exit 2; }
cd -- "$MHINET_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
command=("$MHINET_PYTHON" -u -m mhinet.cli evaluate
    --runtime "$MHINET_ROOT/configs/runtime_paths.server.json"
    --checkpoint "$MHINET_CHECKPOINT" --split val --visualization-pairs 1
    --output-dir "$MHINET_ROOT/outputs/evaluation/$(date -u +%Y%m%dT%H%M%SZ)" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
