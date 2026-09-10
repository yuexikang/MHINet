#!/usr/bin/env bash
# 正式 E01 入口：默认物理卡1；不会自动运行 E00 或覆盖已有输出。
set -euo pipefail

MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MHINET_PYTHON="${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}"
GPU_ID="${GPU_ID:-1}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo 'GPU_ID 必须是单个非负整数（物理 GPU 编号）。' >&2
    exit 2
fi
if [[ ! -x "$MHINET_PYTHON" ]]; then
    echo "找不到 Python：$MHINET_PYTHON；请设置 MHINET_PYTHON。" >&2
    exit 2
fi
cd -- "$MHINET_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

command=("$MHINET_PYTHON" -u -m mhinet.cli train
    --runtime "$MHINET_ROOT/configs/runtime_paths.server.json"
    --config "$MHINET_ROOT/configs/e01_heads_v1.2.json"
    --tiny-gate-artifact "$MHINET_ROOT/artifacts/p4_tiny_gate_mcnet_d2.json"
    --output-dir "$MHINET_ROOT/outputs/E01_heads_seed0"
    "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
