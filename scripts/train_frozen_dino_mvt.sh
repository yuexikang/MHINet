#!/usr/bin/env bash
# 新训练从原始预训练权重初始化；同目录自动恢复最新checkpoint。
# 冻结DINO/MVT，训练GHIM head、VGG、CGMDP和MHIR；D1不执行。
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MHINET_PYTHON="${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}"
MHINET_CONFIG="${MHINET_CONFIG:-$MHINET_ROOT/configs/train_frozen_dino_mvt_temporal4.json}"
MHINET_RUNTIME="${MHINET_RUNTIME:-$MHINET_ROOT/configs/runtime_paths.temporal4.server.json}"
MHINET_OUTPUT_DIR="${MHINET_OUTPUT_DIR:-$MHINET_ROOT/outputs/GHIM_joint_frozen_dino_mvt_temporal4_seed0}"
GPU_ID="${GPU_ID:-1}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'GPU_ID必须是单个物理卡编号' >&2; exit 2; }
for arg in "$@"; do
    case "$arg" in
        --runtime|--runtime=*|--config|--config=*|--output-dir|--output-dir=*)
            echo '请用MHINET_RUNTIME/MHINET_CONFIG/MHINET_OUTPUT_DIR环境变量指定路径，避免绕过数据检查' >&2
            exit 2 ;;
    esac
done
cd "$MHINET_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
command=("$MHINET_PYTHON" -u -m mhinet.cli train
    --runtime "$MHINET_RUNTIME" --config "$MHINET_CONFIG"
    --tiny-gate-artifact "$MHINET_ROOT/artifacts/p4_tiny_gate_mcnet_d2.json"
    --output-dir "$MHINET_OUTPUT_DIR" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_ID"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
"$MHINET_PYTHON" -m mhinet.dataio.check_temporal_ready --runtime "$MHINET_RUNTIME"
exec "${command[@]}"
