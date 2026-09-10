#!/usr/bin/env bash
# E00：原始预训练GHIM的H0基线；不训练、不加载E01权重。
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MHINET_PYTHON="${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}"
GPU_ID="${GPU_ID:-1}"
if [[ "${1:-}" == --help ]]; then
    echo '用法: GPU_ID=1 bash scripts/test_e00.sh [--max-pairs N] [--output-dir DIR]'
    echo '默认评估完整test集（1000对）；仅H0指标，不训练、不运行精修。'
    echo '权重来自configs/runtime_paths.server.json，不需要训练checkpoint。'
    echo 'DRY_RUN=1只打印命令。默认输出outputs/E00_h0_seed0，非空目录拒绝覆盖。'
    exit 0
fi
for argument in "$@"; do
    case "$argument" in
        --checkpoint|--checkpoint=*)
            echo 'E00使用登记的预训练GHIM；评估训练权重请用scripts/test.sh。' >&2
            exit 2 ;;
    esac
done
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'GPU_ID必须是单个物理卡编号' >&2; exit 2; }
[[ -x "$MHINET_PYTHON" ]] || { echo "Python不可执行：$MHINET_PYTHON" >&2; exit 2; }
cd -- "$MHINET_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
command=("$MHINET_PYTHON" -u -m mhinet.cli evaluate
    --runtime "$MHINET_ROOT/configs/runtime_paths.server.json"
    --split test --h0-only --visualization-pairs 0
    --output-dir "$MHINET_ROOT/outputs/E00_h0_seed0" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
