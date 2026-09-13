#!/usr/bin/env bash
# 仅冻结DINOv3；BS1×累积4，20k步，默认关闭相关性重计算。
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MHINET_CONFIG="${MHINET_CONFIG:-$MHINET_ROOT/configs/train_frozen_dino_temporal4_ebs4_20k.json}"
export MHINET_OUTPUT_DIR="${MHINET_OUTPUT_DIR:-$MHINET_ROOT/outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0}"
exec bash "$MHINET_ROOT/scripts/train_frozen_dino_mvt.sh" --no-correlation-checkpoint "$@"
