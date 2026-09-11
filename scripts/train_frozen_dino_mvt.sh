#!/usr/bin/env bash
# 新训练从原始预训练权重初始化；同目录自动恢复最新checkpoint。
# 冻结DINO/MVT，训练GHIM head、VGG、CGMDP和MHIR；D1不执行。
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MHINET_CONFIG="${MHINET_CONFIG:-$MHINET_ROOT/configs/train_frozen_dino_mvt.json}"
export MHINET_OUTPUT_DIR="${MHINET_OUTPUT_DIR:-$MHINET_ROOT/outputs/GHIM_joint_frozen_dino_mvt_seed0}"
exec bash "$MHINET_ROOT/scripts/train_e01.sh" "$@"
