#!/usr/bin/env bash
set -euo pipefail
MHINET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_ID="${GPU_ID:-1}"
export MHINET_RUNTIME="$MHINET_ROOT/configs/runtime_paths.quadrant.server.json"
export MHINET_CONFIG="$MHINET_ROOT/configs/train_tier1_b.json"
export MHINET_OUTPUT_DIR="${MHINET_OUTPUT_DIR:-$MHINET_ROOT/outputs/tier1_b_seed0}"
exec bash "$MHINET_ROOT/scripts/train_frozen_dino_mvt.sh" --no-correlation-checkpoint "$@"
