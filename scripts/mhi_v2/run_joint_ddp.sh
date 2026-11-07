#!/usr/bin/env bash
# Launch the joint 3-GPU run. GPU 0 stays idle: CUDA_VISIBLE_DEVICES=1,2,3.
# Usage:  bash scripts/mhi_v2/run_joint_ddp.sh [OUTPUT_DIR] [extra args...]
#   bash scripts/mhi_v2/run_joint_ddp.sh                       # fresh 20-epoch run
#   RESUME=1 bash scripts/mhi_v2/run_joint_ddp.sh              # continue latest.pt
#   bash scripts/mhi_v2/run_joint_ddp.sh outputs/xxx --limit-train 800 --epochs 1
set -euo pipefail

REPO=${REPO:-/home/disk1/MHINet-v2-mainline}
PY=${PY:-/root/miniconda3/envs/loma-repro/bin/python}
CONFIG=${CONFIG:-configs/mhi_v2/joint_rgb512_ddp20.json}

cd "$REPO"
OUT=${1:-outputs/mhi_v2_joint_ddp20_20261105}
shift || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
if [ -f "$OUT/pid.txt" ] && kill -0 "$(cat "$OUT/pid.txt")" 2>/dev/null; then
  echo "already running: pid $(cat "$OUT/pid.txt")" >&2
  exit 1
fi

ARGS=(--config "$CONFIG" --output "$OUT")
if [ "${RESUME:-0}" = "1" ]; then
  ARGS+=(--resume)
fi

setsid nohup "$PY" -m torch.distributed.run \
  --nproc_per_node="${NPROC:-3}" --master_port="${MASTER_PORT:-29517}" \
  --module mhinet.v2.pair_matching.joint_training \
  "${ARGS[@]}" "$@" >> "$OUT/run.log" 2>&1 &
PID=$!
echo "$PID" > "$OUT/pid.txt"
echo "launched pid=$PID output=$OUT log=$OUT/run.log"
