#!/usr/bin/env bash
# Bounded restart watchdog for the joint 3-GPU run.
# Foreground loop: launches the trainer, and if it dies before status=completed,
# relaunches from latest.pt after a pause (at most MAX_ATTEMPTS times).
# Usage: nohup setsid bash scripts/mhi_v2/watch_joint.sh [OUTPUT_DIR] > OUT/watch.log 2>&1 &
set -uo pipefail

REPO=${REPO:-/home/disk1/MHINet-v2-mainline}
PY=${PY:-/root/miniconda3/envs/loma-repro/bin/python}
CONFIG=${CONFIG:-configs/mhi_v2/joint_rgb512_ddp20.json}
OUT=${1:-outputs/mhi_v2_joint_ddp20_20261105}
shift || true
EXTRA=("$@")
MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}

cd "$REPO"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

mkdir -p "$OUT"
echo $$ > "$OUT/watchdog.pid"

state() {
  "$PY" - "$OUT/status.json" <<'EOF' 2>/dev/null || echo ""
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("state", ""))
except Exception:
    print("")
EOF
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -f "$OUT/latest.pt" ]; then
    ARGS=(--config "$CONFIG" --output "$OUT" --resume)
  else
    ARGS=(--config "$CONFIG" --output "$OUT")
  fi
  echo "[watchdog $(date '+%F %T')] attempt $attempt/${MAX_ATTEMPTS}"
  "$PY" -m torch.distributed.run --nproc_per_node="${NPROC:-3}" \
    --master_port="${MASTER_PORT:-29517}" --module mhinet.v2.pair_matching.joint_training \
    "${ARGS[@]}" "${EXTRA[@]}" >> "$OUT/run.log" 2>&1
  code=$?
  current=$(state)
  echo "[watchdog $(date '+%F %T')] exit=$code state=${current:-unknown}"
  if [ "$current" = "completed" ]; then
    echo "[watchdog] completed"
    exit 0
  fi
  if [ "$code" = "0" ] && [ "$current" = "stopped" ]; then
    exit 0
  fi
  sleep "${RESTART_WAIT:-60}"
done
echo "[watchdog] gave up after ${MAX_ATTEMPTS} attempts"
exit 1
