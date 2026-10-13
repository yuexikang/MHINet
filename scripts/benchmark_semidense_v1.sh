#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT [val|test] [32|1024]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${3:-val}"
CHUNK="${4:-32}"
case "$SPLIT" in
  val) MANIFEST=/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2/val/pairs.jsonl ;;
  test) MANIFEST=/home/disk1/Data/datasets/GoogleEarth_test_single_tier3_v1/test/pairs.jsonl ;;
  *) echo 'Split must be val or test' >&2; exit 2 ;;
esac
case "$CHUNK" in 32|1024) ;; *) echo 'Chunk must be 32 or 1024' >&2; exit 2 ;; esac
: "${GPU_ID:?Set GPU_ID to an available physical GPU}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "${MHINET_PYTHON:-/root/miniconda3/envs/loma-repro/bin/python}" \
  "$ROOT/scripts/evaluate_semidense_timed.py" \
  --checkpoint "$1" --output "$2" --manifest "$MANIFEST" \
  --tier 3 --inference-window-chunk "$CHUNK" --verify-first-pair
