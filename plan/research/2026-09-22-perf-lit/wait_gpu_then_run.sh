#!/usr/bin/env bash
# Wait until one office GPU has no compute process other than the persistent
# web-app workers (each well under 300 MiB) and is idle, then run PHASE=extra
# on that GPU only. Never shares or preempts a GPU in use.
# Usage (repository root): wait_gpu_then_run.sh <outdir> <python>
set -u
OUT=$1; PY=$2
mkdir -p "$OUT"
while true; do
  for idx in 0 1; do
    uuid=$(nvidia-smi -i "$idx" --query-gpu=uuid --format=csv,noheader)
    util=$(nvidia-smi -i "$idx" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
      | awk -F', ' -v u="$uuid" '$1==u && $2>300' | wc -l)
    if [ "$busy" -eq 0 ] && [ "$util" -lt 10 ]; then
      echo "$(date -u +%FT%TZ) gpu $idx free (util $util)" >> "$OUT/supervisor.txt"
      CUDA_VISIBLE_DEVICES=$idx PHASE=${GPU_PHASE:-extra} bash plan/research/2026-09-22-perf-lit/run_profile.sh "$OUT" "$PY" gpu
      exit 0
    fi
  done
  sleep 60
done
