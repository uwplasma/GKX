#!/usr/bin/env bash
# Q20 gyaradax lane (office GPU0 only while idle). Usage: run_gyaradax_queue.sh <queue-file>
# Queue lines: "<ky_gx> <rung> <timeout_s>". Before every run the lane requires that
# nvidia-smi --query-compute-apps lists no process on GPU0 (UUID c64146c9...); if any other
# process is there it stops (never shares or waits on a GPU another session uses).
set -u
Q=$1
ROOT=$(cd "$(dirname "$0")" && pwd)
GPU_UUID=GPU-c64146c9-227c-496f-974a-d1a66ec55673
PY=/home/rjorge/venvs/gyaradax/bin/python
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
SUP=$ROOT/logs/supervisor_gyaradax.txt
mkdir -p "$ROOT/logs" "$ROOT/gyaradax"
echo "lane=gyaradax pid=$$ start=$(date -Is) gpu=$GPU_UUID queue=$Q" >> "$SUP"
while read -r ky rung tcap; do
  [ -z "${ky:-}" ] && continue
  case "$ky" in \#*) continue ;; esac
  out=$ROOT/gyaradax/S_ky${ky}_${rung}.txt
  if [ -f "$out" ] && grep -q '^RESULT' "$out"; then continue; fi
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$GPU_UUID" | grep -vc "^$" || true)
  if [ "$others" != "0" ]; then
    echo "STOP gpu0 busy before ky=$ky rung=$rung: $(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | tr '\n' ';')" >> "$SUP"
    break
  fi
  t0=$(date +%s.%N)
  echo "gyaradax ky=$ky rung=$rung start=$(date -Is)" >> "$SUP"
  cd "$ROOT/gyaradax"
  timeout --signal=TERM --kill-after=20s "${tcap}s" /usr/bin/time -v \
    "$PY" -u "$ROOT/gyaradax_salpha.py" --ky-gx "$ky" --rung "$rung" > "$out" 2> "${out%.txt}.time.txt" < /dev/null
  rc=$?
  wall=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $t0}")
  echo "gyaradax ky=$ky rung=$rung rc=$rc wall=$wall end=$(date -Is)" >> "$SUP"
done < "$Q"
echo "lane=gyaradax ALL DONE $(date -Is)" >> "$SUP"
