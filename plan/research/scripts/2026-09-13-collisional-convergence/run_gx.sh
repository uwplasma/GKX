#!/usr/bin/env bash
# Sequential GX supervisor for Q8 (office GPU), run only after the GKX runs.
# Usage: run_gx.sh RUN_DIR GPU_INDEX KEY [KEY ...]
# Each KEY has RUN_DIR/gx/KEY.in. The repaired GX binary is used in place and
# its SHA-256 is checked first. Before each run the chosen GPU must have no
# compute process (never share). One run per key under a 2700 s cap; the log
# is scanned every 10 s for nan/inf tokens (TERM, then KILL after 10 s); device
# memory of the gx process is sampled every 10 s. After each run gamma/omega
# are extracted with gx_fit.py and the restart/big files are deleted. Stop on
# the first nonzero exit, nonfinite token or nonfinite fit. No follow-on runs.
set -u
RUN_DIR=$1
GPU_INDEX=$2
shift 2
GXDIR=/home/rjorge/gkx-nl24-discriminator-20260912.vvmgDD
GXBIN=$GXDIR/gx
GXSHA=96a53403a803e40fe3f9f6d1734779158d8be84d22e13155eb952a9035d70536
PY=/home/rjorge/venvs/gkx-nl/bin/python
FIT=$RUN_DIR/src_stage/plan/research/scripts/2026-09-13-collisional-convergence/gx_fit.py
STATUS="$RUN_DIR/logs/gx_supervisor.txt"
cd "$RUN_DIR/gx" || exit 2
echo "gx supervisor pid $$ host $(hostname) gpu $GPU_INDEX start $(date -Is) keys: $*" >> "$STATUS"
if [ "$(sha256sum "$GXBIN" | cut -d' ' -f1)" != "$GXSHA" ]; then
  echo "STOP gx binary hash mismatch" >> "$STATUS"
  exit 5
fi
for key in "$@"; do
  busy=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')
  if [ -n "${busy// /}" ]; then
    echo "STOP gpu $GPU_INDEX has compute processes ($busy) before $key $(date -Is)" >> "$STATUS"
    exit 4
  fi
  echo "$key start $(date -Is) input sha $(sha256sum "$key.in" | cut -d' ' -f1)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" setsid timeout --signal=TERM --kill-after=10s 2700s \
    /usr/bin/time -v -o "$RUN_DIR/logs/$key.time.txt" "$GXBIN" "$key.in" \
    > "$RUN_DIR/logs/$key.run.log" 2>&1 &
  child=$!
  : > "$RUN_DIR/logs/$key.gpumem.txt"
  reason=completed
  sleep 5
  echo "$key timeout pid $child gx pids $(pgrep -f -- "$GXBIN $key.in" | tr '\n' ' ')" >> "$STATUS"
  while kill -0 "$child" 2> /dev/null; do
    for pid in $(pgrep -f -- "$GXBIN $key.in"); do
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F', ' -v p="$pid" '$1 == p {print $2}' >> "$RUN_DIR/logs/$key.gpumem.txt"
    done
    if grep -Eiq '(^|[^a-z])(nan|inf|infinity)([^a-z]|$)' "$RUN_DIR/logs/$key.run.log"; then
      reason="nonfinite log token"
      kill -TERM -- "-$child" 2> /dev/null || kill -TERM "$child"
      sleep 10
      kill -KILL -- "-$child" 2> /dev/null
      break
    fi
    sleep 10
  done
  wait "$child"
  rc=$?
  echo "$key exit $rc reason $reason end $(date -Is)" >> "$STATUS"
  rm -f "$key.big.nc" "$key.restart.nc"
  if [ "$rc" -ne 0 ] || [ "$reason" != completed ]; then
    echo "STOP at $key" >> "$STATUS"
    exit 1
  fi
  if ! "$PY" "$FIT" "$key.out.nc" "$key" "$key.json" > "$RUN_DIR/logs/$key.fit.txt" 2>&1; then
    echo "STOP fit failed at $key" >> "$STATUS"
    exit 3
  fi
  if ! "$PY" -c "import json,math,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if math.isfinite(r['gamma_phi2_late']) and math.isfinite(r['omega_second_half_mean']) and r['nonfinite_phi2'] == 0 else 3)" "$key.json"; then
    echo "STOP nonfinite fit at $key" >> "$STATUS"
    exit 3
  fi
done
echo "DONE $(date -Is)" >> "$STATUS"
