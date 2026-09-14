#!/usr/bin/env bash
# Sequential supervisor for Q8 (office GPU). Adapted from the Q3 supervisor.
# Usage: run_convergence.sh RUN_DIR GPU_INDEX KEY [KEY ...]
# RUN_DIR holds src_stage/ (git archive of the pinned SHA including this
# directory). GPU_INDEX is the physical GPU chosen by the occupancy policy (no
# compute processes and <5% utilization on two consecutive polls). Before each
# run the chosen GPU must have no compute process (never share); one runner
# process per key under a 2700 s cap; stop on the first nonzero exit or
# nonfinite gamma/omega. No follow-on runs.
set -u
RUN_DIR=$1
GPU_INDEX=$2
shift 2
cd "$RUN_DIR/src_stage" || exit 2
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD/src:$PWD"
export GX_PARITY_REF_DIR=/home/rjorge/gkx-r0-rate-parity-20260905.GtHbRz/matched_refs
PY=/home/rjorge/venvs/gkx-nl/bin/python
MANIFEST=plan/research/scripts/2026-09-13-collisional-convergence/manifest.toml
STATUS="$RUN_DIR/logs/supervisor.txt"
echo "supervisor pid $$ host $(hostname) gpu $GPU_INDEX start $(date -Is) keys: $*" >> "$STATUS"
for key in "$@"; do
  busy=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')
  if [ -n "${busy// /}" ]; then
    echo "STOP gpu $GPU_INDEX has compute processes ($busy) before $key $(date -Is)" >> "$STATUS"
    exit 4
  fi
  JAX_PLATFORMS=cpu "$PY" plan/research/scripts/2026-09-13-collisional-convergence/print_resolved.py \
    "$MANIFEST" "$key" > "$RUN_DIR/logs/$key.resolved.txt" 2>&1
  echo "$key start $(date -Is)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" JAX_PLATFORMS=cuda \
    timeout --signal=TERM --kill-after=10s 2700s /usr/bin/time -v \
    "$PY" tools/comparison/build_gx_parity_matrix.py --manifest "$MANIFEST" \
    --cases "$key" --stem "$RUN_DIR/results/$key" \
    > "$RUN_DIR/logs/$key.stdout.txt" 2> "$RUN_DIR/logs/$key.stderr.txt" &
  child=$!
  sleep 5
  echo "$key timeout pid $child python pids $(pgrep -f -- "--cases $key --stem" | tr '\n' ' ')" >> "$STATUS"
  wait "$child"
  rc=$?
  echo "$key exit $rc end $(date -Is)" >> "$STATUS"
  if [ "$rc" -ne 0 ]; then
    echo "STOP nonzero exit at $key" >> "$STATUS"
    exit "$rc"
  fi
  if ! "$PY" -c "import json,math,sys; r=json.load(open(sys.argv[1]))['cases'][0]['rows'][0]; sys.exit(0 if math.isfinite(r['gamma_gkx']) and math.isfinite(r['omega_gkx']) else 3)" "$RUN_DIR/results/$key.json"; then
    echo "STOP nonfinite or unreadable result at $key" >> "$STATUS"
    exit 3
  fi
  rm -f "$RUN_DIR/results/$key.png"
done
echo "DONE $(date -Is)" >> "$STATUS"
