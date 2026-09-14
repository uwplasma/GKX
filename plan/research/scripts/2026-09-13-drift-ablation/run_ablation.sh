#!/usr/bin/env bash
# Sequential supervisor for the Q3 drift ablation (office GPU).
# Usage: run_ablation.sh RUN_DIR GPU_INDEX KEY [KEY ...]
# RUN_DIR holds src_stage/ (git archive of the pinned SHA plus this directory
# copied to the same relative path). GPU_INDEX is the physical GPU chosen by the
# occupancy policy (no compute processes and <5% utilization on two consecutive
# polls). One runner process per key, 1800 s cap, stop on the first nonzero
# exit or nonfinite gamma/omega. No follow-on runs.
set -u
RUN_DIR=$1
GPU_INDEX=$2
shift 2
cd "$RUN_DIR/src_stage" || exit 2
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD/src:$PWD"
export GX_PARITY_REF_DIR=/home/rjorge/gkx-r0-rate-parity-20260905.GtHbRz/matched_refs
PY=/home/rjorge/venvs/gkx-nl/bin/python
MANIFEST=plan/research/scripts/2026-09-13-drift-ablation/manifest.toml
STATUS="$RUN_DIR/logs/supervisor.txt"
echo "supervisor pid $$ host $(hostname) gpu $GPU_INDEX start $(date -Is) keys: $*" >> "$STATUS"
for key in "$@"; do
  echo "$key start $(date -Is)" >> "$STATUS"
  timeout --signal=TERM --kill-after=10s 1800s /usr/bin/time -v \
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
done
echo "DONE $(date -Is)" >> "$STATUS"
