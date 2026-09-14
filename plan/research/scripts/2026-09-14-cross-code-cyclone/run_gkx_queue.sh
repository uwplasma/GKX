#!/usr/bin/env bash
# Q20 GKX lane (office CPU). Usage: run_gkx_queue.sh <queue-file> <lane> <cpu-list> <threads>
# Queue lines: "<S|M> <ky> <Nl> <Nm> <timeout_s>". Serial fresh processes pinned with taskset,
# nice 10, JAX on CPU in float64; output gkx/<geom>_ky<ky>_nl<Nl>_nm<Nm>.txt (last line RESULT json).
# A case whose .txt already holds a RESULT line is skipped.
set -u
Q=$1; LANE=$2; CPUS=$3; NT=$4
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/rjorge/venvs/gkx-nl/bin/python
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export PYTHONPATH=$ROOT/src:$ROOT
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT XLA_FLAGS=--xla_cpu_multi_thread_eigen=true
SUP=$ROOT/logs/supervisor_$LANE.txt
mkdir -p "$ROOT/logs" "$ROOT/gkx"
echo "lane=$LANE pid=$$ start=$(date -Is) cpus=$CPUS threads=$NT queue=$Q" >> "$SUP"
while read -r geom ky nl nm tcap; do
  [ -z "${geom:-}" ] && continue
  case "$geom" in \#*) continue ;; esac
  out=$ROOT/gkx/${geom}_ky${ky}_nl${nl}_nm${nm}.txt
  if [ -f "$out" ] && grep -q '^RESULT' "$out"; then continue; fi
  t0=$(date +%s.%N)
  echo "gkx $geom ky=$ky Nl=$nl Nm=$nm start=$(date -Is)" >> "$SUP"
  cd "$ROOT"
  timeout --signal=TERM --kill-after=20s "${tcap}s" nice -n 10 /usr/bin/time -v \
    taskset -c "$CPUS" "$PY" "$ROOT/gkx_eigen.py" --geometry "$geom" --ky "$ky" --Nl "$nl" --Nm "$nm" --repo "$ROOT" \
    > "$out" 2> "${out%.txt}.time.txt"
  rc=$?
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
  echo "gkx $geom ky=$ky Nl=$nl Nm=$nm rc=$rc wall=$wall end=$(date -Is)" >> "$SUP"
done < "$Q"
echo "lane=$LANE ALL DONE $(date -Is)" >> "$SUP"
