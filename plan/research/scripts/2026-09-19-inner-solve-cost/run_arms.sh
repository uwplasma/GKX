#!/usr/bin/env bash
# Q21 arm runner: one fresh, cold, single-threaded process per arm, run serially
# from the repository root, under the same process conditions as Q7's
# run_production.sh so that the control and the arms are comparable.
#
#   [SCRIPT=name.py] run_arms.sh OUTDIR "label::arguments" ["label::arguments" ...]
#
# SCRIPT defaults to q21.py; set it to blockthomas.py for the apply-cost probe.
#
# Each entry writes OUTDIR/<label>.txt (with a trailing "exit N" line) and one
# line to OUTDIR/supervisor.txt recording start time, load average, nice level,
# exit status and end time.
set -u
D=plan/research/scripts/2026-09-19-inner-solve-cost
PY=${PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}
SCRIPT=${SCRIPT:-q21.py}
CAP_S=${CAP_S:-2700}
OUT=$1
shift
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

nice_level() {
  for _ in $(seq 1 30); do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$load" 'BEGIN { exit !(l < 20) }'; then echo 10; return; fi
    sleep 30
  done
  echo 19
}

echo "supervisor pid $$ started $(date '+%F %T') cap ${CAP_S}s" >>"$OUT/supervisor.txt"
for entry in "$@"; do
  label=${entry%%::*}
  argv=${entry#*::}
  level=$(nice_level)
  load=$(sysctl -n vm.loadavg)
  echo "$(date '+%F %T') start $label nice $level load $load" >>"$OUT/supervisor.txt"
  # shellcheck disable=SC2086
  /usr/bin/time -l nice -n "$level" perl -e 'alarm shift; exec @ARGV' "$CAP_S" \
    "$PY" "$D/$SCRIPT" $argv --label "$label" >"$OUT/$label.txt" 2>&1
  status=$?
  echo "exit $status" >>"$OUT/$label.txt"
  echo "$(date '+%F %T') end $label exit $status load $(sysctl -n vm.loadavg)" \
    >>"$OUT/supervisor.txt"
done
echo "supervisor pid $$ finished $(date '+%F %T')" >>"$OUT/supervisor.txt"
