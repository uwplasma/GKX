#!/usr/bin/env bash
# Q28 arm runner: one fresh, cold, single-threaded process per arm, run
# serially from the repository root, under the same process conditions as Q7's
# run_production.sh and Q21's run_arms.sh so the three rows are comparable.
#
#   [SCRIPT=name.py] run_arms.sh OUTDIR "label::arguments" ["label::arguments" ...]
#
# SCRIPT defaults to measure.py; set it to diagnose.py for the budget-cap probe.
#
# Each entry writes OUTDIR/<label>.txt (with a trailing "exit N" line) and one
# line to OUTDIR/supervisor.txt recording start time, load average, nice level,
# exit status and end time. The load is recorded at every boundary because this
# host is shared: an arm that ran under a different load is not comparable, and
# the log entry says so rather than quoting its wall time.
set -u
D=plan/research/scripts/2026-09-19-pr3cm-src
PY=${PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}
SCRIPT=${SCRIPT:-measure.py}
CAP_S=${CAP_S:-3600}
OUT=$1
shift
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "supervisor pid $$ started $(date '+%F %T') cap ${CAP_S}s" >>"$OUT/supervisor.txt"
for entry in "$@"; do
  label=${entry%%::*}
  argv=${entry#*::}
  load=$(sysctl -n vm.loadavg)
  echo "$(date '+%F %T') start $label load $load" >>"$OUT/supervisor.txt"
  # shellcheck disable=SC2086
  /usr/bin/time -l perl -e 'alarm shift; exec @ARGV' "$CAP_S" \
    "$PY" "$D/$SCRIPT" $argv --label "$label" >"$OUT/$label.txt" 2>&1
  status=$?
  echo "exit $status" >>"$OUT/$label.txt"
  echo "$(date '+%F %T') end $label exit $status load $(sysctl -n vm.loadavg)" \
    >>"$OUT/supervisor.txt"
done
echo "supervisor pid $$ finished $(date '+%F %T')" >>"$OUT/supervisor.txt"
