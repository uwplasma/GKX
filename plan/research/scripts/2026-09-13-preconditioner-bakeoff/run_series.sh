#!/usr/bin/env bash
# Q7 supervisor for the follow-up scripts: one fresh single-threaded process per entry, serial.
# Usage (repository root): run_series.sh TAG::SCRIPT ARGS... [TAG::SCRIPT ARGS...]
# Each entry is one quoted string "TAG::script.py --arg value ...". Wall cap per run: CAP_S (2400 s).
set -u
D=plan/research/scripts/2026-09-13-preconditioner-bakeoff
PY=${PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}
CAP_S=${CAP_S:-2400}
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

for spec in "$@"; do
  tag=${spec%%::*}
  cmd=${spec#*::}
  level=$(nice_level)
  echo "$(date '+%F %T') start $tag (nice $level, cap ${CAP_S}s, load $(sysctl -n vm.loadavg)) pid $$: $cmd"
  # shellcheck disable=SC2086
  /usr/bin/time -l nice -n "$level" perl -e 'alarm shift; exec @ARGV' "$CAP_S" \
    "$PY" $D/$cmd >"$D/$tag.txt" 2>&1
  status=$?
  echo "exit $status" >>"$D/$tag.txt"
  echo "$(date '+%F %T') end $tag exit $status load $(sysctl -n vm.loadavg)"
done
