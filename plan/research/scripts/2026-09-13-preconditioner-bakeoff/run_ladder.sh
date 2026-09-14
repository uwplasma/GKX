#!/usr/bin/env bash
# Q7 supervisor: one fresh single-threaded process per case, run serially on a shared Mac.
# Usage (from the repository root): plan/research/scripts/2026-09-13-preconditioner-bakeoff/run_ladder.sh CASE[:TAG[:EXTRA ARGS]] ...
set -u
D=plan/research/scripts/2026-09-13-preconditioner-bakeoff
PY=${PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

nice_level() {
  # nice 10 when the 1-min load is below 20; otherwise wait up to 15 min, then nice 19
  for _ in $(seq 1 30); do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$load" 'BEGIN { exit !(l < 20) }'; then echo 10; return; fi
    sleep 30
  done
  echo 19
}

for spec in "$@"; do
  IFS=: read -r case tag extra <<<"$spec"
  tag=${tag:-$case}
  level=$(nice_level)
  echo "$(date '+%F %T') start $tag (case $case, nice $level, load $(sysctl -n vm.loadavg)) pid $$"
  # shellcheck disable=SC2086
  /usr/bin/time -l nice -n "$level" perl -e 'alarm shift; exec @ARGV' 2400 \
    "$PY" "$D/bakeoff.py" --case "$case" --tag "$tag" $extra >"$D/$tag.txt" 2>&1
  status=$?
  echo "exit $status" >>"$D/$tag.txt"
  echo "$(date '+%F %T') end $tag exit $status load $(sysctl -n vm.loadavg)"
done
