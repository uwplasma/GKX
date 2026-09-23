#!/usr/bin/env bash
# SOLVAX-DIRECT step 3 driver: one fresh single-threaded process per row.
# d96 alternates the direct arm with GKX's `adaptive` control (Q28's
# measure.py) A/B/A/B, then runs GKX's pr3-cm shift-invert route once; the
# larger cases run the direct/pr3-cm bench once each.
# Usage (repository root): run_bench.sh <outdir> <python> [cases...]
set -u
OUT=$1; PY=$2; shift 2
CASES=${*:-"d96 r48 r96 c48 prod"}
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
H=plan/research/2026-09-22-solvax-direct
Q=plan/research/scripts/2026-09-19-pr3cm-src
row() {
  label=$1; shift
  echo "$(date -u +%FT%TZ) start $label load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/supervisor.txt"
  /usr/bin/time -v "$PY" "$@" > "$OUT/$label.txt" 2>&1
  echo "exit $?" >> "$OUT/$label.txt"
  echo "$(date -u +%FT%TZ) end $label load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/supervisor.txt"
}
for c in $CASES; do
  if [ "$c" = d96 ]; then
    row A1_adaptive_d96 $Q/measure.py --arm adaptive --case d96 --label A1_adaptive_d96
    row B1_direct_d96 $H/bench.py --case d96 --out "$OUT/B1_direct_d96.json"
    row A2_adaptive_d96 $Q/measure.py --arm adaptive --case d96 --label A2_adaptive_d96
    row B2_direct_d96 $H/bench.py --case d96 --skip pr3 --mumps metis --superlu --out "$OUT/B2_direct_d96.json"
    # GKX's own shift-invert route with pr3-cm, PERF-LIT's settings (run_pr3.sh).
    row C1_pr3_route_d96 $Q/measure.py --arm si --case d96 --krylov-dim 48 --restarts 2 \
      --restart 600 --maxiter 600 --inner-rtol 1e-6 --preconditioner pr3-cm \
      --block-solve block-thomas --label C1_pr3_route_d96
  else
    # Warm shift near the ITG branch (d96's certified lambda is 0.1013-0.2455j).
    extra="--shift 0.1-0.25j --superlu COLAMD"
    [ "$c" = prod ] && extra="$extra --pr3-restart 300 --mumps metis scotch amd"
    [ "$c" = c48 ] && extra="$extra --pr3-restart 300"
    # shellcheck disable=SC2086
    row bench_$c $H/bench.py --case $c --memory-limit-mb 5000 $extra --out "$OUT/bench_$c.json"
  fi
done
