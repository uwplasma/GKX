#!/usr/bin/env bash
# Re-run the Q28 d96 control and pr3-cm arms at the pinned SHA; a Linux version
# of plan/research/scripts/2026-09-19-pr3cm-src/run_arms.sh (single-threaded,
# so the matvec-equivalent counts and single-core times are load-insensitive).
# Usage (repository root): run_pr3.sh <outdir> <python>
set -u
OUT=$1; PY=$2
mkdir -p "$OUT"
D=plan/research/scripts/2026-09-19-pr3cm-src
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SI="--arm si --case d96 --krylov-dim 48 --restarts 2 --restart 600 --maxiter 600 --inner-rtol 1e-6 --preconditioner pr3-cm --block-solve block-thomas"
for entry in "A1_adaptive::--arm adaptive --case d96" "B1_pr3_bt::$SI" \
             "A2_adaptive::--arm adaptive --case d96" "B2_pr3_bt::$SI"; do
  label=${entry%%::*}; argv=${entry#*::}
  echo "$(date -u +%FT%TZ) start $label load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/supervisor.txt"
  # shellcheck disable=SC2086
  /usr/bin/time -v "$PY" "$D/measure.py" $argv --label "$label" > "$OUT/$label.txt" 2>&1
  echo "exit $?" >> "$OUT/$label.txt"
  echo "$(date -u +%FT%TZ) end $label load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/supervisor.txt"
done
