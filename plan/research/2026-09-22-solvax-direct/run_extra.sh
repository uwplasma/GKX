#!/usr/bin/env bash
# SOLVAX-DIRECT follow-up rows: the d96 direct arm again (its first two rows
# predate the left-eigenvector shift fix), alternated with the adaptive
# control, and the adaptive control at r96 for the crossover.
# Usage (repository root): run_extra.sh <outdir> <python>
set -u
OUT=$1; PY=$2
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
D96="--case d96 --skip pr3 --mumps metis --superlu"
row B3_direct_d96 $H/bench.py $D96 --out "$OUT/B3_direct_d96.json"
row A3_adaptive_d96 $Q/measure.py --arm adaptive --case d96 --label A3_adaptive_d96
row B4_direct_d96 $H/bench.py $D96 --out "$OUT/B4_direct_d96.json"
row A4_adaptive_r96 $Q/measure.py --arm adaptive --case r96 --label A4_adaptive_r96
