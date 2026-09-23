#!/usr/bin/env bash
# SOLVAX-DIRECT: growth-rate value+gradient, dense / direct / adaptive, n = Nl*Nm*Nz.
# The direct arm's warm shift is the dense arm's eigenvalue rounded to two
# decimals. Order: dense, direct, adaptive, direct (one fresh process each).
# Usage (repository root): run_gradient.sh <outdir> <python> [nz nl nm]
set -u
OUT=$1; PY=$2; NZ=${3:-96}; NL=${4:-4}; NM=${5:-8}
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
H=plan/research/2026-09-22-solvax-direct
tag=z${NZ}l${NL}m${NM}
row() {
  label=$1; shift
  echo "$(date -u +%FT%TZ) start $label load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/supervisor.txt"
  /usr/bin/time -v "$PY" $H/gradient.py --nz $NZ --nl $NL --nm $NM --out "$OUT/$label.json" "$@" > "$OUT/$label.txt" 2>&1
  echo "exit $?" >> "$OUT/$label.txt"
  echo "$(date -u +%FT%TZ) end $label load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/supervisor.txt"
}
row dense_$tag --arm dense
SHIFT=$("$PY" -c "import json;l=json.load(open('$OUT/dense_$tag.json'))['lambda'];print(complex(round(l[0],2),round(l[1],2)))")
row direct1_$tag --arm direct --shift "$SHIFT"
row adaptive_$tag --arm adaptive
row direct2_$tag --arm direct --shift "$SHIFT"
