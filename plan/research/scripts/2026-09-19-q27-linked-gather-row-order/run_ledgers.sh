#!/bin/bash
# Q27: the full-vs-half HLO ledger, now split into materialized and fused-interior
# bytes, plus XLA's own buffer assignment for the same executables.
#
# usage: run_ledgers.sh <tree> <outdir>
#
# Both layout arms are the same binary: the ky layout is a grid argument, not a
# code difference. The `full` arm must reproduce the ledger #248 and #258
# committed, and `bytes_written` is unchanged by this row so it still does.
#
# The chain-free control is the same deck with `boundary = "periodic"`, kept
# here as a file (cyclone_nonlinear_periodic_control.toml) rather than a scratch
# copy, so the `config` field of its ledger JSON is the path it was read from.
#
# Op counts, byte totals and buffer assignment are properties of the compiled
# module and do not depend on machine load, so no idle host is required and
# none is claimed; `nice -n 10` keeps a co-tenant's job unharmed.
set -u
TREE=$1; OUT=$2; mkdir -p "$OUT"
PY=${PY:-python}
# Repo-relative: the tool cds into TREE, and an absolute path would be written
# into the `config` field of the ledger JSON this repository then commits.
CTRL=plan/research/scripts/2026-09-19-q27-linked-gather-row-order/cyclone_nonlinear_periodic_control.toml
# GKX_JAX_CACHE=0: the persistent compilation cache would let one arm reuse an
# executable another tree compiled, and every number here is a property of a
# compiled module. Each arm compiles cold.
export PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu GKX_JAX_CACHE=0
cd "$TREE"
$PY -c "import gkx; print(gkx.__file__)" > "$OUT/gkx_file.txt"
for layout in full half; do
  for route in diagnostics runtime; do
    for grid in 32 64; do
      if [ $grid = 32 ]; then EXTRA=""; else EXTRA="--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8"; fi
      echo "$(date) $layout $route $grid" >> "$OUT/progress.txt"
      nice -n 10 $PY tools/profiling/profile_runtime_kernels.py nonlinear-step-hlo \
        --route $route --methods rk3,rk4 --ky-layout $layout $EXTRA \
        --hlo-dir "$OUT/hlo_${layout}_${route}_${grid}" \
        --out "$OUT/${layout}_${route}_${grid}.json" \
        > "$OUT/${layout}_${route}_${grid}.log" 2>&1
      echo "$(date) rc=$?" >> "$OUT/progress.txt"
    done
  done
  echo "$(date) periodic_$layout" >> "$OUT/progress.txt"
  nice -n 10 $PY tools/profiling/profile_runtime_kernels.py nonlinear-step-hlo \
    --config "$CTRL" --route diagnostics --methods rk3,rk4 --ky-layout $layout \
    --out "$OUT/periodic_${layout}_diagnostics_32.json" \
    > "$OUT/periodic_${layout}_diagnostics_32.log" 2>&1
  echo "$(date) rc=$?" >> "$OUT/progress.txt"
done
echo ALLDONE >> "$OUT/progress.txt"
