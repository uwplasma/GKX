#!/bin/bash
# Q10 default flip: the corrected HLO ledger, two-sided against half.
# usage: run_ledgers.sh <tree> <outdir>
#
# Both arms are the same tree and both pass --ky-layout explicitly, which is
# what keeps this archive comparable with #248's, #258's and #259's across the
# 2.2 default change: an unflagged run now ledgers the half layout, because
# that is what a run of the deck compiles.
#
# The numbers to read are `materialized_bytes` and the `temp_size_in_bytes`
# of `memory_analysis`, not `bytes_written` (#259): the latter counts every
# concatenate and copy in the module text, and XLA:CPU emits an instruction
# inside a fusion body as index arithmetic on the fused loop without
# allocating for it. `bytes_written` is still recorded, unchanged, so the
# committed archive stays readable.
#
# GKX_JAX_CACHE=0 so each arm compiles cold and the ledger describes the
# module rather than a cache hit. Op counts and byte totals do not depend on
# machine load, so no idle host is required and none is claimed; `nice -n 10`
# keeps a co-tenant's job unharmed.
set -u
TREE=$1; OUT=$2; mkdir -p "$OUT"
PY=${PY:-python}
export PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu GKX_JAX_CACHE=0
cd "$TREE"
$PY -c "import gkx; print(gkx.__file__)" > "$OUT/gkx_file.txt"
for layout in full half; do
  for route in diagnostics runtime; do
    for grid in 32 64; do
      if [ "$grid" = 32 ]; then EXTRA=""; else EXTRA="--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8"; fi
      echo "$(date) $layout $route $grid" >> "$OUT/progress.txt"
      nice -n 10 $PY tools/profiling/profile_runtime_kernels.py nonlinear-step-hlo \
        --route "$route" --methods rk3,rk4 --ky-layout "$layout" $EXTRA \
        --out "$OUT/${layout}_${route}_${grid}.json" \
        > "$OUT/${layout}_${route}_${grid}.log" 2>&1
      echo "$(date) rc=$?" >> "$OUT/progress.txt"
    done
  done
done
echo ALLDONE >> "$OUT/progress.txt"
