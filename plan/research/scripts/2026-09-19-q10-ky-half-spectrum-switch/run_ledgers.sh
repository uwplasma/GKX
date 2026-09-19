#!/bin/bash
# Q10 switch: the HLO ledger with the ky axis two-sided and half.
# usage: run_ledgers.sh <tree> <outdir>
#
# Both arms are the same tree: the layout is a grid argument, not a code
# difference, which is the point -- "before" and "after" here are one binary
# lowering the same deck onto Ny and onto Nyc = 1 + Ny//2 ky rows. The `full`
# arm must reproduce #248's committed ledger exactly; the `half` arm is what
# the switch buys.
#
# Op counts and byte totals do not depend on machine load, so no idle host is
# required and none is claimed; `nice -n 10` keeps a co-tenant's job unharmed.
set -u
TREE=$1; OUT=$2; mkdir -p "$OUT"
PY=${PY:-python}
export PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu
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
done
echo ALLDONE >> "$OUT/progress.txt"
