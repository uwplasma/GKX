#!/bin/bash
# usage: run_ledgers.sh <tree> <outdir>
set -u
TREE=$1; OUT=$2; mkdir -p $OUT
PY=${PY:-python}
export PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu
cd $TREE
$PY -c "import gkx; print(gkx.__file__)" > $OUT/gkx_file.txt
for route in diagnostics runtime; do
  for grid in 32 64; do
    while [ "$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print int($1)}')" -gt 20 ]; do sleep 30; done
    if [ $grid = 32 ]; then EXTRA=""; else EXTRA="--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8"; fi
    echo "$(date) $route $grid" >> $OUT/progress.txt
    nice -n 10 $PY tools/profiling/profile_runtime_kernels.py nonlinear-step-hlo --route $route --methods rk3,rk4 $EXTRA --hlo-dir $OUT/hlo_${route}_${grid} --out $OUT/${route}_${grid}.json > $OUT/${route}_${grid}.log 2>&1
    echo "$(date) done rc=$?" >> $OUT/progress.txt
  done
done
echo ALLDONE >> $OUT/progress.txt
