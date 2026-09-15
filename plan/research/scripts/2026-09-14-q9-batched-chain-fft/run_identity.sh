#!/bin/bash
# usage: run_identity.sh <name> <tree> ... runs rhs_identity f32 and x64 per tree, sequentially
set -u
S=${Q9_DIR:?set Q9_DIR to an evidence directory}
PY=${PY:-python}
mkdir -p $S/ident
while [ $# -ge 2 ]; do
  NAME=$1; TREE=$2; shift 2
  for prec in f32 x64; do
    while [ "$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print int($1)}')" -gt 20 ]; do sleep 30; done
    if [ $prec = x64 ]; then X="JAX_ENABLE_X64=true GKX_X64=1"; else X=""; fi
    echo "$(date) $NAME $prec" >> $S/ident/progress.txt
    (cd $TREE && env $X PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu nice -n 10 $PY $(dirname $0)/rhs_identity.py $S/ident/${NAME}_${prec}.npz > $S/ident/${NAME}_${prec}.log 2>&1)
    echo "$(date) rc=$?" >> $S/ident/progress.txt
  done
done
echo ALLDONE >> $S/ident/progress.txt
