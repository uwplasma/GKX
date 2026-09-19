#!/bin/bash
# Q10 stage-1 gates. usage: run_gates.sh <ref_tree> <new_tree> <outdir>
#
# The contract refactor must not move a number, so every gate below is an A/B
# between an unmodified origin/main tree and the branch tree, run in separate
# processes from the same interpreter.
#
#   ledgers      optimized-HLO op counts and bytes per RHS and RK step
#   identity     every RHS term, the total, the nonlinear RHS and the VJPs
#   trajectory   100-step trajectories across integrators and routes
#
# Q9's harness scripts are reused verbatim; this driver only sequences them.
set -u
REF=$1; NEW=$2; OUT=$3
PY=${PY:-python}
Q9=$(cd "$(dirname "$0")/../2026-09-14-q9-batched-chain-fft" && pwd)
mkdir -p "$OUT"

wait_for_load() {
  while [ "$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print int($1)}')" -gt 20 ]; do sleep 30; done
}

for arm in ref:$REF new:$NEW; do
  NAME=${arm%%:*}; TREE=${arm#*:}
  bash "$Q9/run_ledgers.sh" "$TREE" "$OUT/ledgers_$NAME"
  for prec in f32 x64; do
    if [ $prec = x64 ]; then X="JAX_ENABLE_X64=true GKX_X64=1"; else X=""; fi
    for script in rhs_identity gate_traj; do
      wait_for_load
      echo "$(date) $NAME $prec $script" >> "$OUT/progress.txt"
      (cd "$TREE" && env $X PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu \
        nice -n 10 $PY "$Q9/$script.py" "$OUT/${script}_${NAME}_${prec}.npz" \
        > "$OUT/${script}_${NAME}_${prec}.log" 2>&1)
      echo "$(date) rc=$?" >> "$OUT/progress.txt"
    done
  done
done

for prec in f32 x64; do
  for script in rhs_identity gate_traj; do
    $PY "$Q9/compare_npz.py" "$OUT/${script}_ref_${prec}.npz" \
      "$OUT/${script}_new_${prec}.npz" "$OUT/cmp_${script}_${prec}.json" \
      > "$OUT/cmp_${script}_${prec}.txt" 2>&1
  done
done
$PY "$Q9/ledger_table.py" "$OUT/ledgers_ref" "$OUT/ledgers_new" > "$OUT/ledger_table.txt" 2>&1
echo ALLDONE >> "$OUT/progress.txt"
