#!/bin/bash
# Q10 switch: identity A/B between origin/main and the branch.
# usage: run_identity.sh <ref_tree> <new_tree> <outdir>
#
# The branch's default is the two-sided ky layout, so every gate below must be
# *bitwise*: the half layout is reachable but nothing selects it, and the one
# rule that changed for both layouts -- the flux weight of an even grid's
# Nyquist row -- is masked away by two-thirds dealiasing in every case here.
#
# Q9's harness scripts (rhs_identity.py, gate_traj.py, compare_npz.py) are
# reused verbatim; this driver only sequences them.
#
# The float32 arms pin --xla_cpu_multi_thread_eigen=false: without it the
# XLA:CPU FFT thunk is handed to a thread pool whose reduction order is not
# reproducible run to run, and #248 measured two runs of unmodified `main`
# differing by 1.4e-10 with no code change between them.
#
# Unlike the 2026-09-18 stage-1 driver this one does NOT wait for an idle
# host. Every number it produces is a bitwise comparison or an operation
# count, neither of which depends on machine load; only `nice -n 10` is kept,
# so a co-tenant's job is not slowed down. No wall-clock claim is made from
# these runs.
set -u
REF=$1; NEW=$2; OUT=$3
PY=${PY:-python}
FFT_PIN=${FFT_PIN:---xla_cpu_multi_thread_eigen=false}
Q9=$(cd "$(dirname "$0")/../2026-09-14-q9-batched-chain-fft" && pwd)
mkdir -p "$OUT"

for arm in ref:$REF new:$NEW; do
  NAME=${arm%%:*}; TREE=${arm#*:}
  for prec in f32 x64; do
    if [ $prec = x64 ]; then X="JAX_ENABLE_X64=true GKX_X64=1"; else X=""; fi
    for script in rhs_identity gate_traj; do
      echo "$(date) $NAME $prec $script" >> "$OUT/progress.txt"
      (cd "$TREE" && env $X PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu \
        XLA_FLAGS="$FFT_PIN" \
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
echo ALLDONE >> "$OUT/progress.txt"
