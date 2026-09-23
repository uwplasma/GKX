#!/bin/bash
# #266 split: the default-layout identity gate.
# usage: run_identity_default.sh <ref_tree> <new_tree> <outdir>
#
#   ref   origin/main's tree (two-sided ky; it has no other option).
#   new   the split branch, run with NO layout pin at all: the harness takes
#         GridConfig's shipped default.  Against `ref` this must be bitwise
#         in float32 and x64 -- the split keeps "full" as the default, so a
#         deck that names no ky_layout must run exactly as on main.
#
# This complements ../2026-09-20-q10-ky-half-default/run_identity.sh, which
# pins "full" through sitecustomize.py; that gate proves the opt-out is exact,
# this one proves the default is.  Q9's rhs_identity.py and gate_traj.py are
# reused verbatim so the numbers stay comparable with #248/#254/#258/#259/#266.
# The float32 arms pin --xla_cpu_multi_thread_eigen=false for a reproducible
# CPU FFT reduction order (#248).  No timing claim is made from these runs.
set -u
REF=$1; NEW=$2; OUT=$3
PY=${PY:-python}
FFT_PIN=${FFT_PIN:---xla_cpu_multi_thread_eigen=false}
HERE=$(cd "$(dirname "$0")" && pwd)
Q9=$(cd "$HERE/../2026-09-14-q9-batched-chain-fft" && pwd)
mkdir -p "$OUT"

run_arm() {
  NAME=$1; TREE=$2; PREC=$3; SCRIPT=$4
  if [ "$PREC" = x64 ]; then X="JAX_ENABLE_X64=true GKX_X64=1"; else X="JAX_ENABLE_X64=false"; fi
  echo "$(date) $NAME $PREC $SCRIPT" >> "$OUT/progress.txt"
  (cd "$TREE" && env -u Q10_KY_LAYOUT $X \
    PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu \
    XLA_FLAGS="$FFT_PIN" \
    nice -n 10 $PY "$Q9/$SCRIPT.py" "$OUT/${SCRIPT}_${NAME}_${PREC}.npz" \
    > "$OUT/${SCRIPT}_${NAME}_${PREC}.log" 2>&1)
  echo "$(date) rc=$?" >> "$OUT/progress.txt"
}

for prec in f32 x64; do
  for script in rhs_identity gate_traj; do
    run_arm ref "$REF" "$prec" "$script"
    run_arm new_default "$NEW" "$prec" "$script"
  done
done

for prec in f32 x64; do
  for script in rhs_identity gate_traj; do
    $PY "$Q9/compare_npz.py" "$OUT/${script}_ref_${prec}.npz" \
      "$OUT/${script}_new_default_${prec}.npz" \
      "$OUT/cmp_${script}_new_default_${prec}.json" \
      > "$OUT/cmp_${script}_new_default_${prec}.txt" 2>&1
  done
done
echo ALLDONE >> "$OUT/progress.txt"
