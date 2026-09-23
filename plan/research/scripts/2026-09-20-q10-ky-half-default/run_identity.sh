#!/bin/bash
# Q10 default flip: the opt-out identity gate.
# usage: run_identity.sh <ref_tree> <new_tree> <outdir>
#
#   ref       origin/main's tree.  Two-sided ky; it has no other option.
#   new_full  this branch, with GridConfig.ky_layout pinned to "full".
#             Against `ref` this must be **bitwise**: the documented opt-out
#             has to reproduce a pre-flip run bit for bit, and "no diff in the
#             two-sided route" is a claim about the tree, not about the
#             executable.
#
# There is deliberately no half arm.  Q9's harness seeds its input with
# rng.standard_normal over the grid's own ky row count, so a half-axis run
# starts from a different -- and non-real -- random state, and nothing it
# returns can be compared element by element against `ref`.  An earlier
# version of this driver ran that arm and meant to "record it with its
# numbers"; there were no meaningful numbers to record.  Layout equivalence on
# real fields is gated elsewhere: netcdf_layout_ab.py + compare_nc.py over the
# whole published bundle, cyclone_golden_identity.py over the parity table's
# generator, eigen_branch.py over the eigen routes, and the layout unit tests.
#
# The layout is pinned through sitecustomize.py in this directory rather than
# by editing the harness: Q9's rhs_identity.py and gate_traj.py are reused
# verbatim, as #248, #254, #258 and #259 reuse them, so the identity numbers
# stay comparable across the whole campaign.
#
# The float32 arms pin --xla_cpu_multi_thread_eigen=false: without it the
# XLA:CPU FFT thunk is handed to a thread pool whose reduction order is not
# reproducible run to run, and #248 measured two runs of unmodified `main`
# differing by 1.4e-10 with no code change between them.
#
# No idle host is required and no wall-clock claim is made from these runs:
# every number they produce is a bitwise comparison or an operation count.
# `nice -n 10` keeps a co-tenant's job unharmed.
set -u
REF=$1; NEW=$2; OUT=$3
PY=${PY:-python}
FFT_PIN=${FFT_PIN:---xla_cpu_multi_thread_eigen=false}
HERE=$(cd "$(dirname "$0")" && pwd)
Q9=$(cd "$HERE/../2026-09-14-q9-batched-chain-fft" && pwd)
mkdir -p "$OUT"

run_arm() {
  NAME=$1; TREE=$2; LAYOUT=$3; PREC=$4; SCRIPT=$5
  if [ "$PREC" = x64 ]; then X="JAX_ENABLE_X64=true GKX_X64=1"; else X=""; fi
  echo "$(date) $NAME $PREC $SCRIPT layout=$LAYOUT" >> "$OUT/progress.txt"
  (cd "$TREE" && env $X Q10_KY_LAYOUT="$LAYOUT" \
    PYTHONPATH=$HERE:$TREE/src:$TREE JAX_PLATFORMS=cpu \
    XLA_FLAGS="$FFT_PIN" \
    nice -n 10 $PY "$Q9/$SCRIPT.py" "$OUT/${SCRIPT}_${NAME}_${PREC}.npz" \
    > "$OUT/${SCRIPT}_${NAME}_${PREC}.log" 2>&1)
  echo "$(date) rc=$?" >> "$OUT/progress.txt"
}

for prec in f32 x64; do
  for script in rhs_identity gate_traj; do
    run_arm ref "$REF" full "$prec" "$script"
    run_arm new_full "$NEW" full "$prec" "$script"
  done
done

for prec in f32 x64; do
  for script in rhs_identity gate_traj; do
    $PY "$Q9/compare_npz.py" "$OUT/${script}_ref_${prec}.npz" \
      "$OUT/${script}_new_full_${prec}.npz" "$OUT/cmp_${script}_new_full_${prec}.json" \
      > "$OUT/cmp_${script}_new_full_${prec}.txt" 2>&1
  done
done
echo ALLDONE >> "$OUT/progress.txt"
