#!/bin/zsh
# Q30: the bitwise identity matrix, two trees, both precisions.
#
#   ./run_gate.sh <main-tree> <branch-tree> <out-dir>
#
# Each tree is a GKX checkout; one process per tree per case, so the two never
# share a jax import. Prints one line per case: BITWISE, or the per-output ulp
# distance of value / d/dtprim / d/dfprim.
set -e
MAIN=$1; BRANCH=$2; OUT=$3
PY=${PY:-python}
CYC=examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml
KBM=examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_short.toml
mkdir -p $OUT
cases=(
  "cyc_rk2|--deck $CYC --method rk2"
  "cyc_rk3|--deck $CYC --method rk3"
  "cyc_rk4|--deck $CYC --method rk4"
  "cyc_rk3_half|--deck $CYC --method rk3 --ky-mode half"
  "cyc_tail|--deck $CYC --method rk3 --window-steps 9 --tail-steps 4"
  "cyc_nofft|--deck $CYC --method rk3 --no-compressed-fft"
  "cyc_lagexact|--deck $CYC --method rk3 --laguerre-mode exact"
  "kbm_rk3|--deck $KBM --method rk3"
  "kbm_rk3_half|--deck $KBM --method rk3 --ky-mode half"
  "cyc_rk2_nockpt|--deck $CYC --method rk2 --no-checkpoint"
  "cyc_rk3_nockpt|--deck $CYC --method rk3 --no-checkpoint"
  "cyc_rk4_nockpt|--deck $CYC --method rk4 --no-checkpoint"
  "kbm_rk3_nockpt|--deck $KBM --method rk3 --no-checkpoint"
)
here=${0:a:h}
for prec in f32 x64; do
  if [[ $prec == x64 ]]; then export JAX_ENABLE_X64=1; else unset JAX_ENABLE_X64; fi
  for c in $cases; do
    tag=${c%%|*}; opts=${c#*|}
    (cd $MAIN   && JAX_PLATFORMS=cpu PYTHONPATH=$MAIN/src:$MAIN     $PY $here/gate_bitwise.py $OUT/main_${prec}_$tag.npz   ${=opts} >/dev/null)
    (cd $BRANCH && JAX_PLATFORMS=cpu PYTHONPATH=$BRANCH/src:$BRANCH $PY $here/gate_bitwise.py $OUT/branch_${prec}_$tag.npz ${=opts} >/dev/null)
    $PY - <<PY
import numpy as np
a = np.load("$OUT/main_${prec}_$tag.npz"); b = np.load("$OUT/branch_${prec}_$tag.npz")
ulps = []
for key in ("value", "grad_tprim", "grad_fprim"):
    x = np.asarray(a[key]).ravel(); y = np.asarray(b[key]).ravel()
    signed = np.int32 if x.dtype == np.float32 else np.int64
    ulps.append(int(np.max(np.abs(x.view(signed).astype(np.int64) - y.view(signed).astype(np.int64)))))
verdict = "BITWISE" if max(ulps) == 0 else "ULP " + "/".join(str(u) for u in ulps)
print(f"{'$prec':4s} {'$tag':16s} {verdict}")
PY
  done
done
