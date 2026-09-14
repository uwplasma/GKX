#!/usr/bin/env bash
# Q7 production comparison, same host and same process conditions, run serially from the
# repository root after any other heavy run has finished:
#   1. the runtime-default adaptive route through #232's harness (exact_ladder.py, commit
#      7c8a76194121acbd0611f3d43814b5173e00dbd7, sha256 181b380f...), rung 4 = production chain;
#   2. matrix-free shift-invert with pr3-cm + SOLVAX gcrot recycling (prod_certify.py).
# One fresh single-threaded process each, cold, wall cap CAP_S (2700 s = 45 min).
set -u
D=plan/research/scripts/2026-09-13-preconditioner-bakeoff
PY=${PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}
CAP_S=${CAP_S:-2700}
WAIT_FOR=${WAIT_FOR:-}
PR232=7c8a76194121acbd0611f3d43814b5173e00dbd7
PR232_SHA=181b380fe7c483c7a8f819975849df7165322a957a46cbafd4ae0f3c51f0f160
export PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ -n "$WAIT_FOR" ]; then
  until grep -q "^exit" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
  while pgrep -f "bakeoff.py --case|pr_kz.py --case|prod_solve.py|prod_certify.py" >/dev/null; do sleep 10; done
fi

nice_level() {
  for _ in $(seq 1 30); do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$load" 'BEGIN { exit !(l < 20) }'; then echo 10; return; fi
    sleep 30
  done
  echo 19
}

harness=$(mktemp -t exact_ladder_pr232)
git show "$PR232:plan/research/scripts/2026-09-13-exact-ladder/exact_ladder.py" >"$harness"
if [ "$(shasum -a 256 "$harness" | awk '{print $1}')" != "$PR232_SHA" ]; then
  echo "harness sha mismatch" && exit 2
fi

level=$(nice_level)
echo "$(date '+%F %T') start adaptive_same_host (nice $level, cap ${CAP_S}s, load $(sysctl -n vm.loadavg)) pid $$"
/usr/bin/time -l nice -n "$level" perl -e 'alarm shift; exec @ARGV' "$CAP_S" \
  "$PY" "$harness" --rung 4 --arm default --repo "$PWD" >"$D/adaptive_same_host.txt" 2>&1
status=$?
echo "exit $status" >>"$D/adaptive_same_host.txt"
echo "$(date '+%F %T') end adaptive_same_host exit $status load $(sysctl -n vm.loadavg)"
rm -f "$harness"

CAP_S=$CAP_S "$D/run_series.sh" \
  "prod_certify::prod_certify.py --rule scalar-best --alpha-best -10 --sweeps 3 --inner-rtol 1e-9 --max-restarts 60 --max-steps 30 --target 1e-9"
