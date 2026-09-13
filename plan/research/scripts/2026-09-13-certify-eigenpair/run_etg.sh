#!/bin/zsh
# Q12 ETG residual supervisor: one fresh process per (arm, ky), sequential,
# 30-min wall guard per cell, nice 19, against a detached clean source tree.
# usage: run_etg.sh <repo (clean detached worktree)> <output dir>
set -u
REPO=$1
OUT=$2
SCRIPT=${0:A:h}/etg_residuals.py
PY=/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python
mkdir -p "$OUT"
export PYTHONPATH=$REPO/src:$REPO JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu
cd "$REPO" || exit 1
{
  echo "repo=$REPO sha=$(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')"
  echo "script_sha256=$(shasum -a 256 "$SCRIPT" | cut -d' ' -f1) host=$(hostname) pid=$$"
} >> "$OUT/supervisor.txt"
run() {
  local name=$1
  shift
  echo "START $name $(date -u +%FT%TZ) load=$(sysctl -n vm.loadavg)" >> "$OUT/supervisor.txt"
  gtimeout 1800 nice -n 19 /usr/bin/time -l "$PY" "$SCRIPT" --repo "$REPO" "$@" \
    > "$OUT/$name.txt" 2> "$OUT/$name.err"
  echo "END $name rc=$? $(date -u +%FT%TZ)" >> "$OUT/supervisor.txt"
}
for ky in 10 20 30; do run "etg_default_ky$ky" --arm etg_default --ky $ky; done
for ky in 10 20 30; do run "runtime_default_ky$ky" --arm runtime_default --ky $ky; done
run scan --arm scan
for ky in 10 20 30; do run "time_ky$ky" --arm time --ky $ky; done
for ky in 10 20 30; do run "adaptive_ky$ky" --arm adaptive --ky $ky; done
echo "DONE $(date -u +%FT%TZ)" >> "$OUT/supervisor.txt"
