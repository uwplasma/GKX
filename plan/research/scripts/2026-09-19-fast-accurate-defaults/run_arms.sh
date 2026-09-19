#!/usr/bin/env bash
# Q25 arms. One fresh process per arm, serially, output captured per arm.
# The two full-resolution arms (a1, a2) are the shipped Cyclone deck at its own
# Nl=16/Nm=48; every other arm uses the same deck at Nl=4/Nm=8 so the route and
# the policy under test are the deck's while the rung is cheap enough to repeat.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
PY="${GKX_PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}"
S=plan/research/scripts/2026-09-19-fast-accurate-defaults/defaults.py
O=plan/research/scripts/2026-09-19-fast-accurate-defaults/out
DECK=examples/linear/axisymmetric/cyclone.toml
mkdir -p "$O"
export PYTHONPATH="$REPO/src:$REPO" JAX_PLATFORMS=cpu

arm() {  # arm <label> <x64:0|1> <args...>
  local label="$1"; shift
  local x64="$1"; shift
  echo "### $label"
  if [ "$x64" = "1" ]; then
    JAX_ENABLE_X64=true nice -n 10 "$PY" -W always "$S" linear "$DECK" --label "$label" "$@" \
      > "$O/$label.txt" 2>&1
  else
    nice -n 10 "$PY" -W always "$S" linear "$DECK" --label "$label" "$@" \
      > "$O/$label.txt" 2>&1
  fi
  echo "   exit $?  loadavg $(uptime | sed 's/.*averages: //')"
}

# Precision, at the deck's own resolution.
arm a1-default-f32 0
arm a2-default-x64 1

# Precision A/B at the cheap rung. The same two commands run against a pristine
# origin/main worktree are what show that JAX_ENABLE_X64 did not reach the run.
arm b1-x64-small 1 --Nl 4 --Nm 8

# Eigen route.
arm c1-adaptive-f32      0 --method adaptive      --Nl 4 --Nm 8
arm c2-shift-invert-f32  0 --method shift_invert  --Nl 4 --Nm 8
arm c3-shift-invert-mi400  0 --method shift_invert --shift-maxiter 400  --Nl 4 --Nm 8
arm c4-shift-invert-mi2000 0 --method shift_invert --shift-maxiter 2000 --Nl 4 --Nm 8
arm c5-adaptive-x64      1 --method adaptive      --Nl 4 --Nm 8
arm c6-shift-invert-x64  1 --method shift_invert  --Nl 4 --Nm 8

# Time integrator and step-size policy, on the explicit time path.
arm d1-deck-time         0 --solver explicit_time --Nl 4 --Nm 8
arm d2-timeconfig-rk2-fixed 0 --solver explicit_time --Nl 4 --Nm 8 --time-defaults
arm d3-timeconfig-rk4-fixed 0 --solver explicit_time --Nl 4 --Nm 8 --time-defaults --time-method rk4
arm d4-timeconfig-rk2-adaptive 0 --solver explicit_time --Nl 4 --Nm 8 --time-defaults --fixed-dt false
arm d5-timeconfig-rk4-adaptive 0 --solver explicit_time --Nl 4 --Nm 8 --time-defaults --time-method rk4 --fixed-dt false
arm d6-timeconfig-solver-time  0 --solver time          --Nl 4 --Nm 8 --time-defaults

# Resolution fallback: Nl/Nm omitted fall back to 24/12, not the deck's 16/48.
arm e1-resolution-fallback 0 --Nl 24 --Nm 12

echo DONE
