#!/usr/bin/env bash
# Q28 arms. One fresh process per arm, serially, output captured per arm.
#
# Every arm runs the shipped Cyclone deck, examples/linear/axisymmetric/cyclone.toml.
# The power and time arms use the deck at Nl=4/Nm=8 so the route and the policy
# under test are the deck's while the rung is cheap enough to repeat; their
# accuracy reference is that rung's own certified adaptive eigenpair. The
# resolution arms sweep (Nl, Nm) at the deck's own ky=0.3 and are referenced to
# the tracked GX golden in src/gkx/data/cyclone_reference_adiabatic.csv.
#
# The host was contended throughout (1-minute load 32-57 with four sibling
# lanes), so no arm's wall time is quoted as a result. The cost numbers that
# are quoted are load-independent: propagator applies for the power arms,
# resolved step size / step count / right-hand-side evaluations for the time
# arms, and Nl*Nm for the resolution arms.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
PY="${GKX_PY:-/Users/rogeriojorge/local/venvs/gkx-review-20260913/bin/python}"
S=plan/research/scripts/2026-09-20-default-resolution-and-integrator/measure.py
O=plan/research/scripts/2026-09-20-default-resolution-and-integrator/out
DECK=examples/linear/axisymmetric/cyclone.toml
mkdir -p "$O"
export PYTHONPATH="$REPO/src:$REPO" JAX_PLATFORMS=cpu

arm() {  # arm <subcommand> <label> <args...>
  local sub="$1"; shift
  local label="$1"; shift
  nice -n 10 "$PY" -W always "$S" "$sub" "$DECK" --label "$label" "$@" \
    > "$O/$label.txt" 2>&1
  echo "   exit $?  $label  loadavg $(uptime | sed 's/.*averages: //')"
}

# ---------------------------------------------------------------------------
# Power-iteration ladder. KrylovConfig.power_iters is 200; the public
# dominant_eigenpair(power_iters=...) default is 40. Cost is exactly the
# iteration count (one lax.scan over one _advance_imex2 apply per iteration),
# so the ladder is a pure accuracy-against-applies curve.
# ---------------------------------------------------------------------------
for N in 40 80 200 400 1000 2000 5000 10000; do
  arm power "p-iters$N" --Nl 4 --Nm 8 --power-iters "$N"
done

# ---------------------------------------------------------------------------
# Time integrator and step policy, on the explicit time path, with the
# TimeConfig dataclass defaults (t_max=100, dt=0.1) and the method/fixed_dt
# pairing under test. The two fixed-step arms are expected to fail: dt=0.1 is
# ~7.8x the CFL-stable step for this deck.
# ---------------------------------------------------------------------------
arm time t-rk2-fixed     --solver explicit_time --Nl 4 --Nm 8 --time-method rk2   --fixed-dt true
arm time t-rk4-fixed     --solver explicit_time --Nl 4 --Nm 8 --time-method rk4   --fixed-dt true
arm time t-rk2-adaptive  --solver explicit_time --Nl 4 --Nm 8 --time-method rk2   --fixed-dt false
arm time t-rk4-adaptive  --solver explicit_time --Nl 4 --Nm 8 --time-method rk4   --fixed-dt false
arm time t-rk3-adaptive  --solver explicit_time --Nl 4 --Nm 8 --time-method rk3   --fixed-dt false
# Same-step control: at one stable fixed dt the schemes differ only by order,
# which separates the integrator's truncation error from the step size the CFL
# controller hands each scheme in the adaptive arms above.
arm time t-rk2-dt001     --solver explicit_time --Nl 4 --Nm 8 --time-method rk2 --fixed-dt true --dt 0.01
arm time t-rk4-dt001     --solver explicit_time --Nl 4 --Nm 8 --time-method rk4 --fixed-dt true --dt 0.01

# ---------------------------------------------------------------------------
# Velocity-space resolution ladder, certified adaptive eigensolves at the
# deck's own ky=0.3. (24,12) is the runtime fallback; (12,24) is its transpose
# at identical Nl*Nm; (16,48) is the deck's own pair.
# ---------------------------------------------------------------------------
for pair in "24 12" "12 24" "8 24" "8 32" "12 32" "16 32" "24 24" "16 48"; do
  read -r nl nm <<< "$pair"
  arm resolution "r-nl${nl}-nm${nm}" --Nl "$nl" --Nm "$nm"
done

echo DONE
