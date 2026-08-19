"""Carry a converged state between related runs so repeated work restarts warm.

The persistent JAX compilation cache reuses the *executable* between runs. This
module is its state-side counterpart: it reuses the *answer*. It applies only
where work repeats -- a ky scan, a parameter scan, an optimizer loop -- and
never to a single run, which still builds its own cold initial condition and
pays exactly what it paid before.

Two invariants make the reuse safe rather than merely fast:

* **Scale is removed, not carried.** A converged linear state has grown by
  ``exp(gamma T)`` and would overflow (or, on a damped branch, underflow) if it
  were carried raw. Both the eigensolve and the growth-rate fit are invariant
  to the overall scale of the initial condition, so rescaling loses nothing.
  The rescaling factor is rounded to a power of two, which is exact in binary
  floating point, so the carried state is the converged state to the last bit.
* **A carried state is validated before use.** A non-finite, empty, or
  zero-norm state is refused and the caller falls back to its cold path, so a
  failed neighbour can never poison the point after it.

Warm start is **opt-in**, and the measurements in ``docs/performance.rst`` are
why. On the certified adaptive eigensolver it is correctness-neutral (cold and
warm agree to 1e-6 relative on the Cyclone deck, well inside the certified
residual) but cost-neutral: that solver's work is a fixed-size filtered Arnoldi
whose cost is set by the Krylov dimension and the filter length, not by the
quality of ``v0``, and it already converges in one restart from the analytic
seed. On a fixed-horizon time integration a warm seed *does* pay -- it reaches
the horizon-converged growth rate in half the horizon -- but at any horizon
short of convergence it reports a different number than a cold start, because
it has removed a startup transient the cold run still contains. Neither is a
reason to remove the machinery; both are reasons not to switch it on for
somebody without being asked.

An eigensolver started inside one branch can also converge to that branch, so a
case whose branch structure is not already known should be scanned once cold
and compared point by point before the warm numbers are trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_TINY = 1.0e-300


@dataclass(frozen=True)
class WarmStartPolicy:
    """Whether, and at what amplitude, a converged state is carried forward."""

    enabled: bool = True
    amplitude: float = 1.0

    @classmethod
    def from_config(cls, cfg: Any, *, override: bool | None = None) -> "WarmStartPolicy":
        """Resolve ``[output] warm_start`` with an optional explicit override."""

        configured = bool(getattr(getattr(cfg, "output", None), "warm_start", True))
        return cls(enabled=configured if override is None else bool(override))


def linear_scan_warm_start_refusal(
    *,
    policy: WarmStartPolicy,
    solver_key: str,
    workers: int,
) -> str | None:
    """Return why a linear scan cannot warm start, or ``None`` when it can.

    Refusals are policy, not failure: each names a contract that carrying state
    would break, and the caller runs its existing cold path unchanged.
    """

    if not policy.enabled:
        return "warm start disabled"
    if str(solver_key) == "explicit_time":
        return "solver='explicit_time' cannot return a final state"
    if int(workers) > 1:
        return "independent ky workers must stay independent"
    return None


def resolve_scan_warm_start(args: Any, scan_cfg: Mapping[str, Any]) -> bool | None:
    """Resolve warm start from the CLI flag, then ``[scan]``, else the config.

    ``None`` means "not requested either way" and defers to ``[output]
    warm_start``, so the three surfaces compose without any of them having to
    know the default.
    """

    flag = getattr(args, "warm_start", None)
    if flag is not None:
        return bool(flag)
    if "warm_start" in scan_cfg:
        return bool(scan_cfg["warm_start"])
    return None


def scan_visit_order(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the visit order that puts neighbouring scan points side by side.

    A converged solution is only a good guess for a *nearby* point, so the scan
    is walked in monotone parameter order regardless of the order requested.
    Results are written back into the requested order by the caller, so the
    reported arrays are unchanged.
    """

    return np.argsort(np.asarray(values, dtype=float), kind="stable")


def state_norm(state: Any) -> float | None:
    """Return the L2 norm of a reusable state, or ``None`` if it is unusable."""

    if state is None:
        return None
    arr = np.asarray(state)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr.reshape(-1)))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return norm


def carry_state(state: Any, *, amplitude: float = 1.0) -> np.ndarray | None:
    """Rescale a converged *linear* state for reuse, or refuse it as a guess.

    The returned state differs from ``state`` by one power-of-two factor, which
    is exact in binary floating point: no information is lost and no rounding
    is introduced. ``None`` means the state is unusable and the caller must
    fall back to a cold start. Nonlinear saturated states carry a physical
    amplitude and must not pass through here -- check them with ``state_norm``
    and reuse them unscaled.
    """

    norm = state_norm(state)
    if norm is None:
        return None
    arr = np.asarray(state)
    target = float(amplitude)
    if target <= 0.0:
        raise ValueError("warm-start amplitude must be positive")
    exponent = int(round(float(np.log2(norm / target))))
    return np.asarray(arr * np.ldexp(1.0, -exponent), dtype=arr.dtype)


def relative_change(previous: Any, current: Any) -> float:
    """Return the relative L2 distance between two parameter or geometry vectors."""

    a = np.asarray(previous, dtype=float).reshape(-1)
    b = np.asarray(current, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), _TINY)
    return float(np.linalg.norm(b - a) / scale)


def signature_from_arrays(arrays: Iterable[Any]) -> np.ndarray:
    """Flatten geometry or parameter arrays into one comparable signature."""

    parts = [np.asarray(item, dtype=float).reshape(-1) for item in arrays]
    if not parts:
        raise ValueError("a signature needs at least one array")
    return np.concatenate(parts)


_SIGNATURE_FIELDS = (
    "gradpar_value",
    "bmag_profile",
    "bgrad_profile",
    "gds2_profile",
    "gds21_profile",
    "gds22_profile",
    "cv_profile",
    "gb_profile",
)


def flux_tube_signature(geom: Any) -> np.ndarray:
    """Return the flux-tube profiles a carried saturated state is judged against.

    These are exactly the metric coefficients the nonlinear operator reads, so
    two geometries with the same signature drive the same turbulence and a
    state saturated in one is a valid seed in the other.
    """

    present = [
        getattr(geom, name) for name in _SIGNATURE_FIELDS if hasattr(geom, name)
    ]
    if not present:
        raise ValueError("geometry exposes none of the flux-tube signature profiles")
    return signature_from_arrays(present)


@dataclass(frozen=True)
class SaturationRefreshPolicy:
    """When a carried saturated state stops being a valid seed.

    A saturated nonlinear state belongs to the geometry it was grown in. Two
    things invalidate it, and both are bounded here rather than left to trust:

    * ``geometry_tolerance``: the relative change in the flux-tube geometry
      signature above which the previous attractor is not the current one.
    * ``max_reuse``: how many consecutive warm spin-ups may follow one cold
      spin-up before a cold one is forced regardless of the metric, so a long
      run of individually small steps cannot drift arbitrarily far.

    ``warm_step_fraction`` is the fraction of the cold spin-up budget a warm
    spin-up is given. It is a budget, not a claim of convergence: a warm seed
    starts at saturated amplitude and so skips the linear growth phase, but it
    still has to re-equilibrate, and the cold budget is restored the moment
    either bound above is crossed.
    """

    max_reuse: int = 4
    geometry_tolerance: float = 0.05
    warm_step_fraction: float = 0.25

    def __post_init__(self) -> None:
        if int(self.max_reuse) < 0:
            raise ValueError("max_reuse must be non-negative")
        if float(self.geometry_tolerance) < 0.0:
            raise ValueError("geometry_tolerance must be non-negative")
        if not 0.0 < float(self.warm_step_fraction) <= 1.0:
            raise ValueError("warm_step_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class SaturationPlan:
    """How the next spin-up should be seeded, and why."""

    seed: np.ndarray | None
    steps: int
    warm: bool
    reason: str


@dataclass
class SaturationWarmStart:
    """Decide, and record, whether an optimizer iteration may reuse saturation.

    This deliberately does not change *when* the saturated state is refreshed.
    Callers refresh it at the same points they always did -- accepted stages --
    so the objective stays a fixed function of its inputs for the whole of each
    stage and never becomes a function of the optimizer's within-stage history.
    What changes is only the cost of producing the refreshed state.
    """

    policy: SaturationRefreshPolicy = field(default_factory=SaturationRefreshPolicy)
    state: np.ndarray | None = None
    signature: np.ndarray | None = None
    reuse_count: int = 0

    def plan(self, signature: Any, *, cold_steps: int) -> SaturationPlan:
        """Return the seed and step budget for the next spin-up."""

        steps_cold = int(cold_steps)
        if steps_cold < 1:
            raise ValueError("cold_steps must be positive")
        cold = SaturationPlan(seed=None, steps=steps_cold, warm=False, reason="")
        if state_norm(self.state) is None or self.signature is None:
            return replace(cold, reason="no usable saved state")
        if self.reuse_count >= int(self.policy.max_reuse):
            return replace(cold, reason="reuse budget exhausted")
        change = relative_change(self.signature, signature)
        if change > float(self.policy.geometry_tolerance):
            return replace(cold, reason=f"geometry moved {change:.3g}")
        warm_steps = max(int(round(steps_cold * self.policy.warm_step_fraction)), 1)
        return SaturationPlan(
            seed=np.asarray(self.state),
            steps=warm_steps,
            warm=True,
            reason=f"geometry moved {change:.3g}",
        )

    def record(self, state: Any, signature: Any, *, warm: bool) -> None:
        """Store the new saturated state and count the reuse chain."""

        self.state = np.asarray(state)
        self.signature = np.asarray(signature, dtype=float).reshape(-1)
        self.reuse_count = self.reuse_count + 1 if warm else 0


__all__ = [
    "SaturationPlan",
    "SaturationRefreshPolicy",
    "SaturationWarmStart",
    "WarmStartPolicy",
    "carry_state",
    "flux_tube_signature",
    "linear_scan_warm_start_refusal",
    "relative_change",
    "resolve_scan_warm_start",
    "scan_visit_order",
    "signature_from_arrays",
    "state_norm",
]
