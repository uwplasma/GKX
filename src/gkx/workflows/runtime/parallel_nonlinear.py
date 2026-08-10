"""Route the runtime ``[parallel]`` policy into the nonlinear solver path.

``RuntimeParallelConfig`` used to be read only by :mod:`gkx.workflows.linear`,
so a nonlinear TOML could ask for a sharded run and silently get a serial one.
This module is the nonlinear half of that routing, and it is deliberately
narrow about what it will accept.

Supported today: ``strategy = "shard_map"`` with ``axis = "ky"``. The whole
nonlinear state is placed on a ``ky`` device mesh and the ordinary production
integrator runs on it, so the operator is the production nonlinear RHS rather
than a reduced stand-in. Every other strategy/axis combination raises instead
of falling back to serial.

``axis = "z"`` is rejected for two independent reasons, both measured on this
stack rather than assumed:

* the production parallel-streaming derivative is a spectral FFT along ``z``
  (:func:`gkx.operators.linear.streaming.grad_z_periodic`), so a whole-state
  ``z`` shard does not survive SPMD partitioning;
* :mod:`gkx.operators.nonlinear.device_z` is a reduced diagnostic operator --
  the ``-{phi,g}`` bracket with a model field solve, no streaming, mirror,
  curvature, collisions, or species axis -- so routing a production run through
  it would answer a different physics question.

Routing is fail-closed on numerical identity. With ``strict_identity = true``
the same run is also executed serially and the two answers must agree, or
:class:`NonlinearParallelIdentityError` is raised. The comparison reuses the
identity primitives already used by the device-z gates rather than introducing
a second tolerance convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from gkx.operators.nonlinear.spectral_core import (
    _host_max_abs_rel_error,
    _within_abs_or_rel_tolerance,
)
from gkx.parallel.state import resolve_state_sharding
from gkx.solvers.linear.parallel_common import _resolve_parallel_devices

SUPPORTED_NONLINEAR_STRATEGIES = ("serial", "shard_map")
SUPPORTED_NONLINEAR_AXES = ("ky",)

# Same convention as the device-z identity gates, so a nonlinear routing failure
# and a device-z gate failure mean the same thing.
IDENTITY_ATOL = 5.0e-6
IDENTITY_RTOL = 1.0e-4

_MESH_AXIS_NAME = "d"

_SUPPORT_SUMMARY = (
    "the nonlinear path supports [parallel] strategy='serial' and "
    "strategy='shard_map' with axis='ky'"
)

_Z_AXIS_REASON = (
    "[parallel] axis='z' has no nonlinear runtime route. The production "
    "parallel-streaming derivative is a spectral FFT along z, so a whole-state z "
    "shard does not survive SPMD partitioning, and the device-z pencil route in "
    "gkx.operators.nonlinear.device_z evaluates a reduced diagnostic bracket "
    "operator with no streaming, mirror, curvature, collision, or species terms "
    "rather than the production nonlinear RHS. See docs/parallelization.rst. "
    + _SUPPORT_SUMMARY.capitalize()
    + "."
)

# Scalar traces compared alongside the final state. These are the runtime
# diagnostics a nonlinear run is actually read for, so a route that reproduced
# the state but not the transport traces would still fail the gate.
_IDENTITY_TRACE_FIELDS = ("Wg_t", "Wphi_t", "heat_flux_t", "particle_flux_t")


class NonlinearParallelRoutingError(NotImplementedError):
    """Raised when ``[parallel]`` asks for a nonlinear route that does not exist."""


class NonlinearParallelIdentityError(RuntimeError):
    """Raised when a sharded nonlinear run does not reproduce the serial answer."""


@dataclass(frozen=True)
class NonlinearParallelPlan:
    """Resolved device plan for one sharded nonlinear runtime run."""

    axis: str
    devices: tuple[Any, ...]
    strict_identity: bool
    atol: float = IDENTITY_ATOL
    rtol: float = IDENTITY_RTOL

    @property
    def device_count(self) -> int:
        return len(self.devices)

    def describe(self) -> str:
        strict = "on" if self.strict_identity else "off"
        return (
            f"shard_map nonlinear route on axis='{self.axis}' across "
            f"{self.device_count} devices (strict_identity={strict})"
        )


def _normalized(value: Any, default: str) -> str:
    text = str(default if value is None else value).strip().lower()
    return text.replace("-", "_")


def _reject_independent_worker_options(parallel: Any) -> None:
    """Reject batching knobs that only apply to independent-scan strategies."""

    backend = _normalized(getattr(parallel, "backend", "auto"), "auto")
    if backend not in {"auto", ""}:
        raise NonlinearParallelRoutingError(
            f"[parallel] backend='{backend}' selects an independent worker pool, "
            "which orchestrates separate solver calls and cannot shard one "
            f"nonlinear run. {_SUPPORT_SUMMARY.capitalize()} and backend='auto'."
        )
    if getattr(parallel, "batch_size", None) is not None:
        raise NonlinearParallelRoutingError(
            "[parallel] batch_size applies to independent-scan batching, not to a "
            "single sharded nonlinear run; use num_devices to size the device mesh."
        )


def resolve_nonlinear_parallel_plan(parallel: Any) -> NonlinearParallelPlan | None:
    """Return the device plan for ``cfg.parallel``, or ``None`` for a serial run.

    Raises :class:`NonlinearParallelRoutingError` for any requested combination
    the nonlinear path cannot execute. Silently degrading to serial is the
    defect this function exists to remove.
    """

    if parallel is None:
        return None
    strategy = _normalized(getattr(parallel, "strategy", "serial"), "serial")
    if strategy == "serial":
        return None
    if strategy != "shard_map":
        raise NonlinearParallelRoutingError(
            f"[parallel] strategy='{strategy}' has no nonlinear runtime route. "
            "Independent-work strategies such as 'batch' and 'combined_ky' "
            "orchestrate separate solver calls for k_y scans and ensembles; they "
            "cannot shard a single nonlinear run. "
            f"{_SUPPORT_SUMMARY.capitalize()}."
        )

    axis = _normalized(getattr(parallel, "axis", ""), "")
    if axis == "z":
        raise NonlinearParallelRoutingError(_Z_AXIS_REASON)
    if axis not in SUPPORTED_NONLINEAR_AXES:
        raise NonlinearParallelRoutingError(
            f"[parallel] axis='{axis}' has no nonlinear runtime route. "
            f"{_SUPPORT_SUMMARY.capitalize()}."
        )
    _reject_independent_worker_options(parallel)

    try:
        devices = tuple(
            _resolve_parallel_devices(
                num_devices=getattr(parallel, "num_devices", None)
            )
        )
    except ValueError as exc:  # keep one catchable routing-error type
        raise NonlinearParallelRoutingError(
            f"[parallel] num_devices is not satisfiable: {exc}"
        ) from exc
    if len(devices) < 2:
        raise NonlinearParallelRoutingError(
            "the sharded nonlinear route needs at least two JAX devices, but "
            f"{len(devices)} is visible. Set [parallel] strategy='serial', or start "
            "the process with more devices (on CPU, "
            "XLA_FLAGS=--xla_force_host_platform_device_count=N)."
        )
    return NonlinearParallelPlan(
        axis=axis,
        devices=devices,
        strict_identity=bool(getattr(parallel, "strict_identity", True)),
    )


def shard_nonlinear_state(state: Any, plan: NonlinearParallelPlan) -> Any:
    """Return the initial nonlinear state placed on the plan's device mesh."""

    # ky is the third-from-last axis in both the (Nl, Nm, Nky, Nkx, Nz) and
    # (Ns, Nl, Nm, Nky, Nkx, Nz) layouts.
    extent = int(state.shape[-3])
    if extent % plan.device_count:
        raise NonlinearParallelRoutingError(
            f"[parallel] axis='{plan.axis}' has extent {extent}, which is not "
            f"divisible by the requested {plan.device_count} devices. Choose a "
            "device count that divides the grid, or set strategy='serial'."
        )
    sharding = resolve_state_sharding(
        state,
        plan.axis,
        axis_name=_MESH_AXIS_NAME,
        devices=list(plan.devices),
    )
    if sharding is None:
        raise NonlinearParallelRoutingError(
            f"could not build a '{plan.axis}' sharding for a nonlinear state with "
            f"shape {tuple(state.shape)} on {plan.device_count} devices."
        )
    return jax.device_put(state, sharding)


def _identity_pairs(
    *,
    serial_state: Any,
    sharded_state: Any,
    serial_diagnostics: Any | None,
    sharded_diagnostics: Any | None,
) -> list[tuple[str, Any, Any]]:
    pairs: list[tuple[str, Any, Any]] = [("final_state", serial_state, sharded_state)]
    if serial_diagnostics is None or sharded_diagnostics is None:
        return pairs
    for name in _IDENTITY_TRACE_FIELDS:
        serial_trace = getattr(serial_diagnostics, name, None)
        sharded_trace = getattr(sharded_diagnostics, name, None)
        if serial_trace is not None and sharded_trace is not None:
            pairs.append((name, serial_trace, sharded_trace))
    return pairs


def assert_nonlinear_parallel_identity(
    *,
    serial_state: Any,
    sharded_state: Any,
    serial_diagnostics: Any | None = None,
    sharded_diagnostics: Any | None = None,
    plan: NonlinearParallelPlan,
) -> None:
    """Raise unless the sharded nonlinear run reproduced the serial answer.

    Never returns a differing answer quietly: a violated tolerance is an
    exception naming every observable that drifted and by how much.
    """

    failures: list[str] = []
    for name, serial_value, sharded_value in _identity_pairs(
        serial_state=serial_state,
        sharded_state=sharded_state,
        serial_diagnostics=serial_diagnostics,
        sharded_diagnostics=sharded_diagnostics,
    ):
        abs_error, rel_error = _host_max_abs_rel_error(
            jnp.asarray(serial_value),
            jnp.asarray(sharded_value),
            atol=plan.atol,
        )
        if not _within_abs_or_rel_tolerance(
            abs_error, rel_error, atol=plan.atol, rtol=plan.rtol
        ):
            failures.append(
                f"{name} (max_abs={abs_error:.6e}, max_rel={rel_error:.6e})"
            )
    if not failures:
        return
    raise NonlinearParallelIdentityError(
        f"{plan.describe()} did not reproduce the serial nonlinear answer within "
        f"atol={plan.atol:.3e} rtol={plan.rtol:.3e}: "
        + ", ".join(failures)
        + ". The sharded result is discarded rather than returned."
    )


__all__ = [
    "IDENTITY_ATOL",
    "IDENTITY_RTOL",
    "SUPPORTED_NONLINEAR_AXES",
    "SUPPORTED_NONLINEAR_STRATEGIES",
    "NonlinearParallelIdentityError",
    "NonlinearParallelPlan",
    "NonlinearParallelRoutingError",
    "assert_nonlinear_parallel_identity",
    "resolve_nonlinear_parallel_plan",
    "shard_nonlinear_state",
]
