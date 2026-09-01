"""Route the runtime ``[parallel]`` policy into the nonlinear solver path.

``RuntimeParallelConfig`` used to be read only by :mod:`gkx.workflows.linear`,
so a nonlinear TOML could ask for a sharded run and silently get a serial one.
This module is the nonlinear half of that routing, and it is deliberately
narrow about what it will accept.

Supported today: ``strategy = "shard_map"`` with ``axis = "species_hermite"``
(the production decomposition -- a ``(species, hermite)`` mesh with the
perpendicular plane, Laguerre and ``z`` replicated) or ``axis = "ky"`` (kept as
a routing diagnostic). ``auto = true`` picks the species x Hermite mesh from the
visible devices and reports which one it chose. In both cases the whole
nonlinear state is placed on the mesh and the ordinary production integrator
runs on it, so the operator is the production nonlinear RHS rather than a
reduced stand-in; the audited ``shard_map`` route with named collectives is
:func:`gkx.parallel.integrators.integrate_nonlinear_species_hermite`. Every
other strategy/axis combination raises instead of falling back to serial.

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

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from gkx.operators.nonlinear.spectral_core import (
    _host_max_abs_rel_error,
    _within_abs_or_rel_tolerance,
)
from gkx.parallel.state import resolve_state_sharding
from gkx.solvers_linear_parallel_common import _resolve_parallel_devices

SUPPORTED_NONLINEAR_STRATEGIES = ("serial", "shard_map")
SUPPORTED_NONLINEAR_AXES = ("ky", "species_hermite")

# ``species_hermite`` is the production decomposition: a 2-D device mesh with
# species factored first and Hermite second, the perpendicular plane, Laguerre
# and z replicated. Its aliases exist because the same mesh is the natural
# reading of "velocity space" and of an explicit "(s, m)".
_SPECIES_HERMITE_ALIASES = {
    "species_hermite",
    "velocity",
    "s_m",
    "species",
    "m",
    "hermite",
}

# Same convention as the device-z identity gates, so a nonlinear routing failure
# and a device-z gate failure mean the same thing.
IDENTITY_ATOL = 5.0e-6
IDENTITY_RTOL = 1.0e-4

_MESH_AXIS_NAME = "d"

_SUPPORT_SUMMARY = (
    "the nonlinear path supports [parallel] strategy='serial' and "
    "strategy='shard_map' with axis='species_hermite' (the production "
    "species x Hermite mesh; aliases 'velocity', 's_m', 'species', 'm') or "
    "axis='ky' (routing diagnostic)"
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
    mesh_plan: Any | None = None
    auto: bool = False

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def mesh_shape(self) -> tuple[int, int]:
        if self.mesh_plan is None:
            return (self.device_count, 1)
        return (
            int(self.mesh_plan.chunks.get("s", 1)),
            int(self.mesh_plan.chunks.get("m", 1)),
        )

    def describe(self) -> str:
        strict = "on" if self.strict_identity else "off"
        if self.mesh_plan is None:
            return (
                f"shard_map nonlinear route on axis='{self.axis}' across "
                f"{self.device_count} devices (strict_identity={strict})"
            )
        ns_chunks, nm_chunks = self.mesh_shape
        chosen = "auto-selected " if self.auto else ""
        halo = int(self.mesh_plan.hermite_ghost_depth)
        collectives = "field psum" + (
            f", width-{halo} Hermite ppermute" if nm_chunks > 1 else ", no halo"
        )
        return (
            f"shard_map nonlinear route on a {chosen}species x Hermite mesh "
            f"({ns_chunks} species x {nm_chunks} Hermite) across "
            f"{self.device_count} devices, shard "
            f"{tuple(self.mesh_plan.shard_shape)}, collectives: {collectives} "
            f"(strict_identity={strict})"
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
    if axis in _SPECIES_HERMITE_ALIASES:
        axis = "species_hermite"
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
        auto=bool(getattr(parallel, "auto", False)),
    )


def resolve_species_hermite_mesh(
    state: Any, plan: NonlinearParallelPlan
) -> NonlinearParallelPlan:
    """Attach the ``(species, hermite)`` mesh a state and device count imply.

    Failure is a routing error naming the device counts that *do* factor, so a
    user who asked for three devices is told which numbers work rather than
    only which one did not.
    """

    from gkx.parallel.velocity_plan import (
        build_species_hermite_mesh_plan,
        species_hermite_device_counts,
    )

    shape = tuple(int(x) for x in state.shape)
    try:
        mesh_plan = build_species_hermite_mesh_plan(
            shape, num_devices=plan.device_count
        )
    except ValueError as exc:
        ns = shape[0] if len(shape) == 6 else 1
        nm = shape[2] if len(shape) == 6 else shape[1]
        supported = species_hermite_device_counts(ns, nm)
        raise NonlinearParallelRoutingError(
            f"[parallel] axis='species_hermite' cannot factor "
            f"{plan.device_count} devices over Ns={ns}, Nm={nm}: {exc}. "
            "Species are factored first and the Hermite remainder must divide "
            "Nm exactly, so this grid supports "
            f"{', '.join(str(count) for count in supported)} devices."
        ) from exc
    return replace(plan, mesh_plan=mesh_plan)


def shard_nonlinear_state(state: Any, plan: NonlinearParallelPlan) -> Any:
    """Return the initial nonlinear state placed on the plan's device mesh."""

    if plan.axis == "species_hermite":
        from gkx.parallel.integrators import stage_from_host
        from gkx.parallel.state import resolve_species_hermite_sharding

        resolved = (
            plan
            if plan.mesh_plan is not None
            else resolve_species_hermite_mesh(state, plan)
        )
        return stage_from_host(
            state,
            resolve_species_hermite_sharding(
                state, resolved.mesh_plan, devices=list(plan.devices)
            ),
        )

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
    "resolve_species_hermite_mesh",
    "shard_nonlinear_state",
]
