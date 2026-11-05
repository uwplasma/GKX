"""Route nonlinear parallel policy without dropping requested behavior.

``species_hermite`` uses the production species x Hermite mesh, replicating
Laguerre, perpendicular and parallel spatial axes. Auto mode selects the
largest compatible subset of the device budget; explicit counts fail closed
when they cannot factor. Status reports selected and budgeted devices.

Named collectives serve explicit ``rk2``/``rk3``/``rk4``/``rk4-pi``/``sspx3``
runs with production diagnostics, adaptive chunks, saturation checks and progress.
Fixed-step diagnostics-off runs retain the final-state scan where supported.
Diagnosed preparation reuses the same backend with ``strict_identity=False``;
model overrides remain unsupported. Active collisions, sources, fixed modes,
checkpointing, additional state sharding, unequal species moment counts and
reflectionless closures retain the ordinary solver. Uniform moment stamps must
match the normalized state. Fallback reports whole-state placement and does not
imply distributed RHS work.

``ky`` is a single-process placement diagnostic with replicated work. FFT axes
remain local; unsupported strategies and axes raise with their reasons.

``strict_identity`` compares serial state and transport traces, accepting a
maximum absolute error within ``atol`` or maximum scaled relative error within
``rtol``. Multi-process callers must initialize JAX externally and enter on
every rank; the selected mesh must include each process. Explicit global state
returns gather the full array to every host. Identity checks compare local
shards and exchange only error scalars.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental.multihost_utils import process_allgather

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
    "shard does not survive SPMD partitioning, and the retired device_z pencil "
    "route evaluated a reduced diagnostic bracket operator with no streaming, "
    "mirror, curvature, collision, or species terms rather than the production "
    "nonlinear RHS. See docs/parallelization.rst. "
    + _SUPPORT_SUMMARY.capitalize()
    + "."
)

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
        return int(getattr(self.mesh_plan, "num_devices", len(self.devices)))

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
            f"{self.device_count} of {len(self.devices)} devices, shard "
            f"{tuple(self.mesh_plan.shard_shape)}, collectives: {collectives} "
            f"(strict_identity={strict})"
        )


def _normalized(value: Any, default: str) -> str:
    text = str(default if value is None else value).strip().lower()
    return text.replace("-", "_")


def _process_balanced_devices(devices: Any) -> tuple[Any, ...]:
    groups = [
        [device for device in devices if device.process_index == process]
        for process in range(jax.process_count())
    ]
    return tuple(device for group in zip(*groups) for device in group)


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
    if axis == "ky" and jax.process_count() > 1:
        raise NonlinearParallelRoutingError(
            "axis='ky' is a single-process placement diagnostic"
        )

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
    if bool(getattr(parallel, "auto", False)) and jax.process_count() > 1:
        devices = _process_balanced_devices(jax.devices())[: len(devices)]
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
    """Attach the requested mesh, selecting a feasible subset only in auto mode."""

    from gkx.parallel.velocity_plan import (
        build_species_hermite_mesh_plan,
        species_hermite_device_counts,
    )

    shape = tuple(int(x) for x in state.shape)
    if len(shape) not in (5, 6) or min(shape) < 1:
        raise NonlinearParallelRoutingError(
            f"expected a nonempty 5D or 6D state, got {shape}"
        )
    ns = shape[0] if len(shape) == 6 else 1
    nm = shape[2] if len(shape) == 6 else shape[1]
    balanced = plan.auto and jax.process_count() > 1
    devices = _process_balanced_devices(plan.devices) if balanced else plan.devices
    count = len(devices)
    if plan.auto:
        feasible = species_hermite_device_counts(ns, nm, limit=count)
        count = max(
            (
                n
                for n in feasible
                if n >= 2 and (not balanced or n % jax.process_count() == 0)
            ),
            default=count,
        )
    if jax.process_count() > 1 and {d.process_index for d in devices[:count]} != set(
        range(jax.process_count())
    ):
        raise NonlinearParallelRoutingError(
            "the nonlinear mesh must include a device from every initialized process"
        )
    try:
        mesh_plan = build_species_hermite_mesh_plan(shape, num_devices=count)
    except ValueError as exc:
        supported = species_hermite_device_counts(ns, nm)
        raise NonlinearParallelRoutingError(
            f"[parallel] axis='species_hermite' cannot factor "
            f"{count} devices over Ns={ns}, Nm={nm}: {exc}. "
            "Species are factored first and the Hermite remainder must divide "
            "Nm exactly, so this grid supports "
            f"{', '.join(str(count) for count in supported)} devices."
        ) from exc
    if balanced:
        selected = tuple(sorted(devices[:count], key=lambda d: d.process_index))
        devices = selected + tuple(d for d in plan.devices if d not in selected)
    return replace(plan, mesh_plan=mesh_plan, devices=devices)


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
                state, resolved.mesh_plan, devices=list(resolved.devices)
            ),
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
    # ky is the third-from-last axis of both packed layouts. Its extent is
    # ``Ny`` on a two-sided axis and ``Nyc = 1 + Ny//2`` on a ``ky >= 0`` one
    # (gkx.core_ky_layout), and ``Nyc`` is odd whenever ``Ny`` is a multiple
    # of four. JAX places an array only in equal shares, so an extent the
    # devices do not divide enters replicated on the same mesh (SHARD-PAD,
    # plan F.6). That is not a weaker run than the divisible case: the
    # integrator's initial-state projection hands the scan a replicated state
    # either way, so this route integrates the whole state on every device
    # whatever the extent (see the module docstring).
    if int(state.shape[-3]) % plan.device_count:
        return jax.device_put(state, NamedSharding(sharding.mesh, PartitionSpec()))
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
        reference = np.asarray(jax.device_get(jnp.asarray(serial_value)))
        candidate = jnp.asarray(sharded_value)
        errors = np.zeros(2)
        if reference.shape != candidate.shape:
            errors[:] = np.inf
        else:
            for shard in candidate.addressable_shards:
                local_reference = reference[shard.index]
                error = np.abs(np.asarray(shard.data) - local_reference)
                errors = np.maximum(
                    errors,
                    [
                        np.max(error),
                        np.max(error / np.maximum(np.abs(local_reference), plan.atol)),
                    ],
                )
        if jax.process_count() > 1:
            errors = np.max(process_allgather(errors).reshape(-1, 2), axis=0)
        abs_error, rel_error = map(float, errors)
        if not (abs_error <= float(plan.atol) or rel_error <= float(plan.rtol)):
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


def _supports_species_hermite_policy(cfg, ctx, policy, *, method, plan):
    """Check shared named-collective explicit physics and runtime limits."""

    return (
        plan.axis == "species_hermite"
        and not policy.fixed_mode_on
        and not policy.source_on
        and not cfg.time.checkpoint
        and not cfg.time.collision_split
        and not cfg.time.state_sharding
        and (method is None or method == cfg.time.method)
        and cfg.time.method in {"rk2", "rk3", "rk4", "sspx3", "rk4-pi"}
        and all(
            sp.Nl in (None, ctx.G0.shape[1])
            and sp.Nm in (None, ctx.G0.shape[2])
            and sp.hermite_closure == "hypercollision"
            for sp in cfg.species
            if sp.kinetic
        )
        and (
            float(ctx.terms.collisions) == 0.0
            or not np.any(np.asarray(ctx.params.nu) != 0.0)
        )
    )


def _species_hermite_diagnostic_rhs(
    prepared, policies, *, params, plan, compressed_real_fft, laguerre_mode
):
    """Build the staged named RHS for the shared production diagnostic scan."""

    from gkx.parallel.integrators import _prepare_species_hermite_rhs

    return _prepare_species_hermite_rhs(
        prepared.G0,
        prepared.cache,
        params,
        terms=policies.collision_policy.rhs_terms,
        plan=plan.mesh_plan,
        devices=plan.devices,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )


def _integrate_species_hermite_final_state(
    cfg: Any,
    ctx: Any,
    policy: Any,
    *,
    method: str | None,
    plan: NonlinearParallelPlan,
    status: Any,
    prepare: Callable[[], Any] | None = None,
) -> Any:
    """Use named collectives only when every requested policy is supported."""

    if (
        policy.diagnostics_on
        or not cfg.time.fixed_dt
        or (method or cfg.time.method) == "rk4-pi"
        or str(cfg.time.run_to).strip().lower() != "t_max"
        or policy.show_progress
        or cfg.time.progress_bar
        or (prepare is not None and plan.strict_identity)
        or not _supports_species_hermite_policy(
            cfg, ctx, policy, method=method, plan=plan
        )
    ):
        return None
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.parallel.integrators import _prepare_nonlinear_species_hermite
    from gkx.terms.assembly import compute_fields_cached

    status("using named species-Hermite collectives for the fixed-step trajectory")
    prepared = prepare() if prepare is not None else None
    cache = (
        prepared.cache
        if prepared is not None
        else build_linear_cache(ctx.grid, ctx.geom, ctx.params, *ctx.G0.shape[1:3])
    )
    params, terms = ctx.params, ctx.terms

    advance = _prepare_nonlinear_species_hermite(
        ctx.G0,
        cache,
        params,
        dt=ctx.dt,
        steps=ctx.steps,
        method=cfg.time.method,
        terms=terms,
        plan=plan.mesh_plan,
        devices=plan.devices,
        compressed_real_fft=cfg.time.compressed_real_fft,
        laguerre_mode=policy.laguerre_mode,
    )

    if prepared is not None:
        from gkx.parallel.integrators import stage_from_host
        from gkx.parallel.state import resolve_species_hermite_sharding

        sharding = resolve_species_hermite_sharding(
            prepared.initial_state, plan.mesh_plan, devices=plan.devices
        )
        shape, dtype = prepared.initial_state.shape, prepared.initial_state.dtype

        @jax.jit
        def run_raw(state, advance):
            final = advance(state).state
            fields = compute_fields_cached(final, cache, params, terms=terms)
            empty = jnp.empty((0,), dtype=jnp.real(final).dtype)
            return final, (None, empty, empty), fields

        def execute(state, time_horizon=None):
            if time_horizon is not None or state.shape != shape:
                raise ValueError(
                    "parallel preparation requires its fixed shape and horizon"
                )
            state = jnp.asarray(state, dtype=dtype)
            return run_raw(stage_from_host(state, sharding), advance)

        def reject_dynamic(*_args):
            raise ValueError("parallel preparation does not support model overrides")

        return replace(
            prepared,
            initial_state=stage_from_host(prepared.initial_state, sharding),
            _run_raw=execute,
            _run_dynamic_raw=reject_dynamic,
        )
    sharded_state = advance(ctx.G0).state
    if not bool(jnp.all(jnp.isfinite(sharded_state))):
        raise FloatingPointError(
            "species-Hermite integration produced a non-finite state"
        )
    fields = compute_fields_cached(sharded_state, cache, ctx.params, terms=ctx.terms)
    return sharded_state, fields


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
