"""Sharded fixed-step integrators for multi-device scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from gkx.operators.fluxes import heat_flux_species, particle_flux_species
from gkx.operators.moments import (
    distribution_free_energy,
    electrostatic_field_energy,
)
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    _SPECIES_PARAM_NAMES,
)
from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached
from gkx.operators.nonlinear.projection import _make_compressed_real_fft_projector
from gkx.terms.config import FieldState, TermConfig


_EXPLICIT_METHODS = {"euler", "rk2", "rk3", "rk3_heun", "rk3_classic", "rk4", "sspx3"}
pjit = jax.jit


def _dt_array(dt: float, state_dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(dt, dtype=jnp.real(jnp.empty((), dtype=state_dtype)).dtype)


def _validate_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")


def _validate_explicit_method(method: str) -> str:
    method_key = str(method).strip().lower()
    if method_key not in _EXPLICIT_METHODS:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_heun', 'rk3_classic', 'rk4', 'sspx3'}"
        )
    return method_key


def integrate_linear_sharded(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    terms: LinearTerms | None = None,
    state_sharding: Any | None = None,
) -> jnp.ndarray:
    """Integrate the linear system with a pjit-sharded RK2 loop.

    This is intentionally minimal: it returns the final state only and avoids
    saving time histories to focus on strong scaling of the RHS.
    """

    if terms is None:
        terms = LinearTerms()
    _validate_steps(steps)

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    dt_val = _dt_array(dt, state_dtype)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def step(G, _):
        G = _maybe_shard(G)
        dG, _ = linear_rhs_cached(G, cache, params, terms=terms)
        G_half = G + 0.5 * dt_val * dG
        dG_half, _ = linear_rhs_cached(G_half, cache, params, terms=terms)
        G_next = G + dt_val * dG_half
        return _maybe_shard(G_next), None

    def run(G_init):
        G_final, _ = jax.lax.scan(step, G_init, xs=None, length=steps)
        return G_final

    run_pjit = pjit(
        run,
        in_shardings=state_sharding,
        out_shardings=state_sharding,
    )

    if state_sharding is not None:
        G0 = jax.device_put(G0, state_sharding)
        G0 = _maybe_shard(G0)
    return run_pjit(G0)


def _rk3_classic_update(
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    project_shard: Callable[[jnp.ndarray], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    G1 = stage(G, k1, 1.0)
    k2, _ = rhs(G1)
    G2 = project_shard(0.75 * G + 0.25 * (G1 + dt_val * k2))
    k3, _ = rhs(G2)
    return (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)


def _rk3_heun_update(
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    k2, _ = rhs(stage(G, k1, 1.0 / 3.0))
    k3, _ = rhs(stage(G, k2, 2.0 / 3.0))
    return stage(G, k3, 0.75) + 0.25 * dt_val * k1


def _rk4_update(
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    k2, _ = rhs(stage(G, k1, 0.5))
    k3, _ = rhs(stage(G, k2, 0.5))
    k4, _ = rhs(stage(G, k3, 1.0))
    return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _nonlinear_explicit_update(
    method_key: str,
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    project_shard: Callable[[jnp.ndarray], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    if method_key == "euler":
        return G + dt_val * k1
    if method_key == "rk2":
        k2, _ = rhs(stage(G, k1, 0.5))
        return G + dt_val * k2
    if method_key == "rk3_classic":
        return _rk3_classic_update(
            G,
            k1,
            rhs=rhs,
            stage=stage,
            project_shard=project_shard,
            dt_val=dt_val,
        )
    if method_key in {"rk3", "rk3_heun"}:
        return _rk3_heun_update(G, k1, rhs=rhs, stage=stage, dt_val=dt_val)
    if method_key == "rk4":
        return _rk4_update(G, k1, rhs=rhs, stage=stage, dt_val=dt_val)
    return _rk3_classic_update(
        G,
        k1,
        rhs=rhs,
        stage=stage,
        project_shard=project_shard,
        dt_val=dt_val,
    )


@lru_cache(maxsize=64)
def _compiled_nonlinear_sharded_runner(
    *,
    pjit_fn: Callable[..., Any],
    rhs_fn: Callable[..., tuple[jnp.ndarray, FieldState]],
    method_key: str,
    steps: int,
    terms: TermConfig,
    state_sharding: Any | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
    return_fields: bool,
    projector: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> Callable[[jnp.ndarray, LinearCache, LinearParams, jnp.ndarray], Any]:
    """Compile one reusable nonlinear runner without capturing large arrays."""

    def run(
        G_init: jnp.ndarray,
        cache: LinearCache,
        params: LinearParams,
        dt_val: jnp.ndarray,
    ) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
        state_dtype = jnp.result_type(G_init, jnp.complex64)

        def maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
            if state_sharding is None:
                return state
            return jax.lax.with_sharding_constraint(state, state_sharding)

        def project_shard(state: jnp.ndarray) -> jnp.ndarray:
            if projector is not None:
                state = projector(state)
            return maybe_shard(jnp.asarray(state, dtype=state_dtype))

        def rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
            dG, fields = rhs_fn(
                state,
                cache,
                params,
                terms,
                compressed_real_fft=compressed_real_fft,
                laguerre_mode=laguerre_mode,
            )
            return jnp.asarray(dG, dtype=state_dtype), fields

        def stage(
            state: jnp.ndarray, increment: jnp.ndarray, scale: float
        ) -> jnp.ndarray:
            return project_shard(
                state + jnp.asarray(scale, dtype=dt_val.dtype) * dt_val * increment
            )

        def step(
            G: jnp.ndarray, _unused: None
        ) -> tuple[jnp.ndarray, FieldState | None]:
            G = project_shard(G)
            k1, _ = rhs(G)
            G_next = _nonlinear_explicit_update(
                method_key,
                G,
                k1,
                rhs=rhs,
                stage=stage,
                project_shard=project_shard,
                dt_val=dt_val,
            )
            G_next = project_shard(G_next)
            if not return_fields:
                return G_next, None
            _dG_next, fields_next = rhs(G_next)
            return G_next, fields_next

        G_final, fields_t = jax.lax.scan(step, G_init, xs=None, length=steps)
        if return_fields:
            return G_final, cast(FieldState, fields_t)
        return G_final

    output_sharding = None if return_fields else state_sharding
    return pjit_fn(
        run,
        in_shardings=(state_sharding, None, None, None),
        out_shardings=output_sharding,
    )


def integrate_nonlinear_sharded(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str = "rk2",
    terms: TermConfig | None = None,
    state_sharding: Any | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    return_fields: bool = True,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system with an explicit pjit-sharded scan.

    The state array can be partitioned along a ``resolve_state_sharding`` axis
    such as ``ky`` or ``kx``. This is a diagnostic whole-state sharding
    primitive for identity gates and profiler localization. It is not a
    production nonlinear domain decomposition or speedup claim until the exact
    workload has communication-complete identity, conservation, transport, and
    profiler gates. Domain-sharding identity reports are metadata gates only;
    they do not authorize routing through this whole-state integrator.
    """

    _validate_steps(steps)
    method_key = _validate_explicit_method(method)
    state_dtype = jnp.result_type(G0, jnp.complex64)
    G_init = jnp.asarray(G0, dtype=state_dtype)
    projector = (
        _make_compressed_real_fft_projector(
            ny_full=int(cache.ky.size), nx=int(cache.kx.size)
        )
        if compressed_real_fft
        else None
    )
    if state_sharding is not None:
        G_init = jax.device_put(G_init, state_sharding)
    runner = _compiled_nonlinear_sharded_runner(
        pjit_fn=pjit,
        rhs_fn=nonlinear_rhs_cached,
        method_key=method_key,
        steps=steps,
        terms=terms or TermConfig(),
        state_sharding=state_sharding,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        return_fields=return_fields,
        projector=projector,
    )
    return runner(G_init, cache, params, _dt_array(dt, state_dtype))


# ---------------------------------------------------------------------------
# Production species x Hermite shard_map route
# ---------------------------------------------------------------------------
#
# One device owns a species block and a Hermite block; the perpendicular plane,
# Laguerre and z stay whole. That choice is what keeps the shard-local kernels
# the *serial* kernels: every FFT in the bracket and in parallel streaming runs
# over an axis this mesh never splits, so a shard is a slab the production RHS
# already knows how to evaluate, and route overhead is structural rather than
# tuned. See docs/parallelization.rst for the evidence behind the choice.
#
# Collectives, and nothing else:
#   C1  psum   masked m = 0..3 moment head, then the species sum inside the
#              field solve            -> quasineutrality, Ampere, fluxes
#   C2  ppermute x2  one width-2 Hermite halo covering m+-1 (streaming, mirror)
#              and m+-2 (curvature) in a single exchange, only when m is split
#   C5  psum   scalar diagnostics, accumulated in the scan carry
# Any all-to-all in the HLO of this route is a bug, not a cost.

# Cache arrays carrying a species axis at dimension 0.
_SPECIES_CACHE_NAMES = ("Jl", "JlB", "b", "laguerre_j0", "laguerre_j1_over_alpha")

_DIAGNOSTIC_MOMENT_ROWS = 4


@dataclass(frozen=True)
class SpeciesHermiteRun:
    """Result of one sharded nonlinear trajectory."""

    state: jnp.ndarray
    traces: dict[str, jnp.ndarray]
    plan: Any
    mesh_shape: tuple[int, int]

    def describe(self) -> str:
        ns_chunks, nm_chunks = self.mesh_shape
        halo = int(self.plan.hermite_ghost_depth)
        collectives = "field psum" + (
            f" + width-{halo} Hermite ppermute" if nm_chunks > 1 else ""
        )
        return (
            f"species x Hermite mesh ({ns_chunks}, {nm_chunks}) on "
            f"{ns_chunks * nm_chunks} devices, shard "
            f"{tuple(self.plan.shard_shape)}, collectives: {collectives}"
        )


def _species_hermite_inputs(cache, params, ns, real_dtype, plan, hermite_count):
    """Return the sharded leaves and their specs for one mesh placement."""

    from jax.sharding import PartitionSpec
    from gkx.operators.linear.params import _as_species_array
    from gkx.parallel.velocity_hermite import (
        hermite_cache_partition_specs,
        hermite_window_cache_arrays,
        hermite_window_indices,
    )

    nm_chunks = int(plan.chunks.get("m", 1))
    indices = hermite_window_indices(
        hermite_count,
        chunks=nm_chunks,
        ghost_depth=int(plan.hermite_ghost_depth),
    )
    windowed = hermite_window_cache_arrays(cache, indices, hermite_count=hermite_count)
    window_specs = hermite_cache_partition_specs(windowed, "m")
    species_cache = {
        name: getattr(cache, name)
        for name in _SPECIES_CACHE_NAMES
        if getattr(cache, name, None) is not None
    }
    species_specs = {
        name: PartitionSpec("s", *([None] * (value.ndim - 1)))
        for name, value in species_cache.items()
    }
    species_params = {
        name: _as_species_array(getattr(params, name), ns, name).astype(real_dtype)
        for name in _SPECIES_PARAM_NAMES
    }
    return indices, windowed, window_specs, species_cache, species_specs, species_params


def _fused_scalar_diagnostics(
    local,
    head,
    fields,
    cache,
    params,
    grid,
    *,
    vol_fac,
    flux_fac,
    mesh_axes,
    global_species,
):
    """Return the four gated scalar traces from state the RHS already built.

    Fused deliberately. Recomputing these outside the timed route costs up to
    118x the step because the observable path re-runs the bracket; here they are
    reductions over arrays that are already resident, followed by one scalar
    ``psum`` each. ``Wg`` reduces over both mesh axes (every shard owns a
    distinct slab of it); ``Wphi`` and the fluxes reduce over species only,
    because the head and the fields are already Hermite-complete on every shard.
    """

    zero = jnp.zeros_like(fields.phi)
    apar = fields.apar if fields.apar is not None else zero
    bpar = fields.bpar if fields.bpar is not None else zero

    def owner_sum(value):
        # The head and the fields are already Hermite-complete on every shard,
        # so every Hermite block holds the same species contribution. Masking to
        # one block and reducing over the whole mesh is exact -- the others add
        # a true zero -- and it leaves the result provably invariant over both
        # mesh axes, which a species-only psum does not.
        return jax.lax.psum(value * _mesh_owner_mask(mesh_axes, value), mesh_axes)

    wg = jax.lax.psum(distribution_free_energy(local, grid, params, vol_fac), mesh_axes)
    wphi = owner_sum(electrostatic_field_energy(fields.phi, cache, params, vol_fac))
    heat = owner_sum(
        jnp.sum(
            heat_flux_species(
                head, fields.phi, apar, bpar, cache, grid, params, flux_fac
            )
        )
    )
    particle = owner_sum(
        jnp.sum(
            particle_flux_species(
                head,
                fields.phi,
                apar,
                bpar,
                cache,
                grid,
                params,
                flux_fac,
                global_species=global_species,
            )
        )
    )
    return wg, wphi, heat, particle


def _mesh_owner_mask(mesh_axes, like):
    """Return 1 on the first Hermite block and 0 elsewhere, varying over the mesh.

    The trivially-true species factor is deliberate: it makes the mask -- and so
    the masked scalar -- formally vary over both mesh axes, which is what lets a
    single ``psum`` over the whole mesh type-check for a quantity that is
    already Hermite-complete.
    """

    species_axis, hermite_axis = mesh_axes
    mask = (jax.lax.axis_index(hermite_axis) == 0) & (
        jax.lax.axis_index(species_axis) >= 0
    )
    return mask.astype(jnp.asarray(like).dtype)


def _unfused(value):
    """Keep ``value`` in its own kernel instead of the consumer's.

    Measured on 2 x RTX A4000 (jax 0.11.1) at ``(2, 8, 16, 64, 64, 32)``. Left
    alone, XLA folds the whole linear operator -- every shifted, scaled copy of
    the slab that streaming, mirror, curvature and the diamagnetic drive
    contribute -- into the elementwise ``add`` that joins it to the nonlinear
    bracket, and emits one ``kLoop`` fusion with 29 operands that re-reads the
    shard once per shifted term. That fusion is the single most expensive
    kernel in the step and it runs at roughly 40% of the throughput the
    two-kernel form reaches.

    The cost model tips over on shard *shape*, not on shard size, which is why
    it is a sharding bug rather than a small-problem effect: on one device with
    two species the fusion costs 11.0 ms/step, and on the one-species shard a
    two-device mesh gives it, the same fusion costs 37.4 ms/step -- 3.4x longer
    on half the data. Per-species cost is otherwise flat (49.2, 49.1 and
    47.1 ms/step/species at Ns = 2, 3, 4) and jumps to 69.7 at Ns = 1, so a
    two-species run on two devices could never scale past 1.41x however cheap
    the collectives were. Splitting the bracket out takes the shard from
    69.7 to 46.7 ms/step and the two-device mesh from 75.8 to 47.4.

    ``optimization_barrier`` is an identity, so this cannot move the answer on
    its own; it can only change which multiply-adds XLA contracts, and the
    identity gates below cover that.
    """

    return jax.lax.optimization_barrier(value)


def _species_hermite_local_rhs(
    local,
    *,
    cache,
    params,
    term_cfg,
    linear_cfg,
    index,
    hermite_count,
    nm_chunks,
    ghost_depth,
    real_dtype,
    compressed_real_fft,
    laguerre_mode,
):
    """Evaluate the production nonlinear RHS on one shard of the mesh."""

    from gkx.operators.linear.cache_model import HermiteWindow
    from gkx.parallel.velocity_hermite import (
        hermite_field_moment_head,
        hermite_halo_extend,
        hermite_halo_interior,
    )
    from gkx.terms.assembly import assemble_rhs_cached_with_fields
    from gkx.terms.fields import _solve_fields_impl
    from gkx.terms.nonlinear import nonlinear_em_contribution

    # ``index`` covers the *extended* slab; the head is built from the rows the
    # shard actually owns, so it reads the interior window.
    interior = (
        slice(ghost_depth, ghost_depth + local.shape[2])
        if nm_chunks > 1
        else slice(None)
    )
    head = hermite_field_moment_head(
        local,
        index=index[interior],
        rows=(
            _DIAGNOSTIC_MOMENT_ROWS
            if nm_chunks > 1
            else min(_DIAGNOSTIC_MOMENT_ROWS, local.shape[2])
        ),
        chunks=nm_chunks,
        axis_name="m",
    )
    fields = _solve_fields_impl(
        head,
        cache,
        params,
        charge=params.charge_sign,
        density=params.density,
        temp=params.temp,
        mass=params.mass,
        tz=params.tz,
        vth=params.vth,
        fapar=jnp.asarray(params.fapar, dtype=real_dtype),
        w_bpar=jnp.asarray(term_cfg.bpar, dtype=real_dtype),
        axis_name="s",
    )
    extended = hermite_halo_extend(
        local, chunks=nm_chunks, ghost_depth=ghost_depth, axis_name="m"
    )
    window = HermiteWindow(index=index, total=hermite_count)
    dG = assemble_rhs_cached_with_fields(
        extended,
        cache,
        params,
        fields,
        terms=linear_cfg,
        force_electrostatic_fields=term_cfg.apar == 0.0 and term_cfg.bpar == 0.0,
        hermite_window=window,
    )
    dG = hermite_halo_interior(dG, chunks=nm_chunks, ghost_depth=ghost_depth)
    if term_cfg.nonlinear == 0.0:
        return dG, fields, head
    bracket = _unfused(
        nonlinear_em_contribution(
            local,
            phi=fields.phi,
            apar=fields.apar if term_cfg.apar != 0.0 else None,
            bpar=fields.bpar if term_cfg.bpar != 0.0 else None,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=params.tz,
            vth=params.vth,
            sqrt_m=cache.sqrt_m[:, interior],
            sqrt_m_p1=cache.sqrt_m_p1[:, interior],
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=jnp.asarray(term_cfg.nonlinear, dtype=real_dtype),
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )
    )
    return dG + bracket, fields, head


def stage_from_host(value: Any, sharding: Any) -> Any:
    """Place ``value`` on ``sharding`` from host memory rather than resharding.

    Handing ``device_put`` an array that is already committed to one device asks
    the runtime to reshard it across the mesh, and on the two-GPU box that path
    silently produces a wrong answer: the sharded route drops to a max relative
    error of 1.0 against serial while a one-device mesh on the same GPU is exact.
    Round-tripping through the host is the reliable placement, and it happens
    once per run rather than once per step.
    """

    if isinstance(value, jax.core.Tracer):
        return value
    if getattr(value, "sharding", None) == sharding:
        # Already exactly where it belongs: re-staging would only add a host
        # round trip per call, which on a repeated route dominates the step.
        return value
    return jax.device_put(np.asarray(jax.device_get(value)), sharding)


def _reject_unsharded_hermite_terms(term_cfg, params, plan) -> None:
    """Fail closed on the one term family a split Hermite axis cannot serve yet.

    The conserving-collision correction reads the ``m = 0, 1, 2`` moments of the
    *local* slab. Those are the global moments only on the Hermite block that
    owns them, so with the Hermite axis split the operator would quietly become a
    different one on every other shard. Species sharding is unaffected -- and
    species-first factoring means a two-device box with two species never enters
    this branch.
    """

    if int(plan.chunks.get("m", 1)) < 2:
        return
    nu = np.asarray(jax.device_get(getattr(params, "nu", 0.0)), dtype=float)
    if float(term_cfg.collisions) == 0.0 or not np.any(nu != 0.0):
        return
    raise NotImplementedError(
        "conserving collisions need the m = 0, 1, 2 moments summed across the "
        "Hermite mesh axis, which this route does not yet reduce; run with "
        f"{int(plan.chunks.get('s', 1))} devices for a species-only mesh, or "
        "with terms.collisions = 0."
    )


def _resolve_species_hermite_placement(
    G0, cache, params, *, plan, devices, num_devices
):
    """Return the mesh, plan, and every sharded leaf for one placement."""

    from gkx.parallel.state import (
        build_species_hermite_mesh,
        species_hermite_state_spec,
    )
    from gkx.parallel.velocity_plan import build_species_hermite_mesh_plan
    from gkx.solvers_linear_parallel_common import _resolve_parallel_devices

    if G0.ndim != 6:
        raise ValueError(
            "the species x Hermite route requires a 6D (Ns, Nl, Nm, Nky, Nkx, Nz) state"
        )
    device_list = (
        list(devices)
        if devices is not None
        else list(_resolve_parallel_devices(num_devices=num_devices))
    )
    resolved = plan or build_species_hermite_mesh_plan(
        tuple(G0.shape), num_devices=len(device_list)
    )
    mesh = build_species_hermite_mesh(resolved, devices=device_list)
    return mesh, resolved, species_hermite_state_spec(6)


def _species_hermite_mapped(
    G0,
    cache,
    params,
    grid,
    *,
    term_cfg,
    plan,
    mesh,
    state_spec,
    compressed_real_fft,
    laguerre_mode,
    vol_fac,
    flux_fac,
):
    """Return the shard-mapped step callable plus its sharded argument tuple."""

    import jax.sharding as jsharding

    ns = int(G0.shape[0])
    hermite_count = int(G0.shape[2])
    real_dtype = jnp.real(jnp.empty((), dtype=G0.dtype)).dtype
    nm_chunks = int(plan.chunks.get("m", 1))
    ghost_depth = int(plan.hermite_ghost_depth)
    (
        indices,
        windowed,
        window_specs,
        species_cache,
        species_specs,
        species_params,
    ) = _species_hermite_inputs(cache, params, ns, real_dtype, plan, hermite_count)
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    index_spec = jsharding.PartitionSpec("m")
    names = tuple(species_cache) + tuple(windowed) + tuple(_SPECIES_PARAM_NAMES)
    values = (
        tuple(species_cache.values())
        + tuple(windowed.values())
        + tuple(species_params[name] for name in _SPECIES_PARAM_NAMES)
    )
    specs = (
        tuple(species_specs[name] for name in species_cache)
        + tuple(window_specs[name] for name in windowed)
        + (jsharding.PartitionSpec("s"),) * len(_SPECIES_PARAM_NAMES)
    )
    n_cache = len(species_cache) + len(windowed)
    mesh_axes = ("s", "m")

    def local_step(local, index, *leaves):
        shard_cache = replace(cache, **dict(zip(names[:n_cache], leaves[:n_cache])))
        shard_params = replace(
            params, **dict(zip(_SPECIES_PARAM_NAMES, leaves[n_cache:], strict=True))
        )
        dG, fields, head = _species_hermite_local_rhs(
            local,
            cache=shard_cache,
            params=shard_params,
            term_cfg=term_cfg,
            linear_cfg=linear_cfg,
            index=index,
            hermite_count=hermite_count,
            nm_chunks=nm_chunks,
            ghost_depth=ghost_depth,
            real_dtype=real_dtype,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )
        scalars = (
            _fused_scalar_diagnostics(
                local,
                head,
                fields,
                shard_cache,
                shard_params,
                grid,
                vol_fac=vol_fac,
                flux_fac=flux_fac,
                mesh_axes=mesh_axes,
                global_species=ns,
            )
            if vol_fac is not None and flux_fac is not None
            else None
        )
        return dG, scalars

    sharded = tuple(
        stage_from_host(value, jsharding.NamedSharding(mesh, spec))
        for value, spec in zip(values, specs, strict=True)
    )
    index_array = jax.device_put(
        np.asarray(indices, dtype=np.int32),
        jsharding.NamedSharding(mesh, index_spec),
    )
    return local_step, (index_array,) + sharded, (index_spec,) + specs, state_spec


def species_hermite_nonlinear_rhs(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    plan: Any | None = None,
    devices: Any | None = None,
    num_devices: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> jnp.ndarray:
    """Evaluate one production nonlinear RHS on a species x Hermite mesh."""

    term_cfg = terms or TermConfig()
    state_dtype = jnp.result_type(G0, jnp.complex64)
    state = jnp.asarray(G0, dtype=state_dtype)
    mesh, resolved, state_spec = _resolve_species_hermite_placement(
        state, cache, params, plan=plan, devices=devices, num_devices=num_devices
    )
    _reject_unsharded_hermite_terms(term_cfg, params, resolved)
    local_step, leaves, specs, state_spec = _species_hermite_mapped(
        state,
        cache,
        params,
        None,
        term_cfg=term_cfg,
        plan=resolved,
        mesh=mesh,
        state_spec=state_spec,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        vol_fac=None,
        flux_fac=None,
    )

    def rhs_only(local, *rest):
        return local_step(local, *rest)[0]

    mapped = jax.shard_map(
        rhs_only,
        mesh=mesh,
        in_specs=(state_spec,) + specs,
        out_specs=state_spec,
        axis_names={"s", "m"},
    )
    from jax.sharding import NamedSharding

    return jax.jit(mapped)(
        stage_from_host(state, NamedSharding(mesh, state_spec)), *leaves
    )


_TRACE_NAMES = ("Wg_t", "Wphi_t", "heat_flux_t", "particle_flux_t")


def integrate_nonlinear_species_hermite(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str = "rk2",
    terms: TermConfig | None = None,
    grid: Any | None = None,
    vol_fac: jnp.ndarray | None = None,
    flux_fac: jnp.ndarray | None = None,
    plan: Any | None = None,
    devices: Any | None = None,
    num_devices: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> SpeciesHermiteRun:
    """Integrate the nonlinear system on a species x Hermite ``shard_map`` mesh.

    The whole trajectory is one jitted ``lax.scan`` on the mesh, and the scalar
    traces are stacked from inside that scan rather than recomputed afterwards.
    Supplying ``grid`` with ``vol_fac``/``flux_fac`` turns the traces on; without
    them the route is compute-only and returns an empty trace map.
    """

    _validate_steps(steps)
    method_key = _validate_explicit_method(method)
    term_cfg = terms or TermConfig()
    state_dtype = jnp.result_type(G0, jnp.complex64)
    state = jnp.asarray(G0, dtype=state_dtype)
    mesh, resolved, state_spec = _resolve_species_hermite_placement(
        state, cache, params, plan=plan, devices=devices, num_devices=num_devices
    )
    _reject_unsharded_hermite_terms(term_cfg, params, resolved)
    record = grid is not None and vol_fac is not None and flux_fac is not None
    local_step, leaves, specs, state_spec = _species_hermite_mapped(
        state,
        cache,
        params,
        grid,
        term_cfg=term_cfg,
        plan=resolved,
        mesh=mesh,
        state_spec=state_spec,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        vol_fac=vol_fac if record else None,
        flux_fac=flux_fac if record else None,
    )
    projector = (
        _make_compressed_real_fft_projector(
            ny_full=int(cache.ky.size), nx=int(cache.kx.size)
        )
        if compressed_real_fft
        else None
    )
    dt_val = _dt_array(dt, state_dtype)
    scalar_spec = jax.sharding.PartitionSpec()

    def local_scan(local, index, *rest):
        def rhs(value):
            return local_step(value, index, *rest)

        def stage(value, increment, scale):
            nxt = value + jnp.asarray(scale, dtype=dt_val.dtype) * dt_val * increment
            return _project_local(nxt, projector, state_dtype)

        def step(carry, _unused):
            value = _project_local(carry, projector, state_dtype)
            k1, scalars = rhs(value)
            nxt = _nonlinear_explicit_update(
                method_key,
                value,
                k1,
                rhs=lambda arg: (rhs(arg)[0], None),
                stage=stage,
                project_shard=lambda arg: _project_local(arg, projector, state_dtype),
                dt_val=dt_val,
            )
            nxt = _project_local(nxt, projector, state_dtype)
            return nxt, (scalars if record else None)

        final, stacked = jax.lax.scan(step, local, xs=None, length=steps)
        return (final, stacked) if record else (final, None)

    out_specs = (
        (state_spec, (scalar_spec,) * len(_TRACE_NAMES))
        if record
        else (state_spec, None)
    )
    mapped = jax.shard_map(
        local_scan,
        mesh=mesh,
        in_specs=(state_spec,) + specs,
        out_specs=out_specs,
        axis_names={"s", "m"},
    )
    from jax.sharding import NamedSharding

    final, stacked = jax.jit(mapped)(
        stage_from_host(state, NamedSharding(mesh, state_spec)), *leaves
    )
    traces = dict(zip(_TRACE_NAMES, stacked, strict=True)) if record else {}
    return SpeciesHermiteRun(
        state=final,
        traces=traces,
        plan=resolved,
        mesh_shape=(
            int(resolved.chunks.get("s", 1)),
            int(resolved.chunks.get("m", 1)),
        ),
    )


def _project_local(value, projector, dtype):
    """Apply the Hermitian projector to one shard, if the run uses one."""

    if projector is not None:
        value = projector(value)
    return jnp.asarray(value, dtype=dtype)


__all__ = [
    "SpeciesHermiteRun",
    "stage_from_host",
    "integrate_linear_sharded",
    "integrate_nonlinear_sharded",
    "integrate_nonlinear_species_hermite",
    "species_hermite_nonlinear_rhs",
]
