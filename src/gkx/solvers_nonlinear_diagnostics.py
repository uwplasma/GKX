"""Nonlinear integration: cached RHS, state integrators and diagnostic drivers.

The explicit and IMEX diagnostic drivers call the kernels as module-level
names, so tests patch them on this module.
"""

from __future__ import annotations

import contextlib
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import warnings
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, NamedTuple
from gkx.config import method_error, resolve_cfl_fac
from gkx.core_grid import SpectralGrid, _gyrokinetic_moment_shape
from gkx.core_ky_layout import source_ny_full
from gkx.diagnostics_contract import ResolvedDiagnostics, SimulationDiagnostics
from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.operators.collision import CollisionOperator
from gkx.operators.fluxes import (
    heat_flux_species,
    heat_flux_total,
    particle_flux_species,
    turbulent_heating_species,
)
from gkx.operators.linear.cache_builder import (
    build_linear_cache,
    update_linear_cache_for_sheared_kx,
)
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.linked import (
    linked_cover_mask_from_cache,
    mask_supplied_state,
    project_to_linked_cover,
)
from gkx.operators.linear.params import LinearParams, _x64_enabled
from gkx.operators.moments import (
    distribution_free_energy,
    distribution_free_energy_resolved,
    electrostatic_field_energy,
    electrostatic_field_energy_resolved,
    fieldline_quadrature_weights,
    heat_flux_channel_resolved_species,
    magnetic_vector_potential_energy,
    magnetic_vector_potential_energy_resolved,
    particle_flux_channel_resolved_species,
    phi2_resolved,
    turbulent_heating_resolved_species,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
)
from gkx.operators.nonlinear.collisions import (
    _apply_collision_split,
    _collision_damping,
)
from gkx.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    make_nonlinear_diagnostic_tuple_fn,
)
from gkx.operators.nonlinear.policies import (
    IMEXLinearOperator,
    _diagnostic_omega_mode_mask,
    _nonlinear_cfl_frequency_components,
    build_nonlinear_collision_split_policy,
    build_nonlinear_diagnostic_setup,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)
from gkx.operators.nonlinear.projection import (
    _make_compressed_real_fft_projector,
    _make_hermitian_projector,
    advance_shearing_coordinates,
    hermitian_projector_for_signature,
    hermitian_projector_signature,
)
from gkx.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from gkx.solvers_linear_implicit import (
    ImplicitSolveStats,
    _build_implicit_operator,
    _empty_implicit_solve_stats,
)
from gkx.solvers_nonlinear_explicit import (
    advance_explicit_nonlinear_state,
    cached_jit,
    checkpointed_explicit_scan,
    integrate_nonlinear_scan,
    make_explicit_diagnostic_step,
    run_explicit_diagnostic_scan,
)
from gkx.solvers_nonlinear_imex import (
    DiagnosticFn,
    DiagnosticStepFn,
    NonlinearTermFn,
    ProjectFn,
    SolveStepFn,
    StatsSolveStepFn,
    collisions_bind,
    integrate_cached_imex_scan,
    make_imex_diagnostic_step,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    make_imex_solve_step_with_stats,
    run_imex_diagnostic_scan,
    solve_imex_step,
    solve_imex_step_with_stats,
)
from gkx.solvers_time_explicit import (
    _diagnostic_midplane_index,
    _instantaneous_growth_rate_step,
    _laguerre_velocity_max,
    _linear_frequency_bound,
)
from gkx.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from gkx.terms.config import FieldState, TermConfig
from gkx.terms.nonlinear import nonlinear_em_contribution


@dataclass(frozen=True)
class _ExplicitPreparedState:
    term_cfg: TermConfig
    setup: Any
    cache: Any
    project_state: Callable[..., Any]
    G0: jnp.ndarray
    state_dtype: Any
    real_dtype: Any


@dataclass(frozen=True)
class _ExplicitRuntimePolicies:
    time_step_policy: Any
    collision_policy: Any


@dataclass(frozen=True)
class _ExplicitDiagnosticOptions:
    method: str
    dt: float
    steps: int
    checkpoint: bool
    sample_stride: int
    diagnostics_stride: int
    use_dealias_mask: bool
    z_index: int | None
    compressed_real_fft: bool
    laguerre_mode: str
    omega_ky_index: int | None
    omega_kx_index: int | None
    flux_scale: float
    wphi_scale: float
    fixed_dt: bool
    dt_min: float
    dt_max: float | None
    time_horizon: float | None
    cfl: float
    cfl_fac: float | None
    collision_split: bool
    collision_scheme: str
    fixed_mode_ky_index: int | None
    fixed_mode_kx_index: int | None
    external_phi: jnp.ndarray | float | None
    resolved_diagnostics: bool
    show_progress: bool


@dataclass(frozen=True)
class _ExplicitScanComponents:
    prepared: _ExplicitPreparedState
    policies: _ExplicitRuntimePolicies
    step: Callable[..., Any]
    compute_diag_from_state: Callable[..., Any]
    cover_mask: Any = None


@dataclass(frozen=True)
class PreparedExplicitNonlinearDiagnostics:
    """Reusable compiled explicit nonlinear diagnostic simulation.

    Geometry, field operators, and static numerical policy are prepared once.
    Calls to :meth:`run` may supply a new initial state with the same shape and
    dtype without rebuilding the scan closure. Fixed-step sensitivity studies
    may instead pass matched geometry, cache, and parameter PyTrees.

    This object and :func:`integrate_nonlinear_explicit_diagnostics_state`
    compile one graph and return bitwise identical arrays for the same inputs
    (plan §5.3, queue row Q18). The one difference that remains is the state
    dtype: this object freezes the dtype of the ``G0`` it was prepared with and
    casts every later state to it, while the function entry point takes the
    dtype of the ``G0`` it is called with. Feeding a promoted state to one and
    not the other is a precision change, not a route difference.

    **Supplied states** (queue row Q23). On a linked (twist-shift) deck the
    ``kx`` rows outside the linked chains are decoupled, and the runtime's own
    initial conditions never populate them. Every state this object is given --
    the ``G0`` it was prepared with and any ``initial_state`` passed to
    :meth:`run` or :meth:`run_arrays` -- is projected onto the chain cover
    before it is integrated, because the ExB bracket takes the whole state to
    real space and dealiases only its output, so off-chain content aliases back
    onto the chain rows. The projection is a mask fixed by the deck's topology,
    not a check on the state's values, so it holds under ``jit`` and under
    reverse-mode AD; gradients with respect to a supplied state are exactly
    zero on the off-chain rows and unchanged elsewhere. Periodic and full-cover
    decks are untouched and trace exactly as before.
    """

    initial_state: jnp.ndarray
    geometry: Any
    cache: LinearCache
    params: LinearParams
    _run_raw: Callable[..., tuple[Any, Any, Any]]
    _run_dynamic_raw: Callable[
        [jnp.ndarray, Any, LinearCache, LinearParams], tuple[Any, Any, Any]
    ]
    _finalize: Callable[..., SimulationDiagnostics]
    stride: int
    sampled_scan: bool
    resolved_diagnostics: bool
    fixed_dt: bool
    cfl_scales: Any = None
    cover_mask: Any = None

    def run_arrays(
        self,
        initial_state: jnp.ndarray | None = None,
        *,
        geometry: Any | None = None,
        cache: LinearCache | None = None,
        params: LinearParams | None = None,
        time_horizon: float | None = None,
    ) -> tuple[jnp.ndarray, tuple[Any, Any, Any], FieldState]:
        """Run the compiled scan without host conversion or artifact assembly.

        This method is the differentiable Python boundary. The initial state is
        dynamic. Fixed-step runs may also receive a matched ``cache``/``params``
        pair for parameter differentiation. A changed ``geometry`` requires
        that pair. Grid layout and numerical policy remain fixed by
        :func:`prepare_nonlinear_explicit_diagnostics`; adaptive runs currently
        support state changes but reject traced model overrides.

        A supplied ``initial_state`` is projected onto the linked chain cover
        first, as the class docstring describes. The prepared state was
        projected once at preparation, so passing ``None`` runs exactly the
        graph it ran before.

        ``time_horizon`` (adaptive runs) stops stepping once the run time
        reaches it. It is a traced operand, so changing it between calls
        reuses the compiled scan; it overrides any prepared horizon.
        """

        if (cache is None) != (params is None):
            raise ValueError("cache and params must be supplied together")
        if geometry is not None and cache is None:
            raise ValueError(
                "dynamic geometry requires matched cache and params inputs"
            )
        if initial_state is None:
            state = self.initial_state
        else:
            # The cover belongs to the operator this call will use, so a
            # dynamic cache supplies its own instead of reusing the prepared
            # one; on a periodic or full-cover deck both are ``None``.
            state = (
                mask_supplied_state(jnp.asarray(initial_state), cache)
                if cache is not None
                else project_to_linked_cover(
                    jnp.asarray(initial_state), self.cover_mask
                )
            )
        if geometry is None and cache is None:
            if time_horizon is None:
                return self._run_raw(jnp.asarray(state))
            return self._run_raw(jnp.asarray(state), jnp.asarray(time_horizon))
        if time_horizon is not None:
            raise ValueError("time_horizon cannot be combined with dynamic inputs")
        if not self.fixed_dt:
            raise ValueError("dynamic geometry, cache, or params require fixed_dt=True")
        geometry_use = self.geometry if geometry is None else geometry
        cache_use = self.cache if cache is None else cache
        params_use = self.params if params is None else params
        return self._run_dynamic_raw(
            jnp.asarray(state), geometry_use, cache_use, params_use
        )

    @property
    def chain(self) -> Any:
        """The chain implicit operator the scan takes as an operand, if any."""
        return getattr(self._run_raw, "keywords", {}).get("chain")

    def with_chain(self, chain: Any) -> PreparedExplicitNonlinearDiagnostics:
        """Same compiled scan with another operator of the same structure."""
        run = self._run_raw
        return replace(self, _run_raw=partial(run.func, chain=chain))  # type: ignore[attr-defined]

    def run(
        self,
        initial_state: jnp.ndarray | None = None,
        *,
        geometry: Any | None = None,
        cache: LinearCache | None = None,
        params: LinearParams | None = None,
        time_horizon: float | None = None,
    ) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
        """Advance one state through the prepared compiled simulation."""

        G_final, scan_diag_out, fields_final = self.run_arrays(
            initial_state,
            geometry=geometry,
            cache=cache,
            params=params,
            time_horizon=time_horizon,
        )
        diag, t, dt_series = scan_diag_out
        diag_out = self._finalize(
            diag,
            t=t,
            dt_series=dt_series,
            stride=self.stride,
            sampled_scan=self.sampled_scan,
            resolved_diagnostics=self.resolved_diagnostics,
            cfl_scales=self.cfl_scales,
        )
        return jnp.asarray(diag_out.t), diag_out, G_final, fields_final


_EXPLICIT_OPTION_FIELDS = tuple(_ExplicitDiagnosticOptions.__annotations__)


def _explicit_options_from_values(values: dict[str, Any]) -> _ExplicitDiagnosticOptions:
    """Pack public keyword values into a single internal options object."""

    return _ExplicitDiagnosticOptions(
        **{key: values[key] for key in _EXPLICIT_OPTION_FIELDS}
    )


def _prepare_explicit_diagnostic_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    *,
    method: str,
    cache: LinearCache | None,
    terms: TermConfig | None,
    use_dealias_mask: bool,
    z_index: int | None,
    compressed_real_fft: bool,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
) -> _ExplicitPreparedState:
    """Prepare geometry, cache, projection, and dtype state for explicit scans."""

    term_cfg = terms or TermConfig()
    if method == "imex":
        raise ValueError(
            "Final-state runtime diagnostics helper only supports explicit methods"
        )
    setup = build_nonlinear_diagnostic_setup(
        G0,
        grid,
        geom,
        params,
        cache=cache,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        compressed_real_fft=compressed_real_fft,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        ensure_geometry_fn=ensure_flux_tube_geometry_data,
        build_cache_fn=build_linear_cache,
        quadrature_weights_fn=fieldline_quadrature_weights,
        omega_mask_fn=_diagnostic_omega_mode_mask,
        midplane_index_fn=_diagnostic_midplane_index,
    )
    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0_projected = setup.project_state(jnp.asarray(G0, dtype=state_dtype))
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    return _ExplicitPreparedState(
        term_cfg=term_cfg,
        setup=setup,
        cache=setup.cache,
        project_state=setup.project_state,
        G0=G0_projected,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
    )


def _build_explicit_runtime_policies(
    prepared: _ExplicitPreparedState,
    grid: SpectralGrid,
    params: LinearParams,
    *,
    method: str,
    dt: float,
    steps: int,
    fixed_dt: bool,
    dt_min: float,
    dt_max: float | None,
    cfl: float,
    cfl_fac: float | None,
    compressed_real_fft: bool,
    collision_split: bool,
) -> _ExplicitRuntimePolicies:
    """Build timestep and collision-splitting policies for explicit scans."""

    time_step_policy = build_nonlinear_time_step_policy(
        grid,
        prepared.setup.geom,
        params,
        prepared.cache,
        method=method,
        dt=dt,
        steps=steps,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        compressed_real_fft=compressed_real_fft,
        real_dtype=prepared.real_dtype,
        resolve_cfl_fac_fn=resolve_cfl_fac,
        linear_frequency_bound_fn=_linear_frequency_bound,
        laguerre_velocity_max_fn=_laguerre_velocity_max,
        cfl_frequency_components_fn=_nonlinear_cfl_frequency_components,
    )
    collision_policy = build_nonlinear_collision_split_policy(
        prepared.cache,
        params,
        prepared.term_cfg,
        prepared.real_dtype,
        squeeze_species=prepared.G0.ndim == 5,
        collision_split=collision_split,
        collision_damping_fn=_collision_damping,
    )
    return _ExplicitRuntimePolicies(
        time_step_policy=time_step_policy,
        collision_policy=collision_policy,
    )


def _make_explicit_rhs_fn(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    params: LinearParams,
    *,
    compressed_real_fft: bool,
    laguerre_mode: str,
    external_phi: jnp.ndarray | float | None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the nonlinear RHS closure with collision policy baked in."""

    def rhs_fn(G):
        return nonlinear_rhs_cached(
            G,
            prepared.cache,
            params,
            policies.collision_policy.rhs_terms,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            external_phi=external_phi,
        )

    return rhs_fn


def _make_explicit_diagnostic_callable(
    prepared: _ExplicitPreparedState,
    grid: SpectralGrid,
    params: LinearParams,
    *,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
    resolved_diagnostics: bool,
) -> Callable[..., Any]:
    """Return the state-to-diagnostic tuple closure for explicit scans."""

    return make_nonlinear_diagnostic_tuple_fn(
        grid=grid,
        cache=prepared.cache,
        params=params,
        vol_fac=prepared.setup.vol_fac,
        flux_fac=prepared.setup.flux_fac,
        mask=prepared.setup.mask,
        z_idx=prepared.setup.z_idx,
        use_dealias=prepared.setup.use_dealias,
        real_dtype=prepared.real_dtype,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        resolved_diagnostics=resolved_diagnostics,
        kernels=_nonlinear_diagnostic_kernels(),
    )


def _make_explicit_scan_step(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    rhs_fn: Callable[..., Any],
    compute_diag_from_state: Callable[..., Any],
    params: LinearParams,
    *,
    method: str,
    diagnostics_stride: int,
    show_progress: bool,
    steps: int,
    external_phi: jnp.ndarray | float | None,
    collision_scheme: str,
    time_horizon: float | None,
) -> Callable[..., Any]:
    """Build the explicit scan-step closure from prepared policies."""

    return make_explicit_diagnostic_step(
        rhs_fn=rhs_fn,
        method=method,
        project_state=prepared.project_state,
        state_dtype=prepared.state_dtype,
        real_dtype=prepared.real_dtype,
        time_step_policy=policies.time_step_policy,
        compute_fields_fn=compute_fields_cached,
        cache=prepared.cache,
        params=params,
        term_cfg=prepared.term_cfg,
        external_phi=external_phi,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=diagnostics_stride,
        select_diagnostics_fn=select_nonlinear_step_diagnostics,
        show_progress=show_progress,
        steps=steps,
        emit_progress_fn=maybe_emit_nonlinear_progress,
        use_collision_split=policies.collision_policy.active,
        damping=policies.collision_policy.damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=_apply_collision_split,
        time_horizon=time_horizon,
    )


def _run_explicit_diagnostic_scan_and_finalize(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    step: Callable[..., Any],
    compute_diag_from_state: Callable[..., Any],
    params: LinearParams,
    *,
    steps: int,
    sample_stride: int,
    diagnostics_stride: int,
    checkpoint: bool,
    resolved_diagnostics: bool,
    external_phi: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Run the explicit scan and convert raw scan output into diagnostics.

    The raw scan runs under ``jax.jit``, which is the reference-route contract
    (plan §5.3, queue row Q18). ``prepare_explicit_nonlinear_diagnostics_impl``
    jits the same function over the same components, so this entry point --
    the one ``run_runtime_nonlinear`` reaches on its fixed-window, chunked and
    sharded routes -- and ``gkx.prepare`` compile one graph and return bitwise
    identical arrays.

    Run outside jit, as this did before, XLA compiles the scan primitive alone
    with the arrays its body closes over as operands, optimizes the field solve
    and the first diagnostic as separate modules, and the two routes then agree
    only to roundoff: on a 100-step Cyclone gate the state differed by up to
    2.1e-7 relative in float32, and two near-cancelling turbulent-heating
    diagnostics by O(1).
    """

    def run_raw(
        initial_state: jnp.ndarray, chain: Any
    ) -> tuple[Any, tuple[Any, Any, Any], Any]:
        with _bound_chain(step, chain):
            return _run_explicit_diagnostic_scan_raw(
                prepared,
                policies,
                step,
                compute_diag_from_state,
                params,
                initial_state=initial_state,
                steps=steps,
                sample_stride=sample_stride,
                diagnostics_stride=diagnostics_stride,
                checkpoint=checkpoint,
                external_phi=external_phi,
            )

    G_final, scan_diag_out, fields_final = cached_jit(run_raw)(
        prepared.G0, _chain_operand(step)
    )
    diag, t, dt_series = scan_diag_out
    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
    diag_out = finalize_nonlinear_scan_diagnostics(
        diag,
        t=t,
        dt_series=dt_series,
        stride=stride,
        sampled_scan=sampled_scan,
        resolved_diagnostics=resolved_diagnostics,
        cfl_scales=getattr(policies.time_step_policy, "cfl_scales", None),
    )
    return jnp.asarray(diag_out.t), diag_out, G_final, fields_final


def _run_explicit_diagnostic_scan_raw(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    step: Callable[..., Any],
    compute_diag_from_state: Callable[..., Any],
    params: LinearParams,
    *,
    initial_state: jnp.ndarray,
    steps: int,
    sample_stride: int,
    diagnostics_stride: int,
    checkpoint: bool,
    external_phi: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, tuple[Any, Any, Any], FieldState]:
    """Run the device scan and return raw arrays for outside-JIT packaging."""

    G0 = prepared.project_state(jnp.asarray(initial_state, dtype=prepared.state_dtype))

    fields0 = compute_fields_cached(
        G0,
        prepared.cache,
        params,
        terms=prepared.term_cfg,
        external_phi=external_phi,
    )
    dt0 = jnp.asarray(
        policies.time_step_policy.update_dt(fields0, policies.time_step_policy.dt_init),
        dtype=prepared.real_dtype,
    )
    diag_zero = cached_jit(compute_diag_from_state)(G0, fields0, G0, fields0, dt0)
    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
    G_final, scan_diag_out = run_explicit_diagnostic_scan(
        step,
        (
            G0,
            G0,
            fields0,
            diag_zero,
            jnp.asarray(0.0, dtype=prepared.real_dtype),
            dt0,
        ),
        steps=steps,
        stride=stride,
        sampled_scan=sampled_scan,
        checkpoint=checkpoint,
        sampled_scan_fn=run_sampled_explicit_diagnostic_scan,
    )

    fields_final = compute_fields_cached(
        G_final,
        prepared.cache,
        params,
        terms=prepared.term_cfg,
        external_phi=external_phi,
    )
    return G_final, scan_diag_out, fields_final


def stiff_linear_split(
    cache: Any, params: Any, terms: Any, compute_fields_fn: Callable[..., Any]
) -> tuple[Callable, Callable, Callable | None, Callable]:
    """Return ``(split_rhs, fields, mirror_rhs, collision_rhs)`` for :func:`build_chain_implicit_linear`.

    Implicit: streaming with its dissipation, the field response and, for
    species much faster than the slowest (electrons), the mirror force, whose
    explicit bound otherwise sets dt (|lambda| 57 against drifts ~3 on Cyclone
    at (4, 8)). It couples Laguerre moments, so those species get ``Nl N``
    blocks. Drifts, the ion mirror and drive stay explicit with the bracket.
    ``mirror_rhs`` is the mirror part of ``split_rhs`` (``None`` without a fast
    species); ``collision_rhs`` is the collision operator alone.
    """
    from gkx.terms.assembly import assemble_rhs_cached_with_fields
    from gkx.terms.config import FieldState

    vth = np.atleast_1d(np.asarray(params.vth, dtype=float))
    fast = (vth > 4.0 * vth.min()).astype(float)
    mirror = jnp.asarray(fast * float(terms.mirror)).reshape(-1, 1, 1, 1, 1, 1)
    off = dict(curvature=0.0, gradb=0.0, diamagnetic=0.0, collisions=0.0, nonlinear=0.0)
    implicit_terms = dataclasses.replace(terms, mirror=mirror, **off)
    mirror_terms = dataclasses.replace(
        implicit_terms,
        streaming=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
    )
    collision_terms = dataclasses.replace(
        mirror_terms, mirror=0.0, collisions=terms.collisions
    )

    def fields(state: jnp.ndarray) -> jnp.ndarray:
        f = compute_fields_fn(state, cache, params, terms=terms)
        zero = jnp.zeros_like(f.phi)
        return jnp.stack(
            [f.phi]
            + [
                zero if a is None else jnp.asarray(a, f.phi.dtype)
                for a in (f.apar, f.bpar)
            ]
        )

    def rhs(term_cfg):
        def split(state: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
            return assemble_rhs_cached_with_fields(
                state, cache, params, FieldState(F[0], F[1], F[2]), terms=term_cfg
            )

        return split

    mirror_rhs = rhs(mirror_terms) if fast.any() else None
    return rhs(implicit_terms), fields, mirror_rhs, rhs(collision_terms)


def species_moments(cache: Any) -> tuple[tuple[int, int], ...] | None:
    """``(Nl, Nm)`` each species keeps in the zero-padded state (``None``: all alike)."""
    return (
        None
        if cache.groups is None
        else tuple(
            (c.Jl.shape[1], c.m.size)
            for c in cache.groups
            for _ in range(c.Jl.shape[0])
        )
    )


def _attach_chain_implicit_linear(
    rhs_fn: Callable[..., Any],
    prepared: _ExplicitPreparedState,
    *,
    params: LinearParams,
    terms: Any,
    compute_fields_fn: Callable[..., Any],
    fixed_dt: bool,
    dt: float,
    scheme: str,
    implicit_collisions: bool = False,
) -> Callable[..., Any]:
    """Return ``rhs_fn`` carrying the per-chain implicit operator of imex-ars*."""

    from gkx.solvers_nonlinear_imex import build_chain_implicit_linear

    if not fixed_dt:
        raise ValueError(
            f"method '{scheme}' factors I - gamma dt L once: set fixed_dt = true "
            "and choose dt from the nonlinear rate (examples/05_kinetic_electrons)"
        )
    G0 = jnp.asarray(prepared.G0)
    if G0.ndim != 6:
        raise ValueError(
            f"method '{scheme}' needs a (species, Nl, Nm, ky, kx, z) state"
        )
    split_rhs, fields, mirror_rhs, collision_rhs = stiff_linear_split(
        prepared.cache, params, terms, compute_fields_fn
    )
    chain_linear = build_chain_implicit_linear(
        split_rhs,
        fields,
        tuple(G0.shape),
        float(dt),
        modes=np.broadcast_to(
            (np.asarray(prepared.cache.ky) >= 0.0)[:, None], G0.shape[3:5]
        ),
        scheme=scheme,
        dtype=G0.dtype,
        mirror_rhs=mirror_rhs,
        collision_rhs=collision_rhs if implicit_collisions else None,
        moments=species_moments(prepared.cache),
    )

    def rhs_with_chain_linear(state: jnp.ndarray) -> Any:
        return rhs_fn(state)

    rhs_with_chain_linear.chain_implicit_linear = chain_linear  # type: ignore[attr-defined]
    # Host-side, for ImexDtControl (the RHS without the chain operand).
    object.__setattr__(chain_linear, "_rhs", (rhs_fn, prepared.project_state))
    return rhs_with_chain_linear


def _build_explicit_scan_closures(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    grid: SpectralGrid,
    params: LinearParams,
    *,
    options: _ExplicitDiagnosticOptions,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build diagnostic scan closures after state and policies are prepared."""

    chain_terms = policies.collision_policy.rhs_terms
    # Chain IMEX: when explicit collisions would bound dt, the Landau operator is
    # a local implicit sub-step instead of part of the RHS.
    implicit_collisions = (
        options.method in {"imex-ars3", "imex-ark3"}
        and options.collision_scheme == "implicit"
        and options.fixed_dt
        and getattr(prepared.cache, "landau_pairs", None) is not None
        and float(chain_terms.collisions) != 0.0
        and jnp.asarray(prepared.G0).ndim == 6
    )
    if implicit_collisions:
        _, fields, _, coll = stiff_linear_split(
            prepared.cache, params, chain_terms, compute_fields_cached
        )
        G0 = jnp.asarray(prepared.G0)
        implicit_collisions = collisions_bind(
            coll,
            fields,
            tuple(G0.shape),
            G0.dtype,
            options.dt,
            species_moments(prepared.cache),
        )
    if implicit_collisions:
        policies = replace(
            policies,
            collision_policy=replace(
                policies.collision_policy,
                rhs_terms=replace(chain_terms, collisions=0.0),
            ),
        )
    rhs_fn = _make_explicit_rhs_fn(
        prepared,
        policies,
        params,
        compressed_real_fft=options.compressed_real_fft,
        laguerre_mode=options.laguerre_mode,
        external_phi=options.external_phi,
    )
    if options.method in {"imex-ars3", "imex-ark3"}:
        rhs_fn = _attach_chain_implicit_linear(
            rhs_fn,
            prepared,
            params=params,
            terms=chain_terms,
            compute_fields_fn=compute_fields_cached,
            fixed_dt=options.fixed_dt,
            dt=options.dt,
            scheme=options.method,
            implicit_collisions=implicit_collisions,
        )
    compute_diag_from_state = _make_explicit_diagnostic_callable(
        prepared,
        grid,
        params=params,
        omega_ky_index=options.omega_ky_index,
        omega_kx_index=options.omega_kx_index,
        flux_scale=options.flux_scale,
        wphi_scale=options.wphi_scale,
        resolved_diagnostics=options.resolved_diagnostics,
    )
    make_step = partial(
        _make_explicit_scan_step,
        prepared,
        policies,
        rhs_fn,
        compute_diag_from_state,
        params,
        method=options.method,
        # Refresh diagnostics on the rows the output keeps: a sample stride
        # that is not a multiple of diagnostics_stride would otherwise retain
        # rows whose diagnostics belong to an earlier step.
        diagnostics_stride=max(options.sample_stride, options.diagnostics_stride, 1),
        show_progress=options.show_progress,
        steps=options.steps,
        external_phi=options.external_phi,
        collision_scheme=options.collision_scheme,
    )
    step = make_step(time_horizon=options.time_horizon)
    # The adaptive chunk loop changes only the horizon between chunks; this
    # lets a prepared run take it as a traced operand instead of recompiling.
    step.with_horizon = make_step  # type: ignore[attr-defined]
    step.rhs_fn = rhs_fn  # type: ignore[attr-defined]
    return step, compute_diag_from_state


@contextlib.contextmanager
def _bound_chain(step: Any, chain: Any) -> Any:
    """Trace with ``chain`` (an operand) as the step's chain implicit operator.

    Captured as constants, the 64x64 (8, 16) factor was 7 GB baked into the
    executable, which also ``cached_jit`` hashed on the host every call.
    """
    rhs = getattr(step, "rhs_fn", None)
    if chain is None:
        yield
        return
    old = rhs.chain_implicit_linear  # type: ignore[union-attr]
    rhs.chain_implicit_linear = chain  # type: ignore[union-attr]
    try:
        yield
    finally:
        rhs.chain_implicit_linear = old  # type: ignore[union-attr]


def _chain_operand(step: Any) -> Any:
    return getattr(getattr(step, "rhs_fn", None), "chain_implicit_linear", None)


def _build_explicit_scan_components(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    *,
    cache: LinearCache | None,
    terms: TermConfig | None,
    options: _ExplicitDiagnosticOptions,
) -> _ExplicitScanComponents:
    """Prepare explicit nonlinear diagnostic closures from packed options."""

    prepared = _prepare_explicit_diagnostic_state(
        G0,
        grid,
        geom,
        params,
        method=options.method,
        cache=cache,
        terms=terms,
        use_dealias_mask=options.use_dealias_mask,
        z_index=options.z_index,
        compressed_real_fft=options.compressed_real_fft,
        fixed_mode_ky_index=options.fixed_mode_ky_index,
        fixed_mode_kx_index=options.fixed_mode_kx_index,
    )
    # Supplied-state contract (queue row Q23). ``G0`` is a state this module did
    # not build, so on a linked deck it can carry kx rows the chains never
    # reach; the bracket would alias them back onto the chain rows. Project it
    # once here, where both the prepared object and the function entry point
    # pass, so the two stay the one graph row Q18 made them. Periodic and
    # full-cover decks get ``None`` and the same object back.
    cover_mask = linked_cover_mask_from_cache(prepared.cache)
    prepared = replace(prepared, G0=project_to_linked_cover(prepared.G0, cover_mask))
    policies = _build_explicit_runtime_policies(
        prepared,
        grid,
        params,
        method=options.method,
        dt=options.dt,
        steps=options.steps,
        fixed_dt=options.fixed_dt,
        dt_min=options.dt_min,
        dt_max=options.dt_max,
        cfl=options.cfl,
        cfl_fac=options.cfl_fac,
        compressed_real_fft=options.compressed_real_fft,
        collision_split=options.collision_split,
    )
    step, compute_diag_from_state = _build_explicit_scan_closures(
        prepared,
        policies,
        grid,
        params,
        options=options,
    )
    return _ExplicitScanComponents(
        prepared=prepared,
        policies=policies,
        step=step,
        compute_diag_from_state=compute_diag_from_state,
        cover_mask=cover_mask,
    )


def _run_explicit_scan_components(
    components: _ExplicitScanComponents,
    params: LinearParams,
    *,
    options: _ExplicitDiagnosticOptions,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Run prepared explicit nonlinear diagnostic components."""

    return _run_explicit_diagnostic_scan_and_finalize(
        components.prepared,
        components.policies,
        components.step,
        components.compute_diag_from_state,
        params,
        steps=options.steps,
        sample_stride=options.sample_stride,
        diagnostics_stride=options.diagnostics_stride,
        checkpoint=options.checkpoint,
        resolved_diagnostics=options.resolved_diagnostics,
        external_phi=options.external_phi,
    )


def prepare_explicit_nonlinear_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    time_horizon: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    resolved_diagnostics: bool = True,
    show_progress: bool = False,
) -> PreparedExplicitNonlinearDiagnostics:
    """Prepare one compile-stable explicit nonlinear diagnostic simulation."""

    options = _explicit_options_from_values(locals())
    components = _build_explicit_scan_components(
        G0,
        grid,
        geom,
        params,
        cache=cache,
        terms=terms,
        options=options,
    )
    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"

    def run_raw(
        initial_state: jnp.ndarray,
        time_horizon: jnp.ndarray | None = None,
        chain: Any = None,
    ) -> tuple[Any, Any, Any]:
        step = components.step
        if time_horizon is not None:
            step = step.with_horizon(time_horizon=time_horizon)  # type: ignore[attr-defined]
        with _bound_chain(components.step, chain):
            return _run_explicit_diagnostic_scan_raw(
                components.prepared,
                components.policies,
                step,
                components.compute_diag_from_state,
                params,
                initial_state=initial_state,
                steps=steps,
                sample_stride=sample_stride,
                diagnostics_stride=diagnostics_stride,
                checkpoint=checkpoint,
                external_phi=external_phi,
            )

    def run_dynamic_raw(
        initial_state: jnp.ndarray,
        dynamic_geometry: Any,
        dynamic_cache: LinearCache,
        dynamic_params: LinearParams,
    ) -> tuple[Any, Any, Any]:
        dynamic_setup = build_nonlinear_diagnostic_setup(
            initial_state,
            grid,
            dynamic_geometry,
            dynamic_params,
            cache=dynamic_cache,
            use_dealias_mask=options.use_dealias_mask,
            z_index=options.z_index,
            compressed_real_fft=options.compressed_real_fft,
            fixed_mode_ky_index=options.fixed_mode_ky_index,
            fixed_mode_kx_index=options.fixed_mode_kx_index,
            ensure_geometry_fn=ensure_flux_tube_geometry_data,
            build_cache_fn=build_linear_cache,
            quadrature_weights_fn=fieldline_quadrature_weights,
            omega_mask_fn=_diagnostic_omega_mode_mask,
            midplane_index_fn=_diagnostic_midplane_index,
        )
        dynamic_prepared = replace(
            components.prepared,
            setup=dynamic_setup,
            cache=dynamic_cache,
            project_state=dynamic_setup.project_state,
        )
        dynamic_policies = _build_explicit_runtime_policies(
            dynamic_prepared,
            grid,
            dynamic_params,
            method=options.method,
            dt=options.dt,
            steps=options.steps,
            fixed_dt=options.fixed_dt,
            dt_min=options.dt_min,
            dt_max=options.dt_max,
            cfl=options.cfl,
            cfl_fac=options.cfl_fac,
            compressed_real_fft=options.compressed_real_fft,
            collision_split=options.collision_split,
        )
        dynamic_step, dynamic_diagnostics = _build_explicit_scan_closures(
            dynamic_prepared,
            dynamic_policies,
            grid,
            dynamic_params,
            options=options,
        )
        return _run_explicit_diagnostic_scan_raw(
            dynamic_prepared,
            dynamic_policies,
            dynamic_step,
            dynamic_diagnostics,
            dynamic_params,
            initial_state=initial_state,
            steps=steps,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            checkpoint=checkpoint,
            external_phi=external_phi,
        )

    chain = _chain_operand(components.step)
    return PreparedExplicitNonlinearDiagnostics(
        initial_state=components.prepared.G0,
        geometry=components.prepared.setup.geom,
        cache=components.prepared.cache,
        params=params,
        _run_raw=jax.jit(run_raw)
        if chain is None
        else partial(jax.jit(run_raw), chain=chain),
        _run_dynamic_raw=jax.jit(run_dynamic_raw),
        _finalize=finalize_nonlinear_scan_diagnostics,
        stride=stride,
        sampled_scan=sampled_scan,
        resolved_diagnostics=resolved_diagnostics,
        fixed_dt=options.fixed_dt,
        cfl_scales=getattr(components.policies.time_step_policy, "cfl_scales", None),
        cover_mask=components.cover_mask,
    )


def _integrate_nonlinear_explicit_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    time_horizon: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    resolved_diagnostics: bool = True,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Integrate an explicit nonlinear run and return diagnostics plus final state."""

    options = _explicit_options_from_values(locals())
    components = _build_explicit_scan_components(
        G0,
        grid,
        geom,
        params,
        cache=cache,
        terms=terms,
        options=options,
    )
    return _run_explicit_scan_components(components, params, options=options)


def _pack_resolved_diagnostics(
    resolved_t: tuple[np.ndarray, ...],
) -> ResolvedDiagnostics:
    return ResolvedDiagnostics(
        Phi2_kxt=resolved_t[0],
        Phi2_kyt=resolved_t[1],
        Phi2_kxkyt=resolved_t[2],
        Phi2_zt=resolved_t[3],
        Phi2_zonal_t=resolved_t[4],
        Phi2_zonal_kxt=resolved_t[5],
        Phi2_zonal_zt=resolved_t[6],
        Phi_zonal_mode_kxt=resolved_t[7],
        Phi_zonal_line_kxt=resolved_t[8],
        Wg_kxst=resolved_t[9],
        Wg_kyst=resolved_t[10],
        Wg_kxkyst=resolved_t[11],
        Wg_zst=resolved_t[12],
        Wg_lmst=resolved_t[13],
        Wphi_kxst=resolved_t[14],
        Wphi_kyst=resolved_t[15],
        Wphi_kxkyst=resolved_t[16],
        Wphi_zst=resolved_t[17],
        Wapar_kxst=resolved_t[18],
        Wapar_kyst=resolved_t[19],
        Wapar_kxkyst=resolved_t[20],
        Wapar_zst=resolved_t[21],
        HeatFlux_kxst=resolved_t[22],
        HeatFlux_kyst=resolved_t[23],
        HeatFlux_kxkyst=resolved_t[24],
        HeatFlux_zst=resolved_t[25],
        HeatFluxES_kxst=resolved_t[26],
        HeatFluxES_kyst=resolved_t[27],
        HeatFluxES_kxkyst=resolved_t[28],
        HeatFluxES_zst=resolved_t[29],
        HeatFluxApar_kxst=resolved_t[30],
        HeatFluxApar_kyst=resolved_t[31],
        HeatFluxApar_kxkyst=resolved_t[32],
        HeatFluxApar_zst=resolved_t[33],
        HeatFluxBpar_kxst=resolved_t[34],
        HeatFluxBpar_kyst=resolved_t[35],
        HeatFluxBpar_kxkyst=resolved_t[36],
        HeatFluxBpar_zst=resolved_t[37],
        ParticleFlux_kxst=resolved_t[38],
        ParticleFlux_kyst=resolved_t[39],
        ParticleFlux_kxkyst=resolved_t[40],
        ParticleFlux_zst=resolved_t[41],
        ParticleFluxES_kxst=resolved_t[42],
        ParticleFluxES_kyst=resolved_t[43],
        ParticleFluxES_kxkyst=resolved_t[44],
        ParticleFluxES_zst=resolved_t[45],
        ParticleFluxApar_kxst=resolved_t[46],
        ParticleFluxApar_kyst=resolved_t[47],
        ParticleFluxApar_kxkyst=resolved_t[48],
        ParticleFluxApar_zst=resolved_t[49],
        ParticleFluxBpar_kxst=resolved_t[50],
        ParticleFluxBpar_kyst=resolved_t[51],
        ParticleFluxBpar_kxkyst=resolved_t[52],
        ParticleFluxBpar_zst=resolved_t[53],
        TurbulentHeating_kxst=resolved_t[54],
        TurbulentHeating_kyst=resolved_t[55],
        TurbulentHeating_kxkyst=resolved_t[56],
        TurbulentHeating_zst=resolved_t[57],
    )


def _sample_indices_with_final(length: int, stride: int) -> slice | np.ndarray:
    """Return strided sample indices while always retaining the final step."""

    n = int(length)
    stride_i = int(max(stride, 1))
    if stride_i <= 1 or n <= 1:
        return slice(None)
    idx = np.arange(0, n, stride_i, dtype=int)
    if idx.size == 0 or int(idx[-1]) != n - 1:
        idx = np.concatenate([idx, np.asarray([n - 1], dtype=int)])
    return idx


def _sample_axis0(arr, indices: slice | np.ndarray):
    return arr[indices, ...]


def sampled_scan_intervals(length: int, stride: int) -> np.ndarray:
    """Return positive scan intervals that retain the requested final sample."""

    sample_idx_raw = _sample_indices_with_final(int(length), int(stride))
    sample_idx = np.asarray(
        sample_idx_raw if not isinstance(sample_idx_raw, slice) else np.arange(length),
        dtype=np.int32,
    )
    sample_steps = sample_idx + np.int32(1)
    return np.diff(
        np.concatenate([np.asarray([0], dtype=np.int32), sample_steps])
    ).astype(np.int32)


def run_sampled_explicit_diagnostic_scan(
    step_fn: Any,
    initial_carry: tuple[Any, Any, Any, Any, Any, Any],
    *,
    steps: int,
    stride: int,
) -> tuple[tuple[Any, Any, Any, Any, Any, Any], tuple[Any, Any, Any]]:
    """Run an explicit diagnostic scan only at retained sample intervals."""

    intervals = sampled_scan_intervals(steps, stride)

    def sample_interval(carry, interval_steps):
        def run_one_step(_i, inner_carry):
            G_i, G_prev_i, fields_prev_i, diag_prev_i, t_i, dt_i, idx_i = inner_carry
            next_carry, _diag_step = step_fn(
                (G_i, G_prev_i, fields_prev_i, diag_prev_i, t_i, dt_i), idx_i
            )
            G_next, G_prev_next, fields_prev_next, diag_next, t_next, dt_next = (
                next_carry
            )
            return (
                G_next,
                G_prev_next,
                fields_prev_next,
                diag_next,
                t_next,
                dt_next,
                idx_i + 1,
            )

        carry_next = jax.lax.fori_loop(0, interval_steps, run_one_step, carry)
        (
            _G_next,
            _G_prev_next,
            _fields_prev_next,
            diag_next,
            t_next,
            dt_next,
            _idx_next,
        ) = carry_next
        return carry_next, (diag_next, t_next, dt_next)

    initial_carry_with_idx = (*initial_carry, jnp.asarray(0, dtype=jnp.int32))
    final_carry, diag_out = jax.lax.scan(
        sample_interval,
        initial_carry_with_idx,
        jnp.asarray(intervals, dtype=jnp.int32),
        length=int(intervals.size),
    )
    (
        G_final,
        G_prev_last,
        fields_prev_last,
        diag_last,
        t_last,
        dt_last,
        _idx_last,
    ) = final_carry
    return (
        G_final,
        G_prev_last,
        fields_prev_last,
        diag_last,
        t_last,
        dt_last,
    ), diag_out


def _sample_resolved_axis0(
    resolved_t: tuple[Any, ...],
    indices: slice | np.ndarray,
    *,
    resolved_to_numpy: bool,
) -> tuple[Any, ...]:
    return tuple(
        _sample_axis0(np.asarray(arr) if resolved_to_numpy else arr, indices)
        for arr in resolved_t
    )


def build_nonlinear_simulation_diagnostics(
    diag: tuple[Any, ...],
    t: Any,
    dt_series: Any,
    *,
    resolved_diagnostics: bool,
    sample_indices: slice | np.ndarray | None = None,
    resolved_to_numpy: bool = False,
    cfl_scales: Any | None = None,
) -> SimulationDiagnostics:
    """Build sampled nonlinear diagnostics from the raw scan output tuple."""

    (
        gamma_t,
        omega_t,
        Wg_t,
        Wphi_t,
        Wapar_t,
        heat_t,
        pflux_t,
        turbulent_heat_t,
        heat_s_t,
        pflux_s_t,
        turbulent_heat_s_t,
        phi_mode_t,
        resolved_t,
    ) = diag

    if sample_indices is not None:
        gamma_t = _sample_axis0(gamma_t, sample_indices)
        omega_t = _sample_axis0(omega_t, sample_indices)
        Wg_t = _sample_axis0(Wg_t, sample_indices)
        Wphi_t = _sample_axis0(Wphi_t, sample_indices)
        Wapar_t = _sample_axis0(Wapar_t, sample_indices)
        heat_t = _sample_axis0(heat_t, sample_indices)
        pflux_t = _sample_axis0(pflux_t, sample_indices)
        turbulent_heat_t = _sample_axis0(turbulent_heat_t, sample_indices)
        heat_s_t = _sample_axis0(heat_s_t, sample_indices)
        pflux_s_t = _sample_axis0(pflux_s_t, sample_indices)
        turbulent_heat_s_t = _sample_axis0(turbulent_heat_s_t, sample_indices)
        phi_mode_t = _sample_axis0(phi_mode_t, sample_indices)
        if resolved_diagnostics:
            resolved_t = _sample_resolved_axis0(
                resolved_t,
                sample_indices,
                resolved_to_numpy=resolved_to_numpy,
            )
        t = _sample_axis0(t, sample_indices)
        dt_series = _sample_axis0(dt_series, sample_indices)

    resolved = _pack_resolved_diagnostics(resolved_t) if resolved_diagnostics else None
    return SimulationDiagnostics(
        t=t,
        dt_t=dt_series,
        dt_mean=jnp.mean(dt_series),
        gamma_t=gamma_t,
        omega_t=omega_t,
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=Wg_t + Wphi_t + Wapar_t,
        heat_flux_species_t=heat_s_t,
        particle_flux_species_t=pflux_s_t,
        turbulent_heating_t=turbulent_heat_t,
        turbulent_heating_species_t=turbulent_heat_s_t,
        phi_mode_t=phi_mode_t,
        cfl_scales=cfl_scales,
        resolved=resolved,
    )


def finalize_nonlinear_scan_diagnostics(
    diag: tuple[Any, ...],
    t: Any,
    dt_series: Any,
    *,
    stride: int,
    sampled_scan: bool = False,
    resolved_diagnostics: bool,
    resolved_to_numpy: bool = False,
    cfl_scales: Any | None = None,
) -> SimulationDiagnostics:
    """Package raw nonlinear scan diagnostics after applying output sampling."""

    sample_indices = None
    if int(stride) > 1 and not sampled_scan:
        sample_indices = _sample_indices_with_final(int(t.shape[0]), int(stride))
    return build_nonlinear_simulation_diagnostics(
        diag,
        t=t,
        dt_series=dt_series,
        resolved_diagnostics=resolved_diagnostics,
        sample_indices=sample_indices,
        resolved_to_numpy=resolved_to_numpy,
        cfl_scales=cfl_scales,
    )


def select_nonlinear_step_diagnostics(
    idx: Any,
    *,
    diagnostics_stride: int,
    diag_prev: Any,
    compute_diag_fn: Any,
    steps: int | None = None,
) -> Any:
    """Return a fresh or reused nonlinear step diagnostic tuple.

    Off-stride steps reuse ``diag_prev``. The last step of a ``steps``-long
    scan is always computed fresh, because the output sampling keeps it as the
    final row even when it falls off-stride; reusing the carry there would
    report the previous stride sample's diagnostics at the final time.
    """

    diag_stride = int(max(diagnostics_stride, 1))
    do_diag = (idx % diag_stride) == 0
    if steps is not None:
        do_diag = do_diag | (idx == int(steps) - 1)
    return jax.lax.cond(
        do_diag,
        lambda _operand: compute_diag_fn(),
        lambda _operand: diag_prev,
        operand=None,
    )


def maybe_emit_nonlinear_progress(
    state: Any,
    *,
    show_progress: bool,
    diag: tuple[Any, ...],
    idx: Any,
    steps: int,
    t_new: Any,
    progress_total: Any,
) -> Any:
    """Emit nonlinear progress callbacks when requested and return ``state``."""

    if not show_progress:
        return state

    from gkx.callbacks import print_callback, should_emit_progress

    gamma_cb, omega_cb = diag[0], diag[1]
    Wg_cb, Wphi_cb = diag[2], diag[3]
    return jax.lax.cond(
        should_emit_progress(idx, steps),
        lambda value: print_callback(
            value,
            idx,
            steps,
            gamma_cb,
            omega_cb,
            Wphi_cb,
            Wg_cb,
            t_new,
            progress_total,
        ),
        lambda value: value,
        state,
    )


@dataclass(frozen=True)
class _IMEXPreparedState:
    term_cfg: TermConfig
    linear_cfg: TermConfig
    setup: Any
    cache: Any
    project_state: ProjectFn
    implicit_operator: Any
    G0: jnp.ndarray
    state_dtype: Any
    real_dtype: Any
    dt_val: jnp.ndarray
    progress_total: jnp.ndarray


@dataclass(frozen=True)
class _IMEXRuntimeOperators:
    collision_policy: Any
    nonlinear_term: NonlinearTermFn
    solve_step: SolveStepFn
    solve_step_with_stats: StatsSolveStepFn


def _prepare_imex_diagnostic_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    cache: LinearCache | None,
    terms: TermConfig | None,
    collision_split: bool,
    implicit_preconditioner: str | None,
    compressed_real_fft: bool,
    use_dealias_mask: bool,
    z_index: int | None,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
) -> _IMEXPreparedState:
    """Prepare fixed-step IMEX state, linear operator, and dtype policy."""

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    if collision_split:
        linear_cfg = replace(linear_cfg, collisions=0.0, hypercollisions=0.0)

    setup = build_nonlinear_diagnostic_setup(
        G0,
        grid,
        geom,
        params,
        cache=cache,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        compressed_real_fft=compressed_real_fft,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        ensure_geometry_fn=ensure_flux_tube_geometry_data,
        build_cache_fn=build_linear_cache,
        quadrature_weights_fn=fieldline_quadrature_weights,
        omega_mask_fn=_diagnostic_omega_mode_mask,
        midplane_index_fn=_diagnostic_midplane_index,
    )
    initial_state_dtype = jnp.result_type(G0, jnp.complex64)
    G0_projected = setup.project_state(jnp.asarray(G0, dtype=initial_state_dtype))
    implicit_operator = build_nonlinear_imex_operator(
        G0_projected,
        setup.cache,
        params,
        dt,
        terms=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
    )
    state_dtype = implicit_operator.state_dtype
    G0_projected = jnp.asarray(G0_projected, dtype=state_dtype)
    if (
        implicit_operator.squeeze_species
        and G0_projected.ndim == len(implicit_operator.shape) - 1
    ):
        G0_projected = G0_projected[None, ...]
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    progress_total = jnp.asarray(float(steps) * float(dt), dtype=real_dtype)
    return _IMEXPreparedState(
        term_cfg=term_cfg,
        linear_cfg=linear_cfg,
        setup=setup,
        cache=setup.cache,
        project_state=setup.project_state,
        implicit_operator=implicit_operator,
        G0=G0_projected,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        dt_val=dt_val,
        progress_total=progress_total,
    )


def _build_imex_runtime_operators(
    prepared: _IMEXPreparedState,
    params: LinearParams,
    *,
    linear_rhs_fn: Callable[..., Any],
    collision_split: bool,
    external_phi: jnp.ndarray | float | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
    implicit_iters: int,
    implicit_relax: float,
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_restart: int,
) -> _IMEXRuntimeOperators:
    """Build collision, nonlinear, and linear solve operators for IMEX scans."""

    collision_policy = build_nonlinear_collision_split_policy(
        prepared.cache,
        params,
        prepared.term_cfg,
        prepared.real_dtype,
        squeeze_species=prepared.implicit_operator.squeeze_species,
        collision_split=collision_split,
        collision_damping_fn=_collision_damping,
    )
    nonlinear_term = make_imex_nonlinear_term(
        prepared.cache,
        params,
        prepared.term_cfg,
        real_dtype=prepared.real_dtype,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        fields_fn=compute_fields_cached,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
    )
    solve_policy = {
        "linear_rhs_fn": linear_rhs_fn,
        "cache": prepared.cache,
        "params": params,
        "linear_cfg": prepared.linear_cfg,
        "external_phi": external_phi,
        "dt_val": prepared.dt_val,
        "implicit_iters": implicit_iters,
        "implicit_relax": implicit_relax,
        "matvec": prepared.implicit_operator.matvec,
        "shape": prepared.implicit_operator.shape,
        "implicit_tol": implicit_tol,
        "implicit_maxiter": implicit_maxiter,
        "implicit_restart": implicit_restart,
        "precond_op": prepared.implicit_operator.precond_op,
    }
    solve_step = make_imex_solve_step(**solve_policy, solve_step_fn=solve_imex_step)
    return _IMEXRuntimeOperators(
        collision_policy=collision_policy,
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        solve_step_with_stats=make_imex_solve_step_with_stats(**solve_policy),
    )


def _make_imex_diagnostic_callable(
    prepared: _IMEXPreparedState,
    grid: SpectralGrid,
    params: LinearParams,
    *,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
) -> DiagnosticFn:
    """Return the state-to-diagnostic tuple closure for fixed-step IMEX scans."""

    return make_nonlinear_diagnostic_tuple_fn(
        grid=grid,
        cache=prepared.cache,
        params=params,
        vol_fac=prepared.setup.vol_fac,
        flux_fac=prepared.setup.flux_fac,
        mask=prepared.setup.mask,
        z_idx=prepared.setup.z_idx,
        use_dealias=prepared.setup.use_dealias,
        real_dtype=prepared.real_dtype,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        resolved_diagnostics=True,
        kernels=_nonlinear_diagnostic_kernels(),
    )


def _make_imex_scan_step(
    prepared: _IMEXPreparedState,
    runtime_ops: _IMEXRuntimeOperators,
    compute_diag_from_state: DiagnosticFn,
    params: LinearParams,
    *,
    method: str,
    diagnostics_stride: int,
    show_progress: bool,
    steps: int,
    external_phi: jnp.ndarray | float | None,
    collision_scheme: str,
) -> DiagnosticStepFn:
    """Build the fixed-step IMEX diagnostic scan step."""

    return make_imex_diagnostic_step(
        method=method,
        nonlinear_term=runtime_ops.nonlinear_term,
        solve_step=runtime_ops.solve_step,
        solve_step_with_stats=runtime_ops.solve_step_with_stats,
        project_state=prepared.project_state,
        state_dtype=prepared.state_dtype,
        real_dtype=prepared.real_dtype,
        dt_val=prepared.dt_val,
        compute_fields_fn=compute_fields_cached,
        cache=prepared.cache,
        params=params,
        term_cfg=prepared.term_cfg,
        external_phi=external_phi,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=diagnostics_stride,
        select_diagnostics_fn=select_nonlinear_step_diagnostics,
        show_progress=show_progress,
        steps=steps,
        progress_total=prepared.progress_total,
        emit_progress_fn=maybe_emit_nonlinear_progress,
        use_collision_split=runtime_ops.collision_policy.active,
        damping=runtime_ops.collision_policy.damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=_apply_collision_split,
    )


def _run_imex_diagnostic_scan_and_finalize(
    prepared: _IMEXPreparedState,
    step: DiagnosticStepFn,
    compute_diag_from_state: DiagnosticFn,
    params: LinearParams,
    *,
    steps: int,
    checkpoint: bool,
    sample_stride: int,
    diagnostics_stride: int,
    external_phi: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, SimulationDiagnostics, ImplicitSolveStats | None]:
    """Run the fixed-step IMEX scan and finalize diagnostics.

    The scan always carries the solve status, as the cached IMEX and implicit
    linear scans do; it is read back here and dropped by the caller that did not
    ask for it, which is what lets XLA remove the arithmetic behind it.
    """

    fields0 = compute_fields_cached(
        prepared.G0,
        prepared.cache,
        params,
        terms=prepared.term_cfg,
        external_phi=external_phi,
    )
    diag_zero = compute_diag_from_state(
        prepared.G0, fields0, prepared.G0, fields0, prepared.dt_val
    )
    _G_final, scan_diag_out, solve_stats = run_imex_diagnostic_scan(
        step,
        (
            prepared.G0,
            prepared.G0,
            fields0,
            diag_zero,
            jnp.asarray(0.0, dtype=prepared.real_dtype),
            _empty_implicit_solve_stats(prepared.state_dtype),
        ),
        steps=steps,
        checkpoint=checkpoint,
        return_solve_stats=True,
    )

    diag, t = scan_diag_out
    dt_series = jnp.ones_like(t) * prepared.dt_val
    stride = int(max(sample_stride, diagnostics_stride, 1))
    diag_out = finalize_nonlinear_scan_diagnostics(
        diag,
        t=t,
        dt_series=dt_series,
        stride=stride,
        resolved_diagnostics=True,
        resolved_to_numpy=True,
    )
    return jnp.asarray(diag_out.t), diag_out, solve_stats


def integrate_imex_nonlinear_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "imex",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """Integrate an IMEX nonlinear run and return diagnostics.

    Returns ``(t, diagnostics)``, or ``(t, diagnostics, stats)`` with
    ``return_solve_stats=True``.
    """
    prepared = _prepare_imex_diagnostic_state(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        cache=cache,
        terms=terms,
        collision_split=collision_split,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
    )
    linear_rhs_fn = _linear_rhs_jit_for_terms(prepared.linear_cfg)
    runtime_ops = _build_imex_runtime_operators(
        prepared,
        params,
        linear_rhs_fn=linear_rhs_fn,
        collision_split=collision_split,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
    )
    compute_diag_from_state = _make_imex_diagnostic_callable(
        prepared,
        grid,
        params=params,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
    )
    step = _make_imex_scan_step(
        prepared,
        runtime_ops,
        compute_diag_from_state,
        params,
        method=method,
        # Refresh diagnostics on the rows the output keeps (see the explicit
        # scan builder).
        diagnostics_stride=max(sample_stride, diagnostics_stride, 1),
        show_progress=show_progress,
        steps=steps,
        external_phi=external_phi,
        collision_scheme=collision_scheme,
    )
    t, diag, solve_stats = _run_imex_diagnostic_scan_and_finalize(
        prepared,
        step,
        compute_diag_from_state,
        params,
        steps=steps,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        external_phi=external_phi,
    )
    if return_solve_stats:
        return t, diag, solve_stats
    return t, diag


_IMEX_METHODS = frozenset({"imex"})

_EXPLICIT_DIAGNOSTIC_OPTION_KEYS = (
    "method",
    "cache",
    "terms",
    "checkpoint",
    "sample_stride",
    "diagnostics_stride",
    "use_dealias_mask",
    "z_index",
    "compressed_real_fft",
    "laguerre_mode",
    "omega_ky_index",
    "omega_kx_index",
    "flux_scale",
    "wphi_scale",
    "fixed_dt",
    "dt_min",
    "dt_max",
    "time_horizon",
    "cfl",
    "cfl_fac",
    "collision_split",
    "collision_scheme",
    "implicit_tol",
    "implicit_maxiter",
    "implicit_iters",
    "implicit_relax",
    "implicit_restart",
    "implicit_preconditioner",
    "fixed_mode_ky_index",
    "fixed_mode_kx_index",
    "external_phi",
    "resolved_diagnostics",
    "show_progress",
)
_IMEX_DIAGNOSTIC_OPTION_KEYS = tuple(
    key
    for key in _EXPLICIT_DIAGNOSTIC_OPTION_KEYS
    if key
    not in {
        "fixed_dt",
        "dt_min",
        "dt_max",
        "time_horizon",
        "cfl",
        "cfl_fac",
        "resolved_diagnostics",
    }
)


def _options_from_scope(scope: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: scope[key] for key in keys}


def _nonlinear_diagnostic_kernels() -> NonlinearDiagnosticKernels:
    """Return the diagnostic kernels passed to the diagnostic tuple builder."""

    return NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_instantaneous_growth_rate_step,
        phi2_resolved=phi2_resolved,
        zonal_phi_mode_kxt=zonal_phi_mode_kxt,
        zonal_phi_line_kxt=zonal_phi_line_kxt,
        distribution_free_energy=distribution_free_energy,
        distribution_free_energy_resolved=distribution_free_energy_resolved,
        electrostatic_field_energy=electrostatic_field_energy,
        electrostatic_field_energy_resolved=electrostatic_field_energy_resolved,
        magnetic_vector_potential_energy=magnetic_vector_potential_energy,
        magnetic_vector_potential_energy_resolved=magnetic_vector_potential_energy_resolved,
        heat_flux_species=heat_flux_species,
        heat_flux_channel_resolved_species=heat_flux_channel_resolved_species,
        particle_flux_species=particle_flux_species,
        particle_flux_channel_resolved_species=particle_flux_channel_resolved_species,
        turbulent_heating_species=turbulent_heating_species,
        turbulent_heating_resolved_species=turbulent_heating_resolved_species,
    )


def integrate_nonlinear_explicit_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    time_horizon: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    resolved_diagnostics: bool = True,
    show_progress: bool = False,
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """Integrate nonlinear system and return runtime diagnostics.

    ``return_solve_stats=True`` appends the implicit solve summary of an IMEX
    run (``None`` for the explicit methods, which take no implicit solve).
    """

    if method in _IMEX_METHODS:
        return integrate_nonlinear_imex_diagnostics(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            return_solve_stats=return_solve_stats,
            **_options_from_scope(locals(), _IMEX_DIAGNOSTIC_OPTION_KEYS),
        )

    t, diag_out, _G_final, _fields_final = (
        _integrate_nonlinear_explicit_diagnostics_impl(
            G0,
            grid,
            geom,
            params,
            dt,
            steps,
            **_options_from_scope(locals(), _EXPLICIT_DIAGNOSTIC_OPTION_KEYS),
        )
    )
    return (t, diag_out, None) if return_solve_stats else (t, diag_out)


def integrate_nonlinear_explicit_diagnostics_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    time_horizon: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    resolved_diagnostics: bool = True,
    show_progress: bool = False,
    compile_cache: dict[Any, Any] | None = None,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Integrate nonlinear system and return runtime diagnostics plus the final state.

    This is the entry point ``run_runtime_nonlinear`` uses on its fixed-window,
    chunked and sharded routes. It runs the same jitted graph as
    :func:`prepare_nonlinear_explicit_diagnostics` and returns bitwise
    identical arrays for the same inputs (plan §5.3, queue row Q18). Prefer the
    prepared object when the same case is run more than once: it keeps the
    compiled graph, while each call here compiles its own.
    """

    if method in _IMEX_METHODS:
        raise ValueError(
            "integrate_nonlinear_explicit_diagnostics_state only supports explicit methods"
        )

    options = _options_from_scope(locals(), _EXPLICIT_DIAGNOSTIC_OPTION_KEYS)
    if compile_cache is None:
        return _integrate_nonlinear_explicit_diagnostics_impl(
            G0, grid, geom, params, dt, steps, **options
        )
    # Chunked runs call this once per chunk with only the state, the horizon
    # and (on the last capped chunk) the step count changing. Keep one
    # prepared graph per static shape and pass the horizon as an operand:
    # rebuilding and re-jitting the closure every chunk cost a full XLA
    # compile per chunk. The caller owns the cache, one per run, so every
    # other option is fixed for its lifetime.
    key = (int(steps), time_horizon is None, bool(show_progress))
    prepared = compile_cache.get(key)
    if prepared is None:
        prepared = prepare_nonlinear_explicit_diagnostics(
            G0, grid, geom, params, dt, steps, **{**options, "time_horizon": None}
        )
        compile_cache[key] = prepared
    return prepared.run(G0, time_horizon=time_horizon)


def prepare_nonlinear_explicit_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    **options: Any,
) -> PreparedExplicitNonlinearDiagnostics:
    """Prepare a reusable explicit diagnostic scan for repeated Python calls.

    ``options`` accepts the same explicit-only keywords as
    :func:`integrate_nonlinear_explicit_diagnostics_state`.
    """

    method = str(options.get("method", "rk3"))
    if method in _IMEX_METHODS:
        raise ValueError("prepared nonlinear diagnostics only support explicit methods")
    return prepare_explicit_nonlinear_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        **options,
    )


def integrate_nonlinear_imex_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "imex",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """IMEX nonlinear integrator with runtime diagnostics.

    Returns ``(t, diagnostics)``, or ``(t, diagnostics, stats)`` with
    ``return_solve_stats=True``. The scan cannot raise on a starved inner
    budget, so ``stats``
    (:class:`~gkx.solvers_linear_implicit.ImplicitSolveStats`) is this route's
    convergence channel; a host-side caller turns it into a refusal with
    :func:`~gkx.solvers_linear_implicit.require_converged_implicit_solves`.
    """

    options = _options_from_scope(locals(), _IMEX_DIAGNOSTIC_OPTION_KEYS)
    return integrate_imex_nonlinear_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        return_solve_stats=return_solve_stats,
        **options,
    )


#: Longest window whose discrete adjoint has been measured to still behave like
#: a gradient on the shipped saturated Cyclone case: the AD/FD ladder in
#: ``scripts/campaigns/nonlinear_gradient_window.py`` (removed; at df2953026) tracks centered differences
#: through 1024 RK3 steps and departs between 1024 and 2048. It is a property of
#: that trajectory's Lyapunov time, not a solver tolerance, so it is a default to
#: warn against and remeasure -- not a hard limit. ``examples/
#: 10_vmex_optimization/run.py`` runs at exactly 1024, one rung below the departure.
DIVERGENCE_KNEE_STEPS = 1024

#: Reverse-pass checkpoint storage allowed before the window falls back to the
#: nested (three-forward) schedule. 4 GiB admits 32x32x24 Nl4/Nm8 at 1024 steps
#: (3.0 GiB measured on an A4000, 1.27x faster than nested).
ADJOINT_MEMORY_BUDGET_BYTES = 4 * 1024**3


def _warn_if_window_exceeds_divergence_knee(
    steps: int, knee: int | None = DIVERGENCE_KNEE_STEPS
) -> None:
    """Warn when a differentiated window runs past the measured knee."""

    if knee is None or steps <= int(knee):
        return
    warnings.warn(
        f"differentiating a {steps}-step window, above the measured divergence "
        f"knee of {knee} steps: past it the windowed adjoint grows with the "
        "leading Lyapunov exponent and is large, reproducible, and not a "
        "descent direction. Remeasure the knee for this case with "
        "scripts/campaigns/nonlinear_gradient_window.py (git show df2953026), then pass "
        "divergence_knee_steps=<measured> or None.",
        RuntimeWarning,
        stacklevel=2,
    )


class ShearedTransportTrace(NamedTuple):
    """Final state and compact heat-flux history from a sheared run.

    ``solve_stats`` is the convergence summary of every implicit GMRES solve of
    a ``method="imex"`` run, and is populated only when the caller asked for it
    with ``return_solve_stats=True``; it stays ``None`` for explicit methods and
    for runs that did not ask, so a graph that does not read it is unchanged.
    """

    final_state: jnp.ndarray
    time: jnp.ndarray
    heat_flux: jnp.ndarray
    solve_stats: ImplicitSolveStats | None = None


def _linear_rhs_jit_for_terms(term_cfg: TermConfig):
    """Return the narrowest compiled linear RHS path compatible with ``term_cfg``."""

    return linear_rhs_jit_for_terms_impl(
        term_cfg,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
    )


def nonlinear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
    *,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    collision_operator: CollisionOperator | None = None,
    radial_phase: jnp.ndarray | None = None,
    differentiable: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """Compute the assembled nonlinear RHS and electromagnetic field state."""

    return nonlinear_rhs_cached_impl(
        G,
        cache,
        params,
        terms,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        collision_operator=collision_operator,
        radial_phase=radial_phase,
        differentiable=differentiable,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
        nonlinear_contribution_fn=nonlinear_em_contribution,
    )


def _nonlinear_rhs_scan(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    compressed_real_fft: bool,
    laguerre_mode: str,
    collision_operator: CollisionOperator | None,
) -> tuple[jnp.ndarray, FieldState]:
    """Stable scan callable; arrays are dynamic while model switches are static."""

    return nonlinear_rhs_cached(
        G,
        cache,
        params,
        term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
    )


def integrate_nonlinear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    show_progress: bool = False,
    return_fields: bool = True,
    collision_operator: CollisionOperator | None = None,
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the nonlinear system using a cached geometry object.

    ``return_solve_stats=True`` appends the IMEX implicit solve summary
    (``None`` for explicit methods, which have no implicit solve).

    ``G0`` is a state this driver did not build, so on a linked (twist-shift)
    deck it is projected onto the chain cover before the run starts, exactly as
    the runtime does at intake (queue row Q19) and as the entry points above
    this one do (Q23). The projection is a ``where`` against a mask fixed by the
    deck's topology, applied once here and outside the compiled scan, so the
    scan's graph is unchanged and periodic or full-cover decks get the same
    array object back.
    """

    term_cfg = terms or TermConfig()
    G0 = mask_supplied_state(G0, cache)
    if method == "imex":
        if collision_operator is not None:
            raise NotImplementedError(
                "custom collision operators currently require explicit nonlinear integration"
            )
        G_out, fields_t, solve_stats = integrate_nonlinear_imex_cached(
            G0,
            cache,
            params,
            dt,
            steps,
            terms=term_cfg,
            checkpoint=checkpoint,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            show_progress=show_progress,
            return_solve_stats=True,
        )
        head = (G_out, fields_t) if return_fields else (G_out,)
        if return_solve_stats:
            return (*head, solve_stats)
        return head if return_fields else G_out

    project_state = None
    if compressed_real_fft:
        project_state = _make_compressed_real_fft_projector(
            ny_full=source_ny_full(cache),
            nx=int(cache.kx.size),
            rows=int(cache.ky.size),
        )

    result = integrate_nonlinear_scan(
        _nonlinear_rhs_scan,
        G0,
        dt,
        steps,
        method=method,
        rhs_args=(cache, params),
        rhs_static_args=(
            term_cfg,
            compressed_real_fft,
            laguerre_mode,
            collision_operator,
        ),
        checkpoint=checkpoint,
        project_state=project_state,
        show_progress=show_progress,
        return_fields=return_fields,
    )
    if not return_solve_stats:
        return result
    return (*result, None) if return_fields else (result, None)


def integrate_nonlinear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    show_progress: bool = False,
    return_fields: bool = True,
    collision_operator: CollisionOperator | None = None,
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the nonlinear system using built-in cache construction.

    A supplied ``G0`` is projected onto the linked chain cover by
    :func:`integrate_nonlinear_cached`, which this driver builds the cache for
    and then delegates to, so both raw drivers take the same view of a state
    they did not build.
    """

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        Nl, Nm = _gyrokinetic_moment_shape(G0)
        cache = build_linear_cache(grid, geom_eff, params, Nl, Nm)
    return integrate_nonlinear_cached(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        show_progress=show_progress,
        return_fields=return_fields,
        collision_operator=collision_operator,
        return_solve_stats=return_solve_stats,
    )


def _identity_state(state: jnp.ndarray) -> jnp.ndarray:
    """Use a stable function identity for the compiled window's projector."""

    return state


@partial(
    jax.jit,
    static_argnames=(
        "dt",
        "count",
        "tail",
        "method",
        "term_cfg",
        "compressed_real_fft",
        "laguerre_mode",
        "collision_operator",
        "checkpoint",
        "projector_signature",
        "memory_budget_bytes",
    ),
)
def _nonlinear_heat_flux_window_total(
    initial_state: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_factor: jnp.ndarray,
    initial_heat: jnp.ndarray,
    indices: jnp.ndarray,
    *,
    dt: float,
    count: int,
    tail: int,
    method: str,
    term_cfg: TermConfig,
    compressed_real_fft: bool,
    laguerre_mode: str,
    collision_operator: CollisionOperator | None,
    checkpoint: bool,
    projector_signature: tuple[int, bool, int] | None,
    memory_budget_bytes: int | None = None,
) -> jnp.ndarray:
    """Compile a reusable heat-flux sum for fixed shapes and static options.

    Arrays are operands, not captured geometry constants. A module-level JIT
    avoids rebuilding eager scan/cond executables on each objective call and
    permits reuse through ``value_and_grad``. Projector layout is static and
    has no derivative; configuration or topology changes may recompile.

    Keep ``total_heat / tail`` outside this graph: fusing the mean into the
    accumulation changed float32 rounding in the reference comparison.
    """

    project_state: Callable[[jnp.ndarray], jnp.ndarray] = (
        _identity_state
        if projector_signature is None
        else hermitian_projector_for_signature(projector_signature)
    )

    def rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        return nonlinear_rhs_cached(
            state,
            cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            differentiable=True,
        )

    def advance(carry: tuple[jnp.ndarray, jnp.ndarray], index: jnp.ndarray):
        state, total_heat = carry
        state = project_state(state)
        derivative, _ = rhs(state)
        next_state = advance_explicit_nonlinear_state(
            state,
            derivative,
            jnp.asarray(dt, dtype=jnp.real(state).dtype),
            method=method,
            rhs_fn=rhs,
            project_state=project_state,
            state_dtype=state.dtype,
        )
        _, fields = rhs(next_state)
        zero = jnp.zeros_like(fields.phi)
        heat = heat_flux_total(
            next_state,
            fields.phi,
            zero if fields.apar is None else fields.apar,
            zero if fields.bpar is None else fields.bpar,
            cache,
            grid,
            params,
            flux_factor,
        )
        include = jnp.asarray(index >= count - tail, dtype=heat.dtype)
        return (next_state, total_heat + include * heat), None

    (_, total_heat), _ = checkpointed_explicit_scan(
        advance,
        (jax.lax.stop_gradient(initial_state), initial_heat),
        indices,
        checkpoint=checkpoint,
        memory_budget_bytes=memory_budget_bytes,
    )
    return total_heat


def nonlinear_heat_flux_window(
    saturated_state: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    method: str = "rk2",
    tail_steps: int | None = None,
    checkpoint: bool = True,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    divergence_knee_steps: int | None = DIVERGENCE_KNEE_STEPS,
    adjoint_memory_budget_bytes: int | None = ADJOINT_MEMORY_BUDGET_BYTES,
) -> jnp.ndarray:
    r"""Mean physical heat flux over a differentiable nonlinear window.

    ``saturated_state`` is deliberately detached: first run GKX to saturation,
    then differentiate this finite window through ``geom`` and ``params``.
    Reverse-mode AD follows the exact discrete Runge--Kutta map. Block
    checkpointing retains :math:`O(\sqrt{N})` distribution states instead of
    :math:`O(N)` for a window of ``N`` steps.

    On a linked (twist-shift) deck ``saturated_state`` is projected onto the
    linked chain cover before the window starts (queue row Q23): it is a state
    this function did not build, and the ExB bracket takes the whole state to
    real space while dealiasing only its output, so off-chain rows would alias
    back onto the chain rows and change the flux this objective returns. The
    projection does not change the gradient contract, which is that
    ``saturated_state`` is detached and only ``geom`` and ``params`` carry
    derivatives. Periodic and full-cover decks are untouched.

    ``collision_operator`` is the same custom model
    :func:`integrate_nonlinear` accepts, and must be passed here too: a run
    saturated with a custom operator and then differentiated without one is a
    derivative of different physics from the trajectory it starts on.

    ``steps`` above ``divergence_knee_steps`` warns. Past the measured knee the
    windowed adjoint grows with the leading Lyapunov exponent and stops being a
    useful design direction; pass ``None`` to silence the check when the knee
    has been remeasured for the case at hand with
    ``scripts/campaigns/nonlinear_gradient_window.py``.

    ``adjoint_memory_budget_bytes`` bounds the reverse pass's checkpoint
    storage. When the block-only schedule fits, each step's forward runs twice
    in a gradient instead of three times; otherwise, or with ``None``, the
    nested :math:`O(\sqrt{N})` schedule is used. It changes cost, not the
    discrete adjoint.

    Host-side geometry/layout preparation stays here. The differentiated scan
    in :func:`_nonlinear_heat_flux_window_total` reuses its executable for
    matching shapes, topology and static options, with geometry arrays passed
    as operands. See ``docs/nonlinear_autodiff.rst`` for scope and evidence.
    """

    count = int(steps)
    tail = count if tail_steps is None else int(tail_steps)
    if count < 1 or not 1 <= tail <= count:
        raise ValueError("steps must be positive and tail_steps must lie within it")
    _warn_if_window_exceeds_divergence_knee(count, divergence_knee_steps)
    if saturated_state.ndim not in (5, 6):
        raise ValueError(
            "saturated_state must have shape (Nl, Nm, Ny, Nx, Nz) or "
            "(Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    offset = saturated_state.ndim - 5
    Nl, Nm = saturated_state.shape[offset : offset + 2]

    term_cfg = terms or TermConfig(nonlinear=1.0)
    geometry = ensure_flux_tube_geometry_data(geom, grid.z)
    cache = build_linear_cache(grid, geometry, params, Nl=Nl, Nm=Nm)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geometry, grid)
    projector_signature = (
        hermitian_projector_signature(
            np.asarray(grid.ky), int(np.asarray(grid.kx).size)
        )
        if compressed_real_fft
        else None
    )
    initial_state = jax.lax.stop_gradient(
        mask_supplied_state(jnp.asarray(saturated_state), cache)
    )
    heat_dtype = jnp.result_type(
        jnp.real(initial_state), flux_factor, *jax.tree_util.tree_leaves(params)
    )
    total_heat = _nonlinear_heat_flux_window_total(
        initial_state,
        cache,
        grid,
        params,
        flux_factor,
        jnp.zeros((), dtype=heat_dtype),
        jnp.arange(count),
        dt=dt,
        count=count,
        tail=tail,
        method=method,
        term_cfg=term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        checkpoint=checkpoint,
        projector_signature=projector_signature,
        memory_budget_bytes=(
            None
            if adjoint_memory_budget_bytes is None
            else int(adjoint_memory_budget_bytes)
        ),
    )
    return total_heat / tail


def _integrate_nonlinear_sheared_scan(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    record_transport: bool = False,
    return_fields: bool = True,
    flux_scale: float = 1.0,
    differentiable: bool = False,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    initial_time: jnp.ndarray | float = 0.0,
    initial_dt: jnp.ndarray | float | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
) -> tuple[jnp.ndarray, Any, ImplicitSolveStats | None]:
    """Run the shared shearing-coordinate scan with optional transport output.

    Returns the final state, the scan output, and the implicit solve status of
    a ``method="imex"`` run (``None`` for the explicit methods, which take no
    implicit solve and so add no carry leaf).
    """

    if str(grid.boundary).lower() not in {"periodic", "linked"} or bool(grid.non_twist):
        raise NotImplementedError(
            "sheared integration requires a periodic or linked standard flux tube"
        )
    if steps < 1:
        raise ValueError("steps must be at least one")
    method_key = str(method).lower()
    if method_key not in {"rk2", "rk3", "imex"}:
        raise method_error(method, {"rk2", "rk3", "imex"})
    if not fixed_dt and not record_transport:
        raise ValueError("adaptive sheared integration requires a transport trace")
    if method_key == "imex" and not fixed_dt:
        raise ValueError("sheared IMEX integration currently requires fixed_dt=True")
    if method_key == "imex" and collision_operator is not None:
        raise NotImplementedError(
            "sheared IMEX does not yet support custom collision operators"
        )
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        nl, nm = _gyrokinetic_moment_shape(G0)
        cache = build_linear_cache(grid, geom_eff, params, nl, nm)

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)
    base_complex_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_complex_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_value = jnp.asarray(dt, dtype=real_dtype)
    initial_time_value = jnp.asarray(initial_time, dtype=real_dtype)
    initial_dt_value = jnp.asarray(
        dt if initial_dt is None else initial_dt, dtype=real_dtype
    )
    rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)
    coordinate_kx = jnp.asarray(cache.kx, dtype=real_dtype)
    coordinate_ky = jnp.asarray(cache.ky, dtype=real_dtype)
    if int(coordinate_kx.size) > 1:
        coordinate_x0 = 1.0 / jnp.abs(coordinate_kx[1] - coordinate_kx[0])
    else:
        coordinate_x0 = jnp.asarray(grid.x0, dtype=real_dtype) / rho_star
    _, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    # Full-complex shearing coordinates must retain the real-field subspace.
    project_state = _make_hermitian_projector(
        np.asarray(grid.ky), int(np.asarray(grid.kx).size)
    )
    time_step_policy = None
    if not fixed_dt:
        time_step_policy = build_nonlinear_time_step_policy(
            grid,
            geom_eff,
            params,
            cache,
            method=method_key,
            dt=dt,
            steps=steps,
            fixed_dt=False,
            dt_min=dt_min,
            dt_max=dt_max,
            cfl=cfl,
            cfl_fac=cfl_fac,
            compressed_real_fft=compressed_real_fft,
            real_dtype=real_dtype,
            resolve_cfl_fac_fn=resolve_cfl_fac,
            linear_frequency_bound_fn=_linear_frequency_bound,
            laguerre_velocity_max_fn=_laguerre_velocity_max,
            cfl_frequency_components_fn=_nonlinear_cfl_frequency_components,
        )

    def coordinates(state: jnp.ndarray, time: jnp.ndarray, previous: jnp.ndarray):
        update = advance_shearing_coordinates(
            state,
            # The linked-boundary cache may adjust the radial spacing to close
            # the twist-shift chain. Working in the cache's normalized units
            # preserves both that adjustment and the periodic convention.
            kx=coordinate_kx,
            ky=coordinate_ky,
            x0=coordinate_x0,
            shear_rate=shear_rate,
            previous_time=previous,
            time=time,
            dealias_mask=grid.dealias_mask,
        )
        return update._replace(state=project_state(update.state))

    def cache_at(update):
        return update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom_eff,
            params,
            update.effective_kx,
        )

    def rhs_at(update):
        updated_cache = cache_at(update)
        derivative, fields = nonlinear_rhs_cached(
            update.state,
            updated_cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            radial_phase=update.phase,
            differentiable=differentiable,
        )
        return derivative, fields, updated_cache

    def sheared_fields(
        state,
        updated_cache,
        field_params,
        *,
        terms=None,
        external_phi=None,
    ):
        return compute_fields_cached(
            state,
            updated_cache,
            field_params,
            terms=terms,
            use_custom_vjp=not differentiable,
            external_phi=external_phi,
        )

    def fields_at(update):
        updated_cache = cache_at(update)
        fields = sheared_fields(
            update.state,
            updated_cache,
            params,
            terms=term_cfg,
        )
        return fields, updated_cache

    def advance(current, derivative, time, dt_local, solve_stats=None):
        """Advance one sheared step, folding any implicit solve into the status.

        ``solve_stats`` rides through the scan carry because a traced step
        cannot raise, exactly as the cached IMEX and implicit linear scans do
        (queue row Q15). Explicit methods pass ``None`` straight back, so they
        add no carry leaf and compile unchanged.
        """

        new_time = time + dt_local
        if method_key == "imex":
            current_cache = cache_at(current)
            nonlinear_term = nonlinear_em_term_cached_impl(
                current.state,
                current_cache,
                params,
                term_cfg,
                external_phi=None,
                compressed_real_fft=compressed_real_fft,
                laguerre_mode=laguerre_mode,
                radial_phase=current.phase,
                fields_fn=sheared_fields,
                nonlinear_contribution_fn=nonlinear_em_contribution,
            )
            endpoint_guess = coordinates(current.state, new_time, time)
            endpoint_rhs = coordinates(
                current.state + dt_local * nonlinear_term,
                new_time,
                time,
            )
            endpoint_cache = cache_at(endpoint_guess)
            operator = build_nonlinear_imex_operator(
                endpoint_guess.state,
                endpoint_cache,
                params,
                dt_local,
                terms=linear_cfg,
                implicit_preconditioner=implicit_preconditioner,
                compressed_real_fft=compressed_real_fft,
                build_implicit_operator_fn=_build_implicit_operator,
            )
            guess = jnp.asarray(endpoint_guess.state, dtype=operator.state_dtype)
            rhs = jnp.asarray(endpoint_rhs.state, dtype=operator.state_dtype)
            if operator.squeeze_species:
                guess = guess[None, ...]
                rhs = rhs[None, ...]
            solution, solve_stats = solve_imex_step_with_stats(
                guess,
                rhs,
                solve_stats,
                linear_rhs_fn=linear_rhs_fn,
                cache=endpoint_cache,
                params=params,
                linear_cfg=linear_cfg,
                external_phi=None,
                dt_val=operator.dt_val,
                implicit_iters=implicit_iters,
                implicit_relax=implicit_relax,
                matvec=operator.matvec,
                shape=operator.shape,
                implicit_tol=implicit_tol,
                implicit_maxiter=implicit_maxiter,
                implicit_restart=implicit_restart,
                precond_op=operator.precond_op,
            )
            if operator.squeeze_species:
                solution = solution[0]
            return (
                endpoint_rhs._replace(state=project_state(solution)),
                new_time,
                solve_stats,
            )
        if method_key == "rk2":
            midpoint_time = time + 0.5 * dt_local
            midpoint = coordinates(
                current.state + 0.5 * dt_local * derivative,
                midpoint_time,
                time,
            )
            midpoint_derivative, _, _ = rhs_at(midpoint)
            derivative_in_step_basis = coordinates(
                midpoint_derivative,
                time,
                midpoint_time,
            ).state
            trial = current.state + dt_local * derivative_in_step_basis
        else:
            stage1_time = time + dt_local / 3.0
            stage1 = coordinates(
                current.state + (dt_local / 3.0) * derivative,
                stage1_time,
                time,
            )
            stage1_derivative, _, _ = rhs_at(stage1)
            stage1_derivative_base = coordinates(
                stage1_derivative,
                time,
                stage1_time,
            ).state
            stage2_time = time + 2.0 * dt_local / 3.0
            stage2 = coordinates(
                current.state + (2.0 * dt_local / 3.0) * stage1_derivative_base,
                stage2_time,
                time,
            )
            stage2_derivative, _, _ = rhs_at(stage2)
            stage2_derivative_base = coordinates(
                stage2_derivative,
                time,
                stage2_time,
            ).state
            trial = (
                current.state
                + 0.25 * dt_local * derivative
                + 0.75 * dt_local * (stage2_derivative_base)
            )
        return coordinates(trial, new_time, time), new_time, solve_stats

    def local_dt(current_fields, dt_previous):
        if time_step_policy is None:
            return dt_value
        return time_step_policy.update_dt(current_fields, dt_previous)

    # The status leaf exists only on the route that takes an implicit solve, so
    # the explicit sheared scans keep exactly the carry they had.
    carries_solve_stats = method_key == "imex"

    def _stats_of(carry, position: int):
        return carry[position] if carries_solve_stats else None

    def _stats_tail(solve_stats):
        return (solve_stats,) if carries_solve_stats else ()

    def state_only_step(carry: tuple[Any, ...], index: jnp.ndarray):
        state, adaptive_time, dt_previous = carry[:3]
        solve_stats = _stats_of(carry, 3)
        fixed_time = (
            initial_time_value + jnp.asarray(index, dtype=real_dtype) * dt_value
        )
        time = fixed_time if fixed_dt else adaptive_time
        current = coordinates(state, time, time)
        if method_key == "imex":
            derivative = jnp.zeros_like(current.state)
            current_fields = None
        else:
            derivative, current_fields, _ = rhs_at(current)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time, solve_stats = advance(
            current, derivative, time, dt_local, solve_stats
        )
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
        ) + _stats_tail(solve_stats)
        return next_carry, None

    def endpoint_step(carry: tuple[Any, ...], index: jnp.ndarray):
        state, adaptive_time, dt_previous, derivative, current_fields = carry[:5]
        solve_stats = _stats_of(carry, 5)
        del index
        time = adaptive_time
        current = coordinates(state, time, time)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time, solve_stats = advance(
            current, derivative, time, dt_local, solve_stats
        )
        if method_key == "imex":
            next_derivative = jnp.zeros_like(advanced.state)
            fields, advanced_cache = fields_at(advanced)
        else:
            next_derivative, fields, advanced_cache = rhs_at(advanced)
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
            jnp.asarray(next_derivative, dtype=state_dtype),
            fields,
        ) + _stats_tail(solve_stats)
        if not record_transport:
            return next_carry, fields
        apar = jnp.zeros_like(fields.phi) if fields.apar is None else fields.apar
        bpar = jnp.zeros_like(fields.phi) if fields.bpar is None else fields.bpar
        heat_flux = heat_flux_species(
            advanced.state,
            fields.phi,
            apar,
            bpar,
            advanced_cache,
            grid,
            params,
            flux_fac,
            use_dealias=True,
            flux_scale=flux_scale,
        )
        return next_carry, (new_time, heat_flux)

    initial_state_carry = (
        jnp.asarray(project_state(G0), dtype=state_dtype),
        initial_time_value,
        initial_dt_value,
    )
    initial_stats = (
        _empty_implicit_solve_stats(state_dtype) if carries_solve_stats else None
    )
    if not record_transport and not return_fields:
        state_final_carry, output = jax.lax.scan(
            state_only_step,
            initial_state_carry + _stats_tail(initial_stats),
            jnp.arange(steps),
        )
        return state_final_carry[0], output, _stats_of(state_final_carry, 3)

    initial_update = coordinates(
        initial_state_carry[0], initial_state_carry[1], initial_state_carry[1]
    )
    if method_key == "imex":
        initial_derivative = jnp.zeros_like(initial_update.state)
        initial_fields, _ = fields_at(initial_update)
    else:
        initial_derivative, initial_fields, _ = rhs_at(initial_update)
    # Some field-solve policies use a lower internal precision. Match the scan
    # carry to the requested state precision just as every subsequent step does.
    initial_endpoint_carry = (
        initial_state_carry
        + (
            jnp.asarray(initial_derivative, dtype=state_dtype),
            initial_fields,
        )
        + _stats_tail(initial_stats)
    )
    endpoint_final_carry, output = jax.lax.scan(
        endpoint_step, initial_endpoint_carry, jnp.arange(steps)
    )
    return endpoint_final_carry[0], output, _stats_of(endpoint_final_carry, 5)


def integrate_nonlinear_sheared(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    differentiable: bool = False,
    return_fields: bool = True,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the standard-flux-tube shearing-coordinate foundation.

    This research path supports midpoint RK2, three-stage
    Heun RK3, and first-order IMEX. Stage states and derivatives are remapped to
    the stage coordinate basis before the RHS and back to the step basis before
    Runge--Kutta combinations. IMEX evaluates the explicit nonlinear term in the
    current basis and rebuilds the implicit linear operator in the endpoint
    basis. ``compressed_real_fft`` evaluates the nonlinear bracket in the
    equivalent canonical shearing-coordinate representation.

    ``return_solve_stats=True`` appends the implicit solve summary of a
    ``method="imex"`` run (:class:`~gkx.solvers_linear_implicit.ImplicitSolveStats`,
    ``None`` for the explicit methods). The traced scan cannot raise on a
    starved inner budget, so this is the convergence channel; a host-side caller
    turns it into a refusal with
    :func:`~gkx.solvers_linear_implicit.require_converged_implicit_solves`.
    """

    final_state, fields, solve_stats = _integrate_nonlinear_sheared_scan(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        shear_rate=shear_rate,
        method=method,
        cache=cache,
        terms=terms,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        compressed_real_fft=compressed_real_fft,
        differentiable=differentiable,
        return_fields=return_fields,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
    )
    head: tuple[Any, ...] = (final_state, fields) if return_fields else (final_state,)
    if return_solve_stats:
        return (*head, solve_stats)
    return head if return_fields else final_state


def integrate_nonlinear_sheared_transport(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    flux_scale: float = 1.0,
    differentiable: bool = True,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    initial_time: jnp.ndarray | float = 0.0,
    initial_dt: jnp.ndarray | float | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    return_solve_stats: bool = False,
) -> ShearedTransportTrace:
    """Integrate a sheared run and record canonical heat flux at every step.

    With ``fixed_dt=False``, ``steps`` is the accepted-step budget and ``time``
    records the resulting nonuniform physical-time grid. ``initial_time`` and
    ``initial_dt`` continue a prior trace without resetting the shearing basis.

    ``return_solve_stats=True`` fills the trace's ``solve_stats`` field with the
    implicit solve summary of a ``method="imex"`` run; it stays ``None``
    otherwise, so a caller that does not ask reads and compiles exactly what it
    did before.
    """

    final_state, samples, solve_stats = _integrate_nonlinear_sheared_scan(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        shear_rate=shear_rate,
        method=method,
        cache=cache,
        terms=terms,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        compressed_real_fft=compressed_real_fft,
        record_transport=True,
        flux_scale=flux_scale,
        differentiable=differentiable,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        initial_time=initial_time,
        initial_dt=initial_dt,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
    )
    time, heat_flux = samples
    return ShearedTransportTrace(
        final_state, time, heat_flux, solve_stats if return_solve_stats else None
    )


def integrate_nonlinear_imex_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    implicit_operator: IMEXLinearOperator | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """IMEX integrator: implicit linear operator, explicit nonlinear term.

    Returns ``(G_out, fields_t)``, or ``(G_out, fields_t, stats)`` with
    ``return_solve_stats=True``; ``stats`` is the carried
    :class:`~gkx.solvers_linear_implicit.ImplicitSolveStats` of every solve.
    """

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)
    return integrate_cached_imex_scan(
        G0,
        cache,
        params,
        dt,
        steps,
        term_cfg=term_cfg,
        linear_cfg=linear_cfg,
        linear_rhs_fn=linear_rhs_fn,
        build_operator_fn=build_nonlinear_imex_operator,
        build_implicit_operator_fn=_build_implicit_operator,
        fields_fn=compute_fields_cached,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
        return_solve_stats=return_solve_stats,
        checkpoint=checkpoint,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
        implicit_operator=implicit_operator,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        show_progress=show_progress,
    )


__all__ = [
    "PreparedExplicitNonlinearDiagnostics",
    "ShearedTransportTrace",
    "_EXPLICIT_DIAGNOSTIC_OPTION_KEYS",
    "_IMEX_DIAGNOSTIC_OPTION_KEYS",
    "_IMEX_METHODS",
    "_integrate_nonlinear_explicit_diagnostics_impl",
    "_linear_rhs_jit_for_terms",
    "_nonlinear_diagnostic_kernels",
    "_options_from_scope",
    "_pack_resolved_diagnostics",
    "_sample_axis0",
    "_sample_indices_with_final",
    "build_nonlinear_simulation_diagnostics",
    "finalize_nonlinear_scan_diagnostics",
    "integrate_imex_nonlinear_diagnostics_impl",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_explicit_diagnostics_state",
    "integrate_nonlinear_imex_cached",
    "integrate_nonlinear_imex_diagnostics",
    "integrate_nonlinear_sheared",
    "integrate_nonlinear_sheared_transport",
    "maybe_emit_nonlinear_progress",
    "nonlinear_heat_flux_window",
    "nonlinear_rhs_cached",
    "prepare_explicit_nonlinear_diagnostics_impl",
    "prepare_nonlinear_explicit_diagnostics",
    "run_sampled_explicit_diagnostic_scan",
    "sampled_scan_intervals",
    "select_nonlinear_step_diagnostics",
]
