"""Nonlinear diagnostic integration drivers.

This module owns the orchestration for explicit nonlinear diagnostic runs.
:mod:`gkx.solvers.nonlinear.diagnostic_integration` injects the concrete
kernels so tests and downstream users can still patch these seams without
keeping the large implementation body inline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp

from gkx.diagnostics import SimulationDiagnostics
from gkx.geometry import FluxTubeGeometryLike
from gkx.core.grid import SpectralGrid
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams
from gkx.terms.config import FieldState, TermConfig
from gkx.solvers.nonlinear.imex_diagnostics import (
    IMEXNonlinearDiagnosticsDeps,
    integrate_imex_nonlinear_diagnostics_impl,
)


@dataclass(frozen=True)
class ExplicitNonlinearDiagnosticsDeps:
    """Patchable kernels used by the explicit diagnostic integrator."""

    ensure_geometry_fn: Callable[..., Any]
    build_cache_fn: Callable[..., Any]
    quadrature_weights_fn: Callable[..., Any]
    omega_mask_fn: Callable[..., Any]
    midplane_index_fn: Callable[..., Any]
    resolve_cfl_fac_fn: Callable[..., Any]
    linear_frequency_bound_fn: Callable[..., Any]
    laguerre_velocity_max_fn: Callable[..., Any]
    cfl_frequency_components_fn: Callable[..., Any]
    collision_damping_fn: Callable[..., Any]
    nonlinear_rhs_fn: Callable[..., Any]
    compute_fields_fn: Callable[..., Any]
    diagnostic_kernels_fn: Callable[..., Any]
    build_diagnostic_setup_fn: Callable[..., Any]
    build_time_step_policy_fn: Callable[..., Any]
    build_collision_split_policy_fn: Callable[..., Any]
    make_diagnostic_tuple_fn: Callable[..., Any]
    make_explicit_step_fn: Callable[..., Any]
    run_explicit_scan_fn: Callable[..., Any]
    run_sampled_explicit_scan_fn: Callable[..., Any]
    finalize_scan_diagnostics_fn: Callable[..., Any]
    select_step_diagnostics_fn: Callable[..., Any]
    emit_progress_fn: Callable[..., Any]
    apply_collision_split_fn: Callable[..., Any]


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


@dataclass(frozen=True)
class PreparedExplicitNonlinearDiagnostics:
    """Reusable compiled explicit nonlinear diagnostic simulation.

    Geometry, field operators, and static numerical policy are prepared once.
    Calls to :meth:`run` may supply a new initial state with the same shape and
    dtype without rebuilding the scan closure. Fixed-step sensitivity studies
    may instead pass matched geometry, cache, and parameter PyTrees.
    """

    initial_state: jnp.ndarray
    geometry: Any
    cache: LinearCache
    params: LinearParams
    _run_raw: Callable[[jnp.ndarray], tuple[Any, Any, Any]]
    _run_dynamic_raw: Callable[
        [jnp.ndarray, Any, LinearCache, LinearParams], tuple[Any, Any, Any]
    ]
    _finalize: Callable[..., SimulationDiagnostics]
    stride: int
    sampled_scan: bool
    resolved_diagnostics: bool
    fixed_dt: bool

    def run_arrays(
        self,
        initial_state: jnp.ndarray | None = None,
        *,
        geometry: Any | None = None,
        cache: LinearCache | None = None,
        params: LinearParams | None = None,
    ) -> tuple[jnp.ndarray, tuple[Any, Any, Any], FieldState]:
        """Run the compiled scan without host conversion or artifact assembly.

        This method is the differentiable Python boundary. The initial state is
        dynamic. Fixed-step runs may also receive a matched ``cache``/``params``
        pair for parameter differentiation. A changed ``geometry`` requires
        that pair. Grid layout and numerical policy remain fixed by
        :func:`prepare_nonlinear_explicit_diagnostics`; adaptive runs currently
        support state changes but reject traced model overrides.
        """

        if (cache is None) != (params is None):
            raise ValueError("cache and params must be supplied together")
        if geometry is not None and cache is None:
            raise ValueError(
                "dynamic geometry requires matched cache and params inputs"
            )
        state = self.initial_state if initial_state is None else initial_state
        if geometry is None and cache is None:
            return self._run_raw(jnp.asarray(state))
        if not self.fixed_dt:
            raise ValueError("dynamic geometry, cache, or params require fixed_dt=True")
        geometry_use = self.geometry if geometry is None else geometry
        cache_use = self.cache if cache is None else cache
        params_use = self.params if params is None else params
        return self._run_dynamic_raw(
            jnp.asarray(state), geometry_use, cache_use, params_use
        )

    def run(
        self,
        initial_state: jnp.ndarray | None = None,
        *,
        geometry: Any | None = None,
        cache: LinearCache | None = None,
        params: LinearParams | None = None,
    ) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
        """Advance one state through the prepared compiled simulation."""

        G_final, scan_diag_out, fields_final = self.run_arrays(
            initial_state, geometry=geometry, cache=cache, params=params
        )
        diag, t, dt_series = scan_diag_out
        diag_out = self._finalize(
            diag,
            t=t,
            dt_series=dt_series,
            stride=self.stride,
            sampled_scan=self.sampled_scan,
            resolved_diagnostics=self.resolved_diagnostics,
        )
        return jnp.asarray(diag_out.t), diag_out, G_final, fields_final


_EXPLICIT_DIAGNOSTIC_OPTION_KEYS = tuple(_ExplicitDiagnosticOptions.__annotations__)


def _explicit_options_from_values(values: dict[str, Any]) -> _ExplicitDiagnosticOptions:
    """Pack public keyword values into a single internal options object."""

    return _ExplicitDiagnosticOptions(
        **{key: values[key] for key in _EXPLICIT_DIAGNOSTIC_OPTION_KEYS}
    )


def _discard_imex_only_options(*_unused: Any) -> None:
    """Document IMEX-only options accepted by the shared public signature."""

    return None


def _prepare_explicit_diagnostic_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
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
    if method in {"imex", "semi-implicit"}:
        raise ValueError(
            "Final-state runtime diagnostics helper only supports explicit methods"
        )
    setup = deps.build_diagnostic_setup_fn(
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
        ensure_geometry_fn=deps.ensure_geometry_fn,
        build_cache_fn=deps.build_cache_fn,
        quadrature_weights_fn=deps.quadrature_weights_fn,
        omega_mask_fn=deps.omega_mask_fn,
        midplane_index_fn=deps.midplane_index_fn,
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
    deps: ExplicitNonlinearDiagnosticsDeps,
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

    time_step_policy = deps.build_time_step_policy_fn(
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
        resolve_cfl_fac_fn=deps.resolve_cfl_fac_fn,
        linear_frequency_bound_fn=deps.linear_frequency_bound_fn,
        laguerre_velocity_max_fn=deps.laguerre_velocity_max_fn,
        cfl_frequency_components_fn=deps.cfl_frequency_components_fn,
    )
    collision_policy = deps.build_collision_split_policy_fn(
        prepared.cache,
        params,
        prepared.term_cfg,
        prepared.real_dtype,
        squeeze_species=prepared.G0.ndim == 5,
        collision_split=collision_split,
        collision_damping_fn=deps.collision_damping_fn,
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
    deps: ExplicitNonlinearDiagnosticsDeps,
    compressed_real_fft: bool,
    laguerre_mode: str,
    external_phi: jnp.ndarray | float | None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the nonlinear RHS closure with collision policy baked in."""

    def rhs_fn(G):
        return deps.nonlinear_rhs_fn(
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
    deps: ExplicitNonlinearDiagnosticsDeps,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
    resolved_diagnostics: bool,
) -> Callable[..., Any]:
    """Return the state-to-diagnostic tuple closure for explicit scans."""

    return deps.make_diagnostic_tuple_fn(
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
        kernels=deps.diagnostic_kernels_fn(),
    )


def _make_explicit_scan_step(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    rhs_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_diag_from_state: Callable[..., Any],
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
    method: str,
    diagnostics_stride: int,
    show_progress: bool,
    steps: int,
    external_phi: jnp.ndarray | float | None,
    collision_scheme: str,
) -> Callable[..., Any]:
    """Build the explicit scan-step closure from prepared policies."""

    return deps.make_explicit_step_fn(
        rhs_fn=rhs_fn,
        method=method,
        project_state=prepared.project_state,
        state_dtype=prepared.state_dtype,
        real_dtype=prepared.real_dtype,
        time_step_policy=policies.time_step_policy,
        compute_fields_fn=deps.compute_fields_fn,
        cache=prepared.cache,
        params=params,
        term_cfg=prepared.term_cfg,
        external_phi=external_phi,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=diagnostics_stride,
        select_diagnostics_fn=deps.select_step_diagnostics_fn,
        show_progress=show_progress,
        steps=steps,
        emit_progress_fn=deps.emit_progress_fn,
        use_collision_split=policies.collision_policy.active,
        damping=policies.collision_policy.damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=deps.apply_collision_split_fn,
    )


def _run_explicit_diagnostic_scan_and_finalize(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    step: Callable[..., Any],
    compute_diag_from_state: Callable[..., Any],
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
    steps: int,
    sample_stride: int,
    diagnostics_stride: int,
    checkpoint: bool,
    resolved_diagnostics: bool,
    external_phi: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Run the explicit scan and convert raw scan output into diagnostics."""

    G_final, scan_diag_out, fields_final = _run_explicit_diagnostic_scan_raw(
        prepared,
        policies,
        step,
        compute_diag_from_state,
        params,
        deps=deps,
        initial_state=prepared.G0,
        steps=steps,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        checkpoint=checkpoint,
        external_phi=external_phi,
    )
    diag, t, dt_series = scan_diag_out
    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
    diag_out = deps.finalize_scan_diagnostics_fn(
        diag,
        t=t,
        dt_series=dt_series,
        stride=stride,
        sampled_scan=sampled_scan,
        resolved_diagnostics=resolved_diagnostics,
    )
    return jnp.asarray(diag_out.t), diag_out, G_final, fields_final


def _run_explicit_diagnostic_scan_raw(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    step: Callable[..., Any],
    compute_diag_from_state: Callable[..., Any],
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
    initial_state: jnp.ndarray,
    steps: int,
    sample_stride: int,
    diagnostics_stride: int,
    checkpoint: bool,
    external_phi: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, tuple[Any, Any, Any], FieldState]:
    """Run the device scan and return raw arrays for outside-JIT packaging."""

    G0 = prepared.project_state(jnp.asarray(initial_state, dtype=prepared.state_dtype))

    fields0 = deps.compute_fields_fn(
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
    diag_zero = compute_diag_from_state(G0, fields0, G0, fields0, dt0)
    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
    G_final, scan_diag_out = deps.run_explicit_scan_fn(
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
        sampled_scan_fn=deps.run_sampled_explicit_scan_fn,
    )

    fields_final = deps.compute_fields_fn(
        G_final,
        prepared.cache,
        params,
        terms=prepared.term_cfg,
        external_phi=external_phi,
    )
    return G_final, scan_diag_out, fields_final


def _build_explicit_scan_closures(
    prepared: _ExplicitPreparedState,
    policies: _ExplicitRuntimePolicies,
    grid: SpectralGrid,
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
    options: _ExplicitDiagnosticOptions,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build diagnostic scan closures after state and policies are prepared."""

    rhs_fn = _make_explicit_rhs_fn(
        prepared,
        policies,
        params,
        deps=deps,
        compressed_real_fft=options.compressed_real_fft,
        laguerre_mode=options.laguerre_mode,
        external_phi=options.external_phi,
    )
    compute_diag_from_state = _make_explicit_diagnostic_callable(
        prepared,
        grid,
        params=params,
        deps=deps,
        omega_ky_index=options.omega_ky_index,
        omega_kx_index=options.omega_kx_index,
        flux_scale=options.flux_scale,
        wphi_scale=options.wphi_scale,
        resolved_diagnostics=options.resolved_diagnostics,
    )
    step = _make_explicit_scan_step(
        prepared,
        policies,
        rhs_fn,
        compute_diag_from_state,
        params,
        deps=deps,
        method=options.method,
        diagnostics_stride=options.diagnostics_stride,
        show_progress=options.show_progress,
        steps=options.steps,
        external_phi=options.external_phi,
        collision_scheme=options.collision_scheme,
    )
    return step, compute_diag_from_state


def _build_explicit_scan_components(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
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
        deps=deps,
        method=options.method,
        cache=cache,
        terms=terms,
        use_dealias_mask=options.use_dealias_mask,
        z_index=options.z_index,
        compressed_real_fft=options.compressed_real_fft,
        fixed_mode_ky_index=options.fixed_mode_ky_index,
        fixed_mode_kx_index=options.fixed_mode_kx_index,
    )
    policies = _build_explicit_runtime_policies(
        prepared,
        grid,
        params,
        deps=deps,
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
        deps=deps,
        options=options,
    )
    return _ExplicitScanComponents(
        prepared=prepared,
        policies=policies,
        step=step,
        compute_diag_from_state=compute_diag_from_state,
    )


def _run_explicit_scan_components(
    components: _ExplicitScanComponents,
    params: LinearParams,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
    options: _ExplicitDiagnosticOptions,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Run prepared explicit nonlinear diagnostic components."""

    return _run_explicit_diagnostic_scan_and_finalize(
        components.prepared,
        components.policies,
        components.step,
        components.compute_diag_from_state,
        params,
        deps=deps,
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
    deps: ExplicitNonlinearDiagnosticsDeps,
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

    _discard_imex_only_options(
        implicit_tol,
        implicit_maxiter,
        implicit_iters,
        implicit_relax,
        implicit_restart,
        implicit_preconditioner,
    )
    options = _explicit_options_from_values(locals())
    components = _build_explicit_scan_components(
        G0,
        grid,
        geom,
        params,
        deps=deps,
        cache=cache,
        terms=terms,
        options=options,
    )
    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"

    def run_raw(initial_state: jnp.ndarray) -> tuple[Any, Any, Any]:
        return _run_explicit_diagnostic_scan_raw(
            components.prepared,
            components.policies,
            components.step,
            components.compute_diag_from_state,
            params,
            deps=deps,
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
        dynamic_setup = deps.build_diagnostic_setup_fn(
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
            ensure_geometry_fn=deps.ensure_geometry_fn,
            build_cache_fn=deps.build_cache_fn,
            quadrature_weights_fn=deps.quadrature_weights_fn,
            omega_mask_fn=deps.omega_mask_fn,
            midplane_index_fn=deps.midplane_index_fn,
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
            deps=deps,
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
            deps=deps,
            options=options,
        )
        return _run_explicit_diagnostic_scan_raw(
            dynamic_prepared,
            dynamic_policies,
            dynamic_step,
            dynamic_diagnostics,
            dynamic_params,
            deps=deps,
            initial_state=initial_state,
            steps=steps,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            checkpoint=checkpoint,
            external_phi=external_phi,
        )

    return PreparedExplicitNonlinearDiagnostics(
        initial_state=components.prepared.G0,
        geometry=components.prepared.setup.geom,
        cache=components.prepared.cache,
        params=params,
        _run_raw=jax.jit(run_raw),
        _run_dynamic_raw=jax.jit(run_dynamic_raw),
        _finalize=deps.finalize_scan_diagnostics_fn,
        stride=stride,
        sampled_scan=sampled_scan,
        resolved_diagnostics=resolved_diagnostics,
        fixed_dt=options.fixed_dt,
    )


def integrate_explicit_nonlinear_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
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

    _discard_imex_only_options(
        implicit_tol,
        implicit_maxiter,
        implicit_iters,
        implicit_relax,
        implicit_restart,
        implicit_preconditioner,
    )
    options = _explicit_options_from_values(locals())
    components = _build_explicit_scan_components(
        G0,
        grid,
        geom,
        params,
        deps=deps,
        cache=cache,
        terms=terms,
        options=options,
    )
    return _run_explicit_scan_components(components, params, deps=deps, options=options)


__all__ = [
    "ExplicitNonlinearDiagnosticsDeps",
    "IMEXNonlinearDiagnosticsDeps",
    "PreparedExplicitNonlinearDiagnostics",
    "integrate_explicit_nonlinear_diagnostics_impl",
    "integrate_imex_nonlinear_diagnostics_impl",
    "prepare_explicit_nonlinear_diagnostics_impl",
]
