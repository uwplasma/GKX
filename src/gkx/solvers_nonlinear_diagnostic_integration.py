"""Diagnostic nonlinear integration entry points and dependency wiring."""

from __future__ import annotations

from typing import Any
import jax.numpy as jnp
import jax
import numpy as np

from gkx.config import resolve_cfl_fac
from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.core_grid import SpectralGrid
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.terms.assembly import compute_fields_cached
from gkx.terms.config import FieldState, TermConfig
from gkx.terms.nonlinear import nonlinear_em_contribution
from gkx.diagnostics_contract import (
    SimulationDiagnostics,
    ResolvedDiagnostics,
)
from gkx.operators.fluxes import (
    heat_flux_species,
    particle_flux_species,
    turbulent_heating_species,
)
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
from gkx.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    make_nonlinear_diagnostic_tuple_fn,
)
from gkx.operators.nonlinear.policies import (
    _diagnostic_omega_mode_mask,
    _nonlinear_cfl_frequency_components,
    build_nonlinear_collision_split_policy,
    build_nonlinear_diagnostic_setup,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)
from gkx.operators.nonlinear.collisions import (
    _apply_collision_split,
    _collision_damping,
)
from gkx.operators.nonlinear.rhs import nonlinear_em_term_cached_impl
from gkx.solvers_nonlinear_diagnostics import (
    ExplicitNonlinearDiagnosticsDeps,
    IMEXNonlinearDiagnosticsDeps,
    PreparedExplicitNonlinearDiagnostics,
    integrate_explicit_nonlinear_diagnostics_impl,
    integrate_imex_nonlinear_diagnostics_impl,
    prepare_explicit_nonlinear_diagnostics_impl,
)
from gkx.solvers_nonlinear_explicit import (
    make_explicit_diagnostic_step,
    run_explicit_diagnostic_scan,
)
from gkx.solvers_nonlinear_imex import (
    make_imex_diagnostic_step,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    make_imex_solve_step_with_stats,
    run_imex_diagnostic_scan,
    solve_imex_step,
)
from gkx.solvers_time_explicit import (
    _diagnostic_midplane_index,
    _instantaneous_growth_rate_step,
    _laguerre_velocity_max,
    _linear_frequency_bound,
)
from gkx.solvers_nonlinear_state_integration import (
    _linear_rhs_jit_for_terms,
    nonlinear_rhs_cached,
)


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


_IMEX_METHODS = {"imex", "semi-implicit"}

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
    """Return diagnostic kernels for dependency-injected nonlinear diagnostics."""

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


def _common_nonlinear_diagnostics_deps() -> dict[str, Any]:
    """Return facade bindings shared by explicit and IMEX diagnostics."""

    return {
        "ensure_geometry_fn": ensure_flux_tube_geometry_data,
        "build_cache_fn": build_linear_cache,
        "quadrature_weights_fn": fieldline_quadrature_weights,
        "omega_mask_fn": _diagnostic_omega_mode_mask,
        "midplane_index_fn": _diagnostic_midplane_index,
        "collision_damping_fn": _collision_damping,
        "compute_fields_fn": compute_fields_cached,
        "diagnostic_kernels_fn": _nonlinear_diagnostic_kernels,
        "build_diagnostic_setup_fn": build_nonlinear_diagnostic_setup,
        "build_collision_split_policy_fn": build_nonlinear_collision_split_policy,
        "make_diagnostic_tuple_fn": make_nonlinear_diagnostic_tuple_fn,
        "finalize_scan_diagnostics_fn": finalize_nonlinear_scan_diagnostics,
        "select_step_diagnostics_fn": select_nonlinear_step_diagnostics,
        "emit_progress_fn": maybe_emit_nonlinear_progress,
        "apply_collision_split_fn": _apply_collision_split,
    }


def _explicit_nonlinear_diagnostics_deps() -> ExplicitNonlinearDiagnosticsDeps:
    """Collect dependencies for explicit diagnostic integration."""

    return ExplicitNonlinearDiagnosticsDeps(
        **_common_nonlinear_diagnostics_deps(),
        resolve_cfl_fac_fn=resolve_cfl_fac,
        linear_frequency_bound_fn=_linear_frequency_bound,
        laguerre_velocity_max_fn=_laguerre_velocity_max,
        cfl_frequency_components_fn=_nonlinear_cfl_frequency_components,
        nonlinear_rhs_fn=nonlinear_rhs_cached,
        build_time_step_policy_fn=build_nonlinear_time_step_policy,
        make_explicit_step_fn=make_explicit_diagnostic_step,
        run_explicit_scan_fn=run_explicit_diagnostic_scan,
        run_sampled_explicit_scan_fn=run_sampled_explicit_diagnostic_scan,
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
    """Integrate nonlinear system and return runtime diagnostics plus final state."""

    options = _options_from_scope(locals(), _EXPLICIT_DIAGNOSTIC_OPTION_KEYS)
    return integrate_explicit_nonlinear_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        deps=_explicit_nonlinear_diagnostics_deps(),
        **options,
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
        deps=_explicit_nonlinear_diagnostics_deps(),
        **options,
    )


def _imex_nonlinear_diagnostics_deps() -> IMEXNonlinearDiagnosticsDeps:
    """Collect dependencies for IMEX diagnostic integration."""

    return IMEXNonlinearDiagnosticsDeps(
        **_common_nonlinear_diagnostics_deps(),
        linear_rhs_for_terms_fn=_linear_rhs_jit_for_terms,
        build_imex_operator_fn=build_nonlinear_imex_operator,
        make_imex_nonlinear_term_fn=make_imex_nonlinear_term,
        make_imex_solve_step_fn=make_imex_solve_step,
        make_imex_solve_step_with_stats_fn=make_imex_solve_step_with_stats,
        solve_imex_step_fn=solve_imex_step,
        make_imex_step_fn=make_imex_diagnostic_step,
        run_imex_scan_fn=run_imex_diagnostic_scan,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
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
        deps=_imex_nonlinear_diagnostics_deps(),
        return_solve_stats=return_solve_stats,
        **options,
    )


__all__ = [
    "_EXPLICIT_DIAGNOSTIC_OPTION_KEYS",
    "_IMEX_METHODS",
    "_IMEX_DIAGNOSTIC_OPTION_KEYS",
    "_explicit_nonlinear_diagnostics_deps",
    "_imex_nonlinear_diagnostics_deps",
    "_integrate_nonlinear_explicit_diagnostics_impl",
    "_nonlinear_diagnostic_kernels",
    "_options_from_scope",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_explicit_diagnostics_state",
    "integrate_nonlinear_imex_diagnostics",
    "prepare_nonlinear_explicit_diagnostics",
    "_pack_resolved_diagnostics",
    "_sample_axis0",
    "_sample_indices_with_final",
    "build_nonlinear_simulation_diagnostics",
    "finalize_nonlinear_scan_diagnostics",
    "maybe_emit_nonlinear_progress",
    "run_sampled_explicit_diagnostic_scan",
    "sampled_scan_intervals",
    "select_nonlinear_step_diagnostics",
]
