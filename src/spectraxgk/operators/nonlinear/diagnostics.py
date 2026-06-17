"""Diagnostic packing and sampling helpers for nonlinear integrations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.metadata import ResolvedDiagnostics, SimulationDiagnostics

__all__ = [
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


def _pack_resolved_diagnostics(resolved_t: tuple[np.ndarray, ...]) -> ResolvedDiagnostics:
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
    return np.diff(np.concatenate([np.asarray([0], dtype=np.int32), sample_steps])).astype(
        np.int32
    )


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
    )


def select_nonlinear_step_diagnostics(
    idx: Any,
    *,
    diagnostics_stride: int,
    diag_prev: Any,
    compute_diag_fn: Any,
) -> Any:
    """Return a fresh or reused nonlinear step diagnostic tuple."""

    diag_stride = int(max(diagnostics_stride, 1))
    do_diag = (idx % diag_stride) == 0
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

    from spectraxgk.utils.callbacks import print_callback, should_emit_progress

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
