"""Explicit diagnostic time integration for the cETG reduced model."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import resolve_cfl_fac
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.diagnostics import SimulationDiagnostics, total_energy
from spectraxgk.solvers.time.explicit import (
    _diagnostic_midplane_index,
    _instantaneous_growth_rate_step,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.integrators import _SSPX3_ADT, _SSPX3_W1, _SSPX3_W2, _SSPX3_W3
from spectraxgk.terms.reduced.cetg_model import CETGModelParams
from spectraxgk.terms.reduced.cetg_rhs import (
    _cetg_linear_omega_max,
    _cetg_nonlinear_omega_components,
    cetg_fields,
    cetg_rhs,
)
from spectraxgk.terms.reduced.cetg_state import (
    _from_internal_state,
    _project_state,
    _to_internal_state,
    _xy_mask,
)

_CETGRHS = Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]]


def _cetg_diag_weight(grid: SpectralGrid, dtype: jnp.dtype) -> jnp.ndarray:
    vol = jnp.ones((grid.z.size,), dtype=dtype)
    vol = vol / jnp.maximum(jnp.sum(vol), jnp.asarray(1.0, dtype=dtype))
    return _xy_mask(grid, dtype) * vol[None, None, None, :]


def _cetg_flux_weight(grid: SpectralGrid, dtype: jnp.dtype) -> jnp.ndarray:
    flux = jnp.ones((grid.z.size,), dtype=dtype)
    flux = flux / jnp.maximum(jnp.sum(flux), jnp.asarray(1.0, dtype=dtype))
    return _xy_mask(grid, dtype) * flux[None, None, None, :]


def _compute_cetg_diag(
    G: jnp.ndarray,
    fields: FieldState,
    phi_prev: jnp.ndarray,
    dt_step: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    *,
    mask: jnp.ndarray,
    z_index: int,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    G_int = _to_internal_state(G)
    phi = fields.phi
    gamma_modes, omega_modes = _instantaneous_growth_rate_step(
        phi, phi_prev, dt_step, z_index=z_index, mask=mask
    )
    real_dtype = jnp.real(jnp.empty((), dtype=phi.dtype)).dtype
    if omega_ky_index is not None:
        ky_i = int(np.clip(omega_ky_index, 0, int(gamma_modes.shape[0]) - 1))
        kx_i = int(np.clip(omega_kx_index or 0, 0, int(gamma_modes.shape[1]) - 1))
        gamma = jnp.nan_to_num(
            gamma_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype)
        )
        omega = jnp.nan_to_num(
            omega_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype)
        )
        phi_mode = phi[ky_i, kx_i, z_index]
    else:
        gamma = jnp.nan_to_num(
            jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)),
            nan=jnp.asarray(0.0, dtype=real_dtype),
        )
        omega = jnp.nan_to_num(
            jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)),
            nan=jnp.asarray(0.0, dtype=real_dtype),
        )
        phi_mode = jnp.asarray(0.0 + 0.0j, dtype=phi.dtype)

    diag_weight = _cetg_diag_weight(grid, real_dtype)
    flux_weight = _cetg_flux_weight(grid, real_dtype)
    W = (
        0.5
        * jnp.asarray(params.pressure, dtype=real_dtype)
        * jnp.sum(jnp.abs(G_int) ** 2 * diag_weight)
    )
    Phi2 = 0.5 * jnp.sum(jnp.abs(phi) ** 2 * diag_weight[0])
    ky = jnp.asarray(grid.ky, dtype=real_dtype)[:, None, None]
    vphi_r = -1j * ky * phi
    qflux_species = jnp.asarray(
        [
            jnp.sum(jnp.real(jnp.conj(vphi_r) * G_int[1]) * flux_weight[0])
            * jnp.asarray(params.pressure, dtype=real_dtype)
        ],
        dtype=real_dtype,
    )
    pflux_species = jnp.zeros((1,), dtype=real_dtype)
    qflux = qflux_species[0]
    pflux = pflux_species[0]
    return (
        gamma,
        omega,
        W,
        Phi2,
        jnp.asarray(0.0, dtype=real_dtype),
        qflux,
        pflux,
        qflux_species,
        pflux_species,
        phi_mode,
    )


def _cetg_explicit_step(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    *,
    dt_local: jnp.ndarray,
    method: str,
    grid: SpectralGrid,
    compressed_real_fft: bool,
    rhs_fn: _CETGRHS,
) -> jnp.ndarray:
    """Advance one projected cETG state with the requested explicit method."""

    def project(G_stage: jnp.ndarray) -> jnp.ndarray:
        return _project_state(
            G_stage,
            grid,
            compressed_real_fft=compressed_real_fft,
        )

    if method == "euler":
        return G_state + dt_local * dG
    if method == "rk2":
        k1 = dG
        k2, _ = rhs_fn(project(G_state + 0.5 * dt_local * k1))
        return G_state + dt_local * k2
    if method == "rk3_classic":
        k1 = dG
        G1 = project(G_state + dt_local * k1)
        k2, _ = rhs_fn(G1)
        G2 = project(0.75 * G_state + 0.25 * (G1 + dt_local * k2))
        k3, _ = rhs_fn(G2)
        return (1.0 / 3.0) * G_state + (2.0 / 3.0) * (G2 + dt_local * k3)
    if method in {"rk3", "rk3_heun"}:
        k1 = dG
        G1 = project(G_state + (dt_local / 3.0) * k1)
        k2, _ = rhs_fn(G1)
        G2 = project(G_state + (2.0 * dt_local / 3.0) * k2)
        k3, _ = rhs_fn(G2)
        G3 = project(G_state + 0.75 * dt_local * k3)
        return G3 + 0.25 * dt_local * k1
    if method == "rk4":
        k1 = dG
        G2 = project(G_state + 0.5 * dt_local * k1)
        k2, _ = rhs_fn(G2)
        G3 = project(G_state + 0.5 * dt_local * k2)
        k3, _ = rhs_fn(G3)
        G4 = project(G_state + dt_local * k3)
        k4, _ = rhs_fn(G4)
        return G_state + (dt_local / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if method == "sspx3":

        def euler_substep(
            G_stage: jnp.ndarray,
            dG_stage: jnp.ndarray | None = None,
        ) -> jnp.ndarray:
            if dG_stage is None:
                dG_stage, _ = rhs_fn(G_stage)
            return project(G_stage + (_SSPX3_ADT * dt_local) * dG_stage)

        # The first SSPx3 Euler substep must use the RHS that selected dt_local.
        G1 = euler_substep(G_state, dG)
        G2_euler = euler_substep(G1)
        G2 = project(
            (1.0 - _SSPX3_W1) * G_state + (_SSPX3_W1 - 1.0) * G1 + G2_euler
        )
        G3 = euler_substep(G2)
        return (
            (1.0 - _SSPX3_W2 - _SSPX3_W3) * G_state
            + _SSPX3_W3 * G1
            + (_SSPX3_W2 - 1.0) * G2
            + G3
        )

    def k10_euler_step(G_stage: jnp.ndarray) -> jnp.ndarray:
        dG_stage, _ = rhs_fn(G_stage)
        return project(G_stage + (dt_local / 6.0) * dG_stage)

    G_q1 = G_state
    G_q2 = G_state
    for _ in range(5):
        G_q1 = k10_euler_step(G_q1)
    G_q2 = 0.04 * G_q2 + 0.36 * G_q1
    G_q1 = 15.0 * G_q2 - 5.0 * G_q1
    for _ in range(4):
        G_q1 = k10_euler_step(G_q1)
    dG_final, _ = rhs_fn(G_q1)
    return G_q2 + 0.6 * G_q1 + 0.1 * dt_local * dG_final


def integrate_cetg_explicit_diagnostics_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    terms: TermConfig,
    *,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    compressed_real_fft: bool = True,
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 1.0,
    cfl_fac: float | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Integrate the cETG model and stream runtime diagnostics."""

    if method not in {
        "euler",
        "rk2",
        "rk3",
        "rk3_classic",
        "rk3_heun",
        "rk4",
        "k10",
        "sspx3",
    }:
        raise ValueError("Unsupported explicit cETG method")

    G0_int = _project_state(
        _to_internal_state(G0), grid, compressed_real_fft=compressed_real_fft
    )
    state_dtype = jnp.result_type(G0_int, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_init = jnp.asarray(dt, dtype=real_dtype)
    dt_min_val = jnp.asarray(dt_min, dtype=real_dtype)
    dt_max_val = jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype)
    cfl_val = jnp.asarray(cfl, dtype=real_dtype)
    cfl_fac_val = jnp.asarray(resolve_cfl_fac(method, cfl_fac), dtype=real_dtype)
    z_idx = _diagnostic_midplane_index(grid.z.size)
    mask = jnp.broadcast_to(
        jnp.asarray(grid.dealias_mask, dtype=bool), (grid.ky.size, grid.kx.size)
    )
    linear_omega = jnp.asarray(_cetg_linear_omega_max(grid, params), dtype=real_dtype)

    def _update_dt(fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if fixed_dt:
            return dt_prev
        omega_nl_x, omega_nl_y = _cetg_nonlinear_omega_components(
            fields_state.phi, grid
        )
        wmax = linear_omega + omega_nl_x + omega_nl_y
        dt_guess = jnp.where(wmax > 0.0, cfl_fac_val * cfl_val / wmax, dt_prev)
        return jnp.asarray(jnp.clip(dt_guess, dt_min_val, dt_max_val), dtype=real_dtype)

    def rhs_fn(G_state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        return cetg_rhs(
            G_state, grid, params, terms, compressed_real_fft=compressed_real_fft
        )

    fields0 = cetg_fields(G0_int, grid, params, apply_kz_dealias=False)
    dt0 = _update_dt(fields0, dt_init)
    diag_zero = _compute_cetg_diag(
        G0_int,
        fields0,
        fields0.phi,
        dt0,
        grid,
        params,
        mask=mask,
        z_index=z_idx,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
    )

    def step(carry, idx):
        G_state, fields_state, phi_prev, diag_prev, t_prev, dt_prev = carry
        dt_local = _update_dt(fields_state, dt_prev)
        dG, _fields_used = cetg_rhs(
            G_state,
            grid,
            params,
            terms,
            compressed_real_fft=compressed_real_fft,
            fields_override=fields_state,
        )
        G_new = _cetg_explicit_step(
            G_state,
            dG,
            dt_local=dt_local,
            method=method,
            grid=grid,
            compressed_real_fft=compressed_real_fft,
            rhs_fn=rhs_fn,
        )

        G_new = _project_state(G_new, grid, compressed_real_fft=compressed_real_fft)
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = jnp.asarray(t_prev + dt_local, dtype=real_dtype)
        fields_new = cetg_fields(G_new, grid, params)

        def _compute_diag(_):
            return _compute_cetg_diag(
                G_new,
                fields_new,
                phi_prev,
                dt_local,
                grid,
                params,
                mask=mask,
                z_index=z_idx,
                omega_ky_index=omega_ky_index,
                omega_kx_index=omega_kx_index,
            )

        def _reuse_diag(_):
            return diag_prev

        do_diag = (idx % int(max(diagnostics_stride, 1))) == 0
        diag = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        return (G_new, fields_new, fields_new.phi, diag, t_new, dt_local), (
            diag,
            t_new,
            dt_local,
        )

    idx = jnp.arange(steps, dtype=jnp.int32)
    scan_step = step
    if show_progress:
        from spectraxgk.utils.callbacks import print_callback, should_emit_progress

        def scan_step(carry, idx):
            carry_out, diag_out = step(carry, idx)
            diag_vals, _t_out, _dt_out = diag_out
            gamma_cb, omega_cb, Wg_cb, Wphi_cb = (
                diag_vals[0],
                diag_vals[1],
                diag_vals[2],
                diag_vals[3],
            )
            _dt_out = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state, idx, steps, gamma_cb, omega_cb, Wphi_cb, Wg_cb, _t_out, None
                ),
                lambda state: state,
                _dt_out,
            )
            return carry_out, diag_out

    (G_final, fields_last, phi_last, _diag_last, _t_last, _dt_last), diag_out = (
        jax.lax.scan(
            scan_step,
            (
                G0_int,
                fields0,
                fields0.phi,
                diag_zero,
                jnp.asarray(0.0, dtype=real_dtype),
                dt0,
            ),
            idx,
            length=steps,
        )
    )
    diag, t, dt_series = diag_out
    (
        gamma_t,
        omega_t,
        Wg_t,
        Wphi_t,
        Wapar_t,
        heat_t,
        pflux_t,
        heat_s_t,
        pflux_s_t,
        phi_mode_t,
    ) = diag

    stride = int(max(sample_stride, diagnostics_stride, 1))
    if stride > 1:
        gamma_t = gamma_t[::stride]
        omega_t = omega_t[::stride]
        Wg_t = Wg_t[::stride]
        Wphi_t = Wphi_t[::stride]
        Wapar_t = Wapar_t[::stride]
        heat_t = heat_t[::stride]
        pflux_t = pflux_t[::stride]
        heat_s_t = heat_s_t[::stride, ...]
        pflux_s_t = pflux_s_t[::stride, ...]
        phi_mode_t = phi_mode_t[::stride]
        t = t[::stride]
        dt_series = dt_series[::stride]

    diag_out_final = SimulationDiagnostics(
        t=t,
        dt_t=dt_series,
        dt_mean=jnp.mean(dt_series)
        if int(np.asarray(dt_series).size)
        else jnp.asarray(0.0, dtype=real_dtype),
        gamma_t=gamma_t,
        omega_t=omega_t,
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=total_energy(Wg_t, Wphi_t, Wapar_t),
        heat_flux_species_t=heat_s_t,
        particle_flux_species_t=pflux_s_t,
        phi_mode_t=phi_mode_t,
    )
    return (
        t,
        diag_out_final,
        _from_internal_state(G_final),
        FieldState(phi=fields_last.phi, apar=None, bpar=None),
    )


__all__ = [
    "_cetg_diag_weight",
    "_cetg_flux_weight",
    "_compute_cetg_diag",
    "integrate_cetg_explicit_diagnostics_state",
]
