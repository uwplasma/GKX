"""Explicit diagnostic time integration for the cETG reduced model."""

from __future__ import annotations

from dataclasses import dataclass
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
_CETGProject = Callable[[jnp.ndarray], jnp.ndarray]
_SUPPORTED_CETG_EXPLICIT_METHODS = {
    "euler",
    "rk2",
    "rk3",
    "rk3_classic",
    "rk3_heun",
    "rk4",
    "k10",
    "sspx3",
}


def _cetg_diag_weight(grid: SpectralGrid, dtype: jnp.dtype) -> jnp.ndarray:
    vol = jnp.ones((grid.z.size,), dtype=dtype)
    vol = vol / jnp.maximum(jnp.sum(vol), jnp.asarray(1.0, dtype=dtype))
    return _xy_mask(grid, dtype) * vol[None, None, None, :]


def _cetg_flux_weight(grid: SpectralGrid, dtype: jnp.dtype) -> jnp.ndarray:
    flux = jnp.ones((grid.z.size,), dtype=dtype)
    flux = flux / jnp.maximum(jnp.sum(flux), jnp.asarray(1.0, dtype=dtype))
    return _xy_mask(grid, dtype) * flux[None, None, None, :]


def _cetg_mode_diagnostics(
    phi: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt_step: jnp.ndarray,
    *,
    mask: jnp.ndarray,
    z_index: int,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        return gamma, omega, phi[ky_i, kx_i, z_index]
    gamma = jnp.nan_to_num(
        jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)),
        nan=jnp.asarray(0.0, dtype=real_dtype),
    )
    omega = jnp.nan_to_num(
        jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)),
        nan=jnp.asarray(0.0, dtype=real_dtype),
    )
    return gamma, omega, jnp.asarray(0.0 + 0.0j, dtype=phi.dtype)


def _cetg_energy_flux_diagnostics(
    G_int: jnp.ndarray,
    phi: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    real_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diag_weight = _cetg_diag_weight(grid, real_dtype)
    flux_weight = _cetg_flux_weight(grid, real_dtype)
    pressure = jnp.asarray(params.pressure, dtype=real_dtype)
    W = 0.5 * pressure * jnp.sum(jnp.abs(G_int) ** 2 * diag_weight)
    Phi2 = 0.5 * jnp.sum(jnp.abs(phi) ** 2 * diag_weight[0])
    ky = jnp.asarray(grid.ky, dtype=real_dtype)[:, None, None]
    vphi_r = -1j * ky * phi
    qflux_species = jnp.asarray(
        [jnp.sum(jnp.real(jnp.conj(vphi_r) * G_int[1]) * flux_weight[0]) * pressure],
        dtype=real_dtype,
    )
    pflux_species = jnp.zeros((1,), dtype=real_dtype)
    return W, Phi2, qflux_species[0], pflux_species[0], qflux_species, pflux_species


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
    real_dtype = jnp.real(jnp.empty((), dtype=phi.dtype)).dtype
    gamma, omega, phi_mode = _cetg_mode_diagnostics(
        phi,
        phi_prev,
        dt_step,
        mask=mask,
        z_index=z_index,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
    )
    W, Phi2, qflux, pflux, qflux_species, pflux_species = (
        _cetg_energy_flux_diagnostics(G_int, phi, grid, params, real_dtype)
    )
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


def _project_cetg_stage(
    G_stage: jnp.ndarray,
    grid: SpectralGrid,
    *,
    compressed_real_fft: bool,
) -> jnp.ndarray:
    return _project_state(
        G_stage,
        grid,
        compressed_real_fft=compressed_real_fft,
    )


def _cetg_step_rk2(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    rhs_fn: _CETGRHS,
    project: _CETGProject,
) -> jnp.ndarray:
    k2, _ = rhs_fn(project(G_state + 0.5 * dt_local * dG))
    return G_state + dt_local * k2


def _cetg_step_rk3_classic(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    rhs_fn: _CETGRHS,
    project: _CETGProject,
) -> jnp.ndarray:
    G1 = project(G_state + dt_local * dG)
    k2, _ = rhs_fn(G1)
    G2 = project(0.75 * G_state + 0.25 * (G1 + dt_local * k2))
    k3, _ = rhs_fn(G2)
    return (1.0 / 3.0) * G_state + (2.0 / 3.0) * (G2 + dt_local * k3)


def _cetg_step_rk3_heun(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    rhs_fn: _CETGRHS,
    project: _CETGProject,
) -> jnp.ndarray:
    G1 = project(G_state + (dt_local / 3.0) * dG)
    k2, _ = rhs_fn(G1)
    G2 = project(G_state + (2.0 * dt_local / 3.0) * k2)
    k3, _ = rhs_fn(G2)
    G3 = project(G_state + 0.75 * dt_local * k3)
    return G3 + 0.25 * dt_local * dG


def _cetg_step_rk4(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    rhs_fn: _CETGRHS,
    project: _CETGProject,
) -> jnp.ndarray:
    G2 = project(G_state + 0.5 * dt_local * dG)
    k2, _ = rhs_fn(G2)
    G3 = project(G_state + 0.5 * dt_local * k2)
    k3, _ = rhs_fn(G3)
    G4 = project(G_state + dt_local * k3)
    k4, _ = rhs_fn(G4)
    return G_state + (dt_local / 6.0) * (dG + 2.0 * k2 + 2.0 * k3 + k4)


def _cetg_step_sspx3(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    rhs_fn: _CETGRHS,
    project: _CETGProject,
) -> jnp.ndarray:
    def euler_substep(
        G_stage: jnp.ndarray,
        dG_stage: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if dG_stage is None:
            dG_stage, _ = rhs_fn(G_stage)
        return project(G_stage + (_SSPX3_ADT * dt_local) * dG_stage)

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


def _cetg_step_k10(
    G_state: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    rhs_fn: _CETGRHS,
    project: _CETGProject,
) -> jnp.ndarray:
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
        return _project_cetg_stage(G_stage, grid, compressed_real_fft=compressed_real_fft)

    if method == "euler":
        return G_state + dt_local * dG
    if method == "rk2":
        return _cetg_step_rk2(G_state, dG, dt_local, rhs_fn, project)
    if method == "rk3_classic":
        return _cetg_step_rk3_classic(G_state, dG, dt_local, rhs_fn, project)
    if method in {"rk3", "rk3_heun"}:
        return _cetg_step_rk3_heun(G_state, dG, dt_local, rhs_fn, project)
    if method == "rk4":
        return _cetg_step_rk4(G_state, dG, dt_local, rhs_fn, project)
    if method == "sspx3":
        return _cetg_step_sspx3(G_state, dG, dt_local, rhs_fn, project)
    return _cetg_step_k10(G_state, dG, dt_local, rhs_fn, project)


@dataclass(frozen=True)
class _CETGScanContext:
    grid: SpectralGrid
    params: CETGModelParams
    terms: TermConfig
    method: str
    compressed_real_fft: bool
    fixed_dt: bool
    state_dtype: jnp.dtype
    real_dtype: jnp.dtype
    dt_min: jnp.ndarray
    dt_max: jnp.ndarray
    cfl: jnp.ndarray
    cfl_fac: jnp.ndarray
    linear_omega: jnp.ndarray
    mask: jnp.ndarray
    z_index: int
    diagnostics_stride: int
    omega_ky_index: int | None
    omega_kx_index: int | None

    def update_dt(self, fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if self.fixed_dt:
            return dt_prev
        omega_nl_x, omega_nl_y = _cetg_nonlinear_omega_components(
            fields_state.phi, self.grid
        )
        wmax = self.linear_omega + omega_nl_x + omega_nl_y
        dt_guess = jnp.where(wmax > 0.0, self.cfl_fac * self.cfl / wmax, dt_prev)
        return jnp.asarray(jnp.clip(dt_guess, self.dt_min, self.dt_max), dtype=self.real_dtype)

    def rhs(
        self,
        G_state: jnp.ndarray,
        *,
        fields_override: FieldState | None = None,
    ) -> tuple[jnp.ndarray, FieldState]:
        return cetg_rhs(
            G_state,
            self.grid,
            self.params,
            self.terms,
            compressed_real_fft=self.compressed_real_fft,
            fields_override=fields_override,
        )

    def fields(self, G_state: jnp.ndarray) -> FieldState:
        return cetg_fields(G_state, self.grid, self.params)

    def diagnostics(
        self,
        G_state: jnp.ndarray,
        fields_state: FieldState,
        phi_prev: jnp.ndarray,
        dt_step: jnp.ndarray,
    ):
        return _compute_cetg_diag(
            G_state,
            fields_state,
            phi_prev,
            dt_step,
            self.grid,
            self.params,
            mask=self.mask,
            z_index=self.z_index,
            omega_ky_index=self.omega_ky_index,
            omega_kx_index=self.omega_kx_index,
        )

    def advance(
        self, G_state: jnp.ndarray, dG: jnp.ndarray, dt_local: jnp.ndarray
    ) -> jnp.ndarray:
        G_new = _cetg_explicit_step(
            G_state,
            dG,
            dt_local=dt_local,
            method=self.method,
            grid=self.grid,
            compressed_real_fft=self.compressed_real_fft,
            rhs_fn=lambda G_stage: self.rhs(G_stage),
        )
        G_new = _project_state(
            G_new,
            self.grid,
            compressed_real_fft=self.compressed_real_fft,
        )
        return jnp.asarray(G_new, dtype=self.state_dtype)


@dataclass(frozen=True)
class _CETGDiagnosticSeries:
    t: jnp.ndarray
    dt_t: jnp.ndarray
    gamma_t: jnp.ndarray
    omega_t: jnp.ndarray
    Wg_t: jnp.ndarray
    Wphi_t: jnp.ndarray
    Wapar_t: jnp.ndarray
    heat_flux_t: jnp.ndarray
    particle_flux_t: jnp.ndarray
    heat_flux_species_t: jnp.ndarray
    particle_flux_species_t: jnp.ndarray
    phi_mode_t: jnp.ndarray

    def sample(self, stride: int) -> "_CETGDiagnosticSeries":
        if stride <= 1:
            return self
        return _CETGDiagnosticSeries(
            t=self.t[::stride],
            dt_t=self.dt_t[::stride],
            gamma_t=self.gamma_t[::stride],
            omega_t=self.omega_t[::stride],
            Wg_t=self.Wg_t[::stride],
            Wphi_t=self.Wphi_t[::stride],
            Wapar_t=self.Wapar_t[::stride],
            heat_flux_t=self.heat_flux_t[::stride],
            particle_flux_t=self.particle_flux_t[::stride],
            heat_flux_species_t=self.heat_flux_species_t[::stride, ...],
            particle_flux_species_t=self.particle_flux_species_t[::stride, ...],
            phi_mode_t=self.phi_mode_t[::stride],
        )

    def to_simulation_diagnostics(self, real_dtype: jnp.dtype) -> SimulationDiagnostics:
        return SimulationDiagnostics(
            t=self.t,
            dt_t=self.dt_t,
            dt_mean=jnp.mean(self.dt_t)
            if int(np.asarray(self.dt_t).size)
            else jnp.asarray(0.0, dtype=real_dtype),
            gamma_t=self.gamma_t,
            omega_t=self.omega_t,
            Wg_t=self.Wg_t,
            Wphi_t=self.Wphi_t,
            Wapar_t=self.Wapar_t,
            heat_flux_t=self.heat_flux_t,
            particle_flux_t=self.particle_flux_t,
            energy_t=total_energy(self.Wg_t, self.Wphi_t, self.Wapar_t),
            heat_flux_species_t=self.heat_flux_species_t,
            particle_flux_species_t=self.particle_flux_species_t,
            phi_mode_t=self.phi_mode_t,
        )


def _validate_cetg_method(method: str) -> None:
    if method not in _SUPPORTED_CETG_EXPLICIT_METHODS:
        raise ValueError("Unsupported explicit cETG method")


def _project_cetg_initial_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    *,
    compressed_real_fft: bool,
) -> tuple[jnp.ndarray, jnp.dtype, jnp.dtype]:
    G0_int = _project_state(
        _to_internal_state(G0), grid, compressed_real_fft=compressed_real_fft
    )
    state_dtype = jnp.result_type(G0_int, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    return G0_int, state_dtype, real_dtype


def _build_cetg_scan_context(
    grid: SpectralGrid,
    params: CETGModelParams,
    terms: TermConfig,
    *,
    method: str,
    compressed_real_fft: bool,
    fixed_dt: bool,
    state_dtype: jnp.dtype,
    real_dtype: jnp.dtype,
    dt: float,
    dt_min: float,
    dt_max: float | None,
    cfl: float,
    cfl_fac: float | None,
    diagnostics_stride: int,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
) -> tuple[_CETGScanContext, jnp.ndarray]:
    dt_init = jnp.asarray(dt, dtype=real_dtype)
    mask = jnp.broadcast_to(
        jnp.asarray(grid.dealias_mask, dtype=bool), (grid.ky.size, grid.kx.size)
    )
    context = _CETGScanContext(
        grid=grid,
        params=params,
        terms=terms,
        method=method,
        compressed_real_fft=compressed_real_fft,
        fixed_dt=fixed_dt,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        dt_min=jnp.asarray(dt_min, dtype=real_dtype),
        dt_max=jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype),
        cfl=jnp.asarray(cfl, dtype=real_dtype),
        cfl_fac=jnp.asarray(resolve_cfl_fac(method, cfl_fac), dtype=real_dtype),
        linear_omega=jnp.asarray(_cetg_linear_omega_max(grid, params), dtype=real_dtype),
        mask=mask,
        z_index=_diagnostic_midplane_index(grid.z.size),
        diagnostics_stride=int(max(diagnostics_stride, 1)),
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
    )
    return context, dt_init


def _initial_cetg_carry(
    G0_int: jnp.ndarray, context: _CETGScanContext, dt_init: jnp.ndarray
):
    fields0 = cetg_fields(
        G0_int, context.grid, context.params, apply_kz_dealias=False
    )
    dt0 = context.update_dt(fields0, dt_init)
    diag_zero = context.diagnostics(G0_int, fields0, fields0.phi, dt0)
    return (
        G0_int,
        fields0,
        fields0.phi,
        diag_zero,
        jnp.asarray(0.0, dtype=context.real_dtype),
        dt0,
    )


def _cetg_scan_step(context: _CETGScanContext):
    def step(carry, idx):
        G_state, fields_state, phi_prev, diag_prev, t_prev, dt_prev = carry
        dt_local = context.update_dt(fields_state, dt_prev)
        dG, _fields_used = context.rhs(G_state, fields_override=fields_state)
        G_new = context.advance(G_state, dG, dt_local)
        t_new = jnp.asarray(t_prev + dt_local, dtype=context.real_dtype)
        fields_new = context.fields(G_new)

        def _compute_diag(_):
            return context.diagnostics(G_new, fields_new, phi_prev, dt_local)

        def _reuse_diag(_):
            return diag_prev

        do_diag = (idx % context.diagnostics_stride) == 0
        diag = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        return (G_new, fields_new, fields_new.phi, diag, t_new, dt_local), (
            diag,
            t_new,
            dt_local,
        )

    return step


def _with_cetg_progress(base_step, *, steps: int):
    from spectraxgk.utils.callbacks import print_callback, should_emit_progress

    def scan_step(carry, idx):
        carry_out, diag_out = base_step(carry, idx)
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

    return scan_step


def _cetg_series_from_scan(diag, t: jnp.ndarray, dt_series: jnp.ndarray) -> _CETGDiagnosticSeries:
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
    return _CETGDiagnosticSeries(
        t=t,
        dt_t=dt_series,
        gamma_t=gamma_t,
        omega_t=omega_t,
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        heat_flux_species_t=heat_s_t,
        particle_flux_species_t=pflux_s_t,
        phi_mode_t=phi_mode_t,
    )


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

    _validate_cetg_method(method)
    G0_int, state_dtype, real_dtype = _project_cetg_initial_state(
        G0, grid, compressed_real_fft=compressed_real_fft
    )
    context, dt_init = _build_cetg_scan_context(
        grid,
        params,
        terms,
        method=method,
        compressed_real_fft=compressed_real_fft,
        fixed_dt=fixed_dt,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        dt=dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        diagnostics_stride=diagnostics_stride,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
    )
    idx = jnp.arange(steps, dtype=jnp.int32)
    scan_step = _cetg_scan_step(context)
    if show_progress:
        scan_step = _with_cetg_progress(scan_step, steps=steps)

    (G_final, fields_last, phi_last, _diag_last, _t_last, _dt_last), diag_out = (
        jax.lax.scan(
            scan_step,
            _initial_cetg_carry(G0_int, context, dt_init),
            idx,
            length=steps,
        )
    )
    diag, t, dt_series = diag_out
    series = _cetg_series_from_scan(diag, t, dt_series).sample(
        int(max(sample_stride, diagnostics_stride, 1))
    )
    return (
        series.t,
        series.to_simulation_diagnostics(real_dtype),
        _from_internal_state(G_final),
        FieldState(phi=fields_last.phi, apar=None, bpar=None),
    )


__all__ = [
    "_cetg_diag_weight",
    "_cetg_flux_weight",
    "_compute_cetg_diag",
    "integrate_cetg_explicit_diagnostics_state",
]
