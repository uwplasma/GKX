"""Dedicated collisional-slab ETG model aligned with legacy GX."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import resolve_cfl_fac
from spectraxgk.diagnostics import SimulationDiagnostics, total_energy
from spectraxgk.geometry import FluxTubeGeometryLike, SlabGeometry
from spectraxgk.gx_integrators import _gx_growth_rate_step, _gx_midplane_index
from spectraxgk.grids import SpectralGrid
from spectraxgk.runtime_config import RuntimeConfig
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.integrators import _SSPX3_ADT, _SSPX3_W1, _SSPX3_W2, _SSPX3_W3
from spectraxgk.terms.nonlinear import _spectral_bracket_multi


@dataclass(frozen=True)
class CETGModelParams:
    """GX cETG coefficients and normalization data."""

    tau_fac: float
    z_ion: float
    gradpar: float
    z0: float
    c1: float
    C12: float
    C23: float
    D_hyper: float
    nu_hyper: float
    pressure: float
    dealias_kz: bool


def _model_key(cfg: RuntimeConfig) -> str:
    return cfg.physics.reduced_model.strip().lower()


def validate_cetg_runtime_config(
    cfg: RuntimeConfig,
    geom: FluxTubeGeometryLike,
    *,
    Nl: int,
    Nm: int,
) -> None:
    """Validate that a runtime config matches the GX cETG model contract."""

    if _model_key(cfg) != "cetg":
        raise ValueError("cETG helpers require physics.reduced_model='cetg'")
    if int(Nl) != 2 or int(Nm) != 1:
        raise ValueError("GX cETG requires exactly Nl=2 and Nm=1")
    if not isinstance(geom, SlabGeometry):
        raise ValueError("GX cETG currently requires geometry.model='slab'")
    if not bool(cfg.physics.electrostatic) or bool(cfg.physics.electromagnetic):
        raise ValueError("GX cETG is electrostatic-only")
    if not bool(cfg.physics.adiabatic_ions):
        raise ValueError("GX cETG requires adiabatic_ions=true")
    kinetic = tuple(s for s in cfg.species if bool(s.kinetic))
    if len(kinetic) != 1:
        raise ValueError("GX cETG requires exactly one kinetic species")
    if float(kinetic[0].charge) >= 0.0:
        raise ValueError("GX cETG requires the kinetic species to be an electron")


def build_cetg_model_params(
    cfg: RuntimeConfig,
    geom: FluxTubeGeometryLike,
    *,
    Nl: int,
    Nm: int,
) -> CETGModelParams:
    """Build the GX cETG coefficient set from the runtime config."""

    validate_cetg_runtime_config(cfg, geom, Nl=Nl, Nm=Nm)
    if not isinstance(geom, SlabGeometry):
        raise ValueError("GX cETG currently requires geometry.model='slab'")
    kinetic = tuple(s for s in cfg.species if bool(s.kinetic))
    electron = kinetic[0]
    z_ion = float(cfg.physics.z_ion)
    tau_fac = float(cfg.physics.tau_fac if cfg.physics.tau_fac is not None else cfg.physics.tau_e)
    denom = 1.0 + 61.0 / (np.sqrt(128.0) * z_ion) + 9.0 / (2.0 * z_ion * z_ion)
    c1 = (217.0 / 64.0 + 151.0 / (np.sqrt(128.0) * z_ion) + 9.0 / (2.0 * z_ion * z_ion)) / denom
    c2 = 2.5 * (33.0 / 16.0 + 45.0 / (np.sqrt(128.0) * z_ion)) / denom
    c3 = 25.0 / 4.0 * (13.0 / 4.0 + 45.0 / (np.sqrt(128.0) * z_ion)) / denom - c2 * c2 / c1
    C12 = 1.0 + c2 / c1
    C23 = c3 / c1 + C12 * C12
    D_hyper = float(cfg.collisions.D_hyper if float(cfg.terms.hyperdiffusion) != 0.0 else 0.0)
    nu_hyper = float(cfg.collisions.nu_hyper) if float(cfg.collisions.nu_hyper) > 0.0 else 2.0
    return CETGModelParams(
        tau_fac=tau_fac,
        z_ion=z_ion,
        gradpar=float(geom.gradpar()),
        z0=float(geom.z0) if geom.z0 is not None else float(1.0 / float(geom.gradpar())),
        c1=float(c1),
        C12=float(C12),
        C23=float(C23),
        D_hyper=D_hyper,
        nu_hyper=float(nu_hyper),
        pressure=float(electron.density * electron.temperature),
        dealias_kz=bool(cfg.expert.dealias_kz),
    )


def _to_internal_state(G: jnp.ndarray) -> jnp.ndarray:
    G_arr = jnp.asarray(G)
    if G_arr.ndim == 4 and G_arr.shape[0] == 2:
        return G_arr
    if G_arr.ndim != 6 or G_arr.shape[0] != 1 or G_arr.shape[1] != 2 or G_arr.shape[2] != 1:
        raise ValueError("cETG state must have shape (1, 2, 1, Ny, Nx, Nz) or (2, Ny, Nx, Nz)")
    return jnp.stack([G_arr[0, 0, 0], G_arr[0, 1, 0]], axis=0)


def _from_internal_state(G: jnp.ndarray) -> jnp.ndarray:
    G_arr = jnp.asarray(G)
    if G_arr.ndim != 4 or G_arr.shape[0] != 2:
        raise ValueError("internal cETG state must have shape (2, Ny, Nx, Nz)")
    return G_arr[None, :, None, :, :, :]


def _xy_mask(grid: SpectralGrid, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(grid.dealias_mask, dtype=dtype)[None, :, :, None]


def _kz_grid(grid: SpectralGrid) -> jnp.ndarray:
    z = np.asarray(grid.z, dtype=float)
    if z.size < 2:
        return jnp.zeros((z.size,), dtype=float)
    dz = float(z[1] - z[0])
    return 2.0 * jnp.pi * jnp.fft.fftfreq(z.size, d=dz)


def _kz_mask(grid: SpectralGrid, dtype: jnp.dtype, *, dealias_kz: bool) -> jnp.ndarray:
    if not dealias_kz:
        return jnp.ones((grid.z.size,), dtype=dtype)
    kz_frac = jnp.fft.fftfreq(grid.z.size)
    return jnp.asarray(jnp.abs(kz_frac) < (1.0 / 3.0), dtype=dtype)


def _apply_kz_filter(arr: jnp.ndarray, grid: SpectralGrid, *, dealias_kz: bool) -> jnp.ndarray:
    if not dealias_kz or int(grid.z.size) <= 1:
        return arr
    arr_k = jnp.fft.fft(arr, axis=-1)
    mask = _kz_mask(grid, arr_k.real.dtype, dealias_kz=True)
    # Legacy GX periodic-z dealias uses CUFFT forward/inverse pairs without
    # the 1/N rescale on the inverse, so the filtered field carries an Nz factor.
    return jnp.fft.ifft(arr_k * mask, axis=-1) * jnp.asarray(float(grid.z.size), dtype=arr_k.real.dtype)


def _dz2(arr: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    if int(grid.z.size) <= 1:
        return jnp.zeros_like(arr)
    kz = _kz_grid(grid).astype(jnp.real(arr).dtype)
    arr_k = jnp.fft.fft(arr, axis=-1)
    return jnp.fft.ifft(-(kz**2) * arr_k, axis=-1)


def _use_hermitian_reconstruction(grid: SpectralGrid, *, gx_real_fft: bool) -> bool:
    return bool(gx_real_fft) and bool(np.any(np.asarray(grid.ky, dtype=float) < 0.0))


def _project_state(
    G: jnp.ndarray,
    grid: SpectralGrid,
    *,
    gx_real_fft: bool,
) -> jnp.ndarray:
    G_proj = jnp.asarray(G)
    G_proj = G_proj * _xy_mask(grid, jnp.real(G_proj).dtype)

    if not _use_hermitian_reconstruction(grid, gx_real_fft=gx_real_fft):
        return G_proj
    ny_full = int(grid.ky.size)
    nyc = ny_full // 2 + 1
    if nyc <= 2:
        return G_proj
    pos = G_proj[:, :nyc, :, :]
    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    neg = jnp.conj(pos[:, 1:neg_hi, :, :])[:, ::-1, :, :]
    nx = int(grid.kx.size)
    if nx > 1:
        kx_neg = jnp.concatenate(
            [jnp.asarray([0], dtype=jnp.int32), jnp.arange(nx - 1, 0, -1, dtype=jnp.int32)]
        )
        neg = neg[:, :, kx_neg, :]
    return jnp.concatenate([pos, neg], axis=1)


def cetg_fields(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    *,
    apply_kz_dealias: bool = True,
) -> FieldState:
    """Solve the cETG electrostatic field equation."""

    G_int = _to_internal_state(G)
    phi = -jnp.asarray(params.tau_fac, dtype=jnp.real(G_int).dtype) * G_int[0]
    phi = phi * _xy_mask(grid, jnp.real(phi).dtype)[0]
    if apply_kz_dealias:
        phi = _apply_kz_filter(phi, grid, dealias_kz=params.dealias_kz)
    return FieldState(phi=phi, apar=None, bpar=None)


def _cetg_linear_rhs(
    G: jnp.ndarray,
    fields: FieldState,
    terms: TermConfig,
    grid: SpectralGrid,
    params: CETGModelParams,
) -> jnp.ndarray:
    G_int = _to_internal_state(G)
    density = G_int[0]
    temperature = G_int[1]
    phi = fields.phi
    gpar2 = jnp.asarray(params.gradpar * params.gradpar, dtype=jnp.real(G_int).dtype)
    c1 = jnp.asarray(params.c1, dtype=jnp.real(G_int).dtype)
    C12 = jnp.asarray(params.C12, dtype=jnp.real(G_int).dtype)
    C23 = jnp.asarray(params.C23, dtype=jnp.real(G_int).dtype)

    rhs0 = 0.5 * gpar2 * c1 * (density + C12 * temperature - phi)
    rhs1 = (gpar2 / 3.0) * c1 * (C12 * density + C23 * temperature - C12 * phi)
    rhs = jnp.stack([_dz2(rhs0, grid), _dz2(rhs1, grid)], axis=0)

    ky = jnp.asarray(grid.ky, dtype=jnp.real(G_int).dtype)[:, None, None]
    rhs = rhs.at[1].add(-0.5j * ky * phi)
    if float(terms.hyperdiffusion) != 0.0 and float(params.D_hyper) != 0.0:
        kx = jnp.asarray(grid.kx, dtype=jnp.real(G_int).dtype)[None, :, None]
        k2 = kx * kx + ky * ky
        Dfac = jnp.asarray(params.D_hyper, dtype=jnp.real(G_int).dtype) * (
            k2 ** jnp.asarray(params.nu_hyper, dtype=jnp.real(G_int).dtype)
        )
        rhs = rhs - jnp.asarray(float(terms.hyperdiffusion), dtype=jnp.real(G_int).dtype) * Dfac[None, ...] * G_int
    rhs = rhs * _xy_mask(grid, jnp.real(rhs).dtype)
    return _apply_kz_filter(rhs, grid, dealias_kz=params.dealias_kz)


def _cetg_nonlinear_rhs(
    G: jnp.ndarray,
    fields: FieldState,
    grid: SpectralGrid,
    *,
    gx_real_fft: bool,
) -> jnp.ndarray:
    G_int = _to_internal_state(G)
    phi = fields.phi
    bracket = _spectral_bracket_multi(
        G_int,
        phi[None, ...],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(grid.kxfac, dtype=jnp.real(G_int).dtype),
        gx_real_fft=gx_real_fft,
    )[0]
    bracket = 0.5 * bracket
    bracket = bracket * _xy_mask(grid, jnp.real(bracket).dtype)
    return bracket


def cetg_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    terms: TermConfig,
    *,
    gx_real_fft: bool,
    fields_override: FieldState | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Return the full cETG RHS and the electrostatic fields."""

    G_int = _to_internal_state(G)
    fields = cetg_fields(G_int, grid, params) if fields_override is None else fields_override
    rhs = _cetg_linear_rhs(G_int, fields, terms, grid, params)
    if float(terms.nonlinear) != 0.0:
        rhs = rhs + jnp.asarray(float(terms.nonlinear), dtype=jnp.real(rhs).dtype) * _cetg_nonlinear_rhs(
            G_int,
            fields,
            grid,
            gx_real_fft=gx_real_fft,
        )
    rhs = _project_state(rhs, grid, gx_real_fft=gx_real_fft)
    return rhs, fields


def _cetg_linear_omega_max(grid: SpectralGrid, params: CETGModelParams) -> float:
    ny = int(grid.ky.size)
    nz = int(grid.z.size)
    ky_max = float(abs(np.asarray(grid.ky, dtype=float)[(ny - 1) // 3])) if ny > 1 else 0.0
    z0 = abs(float(params.z0))
    kz_max = (float(nz) / 3.0) * float(params.gradpar) / z0
    cfac = 0.5 * float(params.c1) * float(np.sqrt(1.0 + (params.C12 - 1.0)))
    return float(cfac * np.sqrt(max(ky_max, 0.0)) * kz_max)


def _cetg_nonlinear_omega_components(
    phi: jnp.ndarray,
    grid: SpectralGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    complex_dtype = jnp.result_type(phi, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)
    fft_norm = float(grid.ky.size * grid.kx.size)
    ifft_scale = jnp.asarray(fft_norm, dtype=real_dtype)
    kx = jnp.asarray(grid.kx_grid, dtype=real_dtype)
    ky = jnp.asarray(grid.ky_grid, dtype=real_dtype)
    dphi_dx = jnp.fft.ifft2(imag * kx[:, :, None] * phi, axes=(-3, -2)) * ifft_scale
    dphi_dy = jnp.fft.ifft2(imag * ky[:, :, None] * phi, axes=(-3, -2)) * ifft_scale
    vmax_x = jnp.max(jnp.abs(dphi_dy))
    vmax_y = jnp.max(jnp.abs(dphi_dx))
    nx = int(grid.kx.size)
    ny = int(grid.ky.size)
    kx_max = float(abs(np.asarray(grid.kx, dtype=float)[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(np.asarray(grid.ky, dtype=float)[(ny - 1) // 3])) if ny > 1 else 0.0
    return jnp.asarray(kx_max, dtype=real_dtype) * vmax_x, jnp.asarray(ky_max, dtype=real_dtype) * vmax_y


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
    gamma_modes, omega_modes = _gx_growth_rate_step(phi, phi_prev, dt_step, z_index=z_index, mask=mask)
    real_dtype = jnp.real(jnp.empty((), dtype=phi.dtype)).dtype
    if omega_ky_index is not None:
        ky_i = int(np.clip(omega_ky_index, 0, int(gamma_modes.shape[0]) - 1))
        kx_i = int(np.clip(omega_kx_index or 0, 0, int(gamma_modes.shape[1]) - 1))
        gamma = jnp.nan_to_num(gamma_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype))
        omega = jnp.nan_to_num(omega_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype))
        phi_mode = phi[ky_i, kx_i, z_index]
    else:
        gamma = jnp.nan_to_num(jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)), nan=jnp.asarray(0.0, dtype=real_dtype))
        omega = jnp.nan_to_num(jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)), nan=jnp.asarray(0.0, dtype=real_dtype))
        phi_mode = jnp.asarray(0.0 + 0.0j, dtype=phi.dtype)

    diag_weight = _cetg_diag_weight(grid, real_dtype)
    flux_weight = _cetg_flux_weight(grid, real_dtype)
    W = 0.5 * jnp.asarray(params.pressure, dtype=real_dtype) * jnp.sum(jnp.abs(G_int) ** 2 * diag_weight)
    Phi2 = 0.5 * jnp.sum(jnp.abs(phi) ** 2 * diag_weight[0])
    ky = jnp.asarray(grid.ky, dtype=real_dtype)[:, None, None]
    vphi_r = -1j * ky * phi
    qflux_species = jnp.asarray(
        [jnp.sum(jnp.real(jnp.conj(vphi_r) * G_int[1]) * flux_weight[0]) * jnp.asarray(params.pressure, dtype=real_dtype)],
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


def integrate_cetg_gx_diagnostics_state(
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
    gx_real_fft: bool = True,
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 1.0,
    cfl_fac: float | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Integrate the GX cETG model and stream GX-style diagnostics."""

    if method not in {"euler", "rk2", "rk3", "rk3_classic", "rk3_gx", "rk4", "k10", "sspx3"}:
        raise ValueError("Unsupported explicit cETG method")

    G0_int = _project_state(_to_internal_state(G0), grid, gx_real_fft=gx_real_fft)
    state_dtype = jnp.result_type(G0_int, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_init = jnp.asarray(dt, dtype=real_dtype)
    dt_min_val = jnp.asarray(dt_min, dtype=real_dtype)
    dt_max_val = jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype)
    cfl_val = jnp.asarray(cfl, dtype=real_dtype)
    cfl_fac_val = jnp.asarray(resolve_cfl_fac(method, cfl_fac), dtype=real_dtype)
    z_idx = _gx_midplane_index(grid.z.size)
    mask = jnp.broadcast_to(jnp.asarray(grid.dealias_mask, dtype=bool), (grid.ky.size, grid.kx.size))
    linear_omega = jnp.asarray(_cetg_linear_omega_max(grid, params), dtype=real_dtype)

    def _update_dt(fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if fixed_dt:
            return dt_prev
        omega_nl_x, omega_nl_y = _cetg_nonlinear_omega_components(fields_state.phi, grid)
        wmax = linear_omega + omega_nl_x + omega_nl_y
        dt_guess = jnp.where(wmax > 0.0, cfl_fac_val * cfl_val / wmax, dt_prev)
        return jnp.asarray(jnp.clip(dt_guess, dt_min_val, dt_max_val), dtype=real_dtype)

    def rhs_fn(G_state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        return cetg_rhs(G_state, grid, params, terms, gx_real_fft=gx_real_fft)

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
            gx_real_fft=gx_real_fft,
            fields_override=fields_state,
        )
        if method == "euler":
            G_new = G_state + dt_local * dG
        elif method == "rk2":
            k1 = dG
            G_half = _project_state(G_state + 0.5 * dt_local * k1, grid, gx_real_fft=gx_real_fft)
            k2, _ = rhs_fn(G_half)
            G_new = G_state + dt_local * k2
        elif method == "rk3_classic":
            k1 = dG
            G1 = _project_state(G_state + dt_local * k1, grid, gx_real_fft=gx_real_fft)
            k2, _ = rhs_fn(G1)
            G2 = _project_state(
                0.75 * G_state + 0.25 * (G1 + dt_local * k2),
                grid,
                gx_real_fft=gx_real_fft,
            )
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G_state + (2.0 / 3.0) * (G2 + dt_local * k3)
        elif method in {"rk3", "rk3_gx"}:
            k1 = dG
            G1 = _project_state(G_state + (dt_local / 3.0) * k1, grid, gx_real_fft=gx_real_fft)
            k2, _ = rhs_fn(G1)
            G2 = _project_state(
                G_state + (2.0 * dt_local / 3.0) * k2,
                grid,
                gx_real_fft=gx_real_fft,
            )
            k3, _ = rhs_fn(G2)
            G3 = _project_state(G_state + 0.75 * dt_local * k3, grid, gx_real_fft=gx_real_fft)
            G_new = G3 + 0.25 * dt_local * k1
        elif method == "rk4":
            k1 = dG
            G2 = _project_state(G_state + 0.5 * dt_local * k1, grid, gx_real_fft=gx_real_fft)
            k2, _ = rhs_fn(G2)
            G3 = _project_state(G_state + 0.5 * dt_local * k2, grid, gx_real_fft=gx_real_fft)
            k3, _ = rhs_fn(G3)
            G4 = _project_state(G_state + dt_local * k3, grid, gx_real_fft=gx_real_fft)
            k4, _ = rhs_fn(G4)
            G_new = G_state + (dt_local / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif method == "sspx3":
            def _sspx3_euler_step(
                G_stage: jnp.ndarray,
                dG_stage: jnp.ndarray | None = None,
            ) -> jnp.ndarray:
                if dG_stage is None:
                    dG_stage, _ = rhs_fn(G_stage)
                return _project_state(
                    G_stage + (_SSPX3_ADT * dt_local) * dG_stage,
                    grid,
                    gx_real_fft=gx_real_fft,
                )

            # The first SSPx3 Euler substep must use the carried field state that
            # was already used to pick the adaptive timestep, matching GX's
            # Timestepper::advance contract.
            G1 = _sspx3_euler_step(G_state, dG)
            G2_euler = _sspx3_euler_step(G1)
            G2 = _project_state(
                (1.0 - _SSPX3_W1) * G_state + (_SSPX3_W1 - 1.0) * G1 + G2_euler,
                grid,
                gx_real_fft=gx_real_fft,
            )
            G3 = _sspx3_euler_step(G2)
            G_new = (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G_state
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        else:
            def _k10_euler_step(G_stage: jnp.ndarray) -> jnp.ndarray:
                dG_stage, _ = rhs_fn(G_stage)
                return _project_state(
                    G_stage + (dt_local / 6.0) * dG_stage,
                    grid,
                    gx_real_fft=gx_real_fft,
                )

            G_q1 = G_state
            G_q2 = G_state
            for _ in range(5):
                G_q1 = _k10_euler_step(G_q1)
            G_q2 = 0.04 * G_q2 + 0.36 * G_q1
            G_q1 = 15.0 * G_q2 - 5.0 * G_q1
            for _ in range(4):
                G_q1 = _k10_euler_step(G_q1)
            dG_final, _ = rhs_fn(G_q1)
            G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_local * dG_final

        G_new = _project_state(G_new, grid, gx_real_fft=gx_real_fft)
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
        return (G_new, fields_new, fields_new.phi, diag, t_new, dt_local), (diag, t_new, dt_local)

    idx = jnp.arange(steps, dtype=jnp.int32)
    scan_step = step
    if show_progress:
        from spectraxgk.utils.callbacks import print_callback, should_emit_progress

        def scan_step(carry, idx):
            carry_out, diag_out = step(carry, idx)
            diag_vals, _t_out, _dt_out = diag_out
            gamma_cb, omega_cb, Wg_cb, Wphi_cb = diag_vals[0], diag_vals[1], diag_vals[2], diag_vals[3]
            _dt_out = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(state, idx, steps, gamma_cb, omega_cb, Wphi_cb, Wg_cb, _t_out, None),
                lambda state: state,
                _dt_out,
            )
            return carry_out, diag_out

    (G_final, fields_last, phi_last, _diag_last, _t_last, _dt_last), diag_out = jax.lax.scan(
        scan_step,
        (G0_int, fields0, fields0.phi, diag_zero, jnp.asarray(0.0, dtype=real_dtype), dt0),
        idx,
        length=steps,
    )
    diag, t, dt_series = diag_out
    gamma_t, omega_t, Wg_t, Wphi_t, Wapar_t, heat_t, pflux_t, heat_s_t, pflux_s_t, phi_mode_t = diag

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
        dt_mean=jnp.mean(dt_series) if int(np.asarray(dt_series).size) else jnp.asarray(0.0, dtype=real_dtype),
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
    return t, diag_out_final, _from_internal_state(G_final), FieldState(phi=fields_last.phi, apar=None, bpar=None)
