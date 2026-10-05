"""Reference-run parameter builders shared by the GX comparison tooling.

These reproduce the hypercollision, two-species and initial-condition
settings the tracked GX reference runs used; the solver itself never needs them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from gkx.config import InitializationConfig
from gkx.core_grid import SpectralGrid
from gkx.geometry import FluxTubeGeometryLike
from gkx.operators.linear.params import LinearParams, Species, build_linear_params


REFERENCE_NU_HYPER_L = 0.0

REFERENCE_NU_HYPER_M = 1.0

REFERENCE_P_HYPER_L = 6.0

REFERENCE_P_HYPER_M = 20.0


def _reference_hypercollision_power(nhermite: int | None) -> float:
    if nhermite is None:
        return REFERENCE_P_HYPER_M
    return float(min(REFERENCE_P_HYPER_M, max(int(nhermite) // 2, 1)))


def _apply_reference_hypercollisions(
    params: LinearParams, *, nhermite: int | None = None
) -> LinearParams:
    return replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=REFERENCE_NU_HYPER_L,
        nu_hyper_m=REFERENCE_NU_HYPER_M,
        p_hyper_l=REFERENCE_P_HYPER_L,
        p_hyper_m=_reference_hypercollision_power(nhermite),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    )


def _two_species_params(
    model,
    *,
    kpar_scale: float,
    omega_d_scale: float,
    omega_star_scale: float,
    rho_star: float,
    beta_override: float | None = None,
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    damp_ends_amp: float | None = None,
    damp_ends_widthfrac: float | None = None,
    nhermite: int | None = None,
) -> LinearParams:
    """Build ``LinearParams`` for a two-species kinetic model."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")
    ion_fprim_raw = getattr(model, "fprim_i", None)
    ele_fprim_raw = getattr(model, "fprim_e", None)
    ion_fprim = float(model.fprim) if ion_fprim_raw is None else float(ion_fprim_raw)
    ele_fprim = float(model.fprim) if ele_fprim_raw is None else float(ele_fprim_raw)

    nu_i = float(getattr(model, "nu_i", 0.0))
    nu_e = float(getattr(model, "nu_e", 0.0))
    beta = float(getattr(model, "beta", 1.0e-5))
    if beta_override is not None:
        beta = float(beta_override)

    ion = Species(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=float(getattr(model, "tprim_i", model.tprim_e)),
        fprim=ion_fprim,
        nu=nu_i,
    )
    electron = Species(
        charge=-1.0,
        mass=1.0 / mass_ratio,
        density=1.0,
        temperature=Te_over_Ti,
        tprim=float(model.tprim_e),
        fprim=ele_fprim,
        nu=nu_e,
    )
    params = build_linear_params(
        [ion, electron],
        tau_e=0.0,
        kpar_scale=kpar_scale,
        omega_d_scale=omega_d_scale,
        omega_star_scale=omega_star_scale,
        rho_star=rho_star,
        beta=beta,
        fapar=1.0 if beta > 0.0 else 0.0,
        apar_beta_scale=0.5 if apar_beta_scale is None else float(apar_beta_scale),
        ampere_g0_scale=0.5 if ampere_g0_scale is None else float(ampere_g0_scale),
        bpar_beta_scale=0.5 if bpar_beta_scale is None else float(bpar_beta_scale),
    )
    params = _apply_reference_hypercollisions(params, nhermite=nhermite)
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    envelope = (
        init_cfg.gaussian_envelope_constant
        + init_cfg.gaussian_envelope_sine * np.sin(z - theta0)
    )
    width = init_cfg.gaussian_width
    if width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    return envelope * np.exp(-(((z - theta0) / width) ** 2))


def _build_initial_condition(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    *,
    ky_index: int | Sequence[int] | np.ndarray,
    kx_index: int,
    Nl: int,
    Nm: int,
    init_cfg: InitializationConfig,
) -> jnp.ndarray:
    init_field = init_cfg.init_field.lower()
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    # Moment-normalized initializer amplitudes for init_field="all".
    all_scales = {
        "density": 1.0,
        "upar": 1.0,
        "tpar": 1.0 / np.sqrt(2.0),
        "tperp": 1.0,
        "qpar": 1.0 / np.sqrt(6.0),
        "qperp": 1.0,
    }
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(init_cfg.init_amp)
    ky_idx = np.atleast_1d(np.asarray(ky_index, dtype=int))
    for ky_i in ky_idx:
        if init_cfg.gaussian_init:
            profile = _build_gaussian_profile(
                np.asarray(grid.z),
                kx=float(grid.kx[kx_index]),
                ky=float(grid.ky[ky_i]),
                s_hat=geom.s_hat,
                init_cfg=init_cfg,
            )
            init_vals = amp * profile * (1.0 + 1.0j)
        else:
            init_vals = amp * (1.0 + 1.0j) * np.ones_like(grid.z)
        if grid.ky[ky_i] != 0.0:
            if init_field == "all":
                for field_name, (l_idx, m_idx) in field_map.items():
                    if l_idx < Nl and m_idx < Nm:
                        scale = all_scales.get(field_name, 1.0)
                        G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals * scale
            else:
                l_idx, m_idx = field_map[init_field]
                if l_idx >= Nl or m_idx >= Nm:
                    raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
                G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals
    return jnp.asarray(G0)
