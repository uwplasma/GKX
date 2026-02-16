"""Species helpers for assembling linear gyrokinetic parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import jax.numpy as jnp

from spectraxgk.linear import LinearParams


@dataclass(frozen=True)
class Species:
    """Physical parameters for a kinetic species."""

    charge: float
    mass: float
    density: float
    temperature: float
    tprim: float
    fprim: float
    nu: float = 0.0


def _as_array(values: Iterable[float]) -> jnp.ndarray:
    return jnp.asarray(np.asarray(list(values), dtype=float))


def build_linear_params(
    species: Iterable[Species],
    *,
    tau_e: float = 0.0,
    kpar_scale: float = 1.0,
    omega_d_scale: float = 1.0,
    omega_star_scale: float = 1.0,
    rho_star: float = 1.0,
    beta: float = 0.0,
    fapar: float = 0.0,
    nu_hyper: float = 0.0,
    p_hyper: float = 4.0,
) -> LinearParams:
    """Build LinearParams arrays from a list of species."""

    species = list(species)
    charge = _as_array(s.charge for s in species)
    mass = _as_array(s.mass for s in species)
    density = _as_array(s.density for s in species)
    temp = _as_array(s.temperature for s in species)
    tprim = _as_array(s.tprim for s in species)
    fprim = _as_array(s.fprim for s in species)
    nu = _as_array(s.nu for s in species)

    vth = jnp.sqrt(temp / mass)
    rho = jnp.sqrt(temp * mass) / jnp.abs(charge)
    tz = charge / temp

    return LinearParams(
        charge_sign=charge,
        density=density,
        mass=mass,
        temp=temp,
        vth=vth,
        rho=rho,
        tz=tz,
        R_over_LTi=tprim,
        R_over_Ln=fprim,
        tau_e=tau_e,
        kpar_scale=kpar_scale,
        omega_d_scale=omega_d_scale,
        omega_star_scale=omega_star_scale,
        rho_star=rho_star,
        beta=beta,
        fapar=fapar,
        nu=nu,
        nu_hyper=nu_hyper,
        p_hyper=p_hyper,
    )
