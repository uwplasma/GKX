import jax.numpy as jnp
from typing import Tuple

from spectraxgk.constants import (
    epsilon_0, elementary_charge as e_charge
)
from spectraxgk.io_config import Config, SpeciesCfg

def _pick_debye_species(cfg: Config) -> Tuple[SpeciesCfg, int]:
    """
    Choose species for Debye/ωp normalization:
    - If cfg.grid.debye_species is set (name or index), use it.
    - Else prefer first negative charge (electron-like), else species[0].
    """
    tag = getattr(cfg.grid, "debye_species", None)
    sp_list = cfg.species
    if isinstance(tag, int) and 0 <= tag < len(sp_list):
        return sp_list[tag], tag
    if isinstance(tag, str):
        for i, sp in enumerate(sp_list):
            if sp.name == tag:
                return sp, i
    # default: first q<0 if any
    for i, sp in enumerate(sp_list):
        if float(sp.q) < 0:
            return sp, i
    # fallback
    return sp_list[0], 0


def _compute_wp(sp: SpeciesCfg) -> float:
    """
    ω_p = sqrt( n0 * (q e)^2 / (ε0 m) ), SI units.
    - sp.q is dimensionless charge number (e.g. -1 for electrons, +1 for protons)
    - sp.n0 is in m^-3
    - returns rad/s
    """
    n = float(sp.n0)
    q_si = float(sp.q) * e_charge
    m = float(sp.m)
    return float(jnp.sqrt(n * q_si*q_si / (epsilon_0 * m)))

def _infer_Te_J_from_species(sp) -> float:
    """
    Try to infer electron temperature (Joules) for Debye length.
    If your SpeciesCfg already stores 'Te_eV', use that; otherwise infer from vth if present.
    Assumptions for inference from vth (m/s):
        vth^2 ≈ 2 * kB * T / m  =>  T = m vth^2 / (2 kB)
    """
    Te_eV = getattr(sp, "Te_eV", None)
    if Te_eV is not None:
        return float(Te_eV) * e_charge
    # fallback: infer from vth if it looks like m/s
    vth = float(getattr(sp, "vth", 0.0))
    m   = float(getattr(sp, "m", 9.10938371e-31))
    if vth > 0.0:
        return m * vth * vth / (2.0)  # Joules (since divide by kB happens later)
    # last resort: small T to avoid div-by-zero; you may choose to error instead
    return 1.0 * e_charge  # 1 eV in Joules


def _compute_lambda_D(sp: SpeciesCfg) -> float:
    """
    λ_D = sqrt( ε0 * T_J / ( n0 (q e)^2 ) ), where T_J = (1/2) m v_th^2 (Joules).
    """
    n = float(sp.n0)
    q_si = float(sp.q) * e_charge
    m = float(sp.m)
    vth = float(sp.vth)
    T_J = 0.5 * m * vth * vth
    return float(jnp.sqrt(epsilon_0 * T_J / (n * q_si * q_si)))
