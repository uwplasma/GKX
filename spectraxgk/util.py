import jax.numpy as jnp
from constants import (
    epsilon_0, elementary_charge as e_charge
)

def _pick_debye_species(cfg):
    """
    Choose the species used to define λ_D.
    Priority:
      1) cfg.grid.debye_species as name (str) or index (int)
      2) First species with q < 0 (electron-like)
      3) Fallback: species[0]
    Returns (species_obj, index).
    """
    sp_list = getattr(cfg, "species", [])
    if not sp_list:
        raise ValueError("At least one [[species]] is required to define Debye length.")
    sel = getattr(cfg.grid, "debye_species", None)
    if isinstance(sel, int):
        return sp_list[sel], sel
    if isinstance(sel, str):
        for i, sp in enumerate(sp_list):
            if getattr(sp, "name", f"s{i}") == sel:
                return sp, i
        raise ValueError(f"grid.debye_species='{sel}' not found among [[species]].")
    # default: first with q<0
    for i, sp in enumerate(sp_list):
        if float(getattr(sp, "q", -1.0)) < 0.0:
            return sp, i
    return sp_list[0], 0

def _compute_wp(sp) -> float:
    """
    Plasma frequency ω_p = sqrt(n0 * q^2 / (ε0 * m)) [rad/s]
    Uses species.n0 (m^-3), q (C), m (kg).
    """
    n0 = float(getattr(sp, "n0", 0.0))
    q  = float(getattr(sp, "q", -e_charge))
    m  = float(getattr(sp, "m", 9.10938371e-31))
    if n0 <= 0.0:
        raise ValueError("Plasma frequency requires species.n0 > 0 (in m^-3).")
    return float(jnp.sqrt(n0 * (q*q) / (epsilon_0 * m)))

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


def _compute_lambda_D(sp) -> float:
    """
    Electron Debye length: λ_D = sqrt( ε0 * kB * T / (n0 * q_e^2) ).
    Uses n0 (m^-3) from species and T from _infer_Te_J_from_species(sp).
    """
    n0 = float(getattr(sp, "n0", 0.0))
    if n0 <= 0.0:
        raise ValueError("Debye length requires species.n0 > 0 (in m^-3).")
    Tj = _infer_Te_J_from_species(sp) / 1.0  # Joules
    return float(jnp.sqrt(epsilon_0 * (Tj) / (n0 * e_charge * e_charge)))
