import jax.numpy as jnp

from spectraxgk.constants import (
    boltzmann_constant as kB,
)
from spectraxgk.constants import (
    elementary_charge as e_charge,
)
from spectraxgk.constants import (
    epsilon_0,
    mass_electron,
    mass_proton,
)
from spectraxgk.constants import (
    speed_of_light as c_light,
)
from spectraxgk.io_config import Config, SpeciesCfg


def print_units_banner(
    cfg: Config, sp_ref: SpeciesCfg, idx_ref: int, omega_p: float, tmax_over_wp: float
) -> None:
    """Header lines about ωp-based time normalization and optional Debye-length box length."""
    tmax_seconds = float(cfg.sim.tmax)
    dt = tmax_seconds / float(cfg.sim.nt - 1) if cfg.sim.nt > 1 else tmax_seconds
    print(
        "[units] Using ω_p from species "
        f"'{getattr(sp_ref, 'name', f's{idx_ref}')}' → ω_p = {omega_p:.6e} rad/s\n"
        f"        t_max = {tmax_seconds:.6e} s (= {tmax_over_wp:g}/ω_p), "
        f"Δt = {dt:.6e} s (= {dt * omega_p:.6g}/ω_p)"
    )

    # If user provided L in Debye lengths, echo the conversion that main() already applied.
    if getattr(cfg.grid, "L_lambdaD", None) is not None:
        lambda_d = _compute_lambda_D(sp_ref)
        print(
            f"[units] Using Debye length from species '{getattr(sp_ref, 'name', f's{idx_ref}')}' "
            f"→ λ_D = {lambda_d:.6e} m;  L = {float(cfg.grid.L):.6e} m "
            f"(= {float(cfg.grid.L_lambdaD):g} λ_D)"
        )


def print_sim_summary(cfg: Config) -> None:
    """One place for the verbose simulation summary."""
    print("\n=== Simulation parameters summary ===")
    # Grid/time
    L = float(cfg.grid.L)
    Nx = int(cfg.grid.Nx)
    dx = L / Nx if Nx > 0 else 0.0
    print(f"Box length L = {L:.3e} m")
    print(f"Grid points Nx = {Nx}, dx = {dx:.3e} m")

    tmax = float(cfg.sim.tmax)
    nt = int(cfg.sim.nt)
    dt = tmax / (nt - 1) if nt > 1 else tmax
    print(f"Total time = {tmax:.3e} s, time step dt = {dt:.3e} s")

    # k-grid (summary purposes only)
    if Nx > 0 and L > 0.0:
        kvals = (2.0 * jnp.pi / L) * jnp.arange(-Nx // 2, Nx // 2)
        kmin = float(jnp.min(jnp.abs(kvals[kvals != 0]))) if Nx > 1 else 0.0
        kmax = float(jnp.max(jnp.abs(kvals)))
    else:
        kmin = kmax = 0.0
    print(f"Wavenumber range: k_min = {kmin:.3e} 1/m, k_max = {kmax:.3e} 1/m")

    # Species
    print("\n--- Species ---")
    for sp in cfg.species:
        n0_m3 = float(sp.n0)
        n0_cm3 = n0_m3 * 1e-6
        m_s = float(sp.m)
        vth = float(sp.vth)
        u0 = float(sp.u0)

        # T from vth: T_J = 1/2 m vth^2   (no kB here; that appears in Debye via ε0*kB*T)
        t_j = 0.5 * m_s * (vth**2)
        t_ev = t_j / e_charge

        # Debye length: λ_D = sqrt( ε0 * kB * T / ( n0 e^2 ) ), with T inferred from vth
        lambda_d = float(jnp.sqrt(epsilon_0 * kB * t_j / (n0_m3 * (e_charge**2))))
        # Plasma frequency: ω_p = sqrt( n0 e^2 / (ε0 m) )
        omega_p = float(jnp.sqrt(n0_m3 * (e_charge**2) / (epsilon_0 * m_s)))

        # Mass in multiples
        if m_s < 5 * mass_proton:
            mass_str = f"{(m_s / mass_electron):.2f} m_e"
        else:
            mass_str = f"{(m_s / mass_proton):.2f} m_p"

        drift_frac_c = u0 / c_light

        print(f"Species: {sp.name}")
        print(f"  n0 = {n0_cm3:.3e} cm^-3")
        print(f"  m = {mass_str}")
        print(f"  T = {t_ev:.3f} eV, vth = {vth:.3e} m/s")
        print(f"  u0 = {u0:.3e} m/s ({drift_frac_c:.3e} c)")
        print(f"  lambda_D = {lambda_d:.3e} m,  omega_p = {omega_p:.3e} rad/s")
        if lambda_d > 0:
            print(f"  -> L / lambda_D = {L / lambda_d:.2f},  dx / lambda_D = {dx / lambda_d:.2f}")
            print(
                f"  -> k_min * lambda_D = {kmin * lambda_d:.2e}, k_max * lambda_D = {kmax * lambda_d:.2e}"
            )
        print(f"  -> tmax * ω_p = {tmax * omega_p:.2e},  dt * ω_p = {dt * omega_p:.2e}")
    print("====================================\n")


def _pick_debye_species(cfg: Config) -> tuple[SpeciesCfg, int]:
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
    return float(jnp.sqrt(n * q_si * q_si / (epsilon_0 * m)))


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
    m = float(getattr(sp, "m", 9.10938371e-31))
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
