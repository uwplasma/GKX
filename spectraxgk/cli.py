# Ensure x64 BEFORE importing jax
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from spectraxgk.io_config import read_toml
from spectraxgk.backends import run_fourier, run_dg
from spectraxgk.plots import render_suite_onefigure
from spectraxgk.util import _pick_debye_species, _compute_lambda_D, _compute_wp
from spectraxgk.constants import (
    epsilon_0, elementary_charge as e_charge,
    mass_electron, mass_proton, speed_of_light as c_light,
    boltzmann_constant as kB
)

def main():
    parser = argparse.ArgumentParser(description="Run 1D Vlasov–Poisson experiment")
    parser.add_argument("--input", type=str, default="examples/two_stream.toml",
                        help="Path to TOML config (default: examples/two_stream.toml)")
    args = parser.parse_args()

    cfg = read_toml(args.input)

    # --- Interpret sim.tmax as multiples of 1/ωp (dimensionless) ---
    sp_ref, idx_ref = _pick_debye_species(cfg)
    omega_p = _compute_wp(sp_ref)  # rad/s

    tmax_over_wp = float(cfg.sim.tmax)
    cfg.sim.tmax = tmax_over_wp / omega_p  # seconds

    dt = cfg.sim.tmax / float(cfg.sim.nt - 1) if cfg.sim.nt > 1 else cfg.sim.tmax
    print(
        "[units] Using ω_p from species "
        f"'{getattr(sp_ref, 'name', f's{idx_ref}')}' → ω_p = {omega_p:.6e} rad/s\n"
        f"        t_max = {cfg.sim.tmax:.6e} s (= {tmax_over_wp:g}/ω_p), "
        f"Δt = {dt:.6e} s (= {dt*omega_p:.6g}/ω_p)"
    )

    # --- Allow L specified in Debye lengths ---
    if getattr(cfg.grid, "L_lambdaD", None) is not None:
        lambda_D = _compute_lambda_D(sp_ref)
        cfg.grid.L = float(cfg.grid.L_lambdaD) * lambda_D
        print(f"[units] Using Debye length from species '{getattr(sp_ref, 'name', f's{idx_ref}')}' "
              f"→ λ_D = {lambda_D:.6e} m;  L = {cfg.grid.L:.6e} m "
              f"(= {cfg.grid.L_lambdaD:g} λ_D)")

    # --- Summary (same as you have now) ---
    print("\n=== Simulation parameters summary ===")
    L = float(cfg.grid.L); Nx = int(cfg.grid.Nx); dx = L / Nx
    print(f"Box length L = {L:.3e} m")
    print(f"Grid points Nx = {Nx}, dx = {dx:.3e} m")

    tmax = float(cfg.sim.tmax); nt = int(cfg.sim.nt)
    dt   = tmax / (nt - 1) if nt > 1 else tmax
    print(f"Total time = {tmax:.3e} s, time step dt = {dt:.3e} s")

    kvals = (2.0 * jnp.pi / L) * jnp.arange(-Nx//2, Nx//2)
    kmin = float(jnp.min(jnp.abs(kvals[kvals != 0]))) if Nx > 1 else 0.0
    kmax = float(jnp.max(jnp.abs(kvals))) if Nx > 0 else 0.0
    print(f"Wavenumber range: k_min = {kmin:.3e} 1/m, k_max = {kmax:.3e} 1/m")

    print("\n--- Species ---")
    for sp in cfg.species:
        n0_m3 = float(sp.n0); n0_cm3 = n0_m3 * 1e-6
        m_s = float(sp.m); vth = float(sp.vth); u0 = float(sp.u0)
        T_J = m_s * (vth**2) / (2.0); T_eV = T_J / e_charge
        lambda_D = (epsilon_0 * kB * T_J / (n0_m3 * (e_charge**2)))**0.5
        omega_p = (n0_m3 * (e_charge**2) / (epsilon_0 * m_s))**0.5
        mass_str = f"{(m_s/mass_electron):.2f} m_e" if m_s < 5*mass_proton else f"{(m_s/mass_proton):.2f} m_p"
        drift_frac_c = u0 / c_light

        print(f"Species: {sp.name}")
        print(f"  n0 = {n0_cm3:.3e} cm^-3")
        print(f"  m = {mass_str}")
        print(f"  T = {T_eV:.3f} eV, vth = {vth:.3e} m/s")
        print(f"  u0 = {u0:.3e} m/s ({drift_frac_c:.3e} c)")
        print(f"  lambda_D = {lambda_D:.3e} m,  omega_p = {omega_p:.3e} rad/s")
        print(f"  -> L / lambda_D = {L/lambda_D:.2f},  dx / lambda_D = {dx/lambda_D:.2f}")
        print(f"  -> tmax * ω_p = {tmax*omega_p:.2e},  dt * ω_p = {dt*omega_p:.2e}")
        print(f"  -> k_min * lambda_D = {kmin*lambda_D:.2e}, k_max * lambda_D = {kmax*lambda_D:.2e}")
    print("====================================\n")

    # --- Solve ---
    t0 = time.perf_counter()
    if cfg.sim.mode == "fourier":
        ts, out = run_fourier(cfg)
    elif cfg.sim.mode == "dg":
        ts, out = run_dg(cfg)
    else:
        raise SystemExit("sim.mode must be 'fourier' or 'dg'")
    print(f"[run] solve time: {time.perf_counter() - t0:.2f}s")

    render_suite_onefigure(cfg, ts, out)

if __name__ == "__main__":
    main()
