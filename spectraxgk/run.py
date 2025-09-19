# run.py
# Ensure x64 BEFORE importing jax
import os
os.environ["JAX_ENABLE_X64"] = "true"

import argparse
import time

import jax
jax.config.update("jax_enable_x64", True)

from io_config import read_toml
from backends import run_fourier, run_dg
from plots import render_suite_onefigure
from util import _pick_debye_species, _compute_lambda_D, _compute_wp
from constants import (
    epsilon_0, elementary_charge as e_charge,
    mass_electron, mass_proton, speed_of_light as c_light,
    boltzmann_constant as kB
)

import jax.numpy as jnp

def main():
    parser = argparse.ArgumentParser(description="Run 1D Vlasov–Poisson experiment")
    parser.add_argument("--input", type=str, default="input.toml",
                        help="Path to TOML config (default: input.toml)")
    args = parser.parse_args()

    cfg = read_toml(args.input)

    # --- Interpret sim.tmax as multiples of 1/ωp (dimensionless) ---
    # Use same reference species as Debye selection (unless you want a separate selector).
    sp_ref, idx_ref = _pick_debye_species(cfg)
    omega_p = _compute_wp(sp_ref)  # rad/s

    tmax_over_wp = float(cfg.sim.tmax)           # user-given, in units of 1/ωp
    cfg.sim.tmax = tmax_over_wp / omega_p        # convert to seconds (so backends stay unchanged)

    # Optional: print the conversion so users see what happened
    dt = cfg.sim.tmax / float(cfg.sim.nt - 1) if cfg.sim.nt > 1 else cfg.sim.tmax
    print(
        "[units] Using ω_p from species "
        f"'{getattr(sp_ref, 'name', f's{idx_ref}')}' → ω_p = {omega_p:.6e} rad/s\n"
        f"        t_max = {cfg.sim.tmax:.6e} s (= {tmax_over_wp:g}/ω_p), "
        f"Δt = {dt:.6e} s (= {dt*omega_p:.6g}/ω_p)"
    )

    # --- Allow L specified in Debye lengths ---
    if getattr(cfg.grid, "L_lambdaD", None) is not None:
        sp_ref, idx_ref = _pick_debye_species(cfg)
        lambda_D = _compute_lambda_D(sp_ref)  # meters
        cfg.grid.L = float(cfg.grid.L_lambdaD) * lambda_D
        # Optional: print what we did for clarity
        print(f"[units] Using Debye length from species '{getattr(sp_ref, 'name', f's{idx_ref}')}' "
              f"→ λ_D = {lambda_D:.6e} m;  L = {cfg.grid.L:.6e} m "
              f"(= {cfg.grid.L_lambdaD:g} λ_D)")

    # ===============================
    # === SIMULATION SUMMARY PRINT ===
    # ===============================
    print("\n=== Simulation parameters summary ===")

    # Grid
    L = float(cfg.grid.L)
    Nx = int(cfg.grid.Nx)
    dx = L / Nx
    print(f"Box length L = {L:.3e} m")
    print(f"Grid points Nx = {Nx}, dx = {dx:.3e} m")

    # Time
    tmax = float(cfg.sim.tmax)
    nt   = int(cfg.sim.nt)
    dt   = tmax / (nt - 1)
    print(f"Total time = {tmax:.3e} s, time step dt = {dt:.3e} s")

    # Wavenumbers
    kvals = (2.0 * jnp.pi / L) * jnp.arange(-Nx//2, Nx//2)  # symmetric grid
    kmin = float(jnp.min(jnp.abs(kvals[kvals != 0])))
    kmax = float(jnp.max(jnp.abs(kvals)))
    print(f"Wavenumber range: k_min = {kmin:.3e} 1/m, k_max = {kmax:.3e} 1/m")

    # Species loop
    print("\n--- Species ---")
    for sp in cfg.species:
        n0_m3 = float(sp.n0)          # we interpret n0 in m^-3 already
        n0_cm3 = n0_m3 * 1e-6          # m^-3 -> cm^-3
        m_s = float(sp.m)
        vth = float(sp.vth)
        u0  = float(sp.u0)

        # Debye length λ_D = sqrt(ε₀ k_B T / (n e²)), using T = m vth² / (2 kB)
        T_J = m_s * (vth**2) / (2.0 * kB)
        lambda_D = (epsilon_0 * kB * T_J / (n0_m3 * e_charge**2))**0.5

        # Plasma frequency ω_p = sqrt(n e² / (ε₀ m))
        omega_p = (n0_m3 * e_charge**2 / (epsilon_0 * m_s))**0.5

        # Express mass in multiples of m_e or m_p
        if m_s < 5*mass_proton:
            mass_multiple_me = m_s / mass_electron
            mass_str = f"{mass_multiple_me:.2f} m_e"
        else:
            mass_multiple_mp = m_s / mass_proton
            mass_str = f"{mass_multiple_mp:.2f} m_p"

        drift_frac_c = u0 / c_light
        T_eV = T_J / e_charge

        print(f"Species: {sp.name}")
        print(f"  n0 = {n0_cm3:.3e} cm^-3")
        print(f"  m = {mass_str}")
        print(f"  T = {T_eV:.3f} eV, vth = {vth:.3e} m/s")
        print(f"  u0 = {u0:.3e} m/s ({drift_frac_c:.3e} c)")
        print(f"  lambda_D = {lambda_D:.3e} m,  omega_p = {omega_p:.3e} rad/s")

        # Extra grid/time in normalized units
        print(f"  -> L / lambda_D = {L/lambda_D:.2f},  dx / lambda_D = {dx/lambda_D:.2f}")
        print(f"  -> tmax * ω_p = {tmax*omega_p:.2e},  dt * ω_p = {dt*omega_p:.2e}")
        print(f"  -> k_min * lambda_D = {kmin*lambda_D:.2e}, k_max * lambda_D = {kmax*lambda_D:.2e}")

    print("====================================\n")

    # ===============================
    # === SOLVE ===
    # ===============================
    t0 = time.perf_counter()
    if cfg.sim.mode == "fourier":
        ts, out = run_fourier(cfg)
    elif cfg.sim.mode == "dg":
        ts, out = run_dg(cfg)
    else:
        raise SystemExit("sim.mode must be 'fourier' or 'dg'")
    print(f"[run] solve time: {time.perf_counter() - t0:.2f}s")

    # Plot all diagnostics
    render_suite_onefigure(cfg, ts, out)


if __name__ == "__main__":
    main()
