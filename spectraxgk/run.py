# run.py
"""
Pedagogical entrypoint for 1D Vlasov–Poisson experiments.

- Reads a TOML config (see io_config.py) describing either:
    sim.mode = "fourier"  or  sim.mode = "dg"
- Calls the appropriate backend (run_fourier or run_dg)
- Produces three diagnostics via plots.py:
    1) Phase mixing:   imshow of log |c_n(t)|
    2) Total energy:   proxy W_total = 0.5 <E^2> + 0.5 (C0 + C2) at k=0
    3) Distribution:   animated imshow of f(x, u, t)
                        (x on X-axis, u=v/v_th on Y-axis, colormap = f)

All heavy math stays in JAX; we only convert to NumPy when plotting.
"""

# --- enable x64 before importing jax ---
import os
os.environ["JAX_ENABLE_X64"] = "true"

import time
import argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from io_config import read_toml
from backends import run_fourier, run_dg
from plots import plot_phase_mixing, plot_total_energy, animate_distribution


# ---------------------------------------------------------------------
# Helpers to normalize backend outputs for plotting
# ---------------------------------------------------------------------
def _extract_fourier_outputs(out: dict):
    """
    Normalize Fourier backend dictionary to a canonical tuple:

      returns:
        C_for   : (N, nt) complex  -- Hermite coeffs for *k≈0* (if available),
                                     otherwise mean over k (diagnostics-only)
        C_knt   : (Nk, N, nt) complex or None  -- full bank (for animation)
        Ek_kt   : (Nk, nt) complex or None
        k       : (Nk,) float64 or None

    We prefer C_knt (Nk,N,nt). Backward compatibility: C_nkt (N,k,t) or C_nt (N,t).
    """
    k     = out.get("k", None)
    Ek_kt = out.get("Ek_kt", None)
    if "C_knt" in out:
        C_knt = out["C_knt"]
        if k is not None and jnp.any(jnp.isclose(k, 0.0)):
            j0 = int(jnp.argmin(jnp.abs(k)))
            C_for = C_knt[j0, :, :]
        else:
            C_for = jnp.mean(C_knt, axis=0)
        return C_for, C_knt, Ek_kt, k
    if "C_nkt" in out:
        C_nkt = out["C_nkt"]
        if k is not None and jnp.any(jnp.isclose(k, 0.0)):
            j0 = int(jnp.argmin(jnp.abs(k)))
            C_for = C_nkt[:, j0, :]
        else:
            C_for = jnp.mean(C_nkt, axis=1)
        return C_for, None, Ek_kt, k
    if "C_nt" in out:
        return out["C_nt"], None, Ek_kt, k
    raise KeyError("Fourier backend did not return 'C_knt', 'C_nkt', or 'C_nt'.")


def _extract_dg_outputs(out: dict, L: float | None, Nx: int | None):
    """
    Normalize DG backend dictionary:

      returns:
        C_nxt : (N, Nx, nt) real
        E_xt  : (Nx, nt) real
        x     : (Nx,) float64

    If 'x' is not present in the backend dict, derive a uniform grid from (L, Nx).
    """
    if "C_t" not in out or "E_xt" not in out:
        raise KeyError("DG backend must return 'C_t' and 'E_xt'.")
    C_nxt = out["C_t"]
    E_xt  = out["E_xt"]
    x     = out.get("x", None)

    if x is None:
        if L is None or Nx is None:
            raise ValueError("DG outputs missing 'x' and config lacks grid.L/grid.Nx.")
        x = jnp.linspace(0.0, float(L), int(Nx), endpoint=False, dtype=jnp.float64)

    return C_nxt, E_xt, x


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run 1D Vlasov–Poisson experiment")
    parser.add_argument("--input", type=str, default="input.toml",
                        help="Path to TOML config (default: input.toml)")
    args = parser.parse_args()

    # --- Read config ---
    cfg = read_toml(args.input)
    
   # Plot settings come from [plot] in TOML
    plot_cfg = getattr(cfg, "plot", None)
    if plot_cfg is None:
        raise ValueError("Your TOML must define a [plot] section with plotting parameters.")
    nv       = getattr(plot_cfg, "nv", 257)
    vmax     = getattr(plot_cfg, "vmax", 6.0)
    save_anim= getattr(plot_cfg, "save_anim", None)
    fps      = getattr(plot_cfg, "fps", 30)
    dpi      = getattr(plot_cfg, "dpi", 150)
    no_show  = getattr(plot_cfg, "no_show", False)

    # --- Build a velocity grid for plots (dimensionless u = v/v_th) ---
    v = jnp.linspace(-float(vmax), float(vmax), int(nv), dtype=jnp.float64)

    # --- Run selected backend ---
    t0 = time.perf_counter()
    if cfg.sim.mode == "fourier":
        ts, out = run_fourier(cfg)
    elif cfg.sim.mode == "dg":
        ts, out = run_dg(cfg)
    else:
        raise SystemExit("sim.mode must be 'fourier' or 'dg'")
    print(f"[run] solve: {time.perf_counter() - t0:.2f}s")

    # --- Plot diagnostics (shared) ---
    if cfg.sim.mode == "fourier":
        # Normalize outputs
        C_for, C_knt, Ek_kt, k = _extract_fourier_outputs(out)

        # 1) Phase mixing (k≈0 or avg over k)
        plot_phase_mixing(ts, C_for, title=r"Phase mixing: $|c_n(t)|$ (Fourier)")

        # 2) Total energy (field from Ek_kt; kinetic from C0+C2 at k≈0)
        plot_total_energy(ts, C=C_for, Ek_kt=Ek_kt, E_xt=None, title="Total energy (Fourier)")

        # 3) Animated distribution f(x,u,t) (needs full bank + x-grid)
        #    Reconstruct on a physical x-grid if config specifies a periodic box.
        if C_knt is not None and k is not None and (cfg.grid.L is not None) and (cfg.grid.Nx is not None):
            L, Nx = float(cfg.grid.L), int(cfg.grid.Nx)
            x  = jnp.linspace(0.0, L, Nx, endpoint=False, dtype=jnp.float64)
            ani = animate_distribution(
                ts=ts, v=v, v_th=1.0, x=x,
                mode="fourier",
                C_knt=C_knt, k=k,
                title="f(x,u,t) (Fourier)"
            )
            if save_anim:
                print(f"[run] saving animation to {save_anim} …")
                ani.save(save_anim, dpi=dpi, fps=fps)
        else:
            print("[run] Fourier animation skipped (need C_knt, k, and (grid.L, grid.Nx) in config)")

    else:
        # DG mode
        C_nxt, E_xt, x = _extract_dg_outputs(out, cfg.grid.L, cfg.grid.Nx)

        # 1) Phase mixing (x-avg to k=0)
        plot_phase_mixing(ts, C_nxt, title=r"Phase mixing: $|c_n(t)|$ (DG, k=0)")

        # 2) Total energy (field from E(x,t); kinetic from C0+C2 at k=0)
        plot_total_energy(ts, C=C_nxt, E_xt=E_xt, Ek_kt=None, title="Total energy (DG)")

        # 3) Animated distribution f(x,u,t)
        ani = animate_distribution(
            ts=ts, v=v, v_th=1.0, x=x,
            mode="dg",
            C=C_nxt,
            title="f(x,u,t) (DG)"
        )
        if save_anim:
            print(f"[run] saving animation to {save_anim} …")
            ani.save(save_anim, dpi=dpi, fps=fps)

    if not no_show:
        plt.show()


if __name__ == "__main__":
    main()
