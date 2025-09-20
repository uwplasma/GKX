# diagnostics.py
"""
Diagnostics helpers for 1D Vlasov–Poisson (Fourier/DG), multi-species.

This module centralizes:
  - robust selection of k≈0 slice
  - E(x,t) reconstruction from Fourier (Ek_kt)
  - field energy proxy ⟨E^2⟩/2
  - per-species kinetic energy proxies (from Hermite C0 and C2)
  - convenience wrappers to build the complete energy row inputs

All functions are lightweight and NumPy/JAX-friendly.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from spectraxgk.constants import epsilon_0


# -------------------- robust k≈0 selection --------------------
def pick_k0_index(k: jnp.ndarray) -> int:
    """Return the index of the mode with the smallest |k|."""
    k_np = np.asarray(k)
    return int(np.argmin(np.abs(k_np)))


# -------------------- E(x,t) reconstruction (Fourier) --------------------
def reconstruct_E_xt_from_fourier(
    Ek_kt: jnp.ndarray, k: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    """
    E(x,t) = Re[ sum_k E_k(t) e^{ikx} ]
      Ek_kt: (Nk, nt), k: (Nk,), x: (Nx,)  -> (Nx, nt)
    """
    k = jnp.asarray(k, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    expikx = jnp.exp(1j * (k[:, None] * x[None, :]))  # (Nk, Nx)
    return jnp.real(jnp.transpose(expikx, (1, 0)) @ Ek_kt)  # (Nx, nt)


# -------------------- Field energy proxy --------------------
def _dx_from_grid(L: float, Nx: int) -> float:
    return float(L) / float(Nx)


def _int_x_trapz_uniform(F_x_t: jnp.ndarray, dx: float) -> jnp.ndarray:
    # F_x_t: (Nx, nt) -> integral over x at each t (uniform grid)
    # For periodic/uniform: simple Riemann sum is fine.
    return jnp.sum(F_x_t, axis=0) * dx


def energies_fourier_exact(
    *,
    out: dict,
    species_list,
    L: float,
    Nx: int,
    x: jnp.ndarray,
) -> tuple[list[tuple[str, jnp.ndarray]], jnp.ndarray, jnp.ndarray]:
    """
    Exact energies for Fourier backend by reconstructing to x:

      - per-species kinetic:
          W_kin_s(t) = (n0_s m_s vth_s^2 / (4√2)) * ∫ dx (C0_s + √2 C2_s)
      - field:
          W_field(t) = ∫ dx E(x,t)^2 / (2 ε0)

    Returns:
      species_energy: list[(label, W_kin_s(t))]
      W_field: (nt,)
      E_xt: (Nx, nt)
    """
    if ("C_kSnt" not in out) or ("Ek_kt" not in out) or ("k" not in out):
        raise KeyError("Fourier requires 'C_kSnt','Ek_kt','k' in backend output.")

    C_kSnt = out["C_kSnt"]  # (Nk, S, N, nt)
    Ek_kt = out["Ek_kt"]  # (Nk, nt)
    k = out["k"]  # (Nk,)
    Nk, S, N, nt = C_kSnt.shape
    dx = _dx_from_grid(L, Nx)

    minimum_moments_needed_to_compute_energy = 3
    if minimum_moments_needed_to_compute_energy > N:
        raise ValueError(f"Need N>={minimum_moments_needed_to_compute_energy} for kinetic energy.")

    # Reconstruct E(x,t)
    E_xt = reconstruct_E_xt_from_fourier(Ek_kt, k, x)  # (Nx, nt)

    # Inverse FFT C_s(n,k,t) -> C_s(n,x,t)
    # axis: k is axis=0 in C_kSnt, so we move to (S,N,Nk,nt) then ifft over Nk
    C_Snkt = jnp.transpose(C_kSnt, (1, 2, 0, 3))  # (S,N,Nk,nt)
    C_Snxt = jnp.fft.ifft(C_Snkt, axis=2)  # (S,N,Nx,nt), Nx==Nk if you built k grid from Nx
    C_Snxt = jnp.real(C_Snxt)

    # Kinetic (per species)
    species_energy = []
    for s, sp in enumerate(species_list):
        n0 = float(sp.n0)
        m_s = float(sp.m)
        vth = float(sp.vth)
        # slice C0 + √2 C2
        C0_x_t = C_Snxt[s, 0, :, :]  # (Nx,nt)
        C2_x_t = C_Snxt[s, 2, :, :]
        combo = C0_x_t + (jnp.sqrt(2.0) * C2_x_t)  # (Nx,nt)
        I_t = _int_x_trapz_uniform(combo, dx)  # (nt,)
        pref = n0 * m_s * (vth**2) / (4.0 * jnp.sqrt(2.0))
        W_kin_s = pref * I_t  # (nt,)
        label = getattr(sp, "name", f"s{s}")
        species_energy.append((label, W_kin_s))

    # Field energy
    W_field = _int_x_trapz_uniform(E_xt**2, dx) / (2.0 * epsilon_0)  # (nt,)

    return species_energy, W_field, E_xt


def energies_dg_exact(
    *,
    out: dict,
    species_list,
    L: float,
    Nx: int,
) -> tuple[list[tuple[str, jnp.ndarray]], jnp.ndarray, jnp.ndarray]:
    """
    Exact energies for DG backend directly on x-grid.
    Returns species_energy, W_field, E_xt.
    """
    for key in ("C_St", "E_xt", "x"):
        if key not in out:
            raise KeyError(f"DG requires key '{key}' in backend output.")

    C_St = out["C_St"]  # (S,N,Nx,nt)
    E_xt = out["E_xt"]  # (Nx,nt)
    S, N, Nx_chk, nt = C_St.shape
    assert Nx_chk == Nx, "DG: Nx mismatch"
    minimum_moments_needed_to_compute_energy = 3
    if minimum_moments_needed_to_compute_energy > N:
        raise ValueError(f"Need N>={minimum_moments_needed_to_compute_energy} for kinetic energy.")
    dx = _dx_from_grid(L, Nx)

    species_energy = []
    for s, sp in enumerate(species_list):
        n0 = float(sp.n0)
        m_s = float(sp.m)
        vth = float(sp.vth)
        C0_x_t = C_St[s, 0, :, :]  # (Nx,nt)
        C2_x_t = C_St[s, 2, :, :]
        combo = C0_x_t + (jnp.sqrt(2.0) * C2_x_t)  # (Nx,nt)
        I_t = jnp.sum(combo, axis=0) * dx  # (nt,)
        pref = n0 * m_s * (vth**2) / (4.0 * jnp.sqrt(2.0))
        W_kin_s = pref * I_t
        label = getattr(sp, "name", f"s{s}")
        species_energy.append((label, W_kin_s))

    W_field = (jnp.sum(E_xt**2, axis=0) * dx) / (2.0 * epsilon_0)  # (nt,)

    return species_energy, W_field, E_xt
