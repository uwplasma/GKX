# plots.py
"""
Unified plotting for Fourier–Hermite (FH) and DG runs of 1D Vlasov–Poisson.

Provided plots:
1) Phase mixing:  imshow of |Hermite coefficients| vs time  (n on y, t on x)
   - Accepts either (N, nt) or (N, Nx, nt). If 3D, it takes the k=0 (spatial average) by default.

2) Total energy:
   - Kinetic energy := 0.5 * ( C0_{k=0}(t) + C2_{k=0}(t) )   [as requested]
     (We take the real part; this is a diagnostic “proxy” you asked for.)
   - Field energy   := 0.5 * ∫ E(x,t)^2 dx
     If E(x,t) is given: 0.5 * mean_x(E^2)
     Else if E_k(t) is given: 0.5 * sum_k |E_k|^2 (optionally * L if you want physical units)

3) Distribution function f(x, v, t_sel) = f0(v) + sum_n c_n(x,t_sel) φ_n(v/v_th)
   - You pass either C_nt (N, nt) + assume spatial uniform coefficient (x-constant),
     or C_nxt (N, Nx, nt) which will be used directly.
   - v grid must be provided. If x grid is not provided, we show a 1×Nv panel.

All functions use the same inputs for Fourier and DG; you just pass what you have.
"""


from __future__ import annotations
from typing import Optional, Tuple, Dict, Literal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# -------------------- Hermite basis --------------------
def hermite_basis(u: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Physicists' orthonormal Hermite functions φ_n(u).
    Returns Φ of shape (N, Nu), Φ[n, j] = φ_n(u_j).

    Recurrence:
      φ_0 = π^{-1/4} e^{-u^2/2}
      φ_1 = sqrt(2) u φ_0
      φ_{n+1} = sqrt(2/(n+1)) u φ_n - sqrt(n/(n+1)) φ_{n-1}
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    Nu = u.shape[0]
    Phi = jnp.zeros((N, Nu), dtype=jnp.float64)
    if N == 0:
        return Phi
    phi0 = jnp.exp(-0.5 * u**2) / (jnp.pi**0.25)
    Phi = Phi.at[0].set(phi0)
    if N == 1:
        return Phi
    Phi = Phi.at[1].set(jnp.sqrt(2.0) * u * phi0)

    def body(carry, n):
        phi_nm1, phi_n = carry
        a = jnp.sqrt(2.0 / (n + 1))
        b = jnp.sqrt(n / (n + 1))
        phi_np1 = a * u * phi_n - b * phi_nm1
        return (phi_n, phi_np1), phi_np1

    (_, _), rest = jax.lax.scan(body, (Phi[0], Phi[1]), jnp.arange(1, N - 1))
    if N > 2:
        Phi = Phi.at[2:N].set(rest)
    return Phi


# -------------------- Helpers --------------------
def _k0_spatial_average(C: jnp.ndarray) -> jnp.ndarray:
    """
    Average over x (axis=1) if C is (N, Nx, nt); otherwise return C (N, nt).
    """
    if C.ndim == 2:
        return C
    if C.ndim == 3:
        return jnp.mean(C, axis=1)
    raise ValueError("C must be (N, nt) or (N, Nx, nt).")

def _E2_space_average(
    E_xt: Optional[jnp.ndarray],
    Ek_kt: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """
    Return ⟨E^2⟩_x(t) either from E(x,t) or from Σ_k |E_k|^2.
    """
    if E_xt is not None:
        return jnp.mean(E_xt**2, axis=0)  # (nt,)
    if Ek_kt is not None:
        return jnp.sum(jnp.abs(Ek_kt)**2, axis=0)  # (nt,)
    raise ValueError("Provide E_xt or Ek_kt to compute field energy.")

def _ensure_fig_ax(ax: Optional[plt.Axes]) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


# -------------------- Plot 1: phase mixing --------------------
def plot_phase_mixing(
    ts: jnp.ndarray,
    C: jnp.ndarray,                 # (N, nt) or (N, Nx, nt)
    *,
    ax: Optional[plt.Axes] = None,
    vmin_log: float = -14.0,
    vmax_log: Optional[float] = None,
    title: str = r"Phase mixing: $|c_n(t)|$ (k=0)",
    cmap: str = "viridis",
    savepath: Optional[str] = None,
) -> plt.Axes:
    C = jnp.asarray(C)
    ts = jnp.asarray(ts, dtype=jnp.float64)
    C_nt = _k0_spatial_average(C)
    Z = jnp.log(jnp.abs(C_nt) + 1e-300)

    fig, ax = _ensure_fig_ax(ax)
    im = ax.imshow(
        np.asarray(Z),
        aspect="auto",
        origin="lower",
        extent=[float(ts[0]), float(ts[-1]), 0, int(C_nt.shape[0])],
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin_log,
        vmax=vmax_log,
    )
    fig.colorbar(im, ax=ax, label=r"log $|c_n(t)|$")
    ax.set_xlabel("t")
    ax.set_ylabel("Hermite n")
    ax.set_title(title)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return ax


# -------------------- Plot 2: total energy --------------------
def plot_total_energy(
    ts: jnp.ndarray,
    *,
    C: jnp.ndarray,                     # (N, nt) or (N, Nx, nt)
    E_xt: Optional[jnp.ndarray] = None, # (Nx, nt) (DG)
    Ek_kt: Optional[jnp.ndarray] = None,# (Nk, nt) (Fourier)
    ax: Optional[plt.Axes] = None,
    title: str = "Total energy (proxy)",
    savepath: Optional[str] = None,
    use_logy: bool = True,
) -> plt.Axes:
    ts = jnp.asarray(ts, dtype=jnp.float64)
    C_nt = _k0_spatial_average(jnp.asarray(C))
    if C_nt.shape[0] < 3:
        raise ValueError("Need at least 3 Hermite modes to build kinetic proxy (C0 and C2).")

    C0 = C_nt[0, :]
    C2 = C_nt[2, :]
    W_kin = 0.5 * jnp.real(C0 + C2)
    W_field = 0.5 * _E2_space_average(E_xt, Ek_kt)
    W_total = W_field + W_kin

    fig, ax = _ensure_fig_ax(ax)
    ax.plot(np.asarray(ts), np.asarray(W_field), "b-",  label="Field")
    ax.plot(np.asarray(ts), np.asarray(W_kin),   "r-",  label="Kinetic (C0+C2)/2")
    ax.plot(np.asarray(ts), np.asarray(W_total), "k--", label="Total")
    if use_logy:
        ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("Energy")
    ax.set_title(title); ax.legend(); fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return ax


# -------------------- Plot 3: animated f(x,u,t) --------------------
def _make_f0(u: jnp.ndarray, v_th: float) -> jnp.ndarray:
    return (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * (u / v_th) ** 2)

@jax.jit
def _reconstruct_F_dg_once(C_nx: jnp.ndarray, Phi: jnp.ndarray, f0: jnp.ndarray) -> jnp.ndarray:
    """
    DG frame: C_nx is (N, Nx) real; Phi (N, Nv); f0 (Nv,)
    Returns F_xu (Nx, Nv).
    """
    # F = f0 + (C^T @ Phi)
    return f0[None, :] + (jnp.transpose(C_nx, (1, 0)) @ Phi)

@jax.jit
def _reconstruct_F_fourier_once(C_kn: jnp.ndarray, Phi: jnp.ndarray, expikx: jnp.ndarray, f0: jnp.ndarray) -> jnp.ndarray:
    """
    Fourier frame: C_kn is (Nk, N) complex; Phi (N, Nv); expikx (Nk, Nx) complex; f0 (Nv,)
    Returns F_xu = f0 + Re[ (expikx^T) @ (C_kn @ Phi) ] of shape (Nx, Nv).
    """
    A_kv = C_kn @ Phi                      # (Nk, Nv), complex
    Xv = jnp.real(jnp.transpose(expikx, (1, 0)) @ A_kv)  # (Nx, Nv)
    return f0[None, :] + Xv

def animate_distribution(
    *,
    ts: jnp.ndarray,                         # (nt,)
    v: jnp.ndarray,                          # (Nv,)
    v_th: float,
    x: jnp.ndarray,                          # (Nx,)
    mode: Literal["dg", "fourier"],
    C: Optional[jnp.ndarray] = None,         # DG: (N, Nx, nt)
    C_knt: Optional[jnp.ndarray] = None,     # Fourier: (Nk, N, nt)
    k: Optional[jnp.ndarray] = None,         # Fourier k: (Nk,)
    f0_func=None,
    interval: int = 60,                      # ms per frame
    clim: Optional[Tuple[float,float]] = None,
    title: str = "f(x,u,t)",
) -> animation.FuncAnimation:
    """
    Build an animated imshow of f(x,u,t), x on x-axis, u on y-axis (velocity), color=f.
    - DG: supply mode="dg", C (N,Nx,nt).
    - Fourier: supply mode="fourier", C_knt (Nk,N,nt), k (Nk,), and x grid.

    Returns a matplotlib.animation.FuncAnimation; remember to keep a reference to it.
    """
    ts = jnp.asarray(ts, dtype=jnp.float64)
    v  = jnp.asarray(v,  dtype=jnp.float64)
    x  = jnp.asarray(x,  dtype=jnp.float64)

    if f0_func is None:
        f0 = _make_f0(v, v_th)
    else:
        f0 = jnp.asarray(f0_func(v), dtype=jnp.float64)

    # Hermite basis in velocity
    # For DG: N = C.shape[0]; for Fourier: N = C_knt.shape[1]
    if mode == "dg":
        if C is None:
            raise ValueError("DG mode needs C (N, Nx, nt).")
        N = int(C.shape[0])
        Phi = hermite_basis(v / v_th, N)               # (N, Nv)
    elif mode == "fourier":
        if C_knt is None or k is None:
            raise ValueError("Fourier mode needs C_knt (Nk,N,nt) and k (Nk,).")
        N = int(C_knt.shape[1])
        Phi = hermite_basis(v / v_th, N)
        # Precompute exp(ikx) once
        k = jnp.asarray(k, dtype=jnp.float64)
        expikx = jnp.exp(1j * (k[:, None] * x[None, :]))  # (Nk, Nx)
    else:
        raise ValueError("mode must be 'dg' or 'fourier'.")

    nt = int(ts.shape[0])
    Nx = int(x.shape[0])
    Nv = int(v.shape[0])

    # First frame
    if mode == "dg":
        F0 = _reconstruct_F_dg_once(C[:, :, 0], Phi, f0)   # (Nx, Nv)
    else:
        F0 = _reconstruct_F_fourier_once(C_knt[:, :, 0], Phi, expikx, f0)

    fig, ax = plt.subplots()
    im = ax.imshow(
        np.asarray(F0).T,   # velocity on y-axis => transpose to (Nv, Nx)
        aspect="auto",
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(v[0]), float(v[-1])],
        interpolation="nearest",
        cmap="viridis",
    )
    if clim is not None:
        im.set_clim(*clim)

    cb = fig.colorbar(im, ax=ax, label="f(x,u,t)")
    ax.set_xlabel("x"); ax.set_ylabel("u = v/v_th")
    ttl = ax.set_title(f"{title} @ t={float(ts[0]):.3f}")

    # Frame function
    def update(i):
        if mode == "dg":
            F = _reconstruct_F_dg_once(C[:, :, i], Phi, f0)
        else:
            F = _reconstruct_F_fourier_once(C_knt[:, :, i], Phi, expikx, f0)
        im.set_array(np.asarray(F).T)
        ttl.set_text(f"{title} @ t={float(ts[i]):.3f}")
        return (im, ttl)

    ani = animation.FuncAnimation(fig, update, frames=nt, interval=interval, blit=False)
    fig.tight_layout()
    return ani