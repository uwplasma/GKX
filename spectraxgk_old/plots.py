"""
One-figure plotting for 1D Vlasov–Poisson (Fourier/DG), multi-species.

Layout (always 2 columns):
  Row 1:  [ Energy (per-species + field + total) | E(x,t) imshow ]
  Row s+1 for each species s (s=1..S):
          [ Phase mixing | Animated f_s(x,u,t) ]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax
import matplotlib.pyplot as plt
from matplotlib import animation

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from spectraxgk.constants import speed_of_light as c_light
from spectraxgk.diagnostics import (
    energies_dg_exact,
    energies_fourier_exact,
    pick_k0_index,
)


# -------------------- Hermite basis --------------------
def hermite_basis(u: jnp.ndarray, N: int) -> jnp.ndarray:
    u = jnp.asarray(u, dtype=jnp.float64)
    Phi = jnp.zeros((N, u.shape[0]), dtype=jnp.float64)
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
    if N > 2:  # noqa: PLR2004
        # We already filled φ0 and φ1. `rest` contains exactly N-2 functions (φ2..φ_{N-1}),
        # so place them starting at index 2 up to (but not including) N.
        Phi = Phi.at[2:N].set(rest)
    return Phi


# -------------------- Helpers --------------------
def _maxwellian_shifted(v: jnp.ndarray, n0: float, u0: float, vth: float) -> jnp.ndarray:
    v = jnp.asarray(v, dtype=jnp.float64)
    vth = jnp.asarray(vth, dtype=jnp.float64)
    return n0 * (1.0 / (jnp.sqrt(2.0 * jnp.pi) * vth)) * jnp.exp(-0.5 * ((v - u0) / vth) ** 2)


def _sp_name(species_list, idx: int) -> str:
    return getattr(species_list[idx], "name", f"s{idx}")


# -------------------- Panels --------------------
def plot_phase_mixing(
    ts: jnp.ndarray,
    C_nt_or_nxt: jnp.ndarray,  # (N, nt) or (N, Nx, nt)
    *,
    ax: plt.Axes,
    vmin_log: float = -14.0,
    vmax_log: float | None = None,
    title: str = r"Phase mixing",
    cmap: str = "viridis",
) -> None:
    # If C has shape (N, Nx, nt) (3-D: Hermite × space × time), average over space (axis=1).
    C_nt = jnp.mean(C_nt_or_nxt, axis=1) if C_nt_or_nxt.ndim == 3 else C_nt_or_nxt  # noqa: PLR2004
    Z = jnp.log(jnp.abs(C_nt) + 1e-300)
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
    ax.figure.colorbar(im, ax=ax, label=r"log $|c_n(t)|$")
    ax.set_xlabel("t")
    ax.set_ylabel("Hermite n")
    ax.set_title(title)


def plot_energy_row(
    ts: jnp.ndarray,
    *,
    species_energy: Sequence[tuple[str, jnp.ndarray]],  # list of (label, W_kin_s(t))
    W_field: jnp.ndarray,  # (nt,)
    ax_energy: plt.Axes,
) -> None:
    for label, Wk in species_energy:
        ax_energy.plot(np.asarray(ts), np.asarray(Wk), label=f"Kin({label})")
    W_field_np = np.asarray(W_field)
    ax_energy.plot(np.asarray(ts), W_field_np, "k--", label="Field")
    W_total = W_field_np + np.sum([np.asarray(Wk) for _, Wk in species_energy], axis=0)
    ax_energy.plot(np.asarray(ts), W_total, "r-", label="Total")
    ax_energy.set_yscale("log")
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy conservation")
    ax_energy.legend(loc="best", fontsize=9)


def plot_E_xt_imshow(
    ts: jnp.ndarray,
    x: jnp.ndarray,
    E_xt: jnp.ndarray,
    *,
    ax: plt.Axes,
    title: str = r"$E(x,t)$",
    cmap: str = "RdBu_r",
) -> None:
    im = ax.imshow(
        np.asarray(E_xt),
        aspect="auto",
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(ts[0]), float(ts[-1])],
        interpolation="nearest",
        cmap=cmap,
    )
    ax.figure.colorbar(im, ax=ax, label="E")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)


# -------------------- Distribution animation helpers --------------------
@jax.jit
def _reconstruct_F_dg_once(C_nx: jnp.ndarray, Phi: jnp.ndarray, f0: jnp.ndarray) -> jnp.ndarray:
    return f0[None, :] + (jnp.transpose(C_nx, (1, 0)) @ Phi)


@jax.jit
def _reconstruct_F_fourier_once(
    C_kn: jnp.ndarray, Phi: jnp.ndarray, expikx: jnp.ndarray, f0: jnp.ndarray
) -> jnp.ndarray:
    A_kv = C_kn @ Phi
    Xv = jnp.real(jnp.transpose(expikx, (1, 0)) @ A_kv)
    return f0[None, :] + Xv


def animate_distribution(
    *,
    ts: jnp.ndarray,  # (nt,)
    v: jnp.ndarray,  # (Nv,)
    v_th: float,
    x: jnp.ndarray,  # (Nx,)
    mode: Literal["dg", "fourier"],
    ax: plt.Axes,
    C: jnp.ndarray | None = None,  # DG: (N, Nx, nt)
    C_knt: jnp.ndarray | None = None,  # Fourier: (Nk, N, nt)
    k: jnp.ndarray | None = None,  # Fourier k: (Nk,)
    species_params: dict | None = None,  # one species dict (n0,u0,vth)
    interval: int = 60,
    clim: tuple[float, float] | None = None,
    title: str | None = None,
) -> animation.FuncAnimation:
    if title is None:
        title = "f(x,u,t)"
    ts = jnp.asarray(ts, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    # single-species f0 for that row
    if species_params is not None:
        n0 = float(species_params.get("n0", 1.0))
        u0 = float(species_params.get("u0", 0.0))
        vth_s = float(species_params.get("vth", v_th))
        f0 = _maxwellian_shifted(v, n0, u0, vth_s)
    else:
        f0 = (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * (v / v_th) ** 2)

    # basis
    if mode == "dg":
        if C is None:
            raise ValueError("DG mode needs C (N, Nx, nt).")
        N = int(C.shape[0])
        Phi = hermite_basis(v / v_th, N)
    else:
        if C_knt is None or k is None:
            raise ValueError("Fourier mode needs C_knt (Nk,N,nt) and k (Nk,).")
        N = int(C_knt.shape[1])
        Phi = hermite_basis(v / v_th, N)
        k = jnp.asarray(k, dtype=jnp.float64)
        expikx = jnp.exp(1j * (k[:, None] * x[None, :]))  # (Nk, Nx)

    # first frame
    if mode == "dg":
        F0 = _reconstruct_F_dg_once(C[:, :, 0], Phi, f0)
    else:
        F0 = _reconstruct_F_fourier_once(C_knt[:, :, 0], Phi, expikx, f0)

    im = ax.imshow(
        np.asarray(F0).T,  # velocity on y-axis
        aspect="auto",
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(v[0]), float(v[-1])],
        interpolation="nearest",
        cmap="viridis",
    )
    if clim is not None:
        im.set_clim(*clim)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ttl = ax.set_title(f"{title} @ t={float(ts[0]):.3f}")

    def update(i):
        if mode == "dg":
            F = _reconstruct_F_dg_once(C[:, :, i], Phi, f0)
        else:
            F = _reconstruct_F_fourier_once(C_knt[:, :, i], Phi, expikx, f0)
        im.set_array(np.asarray(F).T)
        ttl.set_text(f"{title} @ t={float(ts[i]):.3f}")
        return (im, ttl)

    ani = animation.FuncAnimation(
        ax.figure, update, frames=int(ts.shape[0]), interval=interval, blit=False
    )
    return ani


# -------------------- One-figure renderer (multi-species only) --------------------
def render_suite_onefigure(cfg, ts: jnp.ndarray, out: dict) -> None:
    """
    Single figure with 2 columns and (1 + S) rows:
      Row 1: [Energy panel | E(x,t) imshow]
      Rows 2..S+1: one row per species: [Phase mixing | Animated distribution]

    Assumes multi-species outputs:
      Fourier: out must contain C_kSnt (Nk,S,N,nt), Ek_kt (Nk,nt), k (Nk,)
      DG:      out must contain C_St (S,N,Nx,nt), E_xt (Nx,nt), x (Nx,)
    """
    plot_cfg = getattr(cfg, "plot", None)
    if plot_cfg is None:
        raise ValueError("cfg.plot is required.")
    species_list = getattr(cfg, "species", [])
    if len(species_list) == 0:
        raise ValueError("Expect at least one [[species]] in the TOML.")

    S = len(species_list)

    nv = int(getattr(plot_cfg, "nv", 257))
    vmin_c = getattr(plot_cfg, "vmin_c", None)
    vmax_c = getattr(plot_cfg, "vmax_c", None)

    if (vmin_c is not None) and (vmax_c is not None):
        vmin = float(vmin_c) * c_light
        vmax = float(vmax_c) * c_light
    else:
        # fallback auto-range based on species
        vmins = [float(sp.u0) - 5 * float(sp.vth) for sp in species_list]
        vmaxs = [float(sp.u0) + 5 * float(sp.vth) for sp in species_list]
        vmin = min(vmins)
        vmax = max(vmaxs)

    v = jnp.linspace(vmin, vmax, nv, dtype=jnp.float64)
    save_anim = getattr(plot_cfg, "save_anim", None)
    fps = getattr(plot_cfg, "fps", 30)
    dpi = getattr(plot_cfg, "dpi", 150)
    no_show = getattr(plot_cfg, "no_show", False)
    fig_width = getattr(plot_cfg, "fig_width", 12.0)
    fig_row_h = getattr(plot_cfg, "fig_row_height", 3.0)

    # x-grid (for E and animations)
    if cfg.grid.L is None or cfg.grid.Nx is None:
        raise ValueError("grid.L and grid.Nx are required for plotting.")
    L, Nx = float(cfg.grid.L), int(cfg.grid.Nx)
    xgrid = jnp.linspace(0.0, L, Nx, endpoint=False, dtype=jnp.float64)

    # Build figure
    nrows = 1 + S
    fig, axes = plt.subplots(
        nrows, 2, figsize=(float(fig_width), float(fig_row_h) * nrows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    anims = []

    # ---------- Row 1: Energy + E(x,t) ----------
    ax_energy = axes[0, 0]
    ax_Ex_t = axes[0, 1]

    L, Nx = float(cfg.grid.L), int(cfg.grid.Nx)
    xgrid = jnp.linspace(0.0, L, Nx, endpoint=False, dtype=jnp.float64)
    if cfg.sim.mode == "fourier":
        species_energy, W_field, E_xt = energies_fourier_exact(
            out=out, species_list=species_list, L=L, Nx=Nx, x=xgrid
        )
        plot_energy_row(ts, species_energy=species_energy, W_field=W_field, ax_energy=ax_energy)

        if E_xt is not None:
            plot_E_xt_imshow(ts, xgrid, E_xt.T, ax=ax_Ex_t)
        else:
            ax_Ex_t.set_title("E(x,t) unavailable")
            ax_Ex_t.axis("off")

    else:
        species_energy, W_field, E_xt = energies_dg_exact(
            out=out, species_list=species_list, L=L, Nx=Nx
        )
        plot_energy_row(ts, species_energy=species_energy, W_field=W_field, ax_energy=ax_energy)
        plot_E_xt_imshow(ts, out["x"], E_xt.T, ax=ax_Ex_t)

    # ---------- Rows 2..S+1: per-species phase mixing + f(x,u,t) ----------
    first_ani = None
    for s in range(S):
        ax_pm = axes[s + 1, 0]
        ax_fx = axes[s + 1, 1]
        sp = species_list[s]
        sp_name = _sp_name(species_list, s)
        vth_s = float(getattr(sp, "vth", 1.0))

        if cfg.sim.mode == "fourier":
            C_kSnt = out["C_kSnt"]  # (Nk,S,N,nt)
            k = out["k"]  # (Nk,)
            # Phase mixing at k≈0
            j0 = pick_k0_index(k)
            C_s_n_t = C_kSnt[j0, s, :, :]  # (N,nt)
            plot_phase_mixing(ts, C_s_n_t, ax=ax_pm, title=f"Phase mix ({sp_name})")

            # Distribution animation (use full bank for that species)
            C_knt_s = C_kSnt[:, s, :, :]  # (Nk,N,nt)
            ani = animate_distribution(
                ts=ts,
                v=v,
                v_th=vth_s,
                x=xgrid,
                mode="fourier",
                ax=ax_fx,
                C_knt=C_knt_s,
                k=k,
                species_params={
                    "n0": getattr(sp, "n0", 1.0),
                    "u0": getattr(sp, "u0", 0.0),
                    "vth": getattr(sp, "vth", 1.0),
                },
                title=f"f(x,u,t) ({sp_name})",
            )
            anims.append(ani)

        else:
            C_St = out["C_St"]  # (S,N,Nx,nt)
            C_s_nxt = C_St[s, :, :, :]  # (N,Nx,nt)
            plot_phase_mixing(ts, C_s_nxt, ax=ax_pm, title=f"Phase mix ({sp_name})")
            ani = animate_distribution(
                ts=ts,
                v=v,
                v_th=vth_s,
                x=xgrid,
                mode="dg",
                ax=ax_fx,
                C=C_s_nxt,
                species_params={
                    "n0": getattr(sp, "n0", 1.0),
                    "u0": getattr(sp, "u0", 0.0),
                    "vth": getattr(sp, "vth", 1.0),
                },
                title=f"f(x,u,t) ({sp_name})",
            )
            anims.append(ani)

        if first_ani is None:
            first_ani = ani

    fig._anims = anims

    if save_anim and anims:
        anims[0].save(save_anim, dpi=dpi, fps=fps)

    if not no_show:
        plt.show()
