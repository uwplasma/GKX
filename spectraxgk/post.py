from __future__ import annotations
from typing import Optional, Tuple

import os

import matplotlib.pyplot as plt
import numpy as np

from .types import Result


def load_result(path: str) -> Result:
    data = np.load(path, allow_pickle=True)
    meta = data["meta"].item() if isinstance(data["meta"].item, object) else dict(data["meta"])  # type: ignore
    return Result(t=data["t"], C=data["C"], meta=meta)


# ---- Helpers for visualization ----
from numpy.polynomial.hermite import hermval  # physicists' Hermite H_n
from numpy.polynomial.laguerre import lagval  # standard Laguerre L_m


def _reconstruct_f_vpar_vperp(
    C_nm: np.ndarray, vpar: np.ndarray, vperp: np.ndarray, vth: float
) -> np.ndarray:
    """Qualitative f(v_parallel, v_perp) from Hermite–Laguerre coefficients.
    Uses H_n(x) and L_m(ρ) with Maxwell weight exp(-x^2-ρ). For quick-look plots.
    """
    Nn, Nm = C_nm.shape
    x = vpar / float(vth)
    rho = (vperp / float(vth)) ** 2
    H = np.empty((Nn, x.size))
    for n in range(Nn):
        coef = np.zeros(n + 1)
        coef[-1] = 1.0
        H[n] = hermval(x, coef)
    L = np.empty((Nm, rho.size))
    for m in range(Nm):
        coef = np.zeros(m + 1)
        coef[-1] = 1.0
        L[m] = lagval(rho, coef)
    weight = np.exp(-(x[:, None] ** 2) - rho[None, :])
    f = np.zeros((x.size, rho.size), dtype=np.complex128)
    for n in range(Nn):
        for m in range(Nm):
            f += C_nm[n, m] * H[n][:, None] * L[m][None, :]
    return (f * weight).real


def _density_x_t(
    C00_t: np.ndarray, k: float, Nx: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build density(x,t) assuming a single Fourier mode with amplitude C_{0,0}(t).
    density(x,t) = Re{ C00(t) * exp(i * k * x) }.
    Returns (x, t, dens) with dens shape (nt, Nx).
    """
    nt = C00_t.shape[0]
    L = (2 * np.pi / abs(k)) if k != 0 else (2 * np.pi)
    x = np.linspace(0.0, L, Nx)
    t = np.linspace(0.0, 1.0, nt)
    # Precompute cos/sin(kx)
    coskx = np.cos(k * x)
    sinkx = np.sin(k * x)
    dens = np.empty((nt, Nx))
    a = C00_t.real
    b = C00_t.imag
    # dens_t(x) = a(t)*cos(kx) - b(t)*sin(kx)
    for i in range(nt):
        dens[i, :] = a[i] * coskx - b[i] * sinkx
    return x, t, dens


# ---- small utilities ----
def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of y(x), same length as x, S[0]=0."""
    out = np.zeros_like(y, dtype=float)
    if len(x) > 1:
        dx = x[1:] - x[:-1]
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def _energy_channels(res: Result) -> dict[str, np.ndarray]:
    """Compute W_f, W_E, sink integral S, and conserved sum W_f+W_E+S."""
    k = float(res.meta.get("grid", {}).get("kpar", 0.0))
    nu = float(res.meta.get("grid", {}).get("nu", res.meta.get("nu", 0.0)))

    # free energy on m=0: (1/4) * sum_n |C_{n,0}|^2
    Cm0 = res.C[:, :, 0]                               # (nt, Nn)
    Wf = 0.25 * np.sum(np.abs(Cm0) ** 2, axis=1)

    # electric field E = i * C00 / k, WE = |E|^2/2
    C00 = res.C[:, 0, 0]
    Efield = (1j * C00 / k) if k != 0.0 else np.zeros_like(C00)
    WE = 0.5 * np.abs(Efield) ** 2

    # collisional power C(t) = (1/2) * nu * sum_n n * |C_{n,0}|^2
    Nn = Cm0.shape[1]
    weights = np.arange(Nn, dtype=float)
    coll_power = 0.5 * nu * np.sum(weights[None, :] * np.abs(Cm0) ** 2, axis=1)

    S = _cumtrapz(coll_power, res.t)                   # integrated sink
    Wsum = Wf + WE + S                                 # should be flat (conserved)

    return dict(Wf=Wf, WE=WE, S=S, Wsum=Wsum)


# ---- panel functions ----
def panel_energy(ax: plt.Axes, res: Result) -> None:
    """[0,0] Energies vs t: W_f, W_E, ∫C dt, and W_f+W_E+∫C dt."""
    ch = _energy_channels(res)
    ax.plot(res.t, ch["Wf"],   label=r"$W_f$", linewidth=1.8)
    ax.plot(res.t, ch["WE"],   label=r"$W_E$", linewidth=1.8)
    ax.plot(res.t, ch["S"],    label=r"$\int_0^t C\,dt$", linestyle="--", linewidth=1.6)
    ax.plot(res.t, ch["Wsum"], label=r"$W_f+W_E+\int_0^t C\,dt$", linestyle=":", linewidth=1.8)
    ax.set_xlabel("t"); ax.set_ylabel("Energy"); ax.grid(True)
    ax.legend(loc="best", ncols=2)
    ax.set_title(r"Energies and collisional sink vs $t$")


def panel_density_xt(ax: plt.Axes, res: Result, Nx: int = 256) -> None:
    """[0,1] density(x,t) from C_{0,0}(t) as imshow (x vertical, t horizontal)."""
    k = float(res.meta.get("grid", {}).get("kpar", 1.0))
    C00 = res.C[:, 0, 0]
    x, _, dens = _density_x_t(C00, k, Nx=Nx)
    im = ax.imshow(dens.T, origin="lower", aspect="auto",
                   extent=[res.t[0], res.t[-1], x[0], x[-1]])
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(r"density $(x,t)$ from $C_{0,0}(t)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_fvparvperp(ax: plt.Axes, res: Result, t_index: int, vth_fallback: float = 1.0) -> None:
    """[1,0] or [1,1] f(v_||, v_⊥) at a given time index (real part)."""
    vth = float(res.meta.get("grid", {}).get("vth", vth_fallback))
    vpar = np.linspace(-4 * vth, 4 * vth, 201)
    vperp = np.linspace(0, 4 * vth, 201)
    f = _reconstruct_f_vpar_vperp(res.C[t_index], vpar, vperp, vth)
    im = ax.imshow(f.T, origin="lower", aspect="auto",
                   extent=[vpar[0], vpar[-1], vperp[0], vperp[-1]])
    ax.set_xlabel(r"$v_\parallel$"); ax.set_ylabel(r"$v_\perp$")
    title_t = "0" if t_index == 0 else "final"
    ax.set_title(rf"$f(v_\parallel, v_\perp)$ at $t={title_t}$ (Re)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_hermite_heatmap_m0(ax: plt.Axes, res: Result) -> None:
    """[2,0] |C_{n,m=0}(t)| as time–Hermite heatmap."""
    C_n_m0_t = np.abs(res.C[:, :, 0])                      # (nt, Nn)
    im = ax.imshow(C_n_m0_t, origin="lower", aspect="auto",
                   extent=[0, C_n_m0_t.shape[1] - 1, res.t[0], res.t[-1]])
    ax.set_xlabel("n"); ax.set_ylabel("t"); ax.set_title(r"$|C_{n,\,m=0}(t)|$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_stacked_coeffs(ax: plt.Axes, res: Result) -> None:
    """[2,1] Stacked heatmap with time on the vertical axis.

    X-axis: concatenated Hermite blocks for m = 0,1,...
    Y-axis: time.
    """
    nt, Nn, Nm = res.C.shape
    blocks = [np.abs(res.C[:, :, m]) for m in range(Nm)]   # each (nt, Nn)
    big = np.concatenate(blocks, axis=1)                   # (nt, Nm*Nn); time is axis=0

    # Show time on vertical axis
    im = ax.imshow(
        big, origin="lower", aspect="auto",
        extent=[0, Nm * Nn, res.t[0], res.t[-1]]
    )

    ax.set_xlabel("n blocks per m")
    ax.set_ylabel("t")
    ax.set_title(r"Stacked $|C_{n,m}(t)|$ (m=0,1,...) blocks")

    # Tick marks at m-block boundaries along X
    xticks = [m * Nn for m in range(Nm + 1)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"m={m}" for m in range(Nm)] + [f"m={Nm}"])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    

def plot_energy(res: Result):
    """Interactive energy panel (same as summary [0,0])."""
    fig, ax = plt.subplots(figsize=(6, 4))
    panel_energy(ax, res)
    fig.tight_layout()
    plt.show()


def save_summary(res: Result, out_png: str) -> str:
    """Save a 3x2 summary figure composed from reusable panel functions.

    Layout:
      (0,0) Energies vs t: W_f, W_E, ∫C dt, and W_f+W_E+∫C dt
      (0,1) density(x,t) imshow (vertical = x, horizontal = t)
      (1,0) f(v_parallel, v_perp) imshow at t=0
      (1,1) f(v_parallel, v_perp) imshow at final time
      (2,0) Hermite heatmap |C_{n,m=0}(t)|
      (2,1) Stacked heatmap of |C_{n,m}(t)| blocks over time
    """
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    panel_energy(axs[0, 0], res)
    panel_density_xt(axs[0, 1], res)
    panel_fvparvperp(axs[1, 0], res, t_index=0)
    panel_fvparvperp(axs[1, 1], res, t_index=-1)
    panel_hermite_heatmap_m0(axs[2, 0], res)
    panel_stacked_coeffs(axs[2, 1], res)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png