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


def plot_energy(res: Result):
    """Interactive plot (does not save)."""
    E = np.sum(np.abs(res.C) ** 2, axis=(1, 2))
    plt.figure()
    plt.plot(res.t, E)
    plt.xlabel("t")
    plt.ylabel(r"$\sum_{n,m} |C_{n,m}|^2$")
    plt.title("Hermite–Laguerre energy proxy")
    plt.grid(True)
    plt.show()


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
    t: Optional[np.ndarray] = None  # time vector provided externally when plotting
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


def save_summary(res: Result, out_png: str) -> str:
    """Save a 3x2 summary figure.

    Layout:
      (0,0) Overlay: Energy proxy (sum |C|^2) & |C_{0,0}| vs t (twin y-axis)
      (0,1) density(x,t) imshow (vertical = x, horizontal = t)
      (1,0) f(v_parallel, v_perp) imshow at t=0
      (1,1) f(v_parallel, v_perp) imshow at final time
      (2,0) Hermite spectrum at final time: sum_m |C_{n,m}(t_final)|^2 vs n
      (2,1) Stacked heatmap of |C_{n,m}(t)| for all m (blocks m=0,1,...) over time
    """
    # Scalars & slices
    E = np.sum(np.abs(res.C) ** 2, axis=(1, 2))
    C00 = res.C[:, 0, 0]

    # Velocity grids for quick-look f(v) plots
    vth = float(res.meta.get("grid", {}).get("vth", 1.0))
    vpar = np.linspace(-4 * vth, 4 * vth, 201)
    vperp = np.linspace(0, 4 * vth, 201)

    # Figure
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    # (0,0): overlay energy and |C00|
    ax00 = axs[0, 0]
    (l1,) = ax00.plot(res.t, E, label="Energy proxy (sum |C|^2)")
    ax00.set_xlabel("t")
    ax00.set_ylabel("E")
    ax00.grid(True)
    ax00t = ax00.twinx()
    (l2,) = ax00t.plot(res.t, np.abs(C00), linestyle="--", label=r"$|C_{0,0}|$")
    ax00t.set_ylabel(r"$|C_{0,0}|$")
    ax00.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best")
    ax00.set_title("Energy & $|C_{0,0}|$ vs t")

    # (0,1): density(x,t) imshow (x vertical, t horizontal)
    k = float(res.meta.get("grid", {}).get("kpar", 1.0))
    x, _, dens = _density_x_t(C00, k, Nx=256)
    im_d = axs[0, 1].imshow(
        dens.T, origin="lower", aspect="auto", extent=[res.t[0], res.t[-1], x[0], x[-1]]
    )
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel("x")
    axs[0, 1].set_title(r"density $(x,t)$ from $C_{0,0}(t)$")
    fig.colorbar(im_d, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # (1,0): f(v_parallel, v_perp) at t=0 (moved from [0,1])
    f0 = _reconstruct_f_vpar_vperp(res.C[0], vpar, vperp, vth)
    im0 = axs[1, 0].imshow(
        f0.T, origin="lower", aspect="auto", extent=[vpar[0], vpar[-1], vperp[0], vperp[-1]]
    )
    axs[1, 0].set_xlabel(r"$v_\parallel$")
    axs[1, 0].set_ylabel(r"$v_\perp$")
    axs[1, 0].set_title(r"$f(v_\parallel, v_\perp)$ at $t=0$ (Re)")
    fig.colorbar(im0, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # (1,1): f(v_parallel, v_perp) at final time
    fT = _reconstruct_f_vpar_vperp(res.C[-1], vpar, vperp, vth)
    imT = axs[1, 1].imshow(
        fT.T, origin="lower", aspect="auto", extent=[vpar[0], vpar[-1], vperp[0], vperp[-1]]
    )
    axs[1, 1].set_xlabel(r"$v_\parallel$")
    axs[1, 1].set_ylabel(r"$v_\perp$")
    axs[1, 1].set_title(r"$f(v_\parallel, v_\perp)$ at final t (Re)")
    fig.colorbar(imT, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # (2,0): |C_{n, m=0}(t)| as time–Hermite heatmap
    # Bottom-row heatmaps precompute
    C_n_m0_t = np.abs(res.C[:, :, 0])  # (nt, Nn)
    # C_n0_m_t = np.abs(res.C[:, 0, :])  # (nt, Nm)
    im1 = axs[2, 0].imshow(
        C_n_m0_t,
        origin="lower",
        aspect="auto",
        extent=[0, C_n_m0_t.shape[1] - 1, res.t[0], res.t[-1]],
    )
    axs[2, 0].set_xlabel("n")
    axs[2, 0].set_ylabel("t")
    axs[2, 0].set_title(r"$|C_{n,\,m=0}(t)|$")
    fig.colorbar(im1, ax=axs[2, 0], fraction=0.046, pad=0.04)

    # (2,1): Stacked heatmap over all m of |C_{n,m}(t)|.
    # Build a tall array by stacking each m-block (shape (nt, Nn)) vertically.
    nt, Nn, Nm = res.C.shape[0], res.C.shape[1], res.C.shape[2]
    blocks = []
    for m in range(Nm):
        blocks.append(np.abs(res.C[:, :, m]))  # (nt, Nn)
    big = np.concatenate(blocks, axis=1)  # (nt, Nm*Nn), time horizontal by default
    # We want time on horizontal axis; display with origin lower.
    im_all = axs[2, 1].imshow(
        big.T, origin="lower", aspect="auto", extent=[res.t[0], res.t[-1], 0, Nm * Nn]
    )
    axs[2, 1].set_xlabel("t")
    axs[2, 1].set_ylabel("n blocks per m")
    axs[2, 1].set_title(r"Stacked $|C_{n,m}(t)|$ (m=0,1,...) blocks")
    # Add y tick labels at block boundaries for readability
    yticks = [m * Nn for m in range(Nm + 1)]
    axs[2, 1].set_yticks(yticks)
    axs[2, 1].set_yticklabels([f"m={m}" for m in range(Nm)] + [f"m={Nm}"])
    fig.colorbar(im_all, ax=axs[2, 1], fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png
