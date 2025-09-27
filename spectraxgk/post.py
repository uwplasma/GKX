from __future__ import annotations
from typing import Optional, Tuple

import os

import matplotlib.pyplot as plt
import numpy as np

from .types import Result


def load_result(path: str) -> Result:
    data = np.load(path, allow_pickle=True)
    meta = data["meta"].item() if isinstance(data["meta"].item, object) else dict(data["meta"])  # type: ignore
    # OPTIONAL: if ks exists at top level, mirror it into meta["grid"]["klist"]
    if "ks" in data.files:
        meta.setdefault("grid", {}).setdefault("klist", data["ks"].tolist())
    return Result(t=data["t"], C=data["C"], meta=meta)


def _get_ks_from_meta(meta: dict) -> Optional[np.ndarray]:
    ks = meta.get("grid", {}).get("klist", None)
    if ks is None:
        # solver saved single-k as "kpar" on linear runs
        kpar = meta.get("grid", {}).get("kpar", meta.get("kpar", None))
        if kpar is None:
            return None
        return np.array([float(kpar)], dtype=float)
    return np.asarray(ks, dtype=float)


def _as_4d(C: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Return C as shape (nt, Nk, Nn, Nm), plus dims."""
    if C.ndim == 3:
        nt, Nn, Nm = C.shape
        C4 = C[:, None, :, :]
        return C4, nt, 1, Nn, Nm
    elif C.ndim == 4:
        nt, Nk, Nn, Nm = C.shape
        return C, nt, Nk, Nn, Nm
    else:
        raise ValueError(f"Unexpected C.ndim={C.ndim}")

def _ks_for(res: Result) -> np.ndarray:
    """Return ks of length Nk that matches C’s second axis."""
    C4, _, Nk, _, _ = _as_4d(res.C)
    ks = _get_ks_from_meta(res.meta)
    if ks is None:
        return np.zeros((Nk,), dtype=float)
    ks = np.asarray(ks, dtype=float).reshape(-1)
    if ks.size != Nk:
        if ks.size > Nk:
            ks = ks[:Nk]
        else:
            ks = np.pad(ks, (0, Nk - ks.size), mode="constant")
    return ks

def _pick_k_index(res: Result) -> int:
    if res.C.ndim != 4:
        return 0
    Nk = res.C.shape[1]
    if Nk == 1:
        return 0
    ks = _ks_for(res)
    return int(np.argmin(np.abs(ks)))

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
    """Compute energies for linear and nonlinear runs.

    Conventions (Laguerre m=0 only, as in the model):
      Wf(t) = (1/4) * sum_{k,n} |C_{k,n,0}(t)|^2
      WE(t) = (1/2) * sum_{k} |E_k(t)|^2,  E_k = i * C_{k,0,0} / k  (guard k=0)
      S(t)  = ∫_0^t  (1/2) * nu * sum_{k,n} n * |C_{k,n,0}|^2  dt'
      WN(t) = ∫_0^t  (1/4) * Re[ sum_{k,n} conj(C_{k,n,0}) * N_{k,n,0} ] dt'
              where N is the nonlinear convolution (only m=0; n>=1)
    """
    C4, nt, Nk, Nn, Nm = _as_4d(res.C)
    # --- ks normalization: ensure length Nk ---
    ks = _ks_for(res)  # <- ensures ks has length Nk
    if ks is None:
        ks = np.zeros((Nk,), dtype=float)
    else:
        ks = np.asarray(ks, dtype=float).reshape(-1)
        if ks.size != Nk:
            if ks.size > Nk:
                ks = ks[:Nk]
            else:
                ks = np.pad(ks, (0, Nk - ks.size), mode="constant")

    nu = float(res.meta.get("grid", {}).get("nu", res.meta.get("nu", 0.0)))
    nonlinear = bool(res.meta.get("sim", {}).get("nonlinear", False))

    # -------------------------
    # Free energy Wf (m=0 only)
    # -------------------------
    a_k_n = C4[:, :, :, 0]                             # (nt, Nk, Nn)
    Wf = 0.25 * np.sum(np.abs(a_k_n) ** 2, axis=(1, 2))  # (nt,)

    # -------------------------
    # Field energy WE (sum over k)
    # -------------------------
    a00_k = a_k_n[:, :, 0]                             # (nt, Nk)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ek = 1j * a00_k / ks[None, :]                 # (nt, Nk); ks=0 -> inf, we set to 0
        Ek = np.where(np.isfinite(Ek), Ek, 0.0)
    WE = 0.5 * np.sum(np.abs(Ek) ** 2, axis=1)         # (nt,)

    # ------------------------------------
    # Collisional sink integral S(t)
    # ------------------------------------
    weights = np.arange(Nn, dtype=float)[None, None, :]  # (1,1,Nn)
    coll_power = 0.5 * nu * np.sum(weights * np.abs(a_k_n) ** 2, axis=(1, 2))  # (nt,)
    S = _cumtrapz(coll_power, res.t)

    # ------------------------------------
    # Nonlinear power integral WN(t)
    # Only if Nk>1 or user saved ks; for Nk=1, N=0 by our model.
    # ------------------------------------
    WN = np.zeros(nt, dtype=float)
    if nonlinear and Nk > 1:
        def _N_from_a(a_t: np.ndarray, ks_vec: np.ndarray) -> np.ndarray:
            """a_t[k,n] -> N_t[k,n] (m=0 only)."""
            a0 = a_t[:, 0]  # (Nk,)
            with np.errstate(divide="ignore", invalid="ignore"):
                E = 1j * a0 / ks_vec
                E = np.where(np.isfinite(E), E, 0.0)
            Eh = np.fft.fft(E, axis=0)  # (Nk,)

            N_t = np.zeros_like(a_t, dtype=np.complex128)
            for n in range(1, a_t.shape[1]):
                Ah = np.fft.fft(a_t[:, n - 1], axis=0)   # (Nk,)
                conv = np.fft.ifft(Eh * Ah, axis=0)      # (Nk,)
                N_t[:, n] = np.sqrt(2.0 * n) * conv
            return N_t

        Pn = np.zeros(nt, dtype=float)
        for it in range(nt):
            N_t = _N_from_a(a_k_n[it], ks)  # (Nk, Nn)
            Pn[it] = 0.25 * np.real(np.sum(np.conj(a_k_n[it]) * N_t))
        WN = _cumtrapz(Pn, res.t)

    # Diagnostics
    Wsum = Wf + WE + S + WN

    return dict(Wf=Wf, WE=WE, S=S, WN=WN, Wsum=Wsum)


# ---- panel functions ----
def panel_energy(ax: plt.Axes, res: Result) -> None:
    """[0,0] Energies vs t: W_f, W_E, ∫C dt, ∫P_N dt, and their sum."""
    ch = _energy_channels(res)
    ax.plot(res.t, ch["Wf"],   label=r"$W_f$", linewidth=1.8)
    ax.plot(res.t, ch["WE"],   label=r"$W_E$", linewidth=1.8)
    ax.plot(res.t, ch["S"],    label=r"$\int_0^t C\,dt$", linestyle="--", linewidth=1.6)
    ax.plot(res.t, ch["WN"],   label=r"$\int_0^t P_N\,dt$", linestyle="--", linewidth=1.6)
    ax.plot(res.t, ch["Wsum"], label=r"$W_f + W_E + \int_0^t(C+P_N)\,dt$", linestyle=":", linewidth=1.8)
    ax.set_xlabel("t"); ax.set_ylabel("Energy"); ax.grid(True)
    ax.legend(loc="best", ncols=2)
    ax.set_title(r"Energy channels vs $t$")


def panel_density_xt(ax: plt.Axes, res: Result, Nx: int = 256) -> None:
    if res.C.ndim == 4:
        kidx = _pick_k_index(res)
        ks = _ks_for(res)
        C00 = res.C[:, kidx, 0, 0]
        k = float(ks[kidx])
    else:
        C00 = res.C[:, 0, 0]
        k = float(res.meta.get("grid", {}).get("kpar", 1.0))

    x, _, dens = _density_x_t(C00, k, Nx=Nx)
    im = ax.imshow(dens.T, origin="lower", aspect="auto",
                   extent=[res.t[0], res.t[-1], x[0], x[-1]])
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(r"density $(x,t)$ from $C_{0,0}(t)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)



def panel_fvparvperp(ax: plt.Axes, res: Result, t_index: int, vth_fallback: float = 1.0) -> None:
    vth = float(res.meta.get("grid", {}).get("vth", vth_fallback))
    vpar = np.linspace(-4 * vth, 4 * vth, 201)
    vperp = np.linspace(0, 4 * vth, 201)

    if res.C.ndim == 4:
        kidx = _pick_k_index(res)
        Cnm = res.C[t_index, kidx, :, :]
    else:
        Cnm = res.C[t_index, :, :]

    f = _reconstruct_f_vpar_vperp(Cnm, vpar, vperp, vth)
    im = ax.imshow(f.T, origin="lower", aspect="auto",
                   extent=[vpar[0], vpar[-1], vperp[0], vperp[-1]])
    ax.set_xlabel(r"$v_\parallel$"); ax.set_ylabel(r"$v_\perp$")
    title_t = "0" if t_index == 0 else "final"
    ax.set_title(rf"$f(v_\parallel, v_\perp)$ at $t={title_t}$ (Re)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_hermite_heatmap_m0(ax: plt.Axes, res: Result) -> None:
    if res.C.ndim == 4:
        kidx = _pick_k_index(res)
        C_n_m0_t = np.abs(res.C[:, kidx, :, 0])   # (nt, Nn)
    else:
        C_n_m0_t = np.abs(res.C[:, :, 0])         # (nt, Nn)

    im = ax.imshow(C_n_m0_t, origin="lower", aspect="auto",
                   extent=[0, C_n_m0_t.shape[1] - 1, res.t[0], res.t[-1]])
    ax.set_xlabel("n"); ax.set_ylabel("t"); ax.set_title(r"$|C_{n,\,m=0}(t)|$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_stacked_coeffs(ax: plt.Axes, res: Result) -> None:
    if res.C.ndim == 4:
        kidx = _pick_k_index(res)
        Ck = res.C[:, kidx, :, :]                  # (nt, Nn, Nm)
    else:
        Ck = res.C                                   # (nt, Nn, Nm)

    nt, Nn, Nm = Ck.shape
    blocks = [np.abs(Ck[:, :, m]) for m in range(Nm)]  # each (nt, Nn)
    big = np.concatenate(blocks, axis=1)               # (nt, Nm*Nn)

    im = ax.imshow(big, origin="lower", aspect="auto",
                   extent=[0, Nm * Nn, res.t[0], res.t[-1]])
    ax.set_xlabel("n blocks per m"); ax.set_ylabel("t")
    ax.set_title(r"Stacked $|C_{n,m}(t)|$ (m=0,1,...) blocks")

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