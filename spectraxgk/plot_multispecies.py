# spectraxgk/plot_multispecies.py
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

def _np(x):
    try:
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)

def plot_multispecies(out: dict, outdir: str = "plots", prefix: str = "run", show: bool = False):
    os.makedirs(outdir, exist_ok=True)

    t = _np(out["time"])
    phi_rms = _np(out.get("phi_rms", np.zeros_like(t)))
    W = _np(out.get("W_total", np.zeros_like(t)))
    Wphi = _np(out.get("W_phi", np.zeros_like(t)))
    Ws = _np(out.get("W_s", None))
    D = _np(out.get("D_coll", np.zeros_like(t)))
    CumD = _np(out.get("Cum_D_coll", None))
    Em = _np(out.get("E_m", None))

    species = out.get("species", [])
    names = [sp.get("name", f"s{i}") for i, sp in enumerate(species)]
    if Ws is None:
        Ns = int(out.get("Ns", 1))
        Ws = np.zeros((len(t), Ns))

    # --- time series figure ---
    fig = plt.figure()
    plt.plot(t, phi_rms)
    plt.xlabel("t")
    plt.ylabel(r"$\phi_{\rm rms}$")
    plt.title("Potential fluctuation amplitude")
    fig.savefig(os.path.join(outdir, f"{prefix}_phi_rms.png"), dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(t, W, label="W_total (0.5||H||^2)")
    if np.any(np.isfinite(Wphi)):
        plt.plot(t, Wphi, label="W_phi (proxy)")
    plt.xlabel("t")
    plt.ylabel("Energy-like")
    plt.legend()
    plt.title("Energy diagnostics")
    fig.savefig(os.path.join(outdir, f"{prefix}_energy.png"), dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    fig = plt.figure()
    for i in range(Ws.shape[1]):
        nm = names[i] if i < len(names) else f"s{i}"
        plt.plot(t, Ws[:, i], label=f"W_{nm}")
    plt.xlabel("t"); plt.ylabel("0.5||H_s||^2")
    plt.legend()
    plt.title("Per-species energy")
    fig.savefig(os.path.join(outdir, f"{prefix}_energy_species.png"), dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(t, D, label="D_coll(t)")
    if CumD is not None:
        plt.plot(t, CumD, label="Cum_D_coll(t)")
    plt.xlabel("t"); plt.ylabel("dissipation")
    plt.legend()
    plt.title("Collision dissipation")
    fig.savefig(os.path.join(outdir, f"{prefix}_collisions.png"), dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    # --- Hermite spectrum heatmaps ---
    if Em is not None and Em.ndim == 3:
        # Em shape: (nt, Ns, Nh)
        nt, Ns, Nh = Em.shape
        for s in range(Ns):
            fig = plt.figure()
            plt.imshow(np.log10(np.maximum(Em[:, s, :], 1e-30)).T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, Nh-1])
            plt.colorbar(label=r"$\log_{10} E_m$")
            nm = names[s] if s < len(names) else f"s{s}"
            plt.xlabel("t"); plt.ylabel("m")
            plt.title(f"Hermite spectrum vs time ({nm})")
            fig.savefig(os.path.join(outdir, f"{prefix}_Em_{nm}.png"), dpi=200, bbox_inches="tight")
            if show: plt.show()
            plt.close(fig)

    # --- 1D phi(z,t) ---
    if "phi_k_line" in out:
        phi_k_line = _np(out["phi_k_line"])  # (nt, Nz) complex
        # Inverse FFT from fftshifted ordering, norm="forward" to match model helpers.
        phi_z = np.fft.ifft(np.fft.ifftshift(phi_k_line, axes=-1), axis=-1, norm="forward")
        fig = plt.figure()
        plt.imshow(np.real(phi_z), aspect="auto", origin="lower",
                   extent=[0, phi_z.shape[1], t[0], t[-1]])
        plt.colorbar(label="Re(phi(z,t))")
        plt.xlabel("z index")
        plt.ylabel("t")
        plt.title("Real-space potential (1D)")
        fig.savefig(os.path.join(outdir, f"{prefix}_phi_zt.png"), dpi=200, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)

    return outdir
