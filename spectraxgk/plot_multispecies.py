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


def _safe_get_time_series(out: dict, key: str, t: np.ndarray, default: float = 0.0):
    if key not in out:
        return np.full_like(t, default, dtype=float)
    arr = _np(out[key])
    if arr.ndim == 0:
        return np.full_like(t, float(arr), dtype=float)
    return arr


def plot_multispecies(out: dict, outdir: str = "plots", prefix: str = "run", show: bool = False):
    """
    Publication-ready (but lightweight) plotting.

    Produces:
      - phi_rms(t)
      - free energy + components
      - per-species kinetic part W_h_s(t)
      - collisions (if provided)
      - Hermite spectrum heatmaps E_m(t,m) per species
      - 1D real-space phi(z,t) if line diagnostics saved
    """
    os.makedirs(outdir, exist_ok=True)

    t = _np(out["time"])

    phi_rms = _safe_get_time_series(out, "phi_rms", t, 0.0)
    W_free = _safe_get_time_series(out, "W_free", t, 0.0)
    W_h = _safe_get_time_series(out, "W_h", t, 0.0)
    W_phi = _safe_get_time_series(out, "W_phi", t, 0.0)

    # per-species kinetic time series
    W_h_s = out.get("W_h_s", None)
    if W_h_s is not None:
        W_h_s = _np(W_h_s)  # (nt,Ns)
    else:
        # fall back to nothing
        W_h_s = None

    Em = out.get("E_m", None)
    if Em is not None:
        Em = _np(Em)  # (nt,Ns,Nh) typically

    species = out.get("species", [])
    names = [sp.get("name", f"s{i}") for i, sp in enumerate(species)]
    if len(names) == 0:
        Ns = int(out.get("Ns", 1))
        names = [f"s{i}" for i in range(Ns)]

    # --- phi_rms ---
    fig = plt.figure()
    plt.plot(t, phi_rms)
    plt.xlabel("t")
    plt.ylabel(r"$\phi_{\rm rms}$")
    plt.title("Potential fluctuation amplitude")
    fig.savefig(os.path.join(outdir, f"{prefix}_phi_rms.png"), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- energy ---
    fig = plt.figure()
    plt.plot(t, W_free, label=r"$W_{\rm free}$")
    plt.plot(t, W_h, label=r"$W_h$")
    plt.plot(t, W_phi, label=r"$W_\phi$")
    plt.xlabel("t")
    plt.ylabel("Free energy (normalized)")
    plt.legend(frameon=False)
    plt.title("Free energy diagnostics")
    fig.savefig(os.path.join(outdir, f"{prefix}_energy.png"), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- per-species kinetic W_h_s ---
    if W_h_s is not None and W_h_s.ndim == 2:
        fig = plt.figure()
        Ns = W_h_s.shape[1]
        for i in range(Ns):
            nm = names[i] if i < len(names) else f"s{i}"
            plt.plot(t, W_h_s[:, i], label=f"{nm}")
        plt.xlabel("t")
        plt.ylabel(r"$W_{h,s}$")
        plt.legend(frameon=False)
        plt.title("Per-species kinetic free energy")
        fig.savefig(os.path.join(outdir, f"{prefix}_energy_species.png"), dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # --- Hermite spectrum heatmaps ---
    if Em is not None and Em.ndim == 3:
        nt, Ns, Nh = Em.shape
        for s in range(Ns):
            fig = plt.figure()
            Em_s = np.maximum(Em[:, s, :], 1e-30)
            plt.imshow(np.log10(Em_s).T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, Nh - 1])
            plt.colorbar(label=r"$\log_{10} E_m$")
            nm = names[s] if s < len(names) else f"s{s}"
            plt.xlabel("t")
            plt.ylabel("Hermite index m")
            plt.title(f"Hermite spectrum vs time ({nm})")
            fig.savefig(os.path.join(outdir, f"{prefix}_Em_{nm}.png"), dpi=220, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # --- 1D phi(z,t) reconstruction if saved ---
    if "phi_k_line_re" in out and "phi_k_line_im" in out:
        phi_k_line = _np(out["phi_k_line_re"]) + 1j * _np(out["phi_k_line_im"])  # (nt,Nz)
        # Inverse FFT from fftshifted ordering along z
        phi_z = np.fft.ifft(np.fft.ifftshift(phi_k_line, axes=-1), axis=-1, norm="forward")

        fig = plt.figure()
        plt.imshow(np.real(phi_z), aspect="auto", origin="lower",
                   extent=[0, phi_z.shape[1], t[0], t[-1]])
        plt.colorbar(label=r"$\Re\,\phi(z,t)$")
        plt.xlabel("z index")
        plt.ylabel("t")
        plt.title("Real-space potential (1D)")
        fig.savefig(os.path.join(outdir, f"{prefix}_phi_zt.png"), dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return outdir
