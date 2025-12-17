# spectraxgk/_plot.py
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot", "plot_probe_velocity_reconstruction"]


def plot(output: dict):
    """
    Publication-ready summary plots for save='diagnostics' output.

    Expected keys (diagnostics mode):
      time, W_g, W_phi, W_total, D_coll, max_abs_G, max_abs_phi
      optional: Cum_D_coll
      optional: probe_G_lm
    """
    t = np.asarray(output["time"])
    Wg = np.asarray(output["W_g"])
    Wphi = np.asarray(output["W_phi"])
    Wtot = np.asarray(output["W_total"])

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    ax.plot(t, Wg, label=r"$W_g$ (moments)")
    ax.plot(t, Wphi, label=r"$W_\phi$ (field)")
    ax.plot(t, Wtot, label=r"$W_{\mathrm{tot}}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("energy proxy")
    ax.legend(frameon=True)
    ax.set_title("Laguerre–Hermite GK diagnostics")
    fig.tight_layout()

    # Dissipation figure
    if "D_coll" in output:
        D = np.asarray(output["D_coll"])
        fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 3.8))
        ax2.plot(t, D, label=r"$D_{\mathrm{coll}}(t) = -\Re\langle H, C(H)\rangle$")
        ax2.set_yscale("log" if np.all(D[D > 0] > 0) else "linear")
        ax2.set_xlabel("t")
        ax2.set_ylabel("collision dissipation proxy")
        ax2.legend(frameon=True)
        if "Cum_D_coll" in output:
            cum = np.asarray(output["Cum_D_coll"])
            ax2b = ax2.twinx()
            ax2b.plot(t, cum, linestyle="--", label=r"$\int_0^t D_{\mathrm{coll}}\,dt$")
            ax2b.set_ylabel("cumulative dissipation")
        fig2.tight_layout()

    # Norms figure
    if "max_abs_G" in output and "max_abs_phi" in output:
        mg = np.asarray(output["max_abs_G"])
        mp = np.asarray(output["max_abs_phi"])
        fig3, ax3 = plt.subplots(1, 1, figsize=(7.5, 3.8))
        ax3.plot(t, mg, label=r"$\max |G_k|$")
        ax3.plot(t, mp, label=r"$\max |\phi_k|$")
        ax3.set_yscale("log")
        ax3.set_xlabel("t")
        ax3.set_ylabel("max-norm")
        ax3.legend(frameon=True)
        fig3.tight_layout()

    plt.show()

    # Optional probe plot
    if "probe_G_lm" in output:
        plot_probe_velocity_reconstruction(output)


def _hermite_functions(v: np.ndarray, mmax: int):
    """
    Orthonormal Hermite *functions* (physicists) with Gaussian weight.
    This matches the common sqrt(m) ladder structure qualitatively.
    For publication-grade normalization you may want to align precisely to your chosen basis;
    for diagnostics/trends this is typically sufficient.
    """
    # Hermite polynomials (physicists): H_0=1, H_1=2x
    H = np.zeros((mmax, v.size))
    H[0] = 1.0
    if mmax > 1:
        H[1] = 2.0 * v
    for m in range(1, mmax - 1):
        H[m + 1] = 2.0 * v * H[m] - 2.0 * m * H[m - 1]

    # Hermite functions psi_m(v) = H_m(v) * exp(-v^2/2) / sqrt(2^m m! sqrt(pi))
    psi = np.zeros_like(H)
    w = np.exp(-0.5 * v**2)
    for m in range(mmax):
        norm = np.sqrt((2.0**m) * np.math.factorial(m) * np.sqrt(np.pi))
        psi[m] = H[m] * w / norm
    return psi


def plot_probe_velocity_reconstruction(output: dict, l_pick: int = 0, vmin=-6, vmax=6, Nv=300):
    """
    Reconstruct a velocity-space slice from the saved probe_G_lm(t) at one k-mode.

    We use only a single Laguerre index (default l=0) and reconstruct along Hermite m.

    output must contain:
      time (T,)
      probe_G_lm (T, lmax, mmax) complex
    """
    t = np.asarray(output["time"])
    G = np.asarray(output["probe_G_lm"])  # (T,lmax,mmax)
    if G.ndim != 3:
        return

    lmax = G.shape[1]
    mmax = G.shape[2]
    if l_pick >= lmax:
        l_pick = 0

    Glm = G[:, l_pick, :]  # (T,mmax)

    v = np.linspace(vmin, vmax, Nv)
    psi = _hermite_functions(v, mmax)  # (mmax,Nv)

    # f(v,t) ~ sum_m Re(G_m(t)) psi_m(v)
    f = (Glm.real @ psi).astype(np.float64)  # (T,Nv)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))
    im = ax.imshow(
        f,
        aspect="auto",
        origin="lower",
        extent=[vmin, vmax, float(t[0]), float(t[-1])],
    )
    ax.set_xlabel(r"$v_\parallel$")
    ax.set_ylabel("t")
    ax.set_title(f"Velocity-space reconstruction from probe (Laguerre l={l_pick})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$f(v_\parallel,t)$ (arb. units)")
    fig.tight_layout()
    plt.show()
