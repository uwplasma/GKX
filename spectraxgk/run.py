"""
Entry point: read input.toml, run the chosen mode, and make basic plots.

Usage:
  python -m fh.run --input examples/landau/input.toml
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from io_config import read_toml
from backends import run_fourier, run_dg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    args = p.parse_args()

    cfg = read_toml(args.input)

    if cfg.sim.mode == "fourier":
        ts, out = run_fourier(cfg)
        k = out["k"]; Ek = out["Ek_kt"]  # (Nk, nt)
        plt.figure()
        for i, ki in enumerate(k):
            plt.plot(ts, np.log(np.abs(Ek[i]) + 1e-300), label=f"k={ki:.3f}")
        plt.xlabel("t"); plt.ylabel("log |E_k|")
        plt.title("Fourier–Hermite: log |E_k(t)|")
        plt.legend(); plt.tight_layout()

    elif cfg.sim.mode == "dg":
        ts, out = run_dg(cfg)
        C_t = out["C_t"]   # (N, Nx, nt)
        E_xt = out["E_xt"] # (Nx, nt)

        # Plot log |c_n| imshow for a few n
        N, Nx, nt = C_t.shape
        nn = min(6, N)
        for n in range(nn):
            plt.figure()
            plt.imshow(np.log(np.abs(C_t[n]) + 1e-12), aspect="auto", origin="lower",
                       extent=[ts[0], ts[-1], 0, Nx], interpolation="nearest", vmin=-12)
            plt.colorbar(label=f"log |c_{n}(x,t)|")
            plt.xlabel("t"); plt.ylabel("cell index")
            plt.title(f"DG: Hermite n={n}")

        # Field
        plt.figure()
        plt.imshow(E_xt, aspect="auto", origin="lower",
                   extent=[ts[0], ts[-1], 0, Nx], interpolation="nearest")
        plt.colorbar(label="E(x,t)")
        plt.xlabel("t"); plt.ylabel("cell index")
        plt.title("DG: Electric field E(x,t)")

        # Energy diagnostic (free energy proxy)
        W_field = 0.5 * np.mean(E_xt**2, axis=0)
        W_kin = 0.5 * np.sum(np.mean(np.abs(C_t[1:, :, :])**2, axis=1), axis=0)
        plt.figure()
        plt.plot(ts, W_field, label="Field")
        plt.plot(ts, W_kin, label="Kinetic")
        plt.plot(ts, W_field + W_kin, "k--", label="Total")
        plt.yscale("log")
        plt.xlabel("t"); plt.ylabel("Energy proxy")
        plt.title("DG: Free-energy proxy")
        plt.legend(); plt.tight_layout()

    else:
        raise SystemExit("sim.mode must be 'fourier' or 'dg'")

    plt.show()

if __name__ == "__main__":
    main()
