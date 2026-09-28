#!/usr/bin/env python3
"""Offline test of a velocity-scaled Hermite basis on converged GKX KBM eigenvectors.

    python scaled_hermite_projection.py <eig.npz> [alphas...]
For every (species, Laguerre l, z) the v_par profile f(v) = sum_m g_m He_m(v) F(v)
(F(v) = exp(-v^2/2)/sqrt(2 pi), He probabilists' Hermite, <He_m He_n>_F = m! delta_mn) is
rebuilt from the saved alpha = 1 coefficients and least-squares projected onto the first N
scaled functions He_k(v/a) F(v/a)/a in the free-energy norm int f^2 / F dv, which is the
norm GKX's Hermite truncation and hypercollisions act in. That norm is finite for the scaled
functions only when a < sqrt(2). Prints the relative truncation error, amplitude-weighted
over (s, l, z), against N for each a. The reference must be converged in Nm (use N << Nm).
"""

import sys

import numpy as np

d = np.load(sys.argv[1])
st = d["state"]  # (Ns, Nl, Nm, ky, kx, z)
alphas = [float(a) for a in sys.argv[2:]] or [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
ns, nl, nm = st.shape[:3]
g = st.reshape(ns, nl, nm, -1)
# GKX stores orthonormal coefficients (streaming couples sqrt(m+1) G_{m+1} + sqrt(m) G_{m-1}),
# so f/F = sum_m G_m psi_m(v) with psi_m = He_m / sqrt(m!), built by the stable recurrence.
v, w = np.polynomial.hermite_e.hermegauss(
    max(3 * nm, 200)
)  # nodes/weights for weight exp(-v^2/2)
w = w / np.sqrt(2 * np.pi)  # now sum w * h = int h F dv


def psi(x, n):
    out = np.zeros((x.size, n))
    out[:, 0] = 1.0
    if n > 1:
        out[:, 1] = x
    for m in range(1, n - 1):
        out[:, m + 1] = (x * out[:, m] - np.sqrt(m) * out[:, m - 1]) / np.sqrt(m + 1)
    return out


def basis_over_F(a, N):
    # phi_k(v)/F(v) at the nodes, phi_k = psi_k(v/a) F(v/a)/a; norm int phi_j phi_k / F = sum w (phi/F)(phi/F)
    ratio = np.exp(-((v / a) ** 2) / 2 + v**2 / 2) / a
    return psi(v / a, N) * ratio[:, None]


fF = np.einsum("vm,slmz->slzv", psi(v, nm), g)
amp = np.sqrt(np.einsum("v,slzv->slz", w, np.abs(fF) ** 2))
tot = np.sqrt((amp**2).sum(axis=(1, 2)))
Ns = [n for n in (2, 4, 6, 8, 12, 16, 24, 32) if n <= nm // 2]
print(f"# {sys.argv[1]} Nl={nl} Nm={nm}; rows species, columns N = {Ns}")
for a in alphas:
    if a >= np.sqrt(2):
        continue
    for s in range(ns):
        errs = []
        for N in Ns:
            P = basis_over_F(a, N)
            W = np.sqrt(w)[:, None]
            Q, _ = np.linalg.qr(W * P)
            x = W[:, 0][None, None, None, :] * fF
            resid = x - np.einsum(
                "vk,slzk->slzv", Q, np.einsum("vk,slzv->slzk", Q.conj(), x)
            )
            errs.append(float(np.sqrt((np.abs(resid[s]) ** 2).sum()) / tot[s]))
        print(f"a={a:.2f} s={s} " + " ".join(f"{e:.2e}" for e in errs))
