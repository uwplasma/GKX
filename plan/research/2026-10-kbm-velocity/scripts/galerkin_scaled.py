#!/usr/bin/env python3
"""Equal-cost test of a velocity-scaled Hermite basis on the exact GKX KBM operator.

    python galerkin_scaled.py <ops/A.npz> <shift_re> <shift_im> [N ...] [--alphas a ...] [--hyper <prefix>]
A is the exact GKX linear operator (sparse_eig.py, KBM_SAVE_A) on (Ns, Nl, M, 1, 1, Nz) with a
large Hermite count M. GKX's Hermite coefficients are orthonormal in the free-energy norm
int f^2/F dv, so the dynamics of an N-function basis {psi_k(v/a) F(v/a)/(a F)} is the Galerkin
projection Q^H A Q, where Q's columns are those functions expanded in the first M psi_m and
orthonormalized. a = 1 reproduces plain truncation at N (the leading N x N block of A). Every
basis has the same N, i.e. the same cost per species, Laguerre moment and z. The growth rate of
the eigenvalue nearest the shift (largest real part among the nearest 6) is printed per (a, N).
The M-term expansion of a scaled function is exact to quadrature error for a < 1 (its Hermite
coefficients decay geometrically as ((1 - a^2)/(1 + a^2))^(k/2)); the printed 'lost' column is
the norm the M-term expansion misses, which must be small for the row to count.

--hyper <prefix> adds GKX's own hypercollision/end-damping difference operator at truncation N,
H_N = A(<prefix>h_N) - A(<prefix>nh_N) (both assembled at Nm = N, KBM_SAVE_A), acting on the N
scaled coefficients exactly as it acts on the N Hermite coefficients; with a = 1 this reproduces
GKX's production truncation-plus-hypercollision operator at Nm = N.
"""

import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

args = sys.argv[1:]
hyper = None
if "--hyper" in args:
    i = args.index("--hyper")
    hyper = args[i + 1]
    args = args[:i] + args[i + 2 :]
alphas = [1.0, 0.9, 0.8, 0.7]
if "--alphas" in args:
    i = args.index("--alphas")
    alphas = [float(a) for a in args[i + 1 :]]
    args = args[:i]
A = sp.load_npz(args[0]).tocsr()
shape = tuple(int(x) for x in np.load(args[0] + ".shape.npy"))
shift = complex(float(args[1]), float(args[2]))
Ns = [int(n) for n in args[3:]] or [4, 6, 8, 12, 16]
ns, nl, M, nz = shape[0], shape[1], shape[2], shape[-1]
v, w = np.polynomial.hermite_e.hermegauss(max(3 * M, 300))
w = w / np.sqrt(2 * np.pi)


def psi(x, n):
    out = np.zeros((x.size, n))
    out[:, 0] = 1.0
    if n > 1:
        out[:, 1] = x
    for m in range(1, n - 1):
        out[:, m + 1] = (x * out[:, m] - np.sqrt(m) * out[:, m - 1]) / np.sqrt(m + 1)
    return out


PM = psi(v, M)
print(f"# {args[0]} shape={shape} shift={shift}")
for a in alphas:
    for N in Ns:
        phi = (
            psi(v / a, N) * (np.exp(-((v / a) ** 2) / 2 + v**2 / 2) / a)[:, None]
        )  # phi_k / F at nodes
        C = PM.T @ (w[:, None] * phi)  # (M, N) coefficients of phi_k in psi_m
        norms = np.sqrt(np.einsum("v,vk->k", w, phi**2))
        lost = float(np.max(np.sqrt(np.maximum(norms**2 - (C**2).sum(0), 0.0)) / norms))
        Cq, _ = np.linalg.qr(C)
        Q = sp.kron(
            sp.identity(ns * nl), sp.kron(sp.csr_matrix(Cq), sp.identity(nz))
        ).tocsr()
        # dense LU: the projected blocks are dense in z and m, so SuperLU fill is worse than dense
        AN = (Q.T @ (A @ Q)).toarray() - shift * np.eye(Q.shape[1])
        if hyper is not None:
            AN += (
                sp.load_npz(f"{hyper}h_4_{N}.npz") - sp.load_npz(f"{hyper}nh_4_{N}.npz")
            ).toarray()
        lu = sla.lu_factor(AN, overwrite_a=True, check_finite=False)
        k = min(6, AN.shape[0] - 2)
        mu = spla.eigs(
            spla.LinearOperator(
                AN.shape, matvec=lambda x: sla.lu_solve(lu, x), dtype=complex
            ),
            k=k,
            which="LM",
            return_eigenvectors=False,
            tol=1e-10,
        )
        lam = shift + 1.0 / mu
        best = lam[np.argmax(lam.real)]
        print(
            f"a={a:.2f} N={N:3d} gamma={best.real:.5f} omega={-best.imag:.5f} lost={lost:.1e}",
            flush=True,
        )
