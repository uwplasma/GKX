#!/usr/bin/env python3
"""Linear eigenpair by shift-invert with an exact Hermite-banded direct factor.

  python bandeig.py <deck> <ky> <Nl> <Nm> [sigma=1.0] [nev=12] [key=value ...]

The single-ky linked operator couples Hermite m only to m +- 2 (streaming,
mirror, A_par: +-1; drifts: +-2; the field response and the collision
moments stay inside that band). Grouping Hermite pairs makes it block
tridiagonal with blocks of size 2 ns Nl Nz. The blocks are assembled from
3 * 2 ns Nl Nz compressed probes of GKX's own RHS (groups three apart never
share a row), checked against the operator on a random vector, factored by
block LU (A - sigma I), and fed to ARPACK in shift-invert mode. The eigenpair
with the largest growth among `nev` candidates is certified by the residual of
GKX's matrix-free operator. Prints RESULT json with timings.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from stiff import linear_setup, load_cfg  # noqa: E402

from gkx.terms.assembly import assemble_rhs_cached  # noqa: E402


def main():
    deck, ky, nl, nm = (
        sys.argv[1],
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
    )
    over = dict(a.split("=", 1) for a in sys.argv[5:])
    sigma = complex(over.pop("sigma", "1.0"))
    nev = int(over.pop("nev", "12"))
    batch = int(over.pop("batch", "32"))
    t0 = time.perf_counter()
    cfg = load_cfg(deck, over)
    cache, params, terms, shape, _ = linear_setup(cfg, ky, nl, nm)
    ns, nz = shape[0], shape[-1]
    n = int(np.prod(shape))
    op1 = jax.jit(
        lambda v: assemble_rhs_cached(v.reshape(shape), cache, params, terms=terms)[
            0
        ].reshape(-1)
    )
    opb = jax.jit(
        jax.vmap(
            lambda v: assemble_rhs_cached(v.reshape(shape), cache, params, terms=terms)[
                0
            ].reshape(-1)
        )
    )
    q = 2  # Hermite pair per block -> block tridiagonal
    nb = -(-nm // q)
    full = np.arange(n).reshape(ns, nl, nm, nz)
    groups = [
        np.moveaxis(full[:, :, q * B : q * B + q, :], 2, 0).reshape(-1)
        for B in range(nb)
    ]
    bs = [g.size for g in groups]
    t_setup = time.perf_counter() - t0

    # ---- compressed probing: column c of every group B = r (mod 3) at once
    t1 = time.perf_counter()
    D = [np.zeros((bs[B], bs[B]), complex) for B in range(nb)]
    Lo = [None] + [np.zeros((bs[B], bs[B - 1]), complex) for B in range(1, nb)]
    Up = [np.zeros((bs[B], bs[B + 1]), complex) for B in range(nb - 1)] + [None]
    probes = 0
    for r in range(3):
        Bs = list(range(r, nb, 3))
        cmax = max(bs[B] for B in Bs) if Bs else 0
        for c0 in range(0, cmax, batch):
            cs = np.arange(c0, min(c0 + batch, cmax))
            V = np.zeros((cs.size, n), complex)
            for B in Bs:
                ok = cs < bs[B]
                V[np.nonzero(ok)[0], groups[B][cs[ok]]] = 1.0
            Y = np.asarray(opb(jnp.asarray(V)))
            probes += cs.size
            for B in Bs:
                ok = np.nonzero(cs < bs[B])[0]
                D[B][:, cs[ok]] = Y[ok][:, groups[B]].T
                if B > 0:
                    Up[B - 1][:, cs[ok]] = Y[ok][:, groups[B - 1]].T
                if B + 1 < nb:
                    Lo[B + 1][:, cs[ok]] = Y[ok][:, groups[B + 1]].T
    t_probe = time.perf_counter() - t1

    def band_mv(x):
        y = np.zeros(n, complex)
        for B in range(nb):
            acc = D[B] @ x[groups[B]]
            if B > 0:
                acc += Lo[B] @ x[groups[B - 1]]
            if B + 1 < nb:
                acc += Up[B] @ x[groups[B + 1]]
            y[groups[B]] = acc
        return y

    rng = np.random.default_rng(0)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    ref = np.asarray(op1(jnp.asarray(x)))
    band_err = float(np.linalg.norm(band_mv(x) - ref) / np.linalg.norm(ref))

    # ---- block LU of A - sigma I
    t2 = time.perf_counter()
    lus, Ls = [], [None]
    for B in range(nb):
        Dk = D[B] - sigma * np.eye(bs[B])
        if B > 0:
            # L_B = Lo_B Dk-1^{-1}  ->  solve Dk-1^T L^T = Lo^T
            LB = sla.lu_solve(lus[B - 1], Lo[B].T, trans=1).T
            Ls.append(LB)
            Dk = Dk - LB @ Up[B - 1]
        lus.append(sla.lu_factor(Dk, overwrite_a=True, check_finite=False))
    t_factor = time.perf_counter() - t2
    mem_gb = 16 * sum(b * b for b in bs) * 3 / 1e9

    def solve(rhs):
        z = [None] * nb
        for B in range(nb):
            v = rhs[groups[B]].astype(complex)
            if B > 0:
                v = v - Ls[B] @ z[B - 1]
            z[B] = v
        x = np.zeros(n, complex)
        xs = [None] * nb
        for B in reversed(range(nb)):
            v = z[B]
            if B + 1 < nb:
                v = v - Up[B] @ xs[B + 1]
            xs[B] = sla.lu_solve(lus[B], v, check_finite=False)
            x[groups[B]] = xs[B]
        return x

    xs = solve(x)
    solve_res = float(np.linalg.norm(band_mv(xs) - sigma * xs - x) / np.linalg.norm(x))

    t3 = time.perf_counter()
    nsolve = [0]

    def mv(v):
        nsolve[0] += 1
        return solve(v)

    lo = spla.LinearOperator((n, n), matvec=mv, dtype=complex)
    theta, vecs = spla.eigs(
        lo,
        k=nev,
        which="LM",
        tol=1e-12,
        ncv=max(2 * nev + 1, 40),
        v0=np.ones(n, complex),
    )
    lam = sigma + 1.0 / theta
    res = []
    for i in range(lam.size):
        v = vecs[:, i]
        Av = np.asarray(op1(jnp.asarray(v)))
        res.append(
            float(np.linalg.norm(Av - lam[i] * v) / (abs(lam[i]) * np.linalg.norm(v)))
        )
    res = np.array(res)
    ok = res < 1e-9
    i = (
        int(np.argmax(np.where(ok, lam.real, -np.inf)))
        if ok.any()
        else int(np.argmax(lam.real))
    )
    t_eig = time.perf_counter() - t3
    print(
        "RESULT",
        json.dumps(
            dict(
                deck=Path(deck).name,
                ky=ky,
                Nl=nl,
                Nm=nm,
                n=n,
                block=bs[0],
                nblocks=nb,
                over=over,
                sigma=[sigma.real, sigma.imag],
                gamma=float(lam[i].real),
                omega=float(-lam[i].imag),
                residual=float(res[i]),
                certified=bool(ok[i]),
                candidates=[
                    [float(a.real), float(a.imag), float(r)] for a, r in zip(lam, res)
                ],
                band_err=band_err,
                solve_res=solve_res,
                probes=probes,
                solves=nsolve[0],
                t_setup=round(t_setup, 1),
                t_probe=round(t_probe, 1),
                t_factor=round(t_factor, 1),
                t_eig=round(t_eig, 1),
                wall_s=round(time.perf_counter() - t0, 1),
                factor_gb=round(mem_gb, 2),
                sha=os.environ.get("GKX_SHA", ""),
                host=f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
