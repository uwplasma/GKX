# ruff: noqa: E402
"""Q7 (plan 5.1 L4): structured preconditioners for B = A - sigma I on the exact operator.

Measurement only; no GKX or SOLVAX source is changed. One case per fresh process.

Operator pieces (SOLVAX column probing of GKX's own RHS, complex128):

* ``A``   the full linear operator (``_apply_operator`` with the deck's terms);
* ``S``   streaming + hypercollisions with phi removed (``external_phi = -phi(G)``):
          the part GKX's Hermite-line solve inverts exactly (checked below);
* ``Db``  curvature + grad-B drift + end damping, phi removed: the exact omega_d(z)
          with its m+-2 / l+-1 couplings, z-local;
* ``Dc``  every term except streaming and hypercollisions, phi kept: drift, mirror,
          diamagnetic drive, end damping and the phi response at that z, z-local;
* ``zS``  the z-diagonal (l,m) blocks of ``S`` (hypercollision symbol mean).

Right preconditioners for B (``M^-1`` applied first, GMRES on B M^-1):

* ``none``, ``hl``          identity; GKX ``build_shift_invert_preconditioner(..., "hermite-line")``
* ``zjac-b/c``              per-(kx,z) dense block solve of ``Db|Dc + zS - sigma``
* ``adi-b/c[a]``            one Peaceman-Rachford step with sigma split evenly between the
                            two parts: ``2a (D - s1)^-1 (S - s1)^-1`` with ``s1 = sigma/2 - a``
                            (so ``2 s1 + 2a = sigma``); ``a = -sigma/2`` gives ``s1 = sigma``,
                            scale ``-sigma`` (the plain ADI product)
* ``pr2-*``, ``pr3-*``      two / three Peaceman-Rachford double sweeps with the same ``a``
* ``ms-b/c``                multiplicative two-stage: ``y = hl(x)``, ``y += zjac(x - B y)``
* ``ms-c-rev``, ``sym-c``   zjac then hl; zjac, hl, zjac (one / two inner matvecs)
* ``perl-lu``, ``ilu``      exact LU per Laguerre index and SciPy ILU(1e-2, fill 3): ceilings

Streaming stays spectral in every structured candidate; D blocks are batched dense
inverses per (kx, z) (identical iterations to any exact factorization of D).
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

T0 = time.perf_counter()
REPO = Path(__file__).resolve().parents[4]

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import solvax
from solvax import sparse_operator_matrix

import gkx
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.runtime import _runtime_linear_dispatch_deps
from gkx.solvers_linear_implicit import _build_shifted_hermite_preconditioner
from gkx.solvers_linear_krylov_algorithms import (
    _apply_operator,
    _normalize,
    build_shift_invert_preconditioner,
)
from gkx.terms.assembly import assemble_rhs_cached, compute_fields_cached
from gkx.terms.config import TermConfig
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.toml import load_runtime_from_toml

FIELDS = (
    "streaming mirror curvature gradb diamagnetic collisions hypercollisions "
    "hyperdiffusion end_damping apar bpar nonlinear"
).split()
CASES = {  # name: (Nx, Ny, Nz, ntheta, nperiod, Nl, Nm)
    "pilot": (8, 16, 16, 16, 1, 4, 8),
    "r16": (1, 24, 16, 16, 1, 4, 8),
    "r32": (1, 24, 32, 32, 1, 8, 16),
    "r48": (1, 24, 48, 16, 2, 8, 16),
    "r64": (1, 24, 64, 64, 1, 8, 32),
    "r96": (1, 24, 96, 32, 2, 8, 24),
    "prod": (1, 24, 96, 32, 2, 16, 48),
}
PILOT_SIGMA = complex(0.09302951, -0.28199404)
PROBE_SIGMA = complex(0.15, -0.26)
GROWTH_OFFSET = 0.05
CAP = 400
PRODUCTION_N = 73728
ADAPTIVE_SECONDS = 761.0
DECK = "examples/linear/axisymmetric/cyclone.toml"


def log(message: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    print(f"[{time.perf_counter() - T0:8.1f}s rss<={rss:.2f}G] {message}", flush=True)


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True
        ).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "dirty_src": bool(run("status", "--porcelain", "src")),
    }


def with_terms(base: TermConfig, **over: float) -> TermConfig:
    return TermConfig(**{f: float(over.get(f, getattr(base, f))) for f in FIELDS})


def median_seconds(fn, x, reps: int = 15) -> float:
    for _ in range(2):
        np.asarray(fn(x))
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        np.asarray(fn(x))
        samples.append(time.perf_counter() - t)
    return float(np.median(samples))


def to_blocks(x, shape):
    ns, nl, nm, ny, nx, nz = shape
    return jnp.transpose(x.reshape(shape), (0, 3, 4, 5, 1, 2)).reshape(
        ns * ny * nx * nz, nl * nm
    )


def from_blocks(y, shape):
    ns, nl, nm, ny, nx, nz = shape
    return jnp.transpose(y.reshape(ns, ny, nx, nz, nl, nm), (0, 4, 5, 1, 2, 3)).reshape(
        -1
    )


def zblocks_from_matrix(M: sp.spmatrix, shape) -> tuple[np.ndarray, float]:
    """Dense (l,m) blocks of the entries sharing (species, ky, kx, z); off-block norm."""

    ns, nl, nm, ny, nx, nz = shape
    coo = M.tocoo()
    rs, rl, rm, ry, rx, rz = np.unravel_index(coo.row, shape)
    cs, cl, cm, cy, cx, cz = np.unravel_index(coo.col, shape)
    same = (rs == cs) & (ry == cy) & (rx == cx) & (rz == cz)
    blk = np.ravel_multi_index((rs, ry, rx, rz), (ns, ny, nx, nz))
    blocks = np.zeros((ns * ny * nx * nz, nl * nm, nl * nm), dtype=np.complex128)
    np.add.at(
        blocks,
        (blk[same], (rl * nm + rm)[same], (cl * nm + cm)[same]),
        coo.data[same],
    )
    total = float(np.linalg.norm(coo.data))
    off = float(np.linalg.norm(coo.data[~same]))
    return blocks, (off / total if total else 0.0)


def zblocks_by_probing(apply, shape, batch: int) -> np.ndarray:
    """Production-style setup: Nl*Nm probes of a z-local operator (ones along kx, z)."""

    ns, nl, nm, ny, nx, nz = shape
    bs, nb = nl * nm, ns * ny * nx * nz
    f = jax.jit(jax.vmap(apply))
    out = np.empty((nb, bs, bs), dtype=np.complex128)
    for start in range(0, bs, batch):
        cols = np.arange(start, min(bs, start + batch))
        probe = np.zeros((cols.size, *shape), dtype=np.complex128)
        probe[np.arange(cols.size), :, cols // nm, cols % nm] = 1.0
        y = np.asarray(f(jnp.asarray(probe)))
        y = y.transpose(0, 1, 4, 5, 6, 2, 3).reshape(cols.size, nb, bs)
        out[:, :, cols] = np.transpose(y, (1, 2, 0))
    return out


def gmres_counts(Smat, minv, b, restart: int, stop: float) -> dict:
    hist: list[float] = []
    op = spla.LinearOperator(
        Smat.shape, matvec=lambda v: Smat @ minv(v), dtype=np.complex128
    )
    y, _ = spla.gmres(
        op,
        b,
        rtol=stop,
        atol=0.0,
        restart=restart,
        maxiter=max(1, CAP // restart),
        callback=hist.append,
        callback_type="pr_norm",
    )
    h = np.asarray(hist, dtype=float)

    def first(tol):
        idx = np.flatnonzero(h <= tol)
        return int(idx[0]) + 1 if idx.size else None

    true = float(np.linalg.norm(b - Smat @ minv(y)) / np.linalg.norm(b))
    return {
        "its_1e5": first(1e-5),
        "its_1e8": first(1e-8),
        "last": float(h[-1]) if h.size else None,
        "true": true,
        "n_its": int(h.size),
    }


def arnoldi_certification(A, lu, sigma, b, kmax: int = 40) -> dict:
    """Exact-LU shift-invert Arnoldi steps until the target Ritz pair certifies."""

    n = b.size
    V = np.zeros((n, kmax + 1), dtype=np.complex128)
    H = np.zeros((kmax + 1, kmax), dtype=np.complex128)
    V[:, 0] = b / np.linalg.norm(b)
    hits: dict = {}
    history = []
    for j in range(kmax):
        w = lu.solve(V[:, j])
        for _ in range(2):
            h = V[:, : j + 1].conj().T @ w
            w = w - V[:, : j + 1] @ h
            H[: j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(w)
        V[:, j + 1] = w / H[j + 1, j]
        theta, Y = np.linalg.eig(H[: j + 1, : j + 1])
        i = int(np.argmax(np.abs(theta)))
        lam = sigma + 1.0 / theta[i]
        u = V[:, : j + 1] @ Y[:, i]
        Au = A @ u
        res = float(
            np.linalg.norm(Au - lam * u)
            / max(np.linalg.norm(Au), abs(lam) * np.linalg.norm(u))
        )
        history.append(res)
        for tol in (1e-6, 1e-9):
            if res <= tol and str(tol) not in hits:
                hits[str(tol)] = {"steps": j + 1, "lambda": [lam.real, lam.imag]}
        if len(hits) == 2:
            break
    return {"hits": hits, "residual_history": history}


def prepare(case: str, ny_override: int | None):
    Nx, Ny, Nz, ntheta, nperiod, Nl, Nm = CASES[case]
    Ny = ny_override or Ny
    cfg0, _ = load_runtime_from_toml(REPO / DECK)
    cfg = replace(
        cfg0,
        grid=replace(
            cfg0.grid, Nx=Nx, Ny=Ny, Nz=Nz, ntheta=ntheta, nperiod=nperiod, jtwist=1
        ),
        time=replace(cfg0.time, damp_ends_rate=0.1),
    )
    deps = _runtime_linear_dispatch_deps().full_deps
    ctx = _prepare_linear_runtime_context(
        cfg,
        deps=deps,
        ky_target=0.3,
        n_laguerre=Nl,
        n_hermite=Nm,
        solver="krylov",
        fit_signal="auto",
        return_state=False,
        initial_state=None,
        status_callback=None,
    )
    seed = jnp.asarray(np.asarray(ctx.initial_state), dtype=jnp.complex128)
    cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, Nl, Nm)
    terms = linear_terms_to_term_config(ctx.terms)
    dims = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "ntheta": ntheta, "nperiod": nperiod}
    dims |= {"Nl": Nl, "Nm": Nm}
    return ctx, seed, cache, terms, dims


class Case:
    def __init__(self, case: str, ny_override: int | None):
        self.name = case
        self.ctx, self.seed, self.cache, self.T0, self.dims = prepare(case, ny_override)
        self.params = self.ctx.params
        self.shape = tuple(self.seed.shape)
        self.n = int(self.seed.size)
        ns, nl, nm, ny, nx, nz = self.shape
        assert ns == 1, "single species only"
        self.bs, self.nb = nl * nm, ns * ny * nx * nz
        self.ky = float(np.asarray(self.ctx.grid.ky).reshape(-1)[0])
        T = self.T0
        self.T_S = with_terms(
            T,
            mirror=0,
            curvature=0,
            gradb=0,
            diamagnetic=0,
            collisions=0,
            hyperdiffusion=0,
            end_damping=0,
        )
        self.T_LINE = with_terms(T, curvature=0, gradb=0, collisions=0)
        self.T_DB = with_terms(
            T,
            streaming=0,
            hypercollisions=0,
            mirror=0,
            diamagnetic=0,
            collisions=0,
            hyperdiffusion=0,
        )
        self.T_DC = with_terms(T, streaming=0, hypercollisions=0)
        covered = (
            np.unique(
                np.concatenate(
                    [np.asarray(m).ravel() for m in self.cache.linked_indices]
                )
            )
            if self.cache.linked_indices
            else np.arange(nx * ny)
        )
        cov_kx = np.isin(np.arange(nx), sorted({int(i) // ny for i in covered}))
        self.cov = np.broadcast_to(
            cov_kx[None, None, None, None, :, None], self.shape
        ).reshape(-1)
        self.b = np.asarray(_normalize(self.seed).reshape(-1))
        self._mv = jax.jit(
            lambda x: _apply_operator(
                x.reshape(self.shape), self.cache, self.params, T
            ).reshape(-1)
        )

    def operator(self, terms: TermConfig, keep_phi: bool):
        cache, params, shape = self.cache, self.params, self.shape

        if keep_phi:

            def apply(G):
                return _apply_operator(G.reshape(shape), cache, params, terms)

        else:

            def apply(G):
                G = G.reshape(shape)
                phi = compute_fields_cached(
                    G, cache, params, terms=terms, use_custom_vjp=False
                ).phi
                dG, _ = assemble_rhs_cached(
                    G,
                    cache,
                    params,
                    terms=terms,
                    use_custom_vjp=False,
                    external_phi=-phi,
                )
                return dG

        return apply

    def assemble(self, terms: TermConfig, keep_phi: bool) -> sp.csr_matrix:
        M = sparse_operator_matrix(
            self.operator(terms, keep_phi), self.seed, batch_size=64
        )
        return M.astype(np.complex128).tocsr()

    # --- preconditioner primitives (JAX, flat -> flat) ---
    def bmv(self, sigma):
        mv = self._mv
        return lambda x: mv(x) - sigma * x

    def hl(self, sigma):
        _, op = build_shift_invert_preconditioner(
            self.seed,
            self.cache,
            self.params,
            self.T0,
            jnp.asarray(sigma, jnp.complex128),
            "hermite-line",
        )
        return jax.jit(op)

    def sinv(self, s1, mean_drift: bool = False):
        """(S - s1)^-1, or (S + mean drift - s1)^-1, on linked rows; -x/s1 elsewhere."""

        terms = with_terms(self.T0, collisions=0) if mean_drift else self.T_LINE
        line = _build_shifted_hermite_preconditioner(
            self.seed,
            self.cache,
            self.params,
            terms,
            jnp.asarray(s1, jnp.complex128),
        )
        cov = jnp.asarray(self.cov)
        return jax.jit(lambda x: jnp.where(cov, line(x), -x / s1))

    def dinv(self, blocks: np.ndarray, shift) -> tuple:
        """Batched dense inverse of (blocks - shift I): (inverse, seconds, bytes)."""

        eye = np.eye(self.bs, dtype=np.complex128)
        t = time.perf_counter()
        inv = jnp.asarray(np.linalg.inv(blocks - shift * eye))
        return inv, time.perf_counter() - t, int(inv.size * 16)

    def compose(self, builder, invs: tuple = ()):
        """jit a composite preconditioner with block inverses as graph arguments."""

        shape = self.shape

        def traced(inverses, x):
            solves = tuple(
                (
                    lambda z, inv=inv: from_blocks(
                        jnp.einsum("bij,bj->bi", inv, to_blocks(z, shape)), shape
                    )
                )
                for inv in inverses
            )
            return builder(*solves)(x)

        jitted = jax.jit(traced)
        return lambda x: jitted(invs, x)


def adi(sinv, dinv, alpha):
    def apply(x):
        return 2.0 * alpha * dinv(sinv(x))

    return apply


def adi_reversed(sinv, dinv, alpha):
    def apply(x):
        return 2.0 * alpha * sinv(dinv(x))

    return apply


def peaceman_rachford(sinv, dinv, alpha, sweeps: int):
    """``sweeps`` PR double sweeps from zero for (S~ + D~) y = x; no extra matvecs."""

    def apply(x):
        v = jnp.zeros_like(x)  # (D~ - a) y of the previous sweep
        y = v
        for _ in range(sweeps):
            r1 = x - v
            y_half = sinv(r1)
            r2 = v + 2.0 * alpha * y_half  # x - (S~ - a) y_half
            y = dinv(r2)
            v = r2 - 2.0 * alpha * y
        return y

    return apply


def two_stage(first, second, bmv):
    def apply(x):
        y = first(x)
        return y + second(x - bmv(y))

    return apply


def three_stage(outer, middle, bmv):
    def apply(x):
        y = outer(x)
        y = y + middle(x - bmv(y))
        return y + outer(x - bmv(y))

    return apply


def host(fn):
    return lambda v: np.asarray(fn(jnp.asarray(v, jnp.complex128)))


def run_case(args) -> dict:
    case = Case(args.case, args.ny)
    rec: dict = {
        "case": case.name,
        "dims": case.dims,
        "n": case.n,
        "shape": case.shape,
        "ky": case.ky,
        "dead_fraction": float(1.0 - case.cov.mean()),
        "blocks": [case.nb, case.bs],
    }
    log(
        f"case {case.name} dims={case.dims} shape={case.shape} n={case.n} ky={case.ky:+.3f} "
        f"dead_rows={rec['dead_fraction']:.3f}"
    )
    x_rand = jnp.asarray(
        np.random.default_rng(0).standard_normal(case.n)
        + 1j * np.random.default_rng(1).standard_normal(case.n)
    )
    t_mv = median_seconds(case._mv, x_rand)
    rec["t_matvec_s"] = t_mv
    log(f"matrix-free matvec {t_mv * 1e3:.3f} ms (jit, warm median)")

    # --- assembly ---
    t = time.perf_counter()
    A = case.assemble(case.T0, keep_phi=True)
    rec["assembly_s"] = time.perf_counter() - t
    A_S = case.assemble(case.T_S, keep_phi=False)
    A_DB = case.assemble(case.T_DB, keep_phi=False)
    A_DC = case.assemble(case.T_DC, keep_phi=True)
    A_DC0 = case.assemble(case.T_DC, keep_phi=False)
    rec["nnz"] = {"A": A.nnz, "S": A_S.nnz, "Db": A_DB.nnz, "Dc": A_DC.nnz}
    R = (A - A_S - A_DC).tocsr()
    R.eliminate_zeros()
    fro = lambda M: float(sp.linalg.norm(M))  # noqa: E731
    rec["remainder_rel_fro"] = fro(R) / fro(A)
    t_csr = median_seconds(lambda v: A @ v, np.asarray(x_rand))
    rec["t_csr_matvec_s"] = t_csr
    log(
        f"assembly {rec['assembly_s']:.1f}s nnz={rec['nnz']} |A-S-Dc|/|A|={rec['remainder_rel_fro']:.3e} "
        f"csr matvec {t_csr * 1e3:.3f} ms"
    )

    blocks_db, off_db = zblocks_from_matrix(A_DB, case.shape)
    blocks_dc, off_dc = zblocks_from_matrix(A_DC, case.shape)
    blocks_dc0, _ = zblocks_from_matrix(A_DC0, case.shape)
    blocks_zs, off_s = zblocks_from_matrix(A_S, case.shape)
    rec["off_block_rel"] = {"Db": off_db, "Dc": off_dc, "S": off_s}
    # structure of Dc: l-bandwidth without phi, rank of the phi part per z
    nl, nm = case.shape[1], case.shape[2]
    lidx = np.repeat(np.arange(nl), nm)
    nz_mask = np.abs(blocks_dc0).max(axis=0) > 0
    rows, cols = np.nonzero(nz_mask)
    phi_part = blocks_dc - blocks_dc0
    sv = np.linalg.svd(phi_part, compute_uv=False)
    ranks = (sv > 1e-12 * max(sv.max(), 1e-300)).sum(axis=1)
    rec["dc_structure"] = {
        "l_bandwidth_no_phi": int(np.abs(lidx[rows] - lidx[cols]).max())
        if rows.size
        else 0,
        "phi_part_rank_max": int(ranks.max()),
        "phi_part_rel_fro": float(np.linalg.norm(phi_part) / np.linalg.norm(blocks_dc)),
    }
    t = time.perf_counter()
    probed = zblocks_by_probing(case.operator(case.T_DC, True), case.shape, batch=64)
    t_probe_cold = time.perf_counter() - t
    t = time.perf_counter()
    probed = zblocks_by_probing(case.operator(case.T_DC, True), case.shape, batch=64)
    rec["probe_setup_s"] = {"cold": t_probe_cold, "warm": time.perf_counter() - t}
    rec["probe_vs_extracted_maxabs"] = float(np.abs(probed - blocks_dc).max())
    log(
        f"off-block rel: {rec['off_block_rel']}  Dc structure {rec['dc_structure']}  "
        f"probing Dc blocks {rec['probe_setup_s']} maxdiff {rec['probe_vs_extracted_maxabs']:.2e}"
    )
    del probed, A_DC0, blocks_dc0, phi_part

    # --- shift placement with the exact route ---
    eye_n = sp.eye(case.n, format="csc", dtype=np.complex128)
    if args.case == "pilot":
        sigma = PILOT_SIGMA
    elif args.sigma is not None:
        sigma = complex(args.sigma)
    else:
        t = time.perf_counter()
        lu_p = spla.splu((A - PROBE_SIGMA * eye_n).tocsc())
        vals, vecs = spla.eigs(
            A,
            k=12,
            sigma=PROBE_SIGMA,
            OPinv=spla.LinearOperator(
                (case.n, case.n), matvec=lu_p.solve, dtype=np.complex128
            ),
            tol=1e-12,
            maxiter=20000,
        )
        del lu_p
        res = np.array(
            [
                np.linalg.norm(A @ vecs[:, i] - vals[i] * vecs[:, i])
                / np.linalg.norm(A @ vecs[:, i])
                for i in range(vals.size)
            ]
        )
        good = vals[res < 1e-8]
        lam_growing = good[np.argmax(good.real)]
        sigma = complex(lam_growing.real + GROWTH_OFFSET, lam_growing.imag)
        rec["probe"] = {
            "sigma": [PROBE_SIGMA.real, PROBE_SIGMA.imag],
            "eigs": [[v.real, v.imag, r] for v, r in zip(vals, res)],
            "seconds": time.perf_counter() - t,
        }
        log(f"growing mode {lam_growing:.6f} -> sigma {sigma:.6f}")
    rec["sigma"] = [sigma.real, sigma.imag]
    S = (A - sigma * eye_n).tocsr()
    exact = None
    if not args.no_exact:
        t = time.perf_counter()
        exact = spla.splu(S.tocsc())
        rec["exact_lu"] = {
            "seconds": time.perf_counter() - t,
            "fill": int(exact.L.nnz + exact.U.nnz),
        }
        vals = spla.eigs(
            A,
            k=4,
            sigma=sigma,
            OPinv=spla.LinearOperator(
                (case.n, case.n), matvec=exact.solve, dtype=np.complex128
            ),
            tol=1e-12,
            maxiter=20000,
            return_eigenvectors=False,
        )
        vals = vals[np.argsort(np.abs(vals - sigma))]
        rec["nearest_eigs"] = [[v.real, v.imag] for v in vals]
        rec["separation"] = float(abs(vals[1] - sigma) / abs(vals[0] - sigma))
        t = time.perf_counter()
        rec["arnoldi"] = arnoldi_certification(A, exact, sigma, case.b)
        rec["arnoldi"]["seconds"] = time.perf_counter() - t
        log(
            f"exact LU {rec['exact_lu']} nearest {vals[:3]} separation {rec['separation']:.3f} "
            f"Arnoldi certification {rec['arnoldi']['hits']}"
        )

    # --- check: GKX line solve inverts S - s1 on every row ---
    s_check = sigma
    sinv_check = case.sinv(s_check)
    y = np.asarray(sinv_check(x_rand))
    xr = np.asarray(x_rand)
    rec["line_solve_defect"] = float(
        np.linalg.norm((A_S - s_check * eye_n) @ y - xr) / np.linalg.norm(xr)
    )
    log(f"|(S - sigma) Sinv x - x|/|x| = {rec['line_solve_defect']:.3e}")

    # --- balanced split: move the z-mean drift diagonal into the streaming stage ---
    shape = case.shape
    ns, nl, nm, ny, nx, nz = shape
    A_DRIFT = case.assemble(with_terms(case.T_DB, end_damping=0), keep_phi=False)
    drift_blocks, _ = zblocks_from_matrix(A_DRIFT, shape)
    del A_DRIFT
    diag = np.diagonal(drift_blocks, axis1=1, axis2=2).reshape(ns, ny, nx, nz, -1)
    mean_drift = np.broadcast_to(diag.mean(axis=3, keepdims=True), diag.shape).copy()
    kx_cov = case.cov.reshape(shape)[0, 0, 0, 0, :, 0]
    mean_drift[:, :, ~kx_cov] = 0.0
    mean_drift_flat = (
        mean_drift.reshape(ns, ny, nx, nz, nl, nm).transpose(0, 4, 5, 1, 2, 3).ravel()
    )
    mean_drift = mean_drift.reshape(case.nb, case.bs)
    blocks_cm = blocks_dc.copy()
    diag_idx = np.arange(case.bs)
    blocks_cm[:, diag_idx, diag_idx] -= mean_drift
    del drift_blocks, diag
    y = np.asarray(case.sinv(sigma, mean_drift=True)(x_rand))
    rec["line_mean_drift_defect"] = float(
        np.linalg.norm((A_S + sp.diags(mean_drift_flat) - sigma * eye_n) @ y - xr)
        / np.linalg.norm(xr)
    )
    log(
        f"|(S + mean drift - sigma) HL x - x|/|x| = {rec['line_mean_drift_defect']:.3e}"
    )

    # --- candidates ---
    bmv = case.bmv(sigma)
    inv_b, t_inv_b, mem_b = case.dinv(blocks_db + blocks_zs, sigma)
    inv_c, t_inv_c, mem_c = case.dinv(blocks_dc + blocks_zs, sigma)
    hl = case.hl(sigma)

    def meta(setup, mem, inner, **extra):
        return {"setup_s": setup, "mem_bytes": mem, "inner_mv": inner} | extra

    cands: dict = {
        "none": (case.compose(lambda: lambda x: x), meta(0.0, 0, 0)),
        "hl": (case.compose(lambda: hl), meta(0.0, 0, 0)),
        "zjac-b": (case.compose(lambda d: d, (inv_b,)), meta(t_inv_b, mem_b, 0)),
        "zjac-c": (case.compose(lambda d: d, (inv_c,)), meta(t_inv_c, mem_c, 0)),
        "ms-b": (
            case.compose(lambda d: two_stage(hl, d, bmv), (inv_b,)),
            meta(t_inv_b, mem_b, 1),
        ),
        "ms-c": (
            case.compose(lambda d: two_stage(hl, d, bmv), (inv_c,)),
            meta(t_inv_c, mem_c, 1),
        ),
        "ms-c-rev": (
            case.compose(lambda d: two_stage(d, hl, bmv), (inv_c,)),
            meta(t_inv_c, mem_c, 1),
        ),
        "sym-c": (
            case.compose(lambda d: three_stage(d, hl, bmv), (inv_c,)),
            meta(t_inv_c, mem_c, 2),
        ),
    }
    alphas = {
        "matched": -sigma / 2,
        "2x": -sigma,
        "4x": -2 * sigma,
        "10x": -5 * sigma,
        "re-0.1": -0.1,
        "re-0.3": -0.3,
        "re-1": -1.0,
        "re-3": -3.0,
        "re-10": -10.0,
    }
    adi_rows = {}
    best = {}

    def score(row):
        c = row[2]
        return (c["its_1e5"] is None, c["its_1e5"] or 0, c["last"] or 1.0)

    for label, blocks, with_mean in (
        ("b", blocks_db, False),
        ("c", blocks_dc, False),
        ("cm", blocks_cm, True),
    ):
        scan = []
        for aname, alpha in alphas.items():
            s1 = sigma / 2 - alpha
            sinv = case.sinv(s1, mean_drift=with_mean)
            inv, t_inv, mem = case.dinv(blocks, s1)
            fn = case.compose(lambda d, s=sinv, a=alpha: adi(s, d, a), (inv,))
            counts = gmres_counts(S, host(fn), case.b, restart=CAP, stop=1e-9)
            scan.append((aname, alpha, counts, sinv, inv, t_inv, mem))
            adi_rows[f"adi-{label}[{aname}]"] = counts
            log(
                f"adi-{label}[{aname}] alpha={complex(alpha):.4f} s1={complex(s1):.4f} {counts}"
            )
        aname, alpha, _, sinv, inv, t_inv, mem = min(scan, key=score)
        scan = None
        best[label] = aname
        m = meta(t_inv, mem, 0, alpha=aname)
        for key, build in (
            ("adi", lambda d, s=sinv, a=alpha: adi(s, d, a)),
            ("rev", lambda d, s=sinv, a=alpha: adi_reversed(s, d, a)),
            ("pr2", lambda d, s=sinv, a=alpha: peaceman_rachford(s, d, a, 2)),
            ("pr3", lambda d, s=sinv, a=alpha: peaceman_rachford(s, d, a, 3)),
        ):
            name = {"adi": f"adi-{label}", "rev": f"adi-{label}-rev"}.get(
                key, f"{key}-{label}"
            )
            cands[name] = (case.compose(build, (inv,)), dict(m))
    rec["adi_scan"] = adi_rows
    rec["adi_best_alpha"] = best

    rows: dict = {}
    for name, (fn, meta) in cands.items():
        t_apply = 0.0 if name == "none" else median_seconds(fn, x_rand, reps=7)
        full = gmres_counts(S, host(fn), case.b, restart=CAP, stop=1e-9)
        r20 = gmres_counts(S, host(fn), case.b, restart=20, stop=1e-6)
        row = dict(meta)
        row |= {
            "apply_s": t_apply,
            "apply_over_matvec": t_apply / t_mv,
            "unrestarted": full,
            "gmres20_its_1e5": r20["its_1e5"],
            "gmres20_last": r20["last"],
        }
        rows[name] = row
        log(
            f"{name:12s} its 1e-5/1e-8={full['its_1e5']}/{full['its_1e8']} (last {full['last']:.1e}, "
            f"true {full['true']:.1e}) GMRES(20)={r20['its_1e5']} apply/matvec={t_apply / t_mv:.2f} "
            f"setup={meta['setup_s']:.3f}s mem={meta['mem_bytes'] / 1e6:.2f}MB"
        )

    if not args.skip_ceilings:
        idx = np.arange(case.n).reshape(case.shape)
        Sc = S.tocsc()
        t = time.perf_counter()
        per_l = []
        for k in range(case.shape[1]):
            r = np.take(idx, k, axis=1).reshape(-1)
            per_l.append((r, spla.splu(Sc[r][:, r].tocsc())))
        t_perl = time.perf_counter() - t

        def perl(v):
            out = np.zeros_like(v)
            for r, lu in per_l:
                out[r] = lu.solve(v[r])
            return out

        fill = sum(lu.L.nnz + lu.U.nnz for _, lu in per_l)
        ceilings = [("perl-lu", perl, t_perl, fill)]
        ilu = None
        for drop, ff in ((1e-2, 3), (1e-3, 5), (1e-4, 10)):
            t = time.perf_counter()
            try:
                ilu = spla.spilu(Sc, drop_tol=drop, fill_factor=ff)
            except (
                RuntimeError
            ) as exc:  # SuperLU ILU can report an exactly singular factor
                rec.setdefault("ilu_failures", []).append([drop, ff, str(exc)])
                log(f"ILU(drop={drop:g}, fill={ff}) failed: {exc}")
                continue
            ceilings.append(
                (
                    f"ilu({drop:g},{ff})",
                    ilu.solve,
                    time.perf_counter() - t,
                    ilu.L.nnz + ilu.U.nnz,
                )
            )
            break
        for name, fn, setup, nnz in ceilings:
            t_apply = median_seconds(fn, np.asarray(x_rand), reps=7)
            full = gmres_counts(S, fn, case.b, restart=CAP, stop=1e-9)
            r20 = gmres_counts(S, fn, case.b, restart=20, stop=1e-6)
            rows[name] = {
                "setup_s": setup,
                "mem_bytes": int(nnz * 20),
                "factor_nnz": int(nnz),
                "inner_mv": 0,
                "apply_s": t_apply,
                "apply_over_matvec": t_apply / t_mv,
                "unrestarted": full,
                "gmres20_its_1e5": r20["its_1e5"],
                "gmres20_last": r20["last"],
            }
            log(
                f"{name:12s} its 1e-5/1e-8={full['its_1e5']}/{full['its_1e8']} GMRES(20)={r20['its_1e5']} "
                f"apply/matvec={t_apply / t_mv:.2f} setup={setup:.2f}s factor nnz={nnz}"
            )
        del per_l, ilu
    rec["candidates"] = rows

    if args.gcrot and exact is not None:
        V = [case.b / np.linalg.norm(case.b)]
        for _ in range(11):
            w = exact.solve(V[-1])
            for _ in range(2):
                for v in V:
                    w = w - np.vdot(v, w) * v
            V.append(w / np.linalg.norm(w))
        mv = case.bmv(sigma)
        gc = {}
        for name in (
            "hl",
            "zjac-c",
            "ms-c",
            "adi-b",
            "adi-c",
            "adi-cm",
            "pr2-c",
            "pr2-cm",
            "pr3-cm",
        ):
            fn = cands[name][0]
            recycle, its, conv = None, [], 0
            t = time.perf_counter()
            for v in V:
                sol = solvax.gcrot(
                    mv,
                    jnp.asarray(v),
                    precond=fn,
                    m=20,
                    k=10,
                    rtol=1e-5,
                    max_restarts=20,
                    recycle=recycle,
                    recycle_strategy="harmonic",
                )
                recycle = sol.recycle
                its.append(int(sol.iterations))
                conv += bool(sol.converged)
            gc[name] = {
                "its": its,
                "converged": conv,
                "total": sum(its),
                "seconds": time.perf_counter() - t,
            }
            log(
                f"gcrot(20,10,harmonic) {name:8s} its={its} converged={conv}/12 total={sum(its)}"
            )
        rec["gcrot"] = gc
    return rec


def run_production_timing(args) -> dict:
    case = Case("prod", args.ny)
    rec: dict = {
        "case": "prod",
        "dims": case.dims,
        "n": case.n,
        "shape": case.shape,
        "ky": case.ky,
    }
    log(
        f"production timing dims={case.dims} n={case.n} ky={case.ky:+.3f} blocks={case.nb}x{case.bs}"
    )
    sigma = complex(0.0930912 + GROWTH_OFFSET, -0.2820327)
    rec["sigma"] = [sigma.real, sigma.imag]
    alpha = complex(args.alpha)
    s1 = sigma / 2 - alpha
    rec["alpha"] = [alpha.real, alpha.imag]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(case.n) + 1j * rng.standard_normal(case.n))
    rec["t_matvec_s"] = t_mv = median_seconds(case._mv, x, reps=9)
    log(f"matvec {t_mv * 1e3:.2f} ms")
    hl = case.hl(sigma)
    s_line = case.sinv(s1)
    s_mean = case.sinv(s1, mean_drift=True)
    rec["t_hl_s"] = median_seconds(hl, x, reps=9)
    rec["t_sinv_s"] = median_seconds(s_line, x, reps=9)
    rec["t_sinv_mean_s"] = median_seconds(s_mean, x, reps=9)
    log(
        f"hl {rec['t_hl_s'] * 1e3:.2f} ms, streaming line {rec['t_sinv_s'] * 1e3:.2f} ms, "
        f"line with mean drift {rec['t_sinv_mean_s'] * 1e3:.2f} ms"
    )
    t = time.perf_counter()
    blocks = zblocks_by_probing(case.operator(case.T_DC, True), case.shape, batch=16)
    rec["probe_setup_s"] = time.perf_counter() - t
    log(f"Dc blocks by {case.bs} probes: {rec['probe_setup_s']:.1f}s")
    inv, t_inv, mem = case.dinv(blocks, s1)
    del blocks
    rec["dense_inverse_s"], rec["dense_inverse_bytes"] = t_inv, mem
    log(f"dense z-block inverse {t_inv:.1f}s {mem / 1e9:.2f} GB")
    bmv = case.bmv(sigma)
    # Apply timings only: every composite reuses the same inverse (identical shapes and
    # flops to its own shifted blocks), so no iteration count is implied here.
    composites = {
        "dinv": lambda d: d,
        "adi": lambda d: adi(s_line, d, alpha),
        "adi-cm": lambda d: adi(s_mean, d, alpha),
        "pr2-cm": lambda d: peaceman_rachford(s_mean, d, alpha, 2),
        "pr3-cm": lambda d: peaceman_rachford(s_mean, d, alpha, 3),
        "ms": lambda d: two_stage(hl, d, bmv),
    }
    rec["t_apply_s"] = {}
    for key, build in composites.items():
        rec["t_apply_s"][key] = median_seconds(case.compose(build, (inv,)), x, reps=7)
        log(f"apply {key}: {rec['t_apply_s'][key] * 1e3:.2f} ms")
    # classical Gram-Schmidt cost per basis vector at this n
    X = np.asarray(
        rng.standard_normal((case.n, 50)) + 1j * rng.standard_normal((case.n, 50))
    )
    w = np.asarray(x)
    t = time.perf_counter()
    for _ in range(5):
        h = X.conj().T @ w
        w = w - X @ h
    rec["t_cgs_per_vector_s"] = (time.perf_counter() - t) / (5 * 50)
    log(f"CGS per vector {rec['t_cgs_per_vector_s'] * 1e3:.3f} ms")
    return rec


TIMING_KEY = {
    "adi-b": "adi",
    "adi-c": "adi",
    "adi-cm": "adi-cm",
    "pr2-c": "pr2-cm",
    "pr2-cm": "pr2-cm",
    "pr3-c": "pr3-cm",
    "pr3-cm": "pr3-cm",
    "ms-c": "ms",
    "ms-b": "ms",
}


def extrapolate(directory: Path, candidates: str) -> None:
    """Power-law fit of iterations over the single-chain ladder -> production estimate."""

    recs = {}
    for path in sorted(directory.glob("*.txt")):
        for line in path.read_text().splitlines():
            if line.startswith("RESULT "):
                r = json.loads(line[len("RESULT ") :])
                recs[r.get("tag") or r["case"]] = r
    prod = recs.get("prod")
    ladder = sorted(
        (r for k, r in recs.items() if k in ("r16", "r32", "r48", "r64", "r96")),
        key=lambda r: r["n"],
    )
    steps = max(
        r["arnoldi"]["hits"].get("1e-09", {}).get("steps", 0)
        for r in ladder
        if "arnoldi" in r
    )
    print(
        "EXTRAPOLATION (estimate, not a measurement) from rungs "
        f"{[(r['case'], r['n']) for r in ladder]}; outer RHS per certified pair = max over "
        f"rungs of exact-LU Arnoldi steps to 1e-9 = {steps}"
    )
    if prod is None:
        print("  no production timing record")
        return
    print(
        f"  production timings (this host, 1 thread): matvec {prod['t_matvec_s'] * 1e3:.1f} ms, "
        f"hl {prod['t_hl_s'] * 1e3:.1f} ms, applies {prod['t_apply_s']}, "
        f"CGS/vector {prod['t_cgs_per_vector_s'] * 1e3:.2f} ms, setup probe "
        f"{prod['probe_setup_s']:.0f} s + inverse {prod['dense_inverse_s']:.0f} s"
    )
    for cand in candidates.split(","):
        for tol_key in ("its_1e5", "its_1e8"):
            pts = [
                (r["n"], r["candidates"][cand]["unrestarted"][tol_key])
                for r in ladder
                if cand in r["candidates"]
                and r["candidates"][cand]["unrestarted"][tol_key]
            ]
            missing = [
                r["case"]
                for r in ladder
                if cand not in r["candidates"]
                or not r["candidates"][cand]["unrestarted"][tol_key]
            ]
            print(f"  {cand} {tol_key}: points {pts} not converged on {missing}")
            if len(pts) < 2 or missing:
                continue
            ln = np.log([p[0] for p in pts])
            li = np.log([p[1] for p in pts])
            slope, icpt = np.polyfit(ln, li, 1)
            last = float(np.diff(li[-2:])[0] / np.diff(ln[-2:])[0])
            t_apply = (
                prod["t_hl_s"] if cand == "hl" else prod["t_apply_s"][TIMING_KEY[cand]]
            )
            setup = (
                0.0 if cand == "hl" else prod["probe_setup_s"] + prod["dense_inverse_s"]
            )
            for label, p, c in (
                ("all-rung fit", slope, icpt),
                ("last-two", last, li[-1] - last * ln[-1]),
            ):
                its = float(np.exp(c + p * np.log(PRODUCTION_N)))
                t_iter = prod["t_matvec_s"] + t_apply
                t_orth = prod["t_cgs_per_vector_s"] * its / 2.0
                total = setup + steps * its * (t_iter + t_orth)
                print(
                    f"    {label}: its ~ n^{p:.2f} -> {its:.0f} at n={PRODUCTION_N}; "
                    f"{steps} RHS x {its:.0f} its x ({t_iter * 1e3:.1f} ms apply+matvec + "
                    f"{t_orth * 1e3:.1f} ms mean orth) + setup {setup:.0f} s = {total:.0f} s "
                    f"(adaptive route, #232 office 12 threads: {ADAPTIVE_SECONDS:.0f} s)"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--sigma", default=None, help="explicit complex shift, skips placement"
    )
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--skip-ceilings", action="store_true")
    parser.add_argument("--gcrot", action="store_true")
    parser.add_argument("--extrapolate", type=Path, default=None)
    parser.add_argument("--candidates", default="hl,adi-cm,pr2-cm,pr3-cm")
    parser.add_argument(
        "--alpha", default="-1", help="PR parameter for production apply timing"
    )
    args = parser.parse_args()
    if args.extrapolate:
        extrapolate(args.extrapolate, args.candidates)
        return
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solvax": solvax.__version__,
        "gkx_file": gkx.__file__,
        "x64": bool(jax.config.jax_enable_x64),
        "devices": str(jax.devices()),
        "threads": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "loadavg": os.getloadavg(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    } | git_state()
    print("ENV " + json.dumps(env), flush=True)
    assert env["x64"], "complex128 required"
    rec = run_production_timing(args) if args.case == "prod" else run_case(args)
    rec["tag"] = args.tag
    rec["wall_s"] = time.perf_counter() - T0
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    print(
        "RESULT "
        + json.dumps(
            rec, default=lambda o: list(o) if isinstance(o, tuple) else str(o)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
