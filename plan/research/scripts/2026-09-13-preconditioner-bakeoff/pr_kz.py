# ruff: noqa: E402
"""Q7 follow-up: per-kz Peaceman-Rachford parameters for pr3-cm from symbol bounds.

Split (as ``pr3-cm`` in ``bakeoff.py``): S' = streaming + hypercollisions + z-mean drift
(diagonal in kz, Hermite-tridiagonal per kz), D' = z-local drift + mirror + local phi +
end damping minus the z-mean drift (dense (l,m) block per (kx,z)); B = S' + D' - sigma.

Two-parameter Peaceman-Rachford sweep (no extra matvec), with Lambda = F^-1 diag(a(kz)) F
commuting with S' and a scalar b for the z-local step::

    (S' - sigma/2 + Lambda) y_half = x - (D' - sigma/2 - Lambda) y
    (D' - sigma/2 + b)      y      = x - (S' - sigma/2 - b) y_half

With a(kz) == b this is ``bakeoff.peaceman_rachford``. Symbol bounds (from GKX's own
line-solve data and the probed blocks):

* s(kz) = |w_stream kpar_scale vth kz| x_max(ladder) + max_m hyper_kz(m) |kz|
* d     = spectral radius of the D' blocks (batched power iteration)
* s1    = s at the smallest nonzero |kz|

Rules (all parameters negative real):

* ``scalar-best``  a = b = the scanned best scalar from bakeoff.py (``--alpha-best``)
* ``scalar-sym``   a = b = -sqrt(s1 d)
* ``kz-geo``       a(kz) = -sqrt(max(s(kz), s1) d), b = -sqrt(s1 d)
* ``kz-geo-best``  a(kz) as kz-geo, b = the scanned best scalar
* ``kz-lin``       a(kz) = -max(s(kz), sqrt(s1 d)), b = -sqrt(s1 d)
* ``kz-geo-half``  0.5 x kz-geo (both parameters)
* ``kz-sqrt-best`` a(kz) = best * sqrt(max(s(kz), s1) / s1), b = best (grows from the optimum)
* ``kz-inv-best``  a(kz) = best * s1 / max(s(kz), s1), b = best (shrinks from the optimum)

Per rule: unrestarted GMRES iterations to 1e-5/1e-8 and GMRES(20) on the assembled
operator, apply time / matvec, and SOLVAX gcrot(m=20, k=10, "harmonic") totals over 12
exact-LU shift-invert Arnoldi RHSs at rtol 1e-5 (``--gcrot``).
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bakeoff as bk
import jax
import jax.numpy as jnp
import numpy as np
import prod_solve as ps
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import solvax

from gkx.solvers_linear_implicit import (
    _build_implicit_preconditioner_data,
    _prepare_implicit_state,
    _solve_tridiagonal_last_axis,
)

RULES = (
    "scalar-best",
    "scalar-sym",
    "kz-geo",
    "kz-geo-best",
    "kz-lin",
    "kz-geo-half",
    "kz-sqrt-best",
    "kz-inv-best",
)


class KzLine:
    """(S' - s1(kz))^-1 on single-link linked rows; -F^-1 (F x / s1(kz)) on dead rows."""

    def __init__(self, case):
        c = case
        kz = np.asarray(c.cache.kz)
        for idx, kz_link in zip(c.cache.linked_indices, c.cache.linked_kz):
            assert np.asarray(idx).shape[1] == 1, "single-link chains only"
            assert np.allclose(np.asarray(kz_link).ravel(), kz), "kz_link must equal kz"
        state = _prepare_implicit_state(c.seed, 1.0, bk.with_terms(c.T0, collisions=0))
        data = _build_implicit_preconditioner_data(c.cache, c.params, state)
        kzj = jnp.asarray(kz, dtype=state.real_dtype)
        kpar = jnp.asarray(c.params.kpar_scale, dtype=state.real_dtype)
        coeff = (
            (data.w_stream * kpar)
            * data.vth[:, None, None, None, None]
            * (data.imag * kzj)[None, None, None, None, :]
        )[..., None]
        self.dl = coeff * data.sqrt_m_line
        self.du = (coeff * data.sqrt_p_line).at[..., -1].set(0.0)
        hyper = data.hyper_kz * jnp.abs(kzj)
        diagonal = jnp.reciprocal(data.precond_full) - hyper
        d = jnp.moveaxis(jnp.mean(diagonal, axis=-1, keepdims=True) + hyper, 2, -1)
        self.D0 = d - 1.0  # dt = 1: the line solve inverts 1 + D0 + L
        self.cov = jnp.asarray(c.cov.reshape(c.shape))
        self.shape = c.shape
        self.kz = kz
        # symbol bounds
        sm = np.asarray(data.sqrt_m_line)
        sp_ = np.asarray(data.sqrt_p_line).copy()
        sp_[-1] = 0.0
        ladder = np.diag(sm[1:], -1) + np.diag(sp_[:-1], 1)
        self.x_max = float(np.abs(np.linalg.eigvals(ladder)).max())
        self.coef = float(
            abs(
                float(np.asarray(data.w_stream))
                * float(np.asarray(c.params.kpar_scale))
                * float(np.asarray(data.vth)[0])
            )
        )
        self.hyper_max = float(np.asarray(data.hyper_kz).max())
        self.s_kz = self.coef * np.abs(kz) * self.x_max + self.hyper_max * np.abs(kz)
        self.s1 = float(np.sort(np.unique(self.s_kz))[1])

    def solve(self, x_flat, s1_kz):
        x = x_flat.reshape(self.shape)
        xf = jnp.fft.fft(x, axis=-1)
        xh = jnp.moveaxis(xf, 2, -1)
        diag = self.D0 + s1_kz[:, None]
        full = xh.shape
        yh = -_solve_tridiagonal_last_axis(
            jnp.broadcast_to(self.dl, full),
            jnp.broadcast_to(diag, full),
            jnp.broadcast_to(self.du, full),
            xh,
        )
        y_cov = jnp.fft.ifft(jnp.moveaxis(yh, -1, 2), axis=-1)
        y_dead = jnp.fft.ifft(-xf / s1_kz, axis=-1)
        return jnp.where(self.cov, y_cov, y_dead).reshape(-1)

    def lam(self, y_flat, a_kz):
        y = y_flat.reshape(self.shape)
        return jnp.fft.ifft(jnp.fft.fft(y, axis=-1) * a_kz, axis=-1).reshape(-1)


def pr_two(line, sigma, a_kz, b, dsolve, sweeps: int):
    """Two-parameter PR sweeps from zero (see module docstring)."""

    a_kz = jnp.asarray(a_kz, dtype=jnp.complex128)
    s1_kz = sigma / 2 - a_kz

    def apply(x):
        w = jnp.zeros_like(x)  # (D~ - Lambda) y
        y = w
        for _ in range(sweeps):
            r1 = x - w
            y_half = line.solve(r1, s1_kz)
            r2 = x - (r1 - line.lam(y_half, a_kz) - b * y_half)
            y = dsolve(r2)
            w = r2 - b * y - line.lam(y, a_kz)
        return y

    return apply


def rule_parameters(rule, line, d_rho, alpha_best):
    s_kz, s1 = line.s_kz, line.s1
    geo = -np.sqrt(np.maximum(s_kz, s1) * d_rho)
    low = -np.sqrt(s1 * d_rho)
    if rule == "scalar-best":
        return np.full_like(s_kz, alpha_best), alpha_best
    if rule == "scalar-sym":
        return np.full_like(s_kz, low), low
    if rule == "kz-geo":
        return geo, low
    if rule == "kz-geo-best":
        return geo, alpha_best
    if rule == "kz-lin":
        return -np.maximum(s_kz, -low), low
    if rule == "kz-geo-half":
        return 0.5 * geo, 0.5 * low
    if rule == "kz-sqrt-best":  # anchored at the scalar optimum, grows as sqrt(s/s1)
        return alpha_best * np.sqrt(np.maximum(s_kz, s1) / s1), alpha_best
    if rule == "kz-inv-best":  # anchored at the scalar optimum, shrinks as s1/s
        return alpha_best * s1 / np.maximum(s_kz, s1), alpha_best
    if rule == "kz-inv-sym":  # symbol bounds only: -sqrt(s1 d) shrinking as s1/s
        return low * s1 / np.maximum(s_kz, s1), low
    raise ValueError(rule)


def spectral_radius(blocks, iters: int = 60) -> float:
    v = np.random.default_rng(0).standard_normal(blocks.shape[:2]) + 0j
    for _ in range(iters):
        v = np.einsum("bij,bj->bi", blocks, v)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return float(np.abs(np.einsum("bi,bij,bj->b", v.conj(), blocks, v)).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("pilot", "r32", "r64", "r96"), required=True)
    parser.add_argument("--sigma", type=complex, default=None)
    parser.add_argument("--alpha-best", type=float, required=True)
    parser.add_argument("--rules", default=",".join(RULES))
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--gcrot", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    t_start = time.perf_counter()
    env = {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "solvax": solvax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "loadavg": os.getloadavg(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bakeoff_sha256": hashlib.sha256(
            (HERE / "bakeoff.py").read_bytes()
        ).hexdigest(),
    } | bk.git_state()
    print("ENV " + json.dumps(env), flush=True)
    assert env["x64"], "complex128 required"

    case = bk.Case(args.case, None)
    sigma = bk.PILOT_SIGMA if args.case == "pilot" else args.sigma
    assert sigma is not None, "--sigma required off the pilot"
    rec: dict = {
        "case": args.case,
        "tag": args.tag,
        "n": case.n,
        "shape": case.shape,
        "ky": case.ky,
        "sigma": [sigma.real, sigma.imag],
        "alpha_best": args.alpha_best,
        "sweeps": args.sweeps,
    }
    bk.log(f"case {args.case} n={case.n} ky={case.ky:+.3f} sigma={sigma:.6f}")
    eye = sp.eye(case.n, format="csr", dtype=np.complex128)
    A = case.assemble(case.T0, keep_phi=True)
    S = (A - sigma * eye).tocsr()
    mean = ps.z_mean_drift(case)
    blocks = bk.zblocks_by_probing(case.operator(case.T_DC, True), case.shape, batch=16)
    idx = np.arange(case.bs)
    blocks[:, idx, idx] -= mean
    line = KzLine(case)
    d_rho = spectral_radius(blocks)
    rec["symbols"] = {
        "coef": line.coef,
        "x_max": line.x_max,
        "hyper_max": line.hyper_max,
        "s_max": float(line.s_kz.max()),
        "s1": line.s1,
        "d_rho": d_rho,
        "sqrt_s1_d": float(np.sqrt(line.s1 * d_rho)),
        "sqrt_smax_d": float(np.sqrt(line.s_kz.max() * d_rho)),
    }
    bk.log(f"symbols {rec['symbols']}")

    # exactness of the per-kz line solve against GKX's own shifted line solve
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(case.n) + 1j * rng.standard_normal(case.n))
    s_test = sigma / 2 + 1.0
    ref = np.asarray(case.sinv(s_test, mean_drift=True)(x))
    mine = np.asarray(
        jax.jit(lambda v: line.solve(v, jnp.full(line.kz.size, s_test)))(x)
    )
    rec["kz_line_vs_gkx_rel"] = float(np.linalg.norm(mine - ref) / np.linalg.norm(ref))
    bk.log(
        f"per-kz line solve vs GKX line solve (constant shift): {rec['kz_line_vs_gkx_rel']:.2e}"
    )
    assert rec["kz_line_vs_gkx_rel"] < 1e-10

    t_mv = bk.median_seconds(case._mv, x, reps=9)
    rec["t_matvec_s"] = t_mv
    exact = spla.splu(S.tocsc()) if args.gcrot else None
    V = []
    if exact is not None:
        V = [case.b / np.linalg.norm(case.b)]
        for _ in range(11):
            w = exact.solve(V[-1])
            for _ in range(2):
                for v in V:
                    w = w - np.vdot(v, w) * v
            V.append(w / np.linalg.norm(w))
        del exact
    bmv = case.bmv(sigma)
    rows: dict = {}
    names = ["hl"] + args.rules.split(",")
    for name in names:
        if name == "hl":
            fn = case.hl(sigma)
            meta = {}
        else:
            a_kz, b = rule_parameters(name, line, d_rho, args.alpha_best)
            inv, t_inv, mem = case.dinv(blocks, sigma / 2 - b)
            fn = case.compose(
                lambda d, a=a_kz, bb=b: pr_two(line, sigma, a, bb, d, args.sweeps),
                (inv,),
            )
            meta = {
                "a_min": float(a_kz.max()),
                "a_max_abs": float(np.abs(a_kz).max()),
                "b": float(b),
                "inverse_s": t_inv,
                "inverse_bytes": mem,
            }
        t_apply = bk.median_seconds(fn, x, reps=5)
        full = bk.gmres_counts(S, bk.host(fn), case.b, restart=bk.CAP, stop=1e-9)
        r20 = bk.gmres_counts(S, bk.host(fn), case.b, restart=20, stop=1e-6)
        row = meta | {
            "apply_over_matvec": t_apply / t_mv,
            "unrestarted": full,
            "gmres20_its_1e5": r20["its_1e5"],
        }
        if V:
            recycle, its, conv = None, [], 0
            t = time.perf_counter()
            for v in V:
                sol = solvax.gcrot(
                    bmv,
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
            row["gcrot"] = {
                "its": its,
                "total": sum(its),
                "converged": conv,
                "seconds": time.perf_counter() - t,
            }
        rows[name] = row
        bk.log(
            f"{name:12s} its 1e-5/1e-8={full['its_1e5']}/{full['its_1e8']} (last {full['last']:.1e}) "
            f"GMRES(20)={r20['its_1e5']} apply/mv={t_apply / t_mv:.1f} "
            f"gcrot={row.get('gcrot', {}).get('total')} ({row.get('gcrot', {}).get('converged')}/12) {meta}"
        )
    rec["candidates"] = rows
    rec["wall_s"] = time.perf_counter() - t_start
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
