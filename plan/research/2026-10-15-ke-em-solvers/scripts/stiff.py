#!/usr/bin/env python3
"""Per-term stiffness spectra and linear IMEX stability of GKX chains.

  python stiff.py linear <deck> <ky> <Nl> <Nm> [key=value ...]
  python stiff.py box    <deck> <Nl> <Nm> [key=value ...] [chains=<spec>]
  python stiff.py arnoldi <deck> <ky> <Nl> <Nm> [key=value ...]

`arnoldi`: production resolution, no dense matrices: the largest |lambda| of
each term (A minus A without it) and of S0 per species block by 40 Arnoldi
steps (as gkx.operators.nonlinear.policies.measured_streaming_frequency).

`linear`: the single-ky linked chain of a linear deck. `box`: chains of a
nonlinear deck's linked box (ky >= 0 rows), captured from the run's own
startup. For every chain the operator is materialized densely (unit probes)
and split by term weight (the RHS is linear in each weight): A (all linear
terms), S0 = GKX's imex split (streaming + dissipation + fields), mirror,
drifts (curvature + grad-B) and, in `linear` mode, every other term.

Printed (one RESULT json per chain): eigenvalue extremes per term and per
species block; for explicit RK3/RK4 on A and for IMEX tableaux with S0,
S0+mirror, S0+mirror+drifts implicit: the largest stable dt on a geometric
ladder (spectral radius of the one-step map, excess growth over the exact
gamma_max below 0.02 + 0.05 |gamma_max|) and the dominant eigenvalue error.
key=value overrides work as in plan/research/2026-10-06-val-ke/scripts/gkx_eig.py.
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.linear.params import linear_terms_to_term_config  # noqa: E402
from gkx.terms.assembly import assemble_rhs_cached  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

# ---------------------------------------------------------------- tableaux
_G3 = 0.4358665215
_B1, _B2 = -1.5 * _G3**2 + 4.0 * _G3 - 0.25, 1.5 * _G3**2 - 5.0 * _G3 + 1.25
_G2 = 1.0 - 2.0**-0.5
_D2 = 1.0 - 1.0 / (2.0 * _G2)


def _ark324():  # Kennedy & Carpenter 2003, ARK3(2)4L[2]SA
    g = F(1767732205903, 4055673282236)
    b = [
        F(1471266399579, 7840856788654),
        F(-4482444167858, 7529755066697),
        F(11266239266428, 11593286722821),
        g,
    ]
    bh = [
        F(2756255671327, 12835298489170),
        F(-10771552573575, 22201958757719),
        F(9247589265047, 10645013368117),
        F(2193209047091, 5459859503100),
    ]
    ae = [
        [0, 0, 0, 0],
        [F(1767732205903, 2027836641118), 0, 0, 0],
        [F(5535828885825, 10492691773637), F(788022342437, 10882634858940), 0, 0],
        [
            F(6485989280629, 16251701735622),
            F(-4246266847089, 9704473918619),
            F(10755448449292, 10357097424841),
            0,
        ],
    ]
    ai = [
        [0, 0, 0, 0],
        [g, g, 0, 0],
        [F(2746238789719, 10658868560708), F(-640167445237, 6845629431997), g, 0],
        b,
    ]
    return ae, ai, b, b, bh


def _ark436():  # Kennedy & Carpenter 2003, ARK4(3)6L[2]SA
    g = F(1, 4)
    b = [
        F(82889, 524892),
        0,
        F(15625, 83664),
        F(69875, 102672),
        F(-2260, 8211),
        g,
    ]
    bh = [
        F(4586570599, 29645900160),
        0,
        F(178811875, 945068544),
        F(814220225, 1159782912),
        F(-3700637, 11593932),
        F(61727, 225920),
    ]
    ae = [
        [0] * 6,
        [F(1, 2), 0, 0, 0, 0, 0],
        [F(13861, 62500), F(6889, 62500), 0, 0, 0, 0],
        [
            F(-116923316275, 2393684061468),
            F(-2731218467317, 15368042101831),
            F(9408046702089, 11113171139209),
            0,
            0,
            0,
        ],
        [
            F(-451086348788, 2902428689909),
            F(-2682348792572, 7519795681897),
            F(12662868775082, 11960479115383),
            F(3355817975965, 11060851509271),
            0,
            0,
        ],
        [
            F(647845179188, 3216320057751),
            F(73281519250, 8382639484533),
            F(552539513391, 3454668386233),
            F(3354512671639, 8306763924573),
            F(4040, 17871),
            0,
        ],
    ]
    ai = [
        [0] * 6,
        [g, g, 0, 0, 0, 0],
        [F(8611, 62500), F(-1743, 31250), g, 0, 0, 0],
        [F(5012029, 34652500), F(-654441, 2922500), F(174375, 388108), g, 0, 0],
        [
            F(15267082809, 155376265600),
            F(-71443401, 120774400),
            F(730878875, 902184768),
            F(2285395, 8070912),
            g,
            0,
        ],
        b[:5] + [g],
    ]
    return ae, ai, b, b, bh


TABLEAUX = {
    "ars343": (  # as in gkx.solvers_nonlinear_imex.ARS_TABLEAUX
        [
            [0, 0, 0, 0],
            [_G3, 0, 0, 0],
            [0.3212788860, 0.3966543747, 0, 0],
            [-0.105858296, 0.5529291479, 0.5529291479, 0],
        ],
        [[0, 0, 0, 0], [0, _G3, 0, 0], [0, (1 - _G3) / 2, _G3, 0], [0, _B1, _B2, _G3]],
        [0, _B1, _B2, _G3],
        [0, _B1, _B2, _G3],
        None,
    ),
    "ars232": (
        [[0, 0, 0], [_G2, 0, 0], [_D2, 1 - _D2, 0]],
        [[0, 0, 0], [0, _G2, 0], [0, 1 - _G2, _G2]],
        [0, 1 - _G2, _G2],
        [0, 1 - _G2, _G2],
        None,
    ),
    "ark324": _ark324(),
    "ark436": _ark436(),
}


def _check_order(name, tab, order):
    ae, ai, be, bi, _ = tab
    for a, b in ((ae, be), (ai, bi)):
        a = np.array(a, float)
        b = np.array(b, float)
        c = a.sum(1)
        conds = [b.sum() - 1, b @ c - 0.5]
        if order >= 3:
            conds += [b @ c**2 - 1 / 3, b @ a @ c - 1 / 6]
        assert max(abs(x) for x in conds) < 1e-9, (name, conds)


for _n, _o in (("ars343", 3), ("ars232", 2), ("ark324", 3), ("ark436", 3)):
    _check_order(_n, TABLEAUX[_n], _o)
TABLEAUX = {
    k: tuple(None if x is None else np.array(x, dtype=float) for x in v)
    for k, v in TABLEAUX.items()
}
COST = {  # (RHS-equivalent evaluations, implicit solves) per step, GKX-style
    "rk3": (3, 0),
    "rk4": (4, 0),
    "ars343": (4, 3),
    "ars232": (3, 2),
    "ark324": (4, 3),
    "ark436": (6, 5),
}
DT_LADDER = 1.0e-4 * 2.0 ** (np.arange(0, 35) / 2.0)  # 1e-4 .. 13


# ------------------------------------------------------------- deck setup
def load_cfg(deck, over):
    cfg, _ = load_runtime_from_toml(deck)
    for key, val in over.items():
        sect, _, field = key.rpartition(".")
        sect = sect or ("grid" if hasattr(cfg.grid, field) else "physics")
        cur = getattr(getattr(cfg, sect), field)
        if isinstance(cur, bool):
            new = val.lower() in ("1", "true")
        elif cur is None or isinstance(cur, (int, float)):
            new = float(val) if cur is None else type(cur)(float(val))
        else:
            new = val
        cfg = replace(cfg, **{sect: replace(getattr(cfg, sect), **{field: new})})
    if "nperiod" in over or "ntheta" in over:
        g = cfg.grid
        cfg = replace(cfg, grid=replace(g, Nz=g.ntheta * (2 * g.nperiod - 1)))
    return cfg


def linear_setup(cfg, ky, nl, nm):
    from gkx.workflows.linear import _prepare_linear_runtime_context

    ctx = _prepare_linear_runtime_context(
        cfg,
        ky_target=ky,
        n_laguerre=nl,
        n_hermite=nm,
        solver="krylov",
        fit_signal="auto",
        return_state=False,
        initial_state=None,
        status_callback=None,
    )
    cache = build_linear_cache(ctx.grid, ctx.geom, ctx.params, nl, nm)
    terms = linear_terms_to_term_config(ctx.terms, nonlinear=0.0)
    shape = tuple(ctx.initial_state.shape)
    idx = np.arange(int(np.prod(shape)))
    return cache, ctx.params, terms, shape, [("ky%.3f" % float(ctx.grid.ky[0]), idx)]


class _Captured(Exception):
    pass


def box_setup(cfg, nl, nm, spec, maxn=8192):
    """Capture (params, cache) at the nonlinear CFL startup, then find chains."""
    import gkx.operators.nonlinear.policies as pol
    from gkx.solvers_nonlinear_imex import _chains
    from gkx.workflows.nonlinear import run_runtime_nonlinear
    from gkx.workflows.runtime.startup import build_runtime_term_config

    got = {}

    def grab(params, cache, *, nl, nm):
        got.update(params=params, cache=cache)
        raise _Captured

    pol.measured_streaming_frequency = grab
    try:
        run_runtime_nonlinear(cfg, Nl=nl, Nm=nm, steps=1)
    except _Captured:
        pass
    params, cache = got["params"], got["cache"]
    terms = replace(build_runtime_term_config(cfg), nonlinear=0.0)
    ky, kx = np.asarray(cache.ky), np.asarray(cache.kx)
    shape = (len(np.atleast_1d(params.vth)), nl, nm, ky.size, kx.size, cache.kz.size)
    modes = np.broadcast_to((ky >= 0)[:, None], (ky.size, kx.size))
    keep = jnp.asarray((ky >= 0)[:, None, None])
    op = jax.jit(
        lambda G: (
            keep
            * assemble_rhs_cached(keep * G, cache, params, terms=terms)[0].reshape(
                shape
            )
        )
    )
    chains = _chains(op, shape, modes, jnp.complex128)
    full = np.arange(int(np.prod(shape))).reshape(shape)
    rows = []
    for c in chains:
        k = np.array([m[0] for m in c])
        x = np.array([m[1] for m in c])
        # dof order (s, l, m, link, z) as gkx.solvers_nonlinear_imex._gather
        rows.append((float(ky[k[0]]), float(np.abs(kx[x]).max()), len(c), k, x))
    print("CHAINS", json.dumps([(r[0], r[1], r[2]) for r in rows]), flush=True)
    per_link = int(np.prod(shape)) // (ky.size * kx.size)
    pick = select_chains([r for r in rows if r[2] * per_link <= maxn], spec)
    out = []
    for kyv, kxm, L, k, x in pick:
        idx = np.moveaxis(full[:, :, :, k, x, :], 3, 0)  # (L, s, l, m, z)
        idx = np.moveaxis(idx, 0, 3).reshape(-1)  # (s, l, m, L, z)
        out.append((f"ky{kyv:.3f}_kxmax{kxm:.3f}_L{L}", idx))
    return cache, params, terms, shape, out


def select_chains(rows, spec):
    """spec 'auto': ky = 0, the two smallest nonzero, the middle and the two
    largest rows; per row the shortest and the longest chain. 'ky=<v>': every
    chain at that ky."""
    if spec.startswith("ky="):
        v = float(spec[3:])
        return [r for r in rows if abs(r[0] - v) < 1e-6]
    kys = sorted({r[0] for r in rows})
    use = {kys[i] for i in (0, 1, 2, len(kys) // 2, -2, -1)}
    pick = []
    for kv in sorted(use):
        rs = sorted([r for r in rows if r[0] == kv], key=lambda r: (r[2], r[1]))
        pick += [rs[0]] if len(rs) == 1 else [rs[0], rs[-1]]
    return pick


# --------------------------------------------------------- dense operators
def dense(cache, params, terms, shape, idx, batch=64):
    n = idx.size
    flat = int(np.prod(shape))
    jidx = jnp.asarray(idx)

    @jax.jit
    def cols(js):
        def one(j):
            v = jnp.zeros(flat, jnp.complex128).at[jidx[j]].set(1.0).reshape(shape)
            return assemble_rhs_cached(v, cache, params, terms=terms)[0].reshape(-1)[
                idx
            ]

        return jax.vmap(one)(js)

    out = np.empty((n, n), complex)
    for s in range(0, n, batch):
        js = np.arange(s, min(s + batch, n))
        js_pad = np.pad(js, (0, batch - js.size), mode="edge")
        out[:, js] = np.asarray(cols(jnp.asarray(js_pad)))[: js.size].T
    return out


def spec_stats(lam):
    lam = np.asarray(lam)
    a = lam[np.argmax(np.abs(lam))]
    return dict(
        max_abs=float(np.abs(lam).max()),
        at=[float(a.real), float(a.imag)],
        max_imag=float(np.abs(lam.imag).max()),
        min_real=float(lam.real.min()),
        max_real=float(lam.real.max()),
    )


# ---------------------------------------------------------- step analysis
def _thr(gmax):
    return 0.02 + 0.05 * abs(gmax)


def explicit_dt(lam, order, lam_dom):
    gmax = float(lam.real.max())
    best = None
    for dt in DT_LADDER:
        z = dt * lam
        p = sum(z**k / np.prod(np.arange(1, k + 1)) for k in range(order + 1))
        if np.log(np.abs(p).max()) / dt - gmax > _thr(gmax):
            break
        best = float(dt)
    return best


def imex_map(tab, dt, S, E):
    ae, ai, be, bi, _ = tab
    n = S.shape[0]
    gam = max(ai[i, i] for i in range(len(be)))
    lu = sla.lu_factor(np.eye(n) - gam * dt * S)

    def apply(y):
        kE, kI = [], []
        for i in range(len(be)):
            r = y + dt * sum(ae[i, j] * kE[j] + ai[i, j] * kI[j] for j in range(i))
            Y = sla.lu_solve(lu, r) if ai[i, i] else r
            kE.append(E @ Y)
            kI.append(S @ Y)
        return y + dt * sum(be[j] * kE[j] + bi[j] * kI[j] for j in range(len(be)))

    return apply


def radius(apply, n):
    if n <= 3072:
        mu = np.linalg.eigvals(apply(np.eye(n, dtype=complex)))
        return mu[np.argmax(np.abs(mu))]
    lo = spla.LinearOperator((n, n), matvec=apply, dtype=complex)
    try:
        mu = spla.eigs(
            lo, k=4, which="LM", tol=1e-7, maxiter=2000, return_eigenvectors=False
        )
    except spla.ArpackNoConvergence as err:
        mu = err.eigenvalues
        if mu.size == 0:
            return np.nan
    return mu[np.argmax(np.abs(mu))]


def imex_dt(tab, S, E, gmax, lam_dom):
    """Bisection on DT_LADDER (stability assumed monotone in dt)."""

    def probe(dt):
        mu = radius(imex_map(tab, dt, S, E), S.shape[0])
        lam = np.log(complex(mu)) / dt if np.isfinite(mu) else complex(np.nan)
        ok = np.isfinite(lam) and lam.real - gmax <= _thr(gmax)
        return ok, lam

    lo, hi, lam_lo = -1, len(DT_LADDER), None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, lam = probe(DT_LADDER[mid])
        if ok:
            lo, lam_lo = mid, lam
        else:
            hi = mid
    if lo < 0:
        return None, None
    # dominant growth error at the largest stable dt (meaningful when the
    # dominant mode grows and so sets the spectral radius)
    return float(DT_LADDER[lo]), [
        float(lam_lo.real - lam_dom.real),
        float(lam_lo.imag - lam_dom.imag),
    ]


def arnoldi_max(op, n, k=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    Q = [v / np.linalg.norm(v)]
    H = np.zeros((k + 1, k), complex)
    for j in range(k):
        w = op(Q[j])
        for _ in range(2):
            for i, qi in enumerate(Q):
                c = np.vdot(qi, w)
                H[i, j] += c
                w = w - c * qi
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j].real < 1e-12 * np.abs(H[: j + 1, j]).max():
            k = j + 1
            break
        Q.append(w / H[j + 1, j])
    return spec_stats(np.linalg.eigvals(H[:k, :k]))


def arnoldi_terms(cache, params, terms, shape):
    n = int(np.prod(shape))
    ns = shape[0]

    def make(t, mask=None):
        f = jax.jit(
            lambda v: assemble_rhs_cached(v.reshape(shape), cache, params, terms=t)[
                0
            ].reshape(-1)
        )
        if mask is None:
            return lambda v: np.asarray(f(jnp.asarray(v)))
        return lambda v: mask * np.asarray(f(jnp.asarray(mask * v)))

    off = dict(mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=0.0, collisions=0.0)
    A = make(terms)
    out = {"A": arnoldi_max(A, n), "S0": arnoldi_max(make(replace(terms, **off)), n)}
    for name, kw in (
        ("streaming", dict(streaming=0.0)),
        ("mirror", dict(mirror=0.0)),
        ("drifts", dict(curvature=0.0, gradb=0.0)),
        ("diamagnetic", dict(diamagnetic=0.0)),
        ("hypercollisions", dict(hypercollisions=0.0)),
        ("end_damping", dict(end_damping=0.0)),
    ):
        B = make(replace(terms, **kw))
        out[name] = arnoldi_max(lambda v, B=B: A(v) - B(v), n)
    for s in range(ns):
        m = np.zeros(shape)
        m[s] = 1.0
        out[f"S0_species{s}"] = arnoldi_max(
            make(replace(terms, **off), m.reshape(-1)), n
        )
        mirror_only = replace(
            terms,
            **{
                **off,
                "streaming": 0.0,
                "hypercollisions": 0.0,
                "end_damping": 0.0,
                "hyperdiffusion": 0.0,
                "mirror": 1.0,
            },
        )
        out[f"mirror_species{s}"] = arnoldi_max(make(mirror_only, m.reshape(-1)), n)
    return out


# -------------------------------------------------------------------- main
def main():
    mode, deck = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    if mode in ("linear", "arnoldi"):
        ky, nl, nm, args = float(args[0]), int(args[1]), int(args[2]), args[3:]
    else:
        ky, nl, nm, args = None, int(args[0]), int(args[1]), args[2:]
    over = dict(a.split("=", 1) for a in args)
    spec = over.pop("chains", "auto")
    maxn = int(over.pop("maxn", 8192))
    schemes = over.pop("schemes", "ars343,ars232,ark324,ark436").split(",")
    variants = over.pop("variants", "S0,S0+mirror,S0+mirror+drifts").split(",")
    cfg = load_cfg(deck, over)
    if mode in ("linear", "arnoldi"):
        cache, params, terms, shape, chains = linear_setup(cfg, ky, nl, nm)
    else:
        cache, params, terms, shape, chains = box_setup(cfg, nl, nm, spec, maxn)
    sha = (
        os.environ.get("GKX_SHA")
        or subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    host = f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}"
    off = dict(mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=0.0, collisions=0.0)
    named = {
        "A": terms,
        "S0": replace(terms, **off),
        "noMirror": replace(terms, mirror=0.0),
        "noDrifts": replace(terms, curvature=0.0, gradb=0.0),
    }
    if mode == "linear":
        for t in (
            "streaming",
            "diamagnetic",
            "collisions",
            "hypercollisions",
            "end_damping",
        ):
            named["no_" + t] = replace(terms, **{t: 0.0})
    ns = shape[0]
    if mode == "arnoldi":
        t0 = time.perf_counter()
        res = arnoldi_terms(cache, params, terms, shape)
        res.update(
            deck=Path(deck).name,
            mode=mode,
            ky=ky,
            Nl=nl,
            Nm=nm,
            over=over,
            sha=sha,
            host=host,
            wall_s=round(time.perf_counter() - t0, 1),
            shape=list(shape),
        )
        print("RESULT", json.dumps(res), flush=True)
        return
    for label, idx in chains:
        t0 = time.perf_counter()
        M = {k: dense(cache, params, v, shape, idx) for k, v in named.items()}
        A = M["A"]
        T = {"mirror": A - M["noMirror"], "drifts": A - M["noDrifts"]}
        T.update({k[3:]: A - v for k, v in M.items() if k.startswith("no_")})
        lam = np.linalg.eigvals(A)
        lam_dom = lam[np.argmax(lam.real)]
        res = dict(
            chain=label,
            n=int(idx.size),
            A=spec_stats(lam),
            dominant=[float(lam_dom.real), float(lam_dom.imag)],
            S0=spec_stats(np.linalg.eigvals(M["S0"])),
            explicit_part=spec_stats(np.linalg.eigvals(A - M["S0"])),
            terms={k: spec_stats(np.linalg.eigvals(v)) for k, v in T.items()},
        )
        blk = idx.size // ns
        res["S0_by_species"] = [
            spec_stats(
                np.linalg.eigvals(
                    M["S0"][s * blk : (s + 1) * blk, s * blk : (s + 1) * blk]
                )
            )
            for s in range(ns)
        ]
        res["dt_max"] = {
            "rk3": explicit_dt(lam, 3, lam_dom),
            "rk4": explicit_dt(lam, 4, lam_dom),
        }
        res["dominant_error"] = {}
        gmax = float(lam.real.max())
        S_of = {
            "S0": M["S0"],
            "S0+mirror": M["S0"] + T["mirror"],
            "S0+mirror+drifts": M["S0"] + T["mirror"] + T["drifts"],
        }
        for v in variants:
            S = S_of[v]
            for sch in schemes:
                dt, err = imex_dt(TABLEAUX[sch], S, A - S, gmax, lam_dom)
                res["dt_max"][f"{sch}[{v}]"] = dt
                res["dominant_error"][f"{sch}[{v}]"] = err
                print("PROGRESS", label, sch, v, dt, flush=True)
        res.update(
            deck=Path(deck).name,
            mode=mode,
            Nl=nl,
            Nm=nm,
            over=over,
            sha=sha,
            host=host,
            wall_s=round(time.perf_counter() - t0, 1),
            shape=list(shape),
        )
        print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
