"""The pr3-cm z-block solve before and after SOLVAX's block-Thomas.

Loads ``solvers_linear_precond_pr3.py`` as it was at the base commit (``BASE``,
the parent of this change) next to the working tree's, against the same GKX,
and measures on ``_pr3_setup``-shaped fixtures (the unit tests' operator, all
terms ``pr3-cm`` splits switched on, linked boundaries):

``full``   the whole ``pr3-cm`` apply (three Peaceman-Rachford sweeps, Hermite
           line solve included): agreement, factor bytes, and interleaved
           A/B/A/B apply time.
``zblock`` the z-block solve alone, four ways on the same operator:
           ``shipped``      the base commit's unrolled dinv/f/q recurrence;
           ``shipped-scan`` that same recurrence as two ``lax.scan``s, which
                            isolates what a scan costs at these block counts;
           ``solvax-scan``  ``solvax.block_thomas_solve_ops`` on the new factors;
           ``new``          the working tree's unrolled substitution on them.

Run from the repository root, single-threaded, float64:

    PYTHONPATH=$PWD/src:$PWD JAX_ENABLE_X64=true GKX_X64=1 JAX_PLATFORMS=cpu \\
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \\
      OMP_NUM_THREADS=1 python plan/research/scripts/2026-09-20-solvax-block-thomas/ab_apply.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = "eeb3481c6"  # parent of the change; origin/main when it was made
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import solvax  # noqa: E402

import gkx.solvers_linear_precond_pr3 as new  # noqa: E402
from gkx.config import CycloneBaseCase, GridConfig  # noqa: E402
from gkx.core_grid import build_spectral_grid  # noqa: E402
from gkx.geometry import SAlphaGeometry  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.linear.params import (  # noqa: E402
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def load_base_module():
    source = _git("show", f"{BASE}:src/gkx/solvers_linear_precond_pr3.py")
    handle = tempfile.NamedTemporaryFile("w", suffix="_pr3_base.py", delete=False)
    handle.write(source)
    handle.close()
    spec = importlib.util.spec_from_file_location("pr3_base", handle.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr3_base"] = module
    spec.loader.exec_module(module)
    return module


old = load_base_module()


def setup(*, Nl, Nm, Nz, Ny=4):
    grid_cfg = GridConfig(
        Nx=1, Ny=Ny, Nz=Nz, Lx=6.0, Ly=6.0, boundary="linked", y0=20.0, jtwist=1
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=0.125,
    )
    cache = build_linear_cache(grid, geom, params, Nl=Nl, Nm=Nm)
    shape = (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    rng = np.random.default_rng(0)
    v0 = jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dtype=jnp.complex128,
    )
    terms = LinearTerms(
        streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0,
        collisions=1.0, hypercollisions=1.0, end_damping=1.0, apar=0.0, bpar=0.0,
    )  # fmt: skip
    return cache, params, v0, linear_terms_to_term_config(terms)


def interleaved(routes: dict, arg, rounds: int, warm: int = 5) -> dict:
    times: dict = {k: [] for k in routes}
    for i in range(rounds):
        # rotate the order every round so no arm always runs first
        order = list(routes)
        order = order[i % len(order) :] + order[: i % len(order)]
        for k in order:
            t0 = time.perf_counter()
            jax.block_until_ready(routes[k](arg))
            times[k].append(time.perf_counter() - t0)
    return {k: sorted(v[warm:]) for k, v in times.items()}


def report(times: dict, base: str) -> None:
    ref = statistics.median(times[base])
    for k, ts in times.items():
        med = statistics.median(ts)
        print(
            f"    {k:13s} median {med * 1e3:8.3f} ms  "
            f"[{ts[0] * 1e3:7.3f}, {ts[-1] * 1e3:7.3f}]  x{med / ref:5.3f}"
        )


def shipped_as_scan(dinv, f, q, nl, nm):
    def solve(x):
        rows = jnp.swapaxes(x.reshape(x.shape[0], nl, nm), 0, 1)

        def fwd(prev, k):
            d, ff, r = k
            y = jnp.einsum("bij,bj->bi", d, r) - jnp.einsum("bij,bj->bi", ff, prev)
            return y, y

        zero = jnp.zeros((x.shape[0], nm), x.dtype)
        _, forward = jax.lax.scan(fwd, zero, (dinv, f, rows))

        def back(nxt, k):
            qq, y = k
            v = y - jnp.einsum("bij,bj->bi", qq, nxt)
            return v, v

        _, out = jax.lax.scan(back, zero, (q, forward), reverse=True)
        return jnp.swapaxes(out, 0, 1).reshape(x.shape[0], nl * nm)

    return solve


def run(nl, nm, nz, rounds):
    cache, params, v0, term_cfg = setup(Nl=nl, Nm=nm, Nz=nz)
    sigma = jnp.asarray(0.05 - 0.2j, dtype=v0.dtype)
    print(f"\n=== (Nl, Nm, Nz) = ({nl}, {nm}, {nz}), Nl*Nm = {nl * nm} ===")

    fo, mo = old.build_pr3_factors(v0, cache, params, term_cfg, sigma)
    fn, mn = new.build_pr3_factors(v0, cache, params, term_cfg, sigma)
    assert mo["block_solve"] == mn["block_solve"] == "block-thomas", (mo, mn)
    assert mo["alpha"] == mn["alpha"] and mo["s1"] == mn["s1"]
    print(f"  coupling half-width {mn['structure']['coupling_halfwidth']:.0f}")

    # full pr3-cm apply
    ao = jax.jit(old.build_pr3_apply(v0, cache, params, term_cfg, fo))
    an = jax.jit(new.build_pr3_apply(v0, cache, params, term_cfg, fn))
    probe = v0.reshape(-1)
    xo, xn = np.asarray(ao(probe)), np.asarray(an(probe))
    print(
        f"  full apply agreement      {np.linalg.norm(xn - xo) / np.linalg.norm(xo):.3e}"
    )
    print(
        f"  factor bytes              {mo['factor_bytes']:,} -> {mn['factor_bytes']:,}"
        f"  ({mo['factor_bytes'] / mn['factor_bytes']:.2f}x smaller)"
    )
    print("  full apply, interleaved:")
    report(interleaved({"base": ao, "new": an}, probe, rounds), "base")

    # z-block solve alone, same operator, four implementations
    ob, nb = fo.blocks, fn.blocks
    schur = nb.schur
    nblocks = int(schur.blocks.shape[0])
    rng = np.random.default_rng(1)
    rhs = jnp.asarray(
        rng.standard_normal((nblocks, nl * nm))
        + 1j * rng.standard_normal((nblocks, nl * nm))
    )

    def solvax_scan(x):
        return jax.vmap(solvax.block_thomas_solve_ops)(
            schur, x.reshape(nblocks, nl, nm)
        ).reshape(nblocks, nl * nm)

    routes = {
        "shipped": jax.jit(old._thomas_solve(ob.dinv, ob.f, ob.q, nl, nm)),
        "shipped-scan": jax.jit(shipped_as_scan(ob.dinv, ob.f, ob.q, nl, nm)),
        "solvax-scan": jax.jit(solvax_scan),
        "new": jax.jit(new._block_thomas_substitute(schur, nl, nm)),
    }
    out = {k: np.asarray(fn_(rhs)) for k, fn_ in routes.items()}
    ref = out["shipped"]
    for k in ("shipped-scan", "solvax-scan", "new"):
        print(
            f"  z-block {k:13s} vs shipped {np.linalg.norm(out[k] - ref) / np.linalg.norm(ref):.3e}"
        )
    print(
        f"  z-block new vs solvax-scan bitwise: {np.array_equal(out['new'], out['solvax-scan'])}"
    )
    print("  z-block solve, interleaved:")
    report(interleaved(routes, rhs, rounds), "shipped")


if __name__ == "__main__":
    print(
        "env",
        {
            "base": BASE,
            "head": _git("rev-parse", "--short", "HEAD").strip(),
            "dirty_src": bool(_git("status", "--porcelain", "src").strip()),
            "jax": jax.__version__,
            "solvax": solvax.__version__,
            "x64": bool(jax.config.read("jax_enable_x64")),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "load_start": os.getloadavg(),
        },
    )
    for dims in ((6, 6, 8), (8, 12, 24), (8, 24, 24), (16, 48, 24)):
        run(*dims, rounds=45)
    print("load_end", os.getloadavg())
