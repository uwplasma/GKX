"""Q6 pilot identity: certified eigenpairs before/after covered-subspace projection.

Usage: python pilot_identity.py <tag> <out_dir>
Runs from the checkout on PYTHONPATH. Writes <out_dir>/<tag>.npz and prints a summary.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import gkx
from gkx.runtime import _runtime_linear_dispatch_deps
from gkx.solvers_linear_krylov import _eigenpair_relative_residual, dominant_eigenpair
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.toml import load_runtime_from_toml

tag, out_dir = sys.argv[1], Path(sys.argv[2])
REPO = Path(gkx.__file__).resolve().parents[2]
print("gkx", gkx.__file__, flush=True)
SIGMA = complex(0.09302951 - 0.28199404j)


def context(nx: int, ny: int):
    cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, Nx=nx, Ny=ny, Nz=16, ntheta=16, nperiod=1, jtwist=1),
        time=replace(cfg.time, damp_ends_rate=0.1),
    )
    deps = _runtime_linear_dispatch_deps().full_deps
    ctx = _prepare_linear_runtime_context(
        cfg,
        deps=deps,
        ky_target=0.3,
        n_laguerre=4,
        n_hermite=8,
        solver="krylov",
        fit_signal="auto",
        return_state=False,
        initial_state=None,
        status_callback=None,
    )
    cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, 4, 8)
    return ctx, cache


results = {}
for label, nx, ny in (("pilot", 8, 16), ("nx1", 1, 16)):
    ctx, cache = context(nx, ny)
    seed = jnp.asarray(np.asarray(ctx.initial_state), dtype=jnp.complex128)
    shape = seed.shape
    Ny, Nx = shape[-3], shape[-2]
    covered = np.zeros(Ny * Nx, dtype=bool)
    for idx in cache.linked_indices:
        covered[np.asarray(idx).reshape(-1)] = True  # flat = ky + Ny * kx
    covered_kx = sorted({int(i) // Ny for i in np.flatnonzero(covered)})
    kx_of = np.broadcast_to(
        np.arange(Nx)[None, None, None, None, :, None], shape
    ).reshape(-1)
    cov = np.isin(kx_of, covered_kx)
    seed_np = np.asarray(seed).reshape(-1)
    print(
        f"[{label}] shape={shape} n={seed.size} covered_kx={covered_kx} "
        f"covered={int(cov.sum())} full_cover={cache.linked_full_cover} "
        f"seed_uncovered_norm2_frac="
        f"{np.linalg.norm(seed_np[~cov]) ** 2 / np.linalg.norm(seed_np) ** 2:.3e}",
        flush=True,
    )
    terms_cfg = linear_terms_to_term_config(ctx.terms)
    routes = [("adaptive", {})]
    if label == "pilot":
        routes.insert(0, ("sparse_shift_invert", {"shift": SIGMA}))
        routes.append(("adaptive_fullseed", {}))
    # Golden-angle start used by gkx.objectives.core: nonzero on every row.
    flat_index = jnp.arange(seed.size, dtype=jnp.float64)
    full_seed = jnp.reshape(
        jnp.exp(1j * (flat_index + 1.0) * jnp.asarray(0.6180339887498948)), shape
    )
    for method_label, extra in routes:
        method = method_label.removesuffix("_fullseed")
        start = full_seed if method_label.endswith("_fullseed") else seed
        messages = []
        t0 = time.perf_counter()
        try:
            eig, vec = dominant_eigenpair(
                start,
                cache,
                ctx.params,
                terms=ctx.terms,
                method=method,
                status_callback=messages.append,
                **extra,
            )
        except Exception as error:  # record the failure mode, keep going
            print(f"[{label}:{method_label}] raised {type(error).__name__}: {error}")
            for m in messages[-3:]:
                print("   status:", m, flush=True)
            continue
        elapsed = time.perf_counter() - t0
        vec_np = np.asarray(vec).reshape(-1)
        res = _eigenpair_relative_residual(eig, vec, cache, ctx.params, terms_cfg)
        unc = np.linalg.norm(vec_np[~cov]) ** 2 / np.linalg.norm(vec_np) ** 2
        eig_c = complex(np.asarray(eig))
        method = method_label
        print(
            f"[{label}:{method}] eig={eig_c.real:+.17e}{eig_c.imag:+.17e}j "
            f"residual={res:.3e} uncovered_weight={unc:.3e} max|uncovered|="
            f"{np.max(np.abs(vec_np[~cov])) if (~cov).any() else 0.0:.3e} "
            f"time={elapsed:.2f}s",
            flush=True,
        )
        for m in messages:
            if "sparse" in m or "adaptive solve" in m:
                print("   status:", m, flush=True)
        results[f"{label}_{method}_eig"] = np.asarray(eig)
        results[f"{label}_{method}_vec"] = np.asarray(vec)
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(out_dir / f"{tag}.npz", **results)
print("wrote", out_dir / f"{tag}.npz", flush=True)
