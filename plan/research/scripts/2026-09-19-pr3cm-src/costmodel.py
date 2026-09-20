# ruff: noqa: E402
"""Q28: the preconditioner apply cost ``c_P``, measured by interleaving.

``measure.py`` times the matvec at the start of an arm and the preconditioner
apply at the end, which on this shared host can be twenty minutes apart. Under
a load that moved between 7 and 70 during this row's runs that is not a ratio
of two comparable numbers, and it showed: four measurements of the same dense
``pr3-cm`` apply at the same rung returned ``c_P`` of 0.96, 6.04, 18.08 and
19.30. A value below 1 is not physical for a batch of dense 192x192 solves
against one matrix-free operator application; it is the matvec having been
timed under a heavier load than the apply.

This script measures the ratio the only way that survives a moving load: one
process, all candidates built up front, then ``--blocks`` rounds that time the
matvec and every apply **back to back** in the same round, rotating the order
each round so no candidate is systematically first. Each round yields one
``c_P`` per candidate from two medians taken seconds apart; the reported value
is the median over rounds and the spread is reported with it.

The iteration counts, certified residuals and factor sizes in ``measure.py``'s
arm files are load-independent and are not re-measured here. Only ``c_P`` is.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-13-preconditioner-bakeoff"))

import jax
import jax.numpy as jnp
import numpy as np

import bakeoff as bk

import gkx
import gkx.solvers_linear_krylov as klv
from gkx.solvers_linear_krylov import _shift_seed
from gkx.solvers_linear_krylov_algorithms import (
    _linked_covered_mode_mask,
    _projected_flat_operator,
    build_shift_invert_preconditioner,
)
from gkx.solvers_linear_precond_pr3 import build_pr3_factors

bk.CASES = dict(bk.CASES) | {
    "d96": (1, 24, 96, 32, 2, 4, 8),
    "d96l8": (1, 24, 96, 32, 2, 8, 24),
}


def _median(fn, x, reps: int) -> float:
    """Median of ``reps`` timed calls, each forced to completion on the host."""

    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(x))
        samples.append(time.perf_counter() - t)
    return float(np.median(samples))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="d96")
    p.add_argument("--shift", default=None)
    p.add_argument("--blocks", type=int, default=9)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--label", default="cost")
    args = p.parse_args()

    case = bk.Case(args.case, None)
    seed = jnp.asarray(case.seed)
    options = dict(method="shift_invert")
    if args.shift:
        options["shift"] = complex(args.shift)
    cfg = klv._normalized_config(dict(klv.KrylovConfig(**options).__dict__))
    sigma, v_init = _shift_seed(
        seed, seed, case.cache, case.params, case.T0, cfg, None, select_overlap=False
    )
    sigma_val = jnp.asarray(sigma, dtype=v_init.dtype)
    covered = _linked_covered_mode_mask(case.cache)
    bk.log(f"{args.case} n={case.n} sigma={complex(np.asarray(sigma_val))}")

    def wrap(raw):
        return jax.jit(_projected_flat_operator(raw, covered, v_init.shape))

    candidates: dict = {}
    for name in ("hermite-line", "field-corrected"):
        _p, raw = build_shift_invert_preconditioner(
            v_init, case.cache, case.params, case.T0, sigma_val, name
        )
        candidates[name] = wrap(raw)
    setup: dict = {}
    for block_solve in ("block-thomas", "dense"):
        t = time.perf_counter()
        factors, meta = build_pr3_factors(
            v_init,
            case.cache,
            case.params,
            case.T0,
            sigma,
            block_solve=block_solve,
        )
        setup[f"pr3-cm:{block_solve}"] = dict(meta, seconds=time.perf_counter() - t)
        _p, raw = build_shift_invert_preconditioner(
            v_init, case.cache, case.params, case.T0, sigma_val, "pr3-cm", factors
        )
        candidates[f"pr3-cm:{block_solve}"] = wrap(raw)

    probe = jnp.asarray(case.b)
    names = ["matvec", *candidates]
    fns = {"matvec": case._mv, **candidates}
    for fn in fns.values():  # warm every graph before any round is timed
        jax.block_until_ready(fn(probe))

    rounds: list[dict] = []
    for block in range(args.blocks):
        order = names[block % len(names) :] + names[: block % len(names)]
        timings = {name: _median(fns[name], probe, args.reps) for name in order}
        t_mv = timings["matvec"]
        row = {
            "block": block,
            "loadavg": os.getloadavg()[0],
            "t_matvec_s": t_mv,
            "c_precond": {
                name: timings[name] / t_mv for name in names if name != "matvec"
            },
        }
        rounds.append(row)
        bk.log(f"block {block} order={order[0]} {row['c_precond']}")

    summary = {}
    for name in names[1:]:
        values = np.array([r["c_precond"][name] for r in rounds])
        summary[name] = {
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "spread_ratio": float(values.max() / max(values.min(), 1e-300)),
        }
    rec = {
        "label": args.label,
        "case": args.case,
        "n": case.n,
        "sigma": [float(np.real(sigma_val)), float(np.imag(sigma_val))],
        "environment": {
            "argv": sys.argv,
            "git": bk.git_state(),
            "gkx": str(Path(gkx.__file__).resolve()),
            "jax": jax.__version__,
            "loadavg": os.getloadavg(),
        },
        "setup": setup,
        "rounds": rounds,
        "c_precond": summary,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "process_seconds": time.perf_counter() - T_PROCESS,
    }
    print("RESULT " + json.dumps(rec, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
