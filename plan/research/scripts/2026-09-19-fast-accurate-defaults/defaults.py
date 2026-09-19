"""Q25 harness: measure the defaults a first GKX run actually takes.

One arm per process. Every arm prints one ``ENV {json}`` line and one
``RESULT {json}`` line, so an arm's record carries the interpreter, the
library versions, the working precision and the worktree the module was
imported from. Nothing here changes a default: the arms pass explicit
overrides and the run reports what the default would have produced.

Arms
----
``linear``    one ``run_runtime_linear`` on a shipped deck, with optional
              ``--method`` (eigen route), ``--solver`` and ``--schedule``
              overrides. Reports gamma, omega, the certified original-operator
              residual, the gate that was applied, and the inner-solve
              statistics of a shift-invert build.
``nonlinear`` one ``run_runtime_nonlinear`` on a shipped deck, with optional
              ``--time-method`` and ``--fixed-dt`` overrides. Reports the
              accepted diagnostics and the step count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def environment(argv: list[str]) -> dict:
    import jax
    import numpy
    import scipy

    import gkx

    src = str((REPO / "src").resolve())
    module = str(Path(gkx.__file__).resolve())
    if not module.startswith(src):
        raise SystemExit(f"gkx imported from {module}, not from {src}")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "argv": argv,
        "gkx_file": module,
        "head": head,
        "dirty_src": bool(dirty),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "loadavg_start": os.getloadavg(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _status_dict(result) -> dict | None:
    status = getattr(result, "eigen_status", None)
    if status is None:
        return None
    return status.as_dict()


def run_linear(args, rec: dict) -> None:
    import gkx
    from gkx.runtime import run_runtime_linear
    from gkx.solvers_linear_krylov import KrylovConfig

    cfg = gkx.load(args.deck)
    time_changes: dict = {}
    if args.time_method:
        time_changes["method"] = args.time_method
    if args.dt is not None:
        time_changes["dt"] = args.dt
    if args.t_max is not None:
        time_changes["t_max"] = args.t_max
    if args.fixed_dt is not None:
        time_changes["fixed_dt"] = args.fixed_dt
    if args.time_defaults:
        # What a deck that omits [time] entirely gets: the TimeConfig dataclass
        # defaults, which are not the ExplicitTimeConfig ones the library uses.
        from gkx.config import TimeConfig

        blank = TimeConfig()
        for field in ("method", "dt", "t_max", "fixed_dt", "sample_stride"):
            time_changes.setdefault(field, getattr(blank, field))
        if args.fixed_dt is not None:
            time_changes["fixed_dt"] = args.fixed_dt
        if args.time_method:
            time_changes["method"] = args.time_method
    if time_changes:
        from dataclasses import replace as _dc_replace

        cfg = cfg.replace(time=_dc_replace(cfg.time, **time_changes))
    rec["time"] = {
        "method": cfg.time.method,
        "dt": cfg.time.dt,
        "t_max": cfg.time.t_max,
        "fixed_dt": cfg.time.fixed_dt,
        "cfl": cfg.time.cfl,
        "cfl_fac": cfg.time.cfl_fac,
    }
    # Mirror what ``gkx <deck>.toml`` resolves. ``run_runtime_linear`` takes
    # ky/Nl/Nm/solver as keyword arguments whose own defaults (0.3, None, None,
    # "auto") are *not* the deck's, and ``Nl=None`` falls back to 24/12 --
    # Hermite-starved for the Cyclone deck, which the deck's own comment records
    # as ~4.5% low in gamma. Reading them from the deck is what makes this a
    # measurement of the shipped defaults rather than of the library signature.
    run_kwargs: dict = {
        "ky_target": float(cfg.run.ky),
        "Nl": int(cfg.run.Nl),
        "Nm": int(cfg.run.Nm),
        "solver": str(cfg.run.solver),
    }
    rec["deck_run"] = dict(run_kwargs)
    if args.solver:
        run_kwargs["solver"] = args.solver
    if args.Nl is not None:
        run_kwargs["Nl"] = args.Nl
    if args.Nm is not None:
        run_kwargs["Nm"] = args.Nm
    kwargs: dict = {}
    if args.method:
        kwargs["method"] = args.method
    if args.schedule:
        kwargs["shift_tol_schedule"] = args.schedule
    if args.shift_tol is not None:
        kwargs["shift_tol"] = args.shift_tol
    if args.shift_maxiter is not None:
        kwargs["shift_maxiter"] = args.shift_maxiter
    krylov_cfg = KrylovConfig(**kwargs) if kwargs else None
    rec["krylov_cfg"] = kwargs
    if krylov_cfg is not None:
        run_kwargs["krylov_cfg"] = krylov_cfg
    if args.legacy_seed:
        # Reproduce the pre-widening behaviour exactly: the runtime used to hand
        # the solver the complex64 seed it assembles, and jnp.asarray does not
        # promote. Passing that same array through the initial_state API, which
        # preserves a caller's dtype, is the one-line A/B for the widening.
        import numpy as _np

        from gkx.core_grid import build_spectral_grid
        from gkx.workflows.runtime import startup as _startup

        grid = build_spectral_grid(cfg.grid)
        geom = _startup.build_runtime_geometry(cfg)
        ky = _np.asarray(grid.ky)
        ky_index = int(_np.argmin(_np.abs(ky - run_kwargs["ky_target"])))
        seed = _startup._build_initial_condition_impl(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=run_kwargs["Nl"],
            Nm=run_kwargs["Nm"],
            nspecies=len([sp for sp in cfg.species if sp.kinetic]),
            build_runtime_linear_params_fn=_startup.build_runtime_linear_params,
        )
        run_kwargs["initial_state"] = _np.asarray(seed, dtype=_np.complex64)
        rec["legacy_seed_dtype"] = "complex64"
    t = time.perf_counter()
    result = run_runtime_linear(cfg, **run_kwargs)
    rec["run_s"] = time.perf_counter() - t
    rec["gamma"] = float(result.gamma)
    rec["omega"] = float(result.omega)
    rec["eigen_status"] = _status_dict(result)


def run_nonlinear(args, rec: dict) -> None:
    import gkx
    from gkx.runtime import run_runtime_nonlinear

    cfg = gkx.load(args.deck)
    changes: dict = {}
    if args.time_method:
        changes["method"] = args.time_method
    if args.fixed_dt is not None:
        changes["fixed_dt"] = args.fixed_dt
    if changes:
        cfg = cfg.replace(time=cfg.time.replace(**changes))
    rec["time_overrides"] = changes
    rec["time"] = {
        "method": cfg.time.method,
        "dt": cfg.time.dt,
        "fixed_dt": cfg.time.fixed_dt,
        "t_max": cfg.time.t_max,
        "cfl": cfg.time.cfl,
        "cfl_fac": cfg.time.cfl_fac,
    }
    t = time.perf_counter()
    result = run_runtime_nonlinear(cfg)
    rec["run_s"] = time.perf_counter() - t
    for name in ("heat_flux", "steps", "t_final", "Q_mean", "Q_sem"):
        value = getattr(result, name, None)
        if value is not None:
            try:
                rec[name] = float(value)
            except (TypeError, ValueError):
                rec[name] = str(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("linear", "nonlinear"))
    p.add_argument("deck")
    p.add_argument("--label", default="")
    p.add_argument("--method", default=None, help="eigen route override")
    p.add_argument("--solver", default=None)
    p.add_argument("--schedule", default=None)
    p.add_argument("--legacy-seed", action="store_true")
    p.add_argument("--time-defaults", action="store_true")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--t-max", type=float, default=None)
    p.add_argument("--Nl", type=int, default=None)
    p.add_argument("--Nm", type=int, default=None)
    p.add_argument("--shift-tol", type=float, default=None)
    p.add_argument("--shift-maxiter", type=int, default=None)
    p.add_argument("--time-method", default=None)
    p.add_argument("--fixed-dt", type=lambda s: s.lower() == "true", default=None)
    args = p.parse_args()

    env = environment(sys.argv[1:])
    print("ENV " + json.dumps(env), flush=True)
    rec: dict = {"label": args.label, "arm": args.arm, "deck": args.deck, "env": env}
    if args.arm == "linear":
        run_linear(args, rec)
    else:
        run_nonlinear(args, rec)
    rec["wall_s"] = time.perf_counter() - T_PROCESS
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
