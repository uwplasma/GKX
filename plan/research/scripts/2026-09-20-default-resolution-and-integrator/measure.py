"""Q28 harness: measure the three defaults Q26 (#257) recorded but could not set.

One arm per process, so an arm's record carries its own interpreter, library
versions, working precision, worktree and host load. Every arm prints one
``ENV {json}`` line and one ``RESULT {json}`` line. Nothing here changes a
default: each arm passes an explicit override and the run reports what that
value would produce, so the defaults can be chosen from the table afterwards.

Arms
----
``power``       one ``run_runtime_linear`` on a shipped deck through the raw
                ``method="power"`` eigen route at an explicit ``power_iters``.
                ``certify=False`` is passed so a rung that fails the outer
                residual gate still *reports* its residual instead of raising:
                that residual is the accuracy number for the rung. Cost is
                load-independent and exact -- the route is a ``lax.scan`` of
                length ``power_iters`` over one ``_advance_imex2`` apply, so
                ``power_iters`` is the propagator-apply count.

``time``        one ``run_runtime_linear`` on the explicit time path with the
                ``TimeConfig`` dataclass defaults, optionally overriding
                ``method`` and ``fixed_dt``. Reports gamma/omega and the
                load-independent cost: the step size the CFL controller
                actually resolved, the step count over the horizon, the stage
                count of the scheme and the resulting right-hand-side
                evaluation count.

``resolution``  one certified adaptive eigensolve at an explicit ``(Nl, Nm)``.
                Reports gamma/omega, the certified original-operator residual
                and the gate, plus the velocity-space degrees of freedom
                ``Nl*Nm`` that set the cost of every propagator apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

# Right-hand-side evaluations per accepted step, by explicit scheme. These are
# the stage counts of the shipped steppers in gkx.solvers_time_explicit.
_STAGES = {"euler": 1, "rk2": 2, "rk3": 3, "sspx3": 3, "rk4": 4}


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
    return None if status is None else status.as_dict()


def _deck_run_kwargs(cfg, args) -> dict:
    """Mirror what ``gkx <deck>.toml`` resolves, not the library signature.

    ``run_runtime_linear`` takes ky/Nl/Nm/solver as keyword arguments whose own
    defaults (0.3, None, None, "auto") are not the deck's, and ``Nl=None``
    falls back to 24/12. Reading them from the deck is what makes an arm a
    measurement of the deck rather than of the signature; the Nl/Nm ladder then
    overrides them explicitly so the rung under test is the one recorded.
    """

    kwargs = {
        "ky_target": float(cfg.run.ky),
        "Nl": int(cfg.run.Nl),
        "Nm": int(cfg.run.Nm),
        "solver": str(cfg.run.solver),
    }
    if args.Nl is not None:
        kwargs["Nl"] = args.Nl
    if args.Nm is not None:
        kwargs["Nm"] = args.Nm
    if args.solver:
        kwargs["solver"] = args.solver
    return kwargs


def _load(args):
    import gkx

    return gkx.load(args.deck)


def run_power(args, rec: dict) -> None:
    """Power-iteration ladder through the shipped runtime entry point."""

    from gkx.runtime import run_runtime_linear
    from gkx.solvers_linear_krylov import KrylovConfig

    cfg = _load(args)
    run_kwargs = _deck_run_kwargs(cfg, args)
    rec["run"] = dict(run_kwargs)
    kcfg = KrylovConfig(
        method="power",
        power_iters=int(args.power_iters),
        power_dt=float(args.power_dt),
        # Report the residual of the rung instead of raising on it. The gate is
        # still computed and recorded, so a rung that would have been rejected
        # is visible as certified=False rather than as a traceback.
        certify=bool(args.certify),
    )
    rec["krylov_cfg"] = {
        "method": kcfg.method,
        "power_iters": kcfg.power_iters,
        "power_dt": kcfg.power_dt,
        "certify": kcfg.certify,
    }
    # Exact and load-independent: the route is one lax.scan of length
    # power_iters over a single _advance_imex2 apply.
    rec["propagator_applies"] = int(kcfg.power_iters)
    t = time.perf_counter()
    result = run_runtime_linear(cfg, krylov_cfg=kcfg, **run_kwargs)
    rec["run_s"] = time.perf_counter() - t
    rec["gamma"] = float(result.gamma)
    rec["omega"] = float(result.omega)
    rec["eigen_status"] = _status_dict(result)


def run_resolution(args, rec: dict) -> None:
    """Certified adaptive eigensolve at one (Nl, Nm) rung."""

    from gkx.runtime import run_runtime_linear
    from gkx.solvers_linear_krylov import KrylovConfig

    cfg = _load(args)
    run_kwargs = _deck_run_kwargs(cfg, args)
    rec["run"] = dict(run_kwargs)
    rec["dof_velocity"] = int(run_kwargs["Nl"]) * int(run_kwargs["Nm"])
    kcfg = KrylovConfig(method="adaptive")
    t = time.perf_counter()
    result = run_runtime_linear(cfg, krylov_cfg=kcfg, **run_kwargs)
    rec["run_s"] = time.perf_counter() - t
    rec["gamma"] = float(result.gamma)
    rec["omega"] = float(result.omega)
    rec["eigen_status"] = _status_dict(result)


def run_time(args, rec: dict) -> None:
    """Time-integrator A/B with load-independent step and RHS-evaluation counts."""

    from dataclasses import replace as _dc_replace

    import numpy as _np

    from gkx.config import TimeConfig, resolve_cfl_fac
    from gkx.runtime import run_runtime_linear

    cfg = _load(args)
    # What a deck that omits [time] entirely gets: the TimeConfig dataclass
    # defaults, which are not the ExplicitTimeConfig ones the library uses.
    blank = TimeConfig()
    changes = {
        field: getattr(blank, field)
        for field in ("method", "dt", "t_max", "fixed_dt", "sample_stride")
    }
    if args.time_method:
        changes["method"] = args.time_method
    if args.fixed_dt is not None:
        changes["fixed_dt"] = args.fixed_dt
    if args.dt is not None:
        changes["dt"] = args.dt
    if args.t_max is not None:
        changes["t_max"] = args.t_max
    cfg = cfg.replace(time=_dc_replace(cfg.time, **changes))
    rec["time"] = {
        "method": cfg.time.method,
        "dt": cfg.time.dt,
        "t_max": cfg.time.t_max,
        "fixed_dt": cfg.time.fixed_dt,
        "cfl": cfg.time.cfl,
        "cfl_fac": cfg.time.cfl_fac,
        "cfl_fac_resolved": resolve_cfl_fac(cfg.time.method, cfg.time.cfl_fac),
        "sample_stride": cfg.time.sample_stride,
    }
    run_kwargs = _deck_run_kwargs(cfg, args)
    rec["run"] = dict(run_kwargs)
    t = time.perf_counter()
    result = run_runtime_linear(cfg, **run_kwargs)
    rec["run_s"] = time.perf_counter() - t
    rec["gamma"] = float(result.gamma)
    rec["omega"] = float(result.omega)
    rec["eigen_status"] = _status_dict(result)
    # Load-independent cost. The sample times are strided by sample_stride, so
    # the accepted step is recovered from their spacing and the step count from
    # the horizon. stages is the scheme's RHS evaluations per accepted step.
    ts = getattr(result, "t", None)
    stages = _STAGES.get(str(cfg.time.method).strip().lower())
    rec["stages_per_step"] = stages
    if ts is not None and _np.asarray(ts).size >= 2:
        ts = _np.asarray(ts, dtype=float)
        stride = int(max(cfg.time.sample_stride, 1))
        dt_resolved = float(_np.median(_np.diff(ts))) / stride
        steps = int(round(float(ts[-1]) / dt_resolved)) if dt_resolved > 0 else None
        rec["dt_resolved"] = dt_resolved
        rec["n_samples"] = int(ts.size)
        rec["t_final"] = float(ts[-1])
        rec["steps"] = steps
        if steps is not None and stages is not None:
            rec["rhs_evals"] = steps * stages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("power", "time", "resolution"))
    p.add_argument("deck")
    p.add_argument("--label", default="")
    p.add_argument("--solver", default=None)
    p.add_argument("--Nl", type=int, default=None)
    p.add_argument("--Nm", type=int, default=None)
    p.add_argument("--power-iters", type=int, default=40)
    p.add_argument("--power-dt", type=float, default=0.01)
    p.add_argument("--certify", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--time-method", default=None)
    p.add_argument("--fixed-dt", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--t-max", type=float, default=None)
    args = p.parse_args()

    env = environment(sys.argv[1:])
    print("ENV " + json.dumps(env), flush=True)
    rec: dict = {"label": args.label, "arm": args.arm, "deck": args.deck, "env": env}
    try:
        {"power": run_power, "time": run_time, "resolution": run_resolution}[args.arm](
            args, rec
        )
    except BaseException as exc:  # recorded, not swallowed: re-raised below
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_s"] = time.perf_counter() - T_PROCESS
        rec["loadavg_end"] = os.getloadavg()
        print("RESULT " + json.dumps(rec, default=str), flush=True)
        raise
    rec["wall_s"] = time.perf_counter() - T_PROCESS
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    if not math.isfinite(rec.get("gamma", math.nan)):
        rec["error"] = "non-finite gamma"
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
