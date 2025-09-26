from __future__ import annotations

import os
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .io_config import FullConfig
from .model import LinearGK
from .operators import (
    StreamingOperator, LenardBernstein, ElectrostaticDrive,
    StreamingRHS, CollisionsRHS, ElectrostaticDriveRHS, 
)
from .post import save_summary
from .types import Result


def _maybe_enable_x64(flag: str):
    if flag.lower() == "x64":
        os.environ.setdefault("JAX_ENABLE_X64", "true")
        jax.config.update("jax_enable_x64", True)


def build_model(cfg: FullConfig) -> LinearGK:
    stream = StreamingOperator(Nn=cfg.grid.Nn, Nm=cfg.grid.Nm,
                               kpar=cfg.grid.kpar, vth=cfg.grid.vth)
    collide = LenardBernstein(Nn=cfg.grid.Nn, Nm=cfg.grid.Nm, nu=cfg.grid.nu)

    drive = None
    terms = [StreamingRHS(stream), CollisionsRHS(collide)]
    if getattr(cfg.grid, "es_drive", False):
        drive = ElectrostaticDrive(Nn=cfg.grid.Nn, Nm=cfg.grid.Nm,
                                   kpar=cfg.grid.kpar, coef=getattr(cfg.grid, "e_coef", 1.0))
        terms.append(ElectrostaticDriveRHS(drive))

    return LinearGK(stream=stream, collide=collide, drive=drive, terms=tuple(terms))


def _make_solver_and_controller(cfg):
    # Map names to Diffrax solvers
    name = cfg.sim.solver.lower()
    solvers = {
        "tsit5": dfx.Tsit5,
        "dopri5": dfx.Dopri5,
        "dopri8": dfx.Dopri8,
        "kvaerno3": dfx.Kvaerno3,
        "kvaerno5": dfx.Kvaerno5,
        "bosh3": dfx.Bosh3,
    }
    solver = solvers.get(name, dfx.Tsit5)()  # default Tsit5

    if cfg.sim.adaptive:
        controller = dfx.PIDController(rtol=cfg.sim.rtol, atol=cfg.sim.atol)
        dt0 = None
    else:
        controller = dfx.ConstantStepSize()
        # If user didn't provide dt, fall back to uniform spacing
        dt0 = cfg.sim.dt if (cfg.sim.dt and cfg.sim.dt > 0) else (cfg.sim.tmax / max(cfg.sim.nt - 1, 1))
    return solver, controller, dt0


def run_simulation(cfg: FullConfig) -> dict:
    _maybe_enable_x64(cfg.sim.precision)
    model = build_model(cfg)

    # Time grid
    ts = jnp.linspace(0.0, cfg.sim.tmax, cfg.sim.nt)

    # Initial condition (PyTree flattened state)
    y0 = model.init_state_real(cfg.ic.kind, cfg.ic.amp, cfg.ic.phase)

    # Diffrax solver
    solver, stepsize_controller, dt0 = _make_solver_and_controller(cfg)
    saveat = dfx.SaveAt(ts=ts)

    @eqx.filter_jit
    def solve(y0):
        term = dfx.ODETerm(model.rhs_real)
        sol = dfx.diffeqsolve(
            term, solver, t0=ts[0], t1=ts[-1], dt0=dt0,
            y0=y0, saveat=saveat, stepsize_controller=stepsize_controller
        )
        return sol

    sol = solve(y0)

    # Collect outputs
    Y = np.asarray(sol.ys)            # (nt, 2*M) REAL
    nt = Y.shape[0]
    Nn, Nm = cfg.grid.Nn, cfg.grid.Nm
    M = Nn * Nm
    Cr = Y[:, :M].reshape(nt, Nn, Nm)
    Ci = Y[:, M:].reshape(nt, Nn, Nm)
    C  = Cr + 1j * Ci                 # complex (nt, Nn, Nm)

    meta = {
        "sim": cfg.sim.__dict__,
        "grid": cfg.grid.__dict__,
        "ic": cfg.ic.__dict__,
        "git": _git_hash_or_none(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": "SPECTRAX-GK 0.0.1",
    }

    os.makedirs(cfg.paths.outdir, exist_ok=True)
    outfile = os.path.join(cfg.paths.outdir, cfg.paths.outfile)
    np.savez_compressed(
        outfile,
        C=C,
        t=np.asarray(ts),
        kpar=cfg.grid.kpar,
        nu=cfg.grid.nu,
        vth=cfg.grid.vth,
        meta=meta,
    )

    # Save a summary figure alongside the NPZ
    base, _ = os.path.splitext(outfile)
    summary_png = base + "_summary.png"
    res = Result(t=np.asarray(ts), C=C, meta=meta)
    save_summary(res, summary_png)

    return {"outfile": outfile, "summary": summary_png, "meta": meta}


def _git_hash_or_none():
    try:
        import subprocess

        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return h.decode().strip()
    except Exception:
        return None
