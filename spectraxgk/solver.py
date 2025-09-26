from __future__ import annotations
import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
from .io_config import FullConfig
from .model import LinearGK
from .operators import StreamingOperator, LenardBernstein, ElectrostaticDrive
from .types import Result
from .post import save_summary
from typing import Optional


def _maybe_enable_x64(flag: str):
    if flag.lower() == "x64":
        os.environ.setdefault("JAX_ENABLE_X64", "true")
        jax.config.update("jax_enable_x64", True)


def build_model(cfg: FullConfig) -> LinearGK:
    stream = StreamingOperator(Nn=cfg.grid.Nn, Nm=cfg.grid.Nm, kpar=cfg.grid.kpar, vth=cfg.grid.vth)
    collide = LenardBernstein(Nn=cfg.grid.Nn, Nm=cfg.grid.Nm, nu=cfg.grid.nu)
    drive: Optional[ElectrostaticDrive] = None
    if getattr(cfg.grid, "es_drive", False):
        drive = ElectrostaticDrive(Nn=cfg.grid.Nn, Nm=cfg.grid.Nm,
                                   kpar=cfg.grid.kpar, coef=getattr(cfg.grid, "e_coef", 1.0))
    return LinearGK(stream=stream, collide=collide, drive=drive)


def run_simulation(cfg: FullConfig) -> dict:
    _maybe_enable_x64(cfg.sim.precision)
    model = build_model(cfg)

    # Time grid
    ts = jnp.linspace(0.0, cfg.sim.tmax, cfg.sim.nt)

    # Initial condition (PyTree flattened state)
    y0 = model.init_state(cfg.ic.kind, cfg.ic.amp, cfg.ic.phase)

    # Diffrax solver
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=ts)
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-9)

    @eqx.filter_jit
    def solve(y0):
        term = dfx.ODETerm(model.rhs)
        sol = dfx.diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=None,
                              y0=y0, saveat=saveat, stepsize_controller=stepsize_controller)
        return sol

    sol = solve(y0)

    # Collect outputs
    Y = np.asarray(sol.ys)  # (nt, Nn*Nm)
    nt = Y.shape[0]
    Nn, Nm = cfg.grid.Nn, cfg.grid.Nm
    C = Y.reshape(nt, Nn, Nm)

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
    np.savez_compressed(outfile, C=C, t=np.asarray(ts), kpar=cfg.grid.kpar, nu=cfg.grid.nu,
                        vth=cfg.grid.vth, meta=meta)

    # Save a summary figure alongside the NPZ
    base, _ = os.path.splitext(outfile)
    summary_png = base + "_summary.png"
    res = Result(t=np.asarray(ts), C=C, meta=meta)
    save_summary(res, summary_png)

    return {"outfile": outfile, "summary": summary_png, "meta": meta}


def _git_hash_or_none():
    try:
        import subprocess
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return h.decode().strip()
    except Exception:
        return None