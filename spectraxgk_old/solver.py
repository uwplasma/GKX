from __future__ import annotations

import os
import time
from typing import Optional, List

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .io_config import FullConfig
from .model import LinearGK
from .operators import StreamingOperator, LenardBernstein, ElectrostaticDrive, NonlinearConvolution, StreamingOperatorKS
from .post import save_summary
from .types import Result, ComplexTerm


def _maybe_enable_x64(flag: str):
    if flag.lower() == "x64":
        os.environ.setdefault("JAX_ENABLE_X64", "true")
        jax.config.update("jax_enable_x64", True)


def build_model(cfg: FullConfig) -> LinearGK:
    Nn, Nm = cfg.grid.Nn, cfg.grid.Nm
    terms: list[ComplexTerm] = []
    drive = None

    if cfg.sim.nonlinear:
        # require klist
        assert cfg.grid.klist is not None and len(cfg.grid.klist) > 0, "Nonlinear run requires [grid].klist."
        ks = jnp.asarray(cfg.grid.klist, dtype=jnp.float64)
        Nk = int(ks.shape[0])
        stream = StreamingOperatorKS(ks=ks, Nn=Nn, Nm=Nm, vth=cfg.grid.vth)
        collide = LenardBernstein(Nn=Nn, Nm=Nm, nu=cfg.grid.nu)
        terms.extend([stream, collide])

        # optional drive: uses single kpar in your current definition; either skip or generalize similarly
        if getattr(cfg.grid, "es_drive", False):
            # (optional) generalize ElectrostaticDrive to KS later; skip for now in nonlinear
            pass

        nl = NonlinearConvolution(ks=ks, Nn=Nn, Nm=Nm, nl_filter=cfg.sim.nl_filter)
        terms.append(nl)
        model = LinearGK(stream=stream, collide=collide, drive=None, terms=tuple(terms), Nk=Nk)
    else:
        # linear single-k path (old behavior)
        stream = StreamingOperator(Nn=Nn, Nm=Nm, kpar=cfg.grid.kpar, vth=cfg.grid.vth)
        collide = LenardBernstein(Nn=Nn, Nm=Nm, nu=cfg.grid.nu)
        terms.extend([stream, collide])
        if getattr(cfg.grid, "es_drive", False):
            drive = ElectrostaticDrive(Nn=Nn, Nm=Nm, kpar=cfg.grid.kpar, coef=getattr(cfg.grid, "e_coef", 1.0))
            terms.append(drive)
        model = LinearGK(stream=stream, collide=collide, drive=drive, terms=tuple(terms), Nk=1)

    return model


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
    Nk = model.Nk  # model carries Nk as a static field
    M  = Nk * Nn * Nm
    Cr = Y[:, :M].reshape(nt, Nk, Nn, Nm)
    Ci = Y[:, M:].reshape(nt, Nk, Nn, Nm)
    C  = Cr + 1j * Ci

    meta = {
        "sim": {**cfg.sim.__dict__, "Nk": model.Nk},
        "grid": cfg.grid.__dict__,
        "ic": cfg.ic.__dict__,
        "git": _git_hash_or_none(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": "SPECTRAX-GK 0.0.1",
    }

    os.makedirs(cfg.paths.outdir, exist_ok=True)
    outfile = os.path.join(cfg.paths.outdir, cfg.paths.outfile)

    if cfg.sim.nonlinear and cfg.grid.klist:
        np.savez_compressed(
            outfile,
            C=C,
            t=np.asarray(ts),
            ks=np.asarray(cfg.grid.klist, dtype=float),  # <— per-mode k list
            nu=cfg.grid.nu,
            vth=cfg.grid.vth,
            meta=np.array(meta, dtype=object),
        )
    else:
        np.savez_compressed(
            outfile,
            C=C,
            t=np.asarray(ts),
            kpar=cfg.grid.kpar,  # <— single-k linear path
            nu=cfg.grid.nu,
            vth=cfg.grid.vth,
            meta=np.array(meta, dtype=object),
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
