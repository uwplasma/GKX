# spectraxgk/_simulation.py
"""
Top-level simulation driver for the slab Laguerre–Hermite GK model.

This version:
  - avoids OOM by default (save="diagnostics")
  - provides rich debug prints
  - supports small "probe" outputs (selected k-mode LH coefficients)
  - saves collision dissipation rate D_coll(t)
"""

from __future__ import annotations

import time as _time
import jax
import jax.numpy as jnp
from functools import partial
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    SubSaveAt,
    PIDController,
    ConstantStepSize,
    TqdmProgressMeter,
    NoProgressMeter,
    Tsit5,
)

from ._initialization import initialize_simulation_parameters
from ._model import rhs_laguerre_hermite_gk, cheap_diagnostics_from_state

__all__ = ["simulation"]


# -----------------------------
# Pretty printing (rich optional)
# -----------------------------
def _get_console():
    try:
        from rich.console import Console
        return Console()
    except Exception:
        return None

_CONSOLE = _get_console()

def _hline():
    if _CONSOLE is not None:
        _CONSOLE.rule()
    else:
        print("-" * 90)

def _p(msg: str):
    if _CONSOLE is not None:
        _CONSOLE.print(msg)
    else:
        print(msg)

def _fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while n >= 1024 and u < len(units) - 1:
        n /= 1024
        u += 1
    return f"{n:.2f} {units[u]}"


def _pack_complex_flat(z: jnp.ndarray) -> jnp.ndarray:
    zf = z.reshape(-1)
    return jnp.concatenate([jnp.real(zf), jnp.imag(zf)], axis=0)


def _unpack_complex_flat(y: jnp.ndarray, complex_shape: tuple[int, ...], Ncomplex: int) -> jnp.ndarray:
    y = y.reshape(-1)  # robust if diffrax returns shape (1,dim) slices later
    re = y[:Ncomplex]
    im = y[Ncomplex: 2 * Ncomplex]
    return (re + 1j * im).reshape(complex_shape)


@partial(jax.jit, static_argnames=("Nh", "Nl", "Ny", "Nx", "Nz", "Ncomplex"))
def vector_field_real(
    t: float,
    y_real: jnp.ndarray,
    params: dict,
    Nh: int,
    Nl: int,
    Ny: int,
    Nx: int,
    Nz: int,
    Ncomplex: int,
) -> jnp.ndarray:
    complex_shape = (Nl, Nh, Ny, Nx, Nz)
    Gk = _unpack_complex_flat(y_real, complex_shape, Ncomplex)
    dGk, _phi = rhs_laguerre_hermite_gk(Gk, params, Nh=Nh, Nl=Nl)
    return _pack_complex_flat(dGk)


def _make_diag_fn(
    Nh: int, Nl: int, Ny: int, Nx: int, Nz: int, Ncomplex: int,
    probe: dict | None,
):
    def _diag_fn(t, y_real, params):
        Gk = _unpack_complex_flat(y_real, (Nl, Nh, Ny, Nx, Nz), Ncomplex)
        d = cheap_diagnostics_from_state(Gk, params)

        if probe is not None:
            ky = int(probe.get("ky", Ny // 2))
            kx = int(probe.get("kx", Nx // 2 + 1))
            kz = int(probe.get("kz", Nz // 2))
            lmax = int(probe.get("lmax", min(Nl, 4)))
            mmax = int(probe.get("mmax", min(Nh, 12)))

            G_probe = Gk[:lmax, :mmax, ky, kx, kz]
            d["probe_G_lm"] = G_probe

        return d
    return _diag_fn


def simulation(
    input_parameters=None,
    Nx=33,
    Ny=33,
    Nz=17,
    Nh=24,
    Nl=8,
    timesteps=200,
    dt=1e-2,
    solver=None,
    adaptive_time_step=True,
    progress=True,
    save: str = "diagnostics",   # "diagnostics" (default), "final", "full"
    save_every: int = 1,
    probe: dict | None = None,
):
    t_wall0 = _time.time()
    _hline()
    _p("[bold cyan]SPECTRAX-GK: Simulation start[/bold cyan]" if _CONSOLE else "SPECTRAX-GK: Simulation start")

    if input_parameters is None:
        input_parameters = {}
    if solver is None:
        solver = Tsit5()

    params = initialize_simulation_parameters(
        input_parameters,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nh=Nh, Nl=Nl,
        timesteps=timesteps,
        dt=dt,
    )

    Ncomplex = int(Nl * Nh * Ny * Nx * Nz)
    bytes_per_state = (2 * Ncomplex) * 8
    _p(f"JAX backend: {jax.default_backend()}")
    _p(f"Grid: (Ny,Nx,Nz)=({Ny},{Nx},{Nz}), Moments: (Nl,Nh)=({Nl},{Nh})")
    _p(f"Packed state length: 2*Ncomplex = {2*Ncomplex:,} reals")
    _p(f"Approx bytes/state: {_fmt_bytes(bytes_per_state)}")
    _p(f"save='{save}', timesteps={timesteps}, dt={dt}, adaptive={adaptive_time_step}, save_every={save_every}")
    if probe is not None:
        _p(f"Probe enabled: {probe}")

    y0 = _pack_complex_flat(params["Gk_0"])

    ts_full = jnp.linspace(0.0, float(params["t_max"]), timesteps)
    ts = ts_full[:: max(1, int(save_every))]

    stepsize_controller = (
        PIDController(rtol=1e-7, atol=1e-9) if adaptive_time_step else ConstantStepSize()
    )

    # Stability/physics: constant-step + nonlinear benefits from smaller dt0.
    # This helps energy tests and makes long constant-step runs much saner.
    dt0 = float(dt)
    if (not adaptive_time_step) and params.get("enable_nonlinear", True):
        dt0 = float(dt) * 0.25  # 4 internal substeps
        _p(f"[yellow]Note:[/yellow] constant-step nonlinear run: using internal dt0={dt0:g} (4x substeps)" if _CONSOLE else f"Note: constant-step nonlinear run: using internal dt0={dt0:g} (4x substeps)")

    term = ODETerm(lambda t, y, args: vector_field_real(t, y, args, Nh, Nl, Ny, Nx, Nz, Ncomplex))

    if save == "diagnostics":
        diag_fn = _make_diag_fn(Nh, Nl, Ny, Nx, Nz, Ncomplex, probe)
        saveat = SaveAt(subs=SubSaveAt(ts=ts, fn=diag_fn))
    elif save == "final":
        # Robust across diffrax versions: save at an explicit ts=[t1]
        t1 = float(params["t_max"])
        saveat = SaveAt(ts=jnp.array([t1], dtype=jnp.float64))
    elif save == "full":
        saveat = SaveAt(ts=ts)
    else:
        raise ValueError(f"Unknown save='{save}'. Use 'diagnostics', 'final', or 'full'.")

    sol = diffeqsolve(
        term,
        solver=solver,
        t0=0.0,
        t1=float(params["t_max"]),
        dt0=dt0,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1_000_000,
        progress_meter=(TqdmProgressMeter() if progress else NoProgressMeter()),
    )

    out = dict(**params)

    if save == "diagnostics":
        out["time"] = sol.ts
        for k, v in sol.ys.items():
            out[k] = v

        t = out["time"]
        D = out.get("D_coll", None)
        if D is not None:
            cum = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float64),
                                  jnp.cumsum(0.5 * (D[1:] + D[:-1]) * (t[1:] - t[:-1]))])
            out["Cum_D_coll"] = cum

    elif save == "final":
        out["time"] = sol.ts
        yT = sol.ys[-1]  # shape (2*Ncomplex,)
        out["Gk_final"] = _unpack_complex_flat(yT, (Nl, Nh, Ny, Nx, Nz), Ncomplex)

    elif save == "full":
        out["time"] = sol.ts
        ys = sol.ys
        out["Gk"] = jax.vmap(lambda y: _unpack_complex_flat(y, (Nl, Nh, Ny, Nx, Nz), Ncomplex))(ys)

    _hline()
    _p("[bold green]SPECTRAX-GK: Simulation finished[/bold green]" if _CONSOLE else "SPECTRAX-GK: Simulation finished")
    _p(f"Wall time: {_time.time() - t_wall0:.2f} s")
    if save == "diagnostics":
        _p(f"Final W_total: {float(out['W_total'][-1]):.6e}")
        _p(f"Final max|G|:   {float(out['max_abs_G'][-1]):.6e}")
        _p(f"Final max|phi|: {float(out['max_abs_phi'][-1]):.6e}")
        if "Cum_D_coll" in out:
            _p(f"Cumulative collision dissipation: {float(out['Cum_D_coll'][-1]):.6e}")
    _hline()
    return out
