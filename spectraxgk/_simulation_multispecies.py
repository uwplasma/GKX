# spectraxgk/_simulation_multispecies.py
from __future__ import annotations

import time as _time
import jax
import jax.numpy as jnp
from functools import partial
from diffrax import (
    diffeqsolve, ODETerm, SaveAt, SubSaveAt,
    ConstantStepSize, PIDController,
    Tsit5, TqdmProgressMeter, NoProgressMeter,
)

from ._initialization_multispecies import initialize_simulation_parameters_multispecies
from ._model_multispecies import rhs_gk_multispecies, cheap_diagnostics_multispecies

__all__ = ["simulation_multispecies"]

def _pack_complex_flat(z: jnp.ndarray) -> jnp.ndarray:
    zf = z.reshape(-1)
    return jnp.concatenate([jnp.real(zf), jnp.imag(zf)], axis=0)

def _unpack_complex_flat(y: jnp.ndarray, shape: tuple[int, ...], Ncomplex: int) -> jnp.ndarray:
    y = y.reshape(-1)
    re = y[:Ncomplex]
    im = y[Ncomplex:2*Ncomplex]
    return (re + 1j*im).reshape(shape)

@partial(jax.jit, static_argnames=("Ns","Nl","Nh","Ny","Nx","Nz","Ncomplex"))
def vector_field_real(t, y, params, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex):
    Gk = _unpack_complex_flat(y, (Ns,Nl,Nh,Ny,Nx,Nz), Ncomplex)
    dGk, _phi = rhs_gk_multispecies(Gk, params, Nh=Nh, Nl=Nl)
    return _pack_complex_flat(dGk)

def _make_diag_fn(Ns,Nl,Nh,Ny,Nx,Nz,Ncomplex):
    def fn(t, y, params):
        Gk = _unpack_complex_flat(y, (Ns,Nl,Nh,Ny,Nx,Nz), Ncomplex)
        return cheap_diagnostics_multispecies(Gk, params)
    return fn

def simulation_multispecies(
    input_parameters=None,
    Nx=33, Ny=33, Nz=17,
    Nh=24, Nl=8,
    timesteps=200,
    dt=1e-2,
    solver=None,
    adaptive_time_step=True,
    progress=True,
    save: str = "diagnostics",
    save_every: int = 1,
):
    t0 = _time.time()
    if input_parameters is None:
        input_parameters = {}
    if solver is None:
        solver = Tsit5()

    params = initialize_simulation_parameters_multispecies(
        input_parameters, Nx=Nx, Ny=Ny, Nz=Nz, Nh=Nh, Nl=Nl, timesteps=timesteps, dt=dt
    )

    # Keep human-readable metadata OUT of jitted params; attach to output only.
    species_meta = list(input_parameters.get("species", []))

    Ns = int(params["Ns"])
    Ncomplex = int(Ns * Nl * Nh * Ny * Nx * Nz)

    y0 = _pack_complex_flat(params["Gk_0"])

    ts_full = jnp.linspace(0.0, float(params["t_max"]), timesteps)
    ts = ts_full[::max(1, int(save_every))]

    stepsize_controller = PIDController(rtol=1e-7, atol=1e-9) if adaptive_time_step else ConstantStepSize()

    term = ODETerm(lambda t, y, args: vector_field_real(t, y, args, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex))

    if save == "diagnostics":
        diag_fn = _make_diag_fn(Ns,Nl,Nh,Ny,Nx,Nz,Ncomplex)
        saveat = SaveAt(subs=SubSaveAt(ts=ts, fn=diag_fn))
    elif save == "final":
        saveat = SaveAt(ts=jnp.array([float(params["t_max"])], dtype=jnp.float64))
    else:
        raise ValueError("save must be 'diagnostics' or 'final' for this multispecies driver.")

    sol = diffeqsolve(
        term,
        solver=solver,
        t0=0.0,
        t1=float(params["t_max"]),
        dt0=float(dt),
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1_000_000,
        progress_meter=(TqdmProgressMeter() if progress else NoProgressMeter()),
    )

    out = dict(**params)
    out["wall_time"] = _time.time() - t0
    out["time"] = sol.ts
    out["species"] = species_meta

    if save == "diagnostics":
        for k, v in sol.ys.items():
            out[k] = v
    else:
        yT = sol.ys[-1]
        out["Gk_final"] = _unpack_complex_flat(yT, (Ns,Nl,Nh,Ny,Nx,Nz), Ncomplex)

    return out
