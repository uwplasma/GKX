# spectraxgk/_simulation_multispecies.py
from __future__ import annotations

import os
import time as _time
from typing import Any, Optional, Protocol, Callable

import jax
import jax.numpy as jnp
from functools import partial
from diffrax import (
    diffeqsolve, ODETerm, SaveAt, SubSaveAt,
    ConstantStepSize, PIDController,
    Tsit5, TqdmProgressMeter, NoProgressMeter,
)

from ._initialization_multispecies import initialize_simulation_parameters_multispecies
from ._model_multispecies import rhs_gk_multispecies, cheap_diagnostics_multispecies, solve_phi_quasineutrality_multispecies, build_Hk_from_Gk_phi

__all__ = ["simulation_multispecies"]

# ---- pretty console (Rich optional) ----
class _ConsoleLike(Protocol):
    def print(self, *args: Any, **kwargs: Any) -> Any: ...

_RICH_AVAIL = False
_MAKE_CONSOLE: Optional[Callable[[], _ConsoleLike]] = None
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    _RICH_AVAIL = True
    _MAKE_CONSOLE = lambda: Console()
except Exception:
    _RICH_AVAIL = False
    _MAKE_CONSOLE = None

_USE_RICH: bool = False
_console: Optional[_ConsoleLike] = None

def init_pretty(prefer_rich: bool = True) -> None:
    global _USE_RICH, _console
    no_color_env = any(k.upper() == "NO_COLOR" for k in os.environ.keys())
    if no_color_env:
        _USE_RICH = False
        _console = None
        return
    if prefer_rich and _RICH_AVAIL and _MAKE_CONSOLE is not None:
        _USE_RICH = True
        _console = _MAKE_CONSOLE()
    else:
        _USE_RICH = False
        _console = None
        if prefer_rich and not _RICH_AVAIL:
            print("Note: nicer terminal output available via `pip install rich`.")

def _p(msg: str) -> None:
    if _USE_RICH and _console is not None:
        _console.print(msg)
    else:
        print(msg)

def _sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


# ---- pack/unpack ----
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

def _make_diag_fn(Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex, diag_config: dict):
    # static “probe indices” resolved outside JIT
    ky = int(diag_config.get("ky", Ny // 2))
    kx = int(diag_config.get("kx", Nx // 2))
    kz = int(diag_config.get("kz", Nz // 2))

    save_phi_line = bool(diag_config.get("save_phi_k_line", False))
    save_den_line = bool(diag_config.get("save_density_k_line", False))
    save_Em = bool(diag_config.get("save_hermite_spectrum", True))

    def fn(t, y, params):
        Gk = _unpack_complex_flat(y, (Ns,Nl,Nh,Ny,Nx,Nz), Ncomplex)

        d = cheap_diagnostics_multispecies(Gk, params)

        # Optional “line” diagnostics (very useful in 1D)
        phi_k = solve_phi_quasineutrality_multispecies(Gk, params)

        if save_phi_line and (Ny == 1 and Nx == 1):
            d["phi_k_line"] = phi_k[0, 0, :]  # (Nz,)

        if save_den_line and (Ny == 1 and Nx == 1):
            Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)
            # density-like moment: n_s(k) ~ n0_s * sum_l J_l * H_{m=0}
            n0 = params["n0_s"]
            Jl_s = params["Jl_s"]
            Hm0 = Hk[:, :, 0, ...]                             # (Ns,Nl,Ny,Nx,Nz)
            num_s = jnp.sum(Jl_s * Hm0, axis=1)                # (Ns,Ny,Nx,Nz)
            d["n_s_k_line"] = (n0[:, None] * num_s[:, 0, 0, :]) # (Ns,Nz)

        if save_Em:
            d["E_m"] = d["E_m"]  # already included; keep explicit

        # Single probe mode (always small)
        d["phi_probe"] = phi_k[ky, kx, kz]
        d["Gm0_probe_s"] = Gk[:, 0, 0, ky, kx, kz]  # (Ns,)
        return d

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
    prefer_rich: bool = True,
    diag_config: Optional[dict] = None,
):
    init_pretty(prefer_rich=prefer_rich)

    t_wall0 = _time.time()
    _p("[bold cyan]SPECTRAX-GK multispecies: start[/bold cyan]" if _USE_RICH else "SPECTRAX-GK multispecies: start")

    if input_parameters is None:
        input_parameters = {}
    if solver is None:
        solver = Tsit5()
    if diag_config is None:
        diag_config = {}

    # Human-readable metadata (not passed into JIT)
    species_meta = list(input_parameters.get("species", []))

    params = initialize_simulation_parameters_multispecies(
        input_parameters, Nx=Nx, Ny=Ny, Nz=Nz, Nh=Nh, Nl=Nl, timesteps=timesteps, dt=dt
    )

    Ns = int(params["Ns"])
    Ncomplex = int(Ns * Nl * Nh * Ny * Nx * Nz)

    # Preflight print
    bytes_per_real = 8 if bool(jax.config.read("jax_enable_x64")) else 4
    bytes_per_state = (2 * Ncomplex) * bytes_per_real
    est_diag_bytes = (timesteps // max(1, int(save_every))) * 64  # scalars only rough

    if _USE_RICH and _console is not None:
        from rich.table import Table
        from rich.panel import Panel
        from rich.box import ROUNDED
        header = Panel.fit(
            "[bold cyan]SPECTRAX-GK multispecies preflight[/bold cyan]\n"
            f"[dim]backend:[/dim] {jax.default_backend()}  [dim]x64:[/dim] {jax.config.read('jax_enable_x64')}",
            box=ROUNDED,
        )
        _console.print(header)

        t = Table(box=ROUNDED, show_lines=False)
        t.add_column("Key", style="bold dim", no_wrap=True)
        t.add_column("Value")
        t.add_row("grid (Ny,Nx,Nz)", f"({Ny},{Nx},{Nz})")
        t.add_row("moments (Nl,Nh)", f"({Nl},{Nh})")
        t.add_row("Ns", f"{Ns}")
        t.add_row("t_max, steps", f"{float(params['t_max'])}, {timesteps}")
        t.add_row("dt, adaptive", f"{dt}, {adaptive_time_step}")
        t.add_row("save, save_every", f"{save}, {save_every}")
        t.add_row("packed state", f"{2*Ncomplex:,} reals")
        t.add_row("bytes/state", _sizeof_fmt(bytes_per_state))
        t.add_row("lambda_D", f"{float(params.get('lambda_D', 0.0))}")
        t.add_row("IC perturb weights", f"{jnp.asarray(params['perturb_weights_s'])}")
        _console.print(t)

        st = Table(title="Species", box=ROUNDED, show_lines=False)
        st.add_column("i")
        st.add_column("name")
        st.add_column("q")
        st.add_column("T")
        st.add_column("n0")
        st.add_column("rho")
        st.add_column("vth")
        st.add_column("nu")
        st.add_column("Upar")
        for i, sp in enumerate(species_meta):
            st.add_row(
                str(i), str(sp.get("name", i)),
                str(sp.get("q")), str(sp.get("T")), str(sp.get("n0")),
                str(sp.get("rho")), str(sp.get("vth")), str(sp.get("nu")), str(sp.get("Upar")),
            )
        _console.print(st)
    else:
        _p("-"*90)
        _p(f"backend={jax.default_backend()} x64={jax.config.read('jax_enable_x64')}")
        _p(f"grid (Ny,Nx,Nz)=({Ny},{Nx},{Nz})  moments (Nl,Nh)=({Nl},{Nh})  Ns={Ns}")
        _p(f"t_max={float(params['t_max'])} timesteps={timesteps} dt={dt} adaptive={adaptive_time_step}")
        _p(f"save={save} save_every={save_every} packed_len={2*Ncomplex} bytes/state={_sizeof_fmt(bytes_per_state)}")
        _p(f"lambda_D={float(params.get('lambda_D', 0.0))}  perturb_weights={jnp.asarray(params['perturb_weights_s']) if 'np' in globals() else params['perturb_weights_s']}")
        _p("-"*90)

    y0 = _pack_complex_flat(params["Gk_0"])

    # Time grid
    ts_full = jnp.linspace(0.0, float(params["t_max"]), timesteps)
    ts = ts_full[::max(1, int(save_every))]

    stepsize_controller = PIDController(rtol=1e-7, atol=1e-9) if adaptive_time_step else ConstantStepSize()

    term = ODETerm(lambda t, y, args: vector_field_real(t, y, args, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex))

    if save == "diagnostics":
        diag_fn = _make_diag_fn(Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex, diag_config)
        saveat = SaveAt(subs=SubSaveAt(ts=ts, fn=diag_fn))
    elif save == "final":
        saveat = SaveAt(ts=jnp.array([float(params["t_max"])], dtype=jnp.float64))
    else:
        raise ValueError("save must be 'diagnostics' or 'final'.")

    # Warmup one RHS eval (compile) + print quick sanity
    t_compile0 = _time.time()
    _ = vector_field_real(0.0, y0, params, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex).block_until_ready()
    _p(f"[dim]JIT compile warmup:[/dim] {_time.time()-t_compile0:.3f} s" if _USE_RICH else f"JIT compile warmup: {_time.time()-t_compile0:.3f} s")

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
    out["wall_time"] = _time.time() - t_wall0
    out["time"] = sol.ts
    out["species"] = species_meta

    if save == "diagnostics":
        for k, v in sol.ys.items():
            out[k] = v

        # Add cumulative collision dissipation if present
        if "D_coll" in out:
            t = out["time"]
            D = out["D_coll"]
            cum = jnp.concatenate([jnp.zeros((1,), dtype=D.dtype),
                                   jnp.cumsum(0.5 * (D[1:] + D[:-1]) * (t[1:] - t[:-1]))])
            out["Cum_D_coll"] = cum

        _p("[bold green]SPECTRAX-GK multispecies: finished[/bold green]" if _USE_RICH else "SPECTRAX-GK multispecies: finished")
        _p(f"wall_time={out['wall_time']:.3f} s  final W_total={float(out['W_total'][-1]):.6e}  final phi_rms={float(out['phi_rms'][-1]):.6e}")
        _p(f"final max|phi|={float(out['max_abs_phi'][-1]):.6e}  final max|G|={float(out['max_abs_G'][-1]):.6e}")

        if "Cum_D_coll" in out:
            _p(f"Cum_D_coll(t_end)={float(out['Cum_D_coll'][-1]):.6e}")

    else:
        yT = sol.ys[-1]
        out["Gk_final"] = _unpack_complex_flat(yT, (Ns,Nl,Nh,Ny,Nx,Nz), Ncomplex)
        _p("[bold green]SPECTRAX-GK multispecies: finished (final state)[/bold green]" if _USE_RICH else "SPECTRAX-GK multispecies: finished (final state)")

    return out
