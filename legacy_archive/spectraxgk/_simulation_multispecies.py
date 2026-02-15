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
from ._model_multispecies import (
    rhs_gk_multispecies,
    cheap_diagnostics_multispecies,
    solve_phi_quasineutrality_multispecies,
    build_Hk_from_Gk_phi,
)

__all__ = ["simulation_multispecies", "_pack_complex_flat", "_unpack_complex_flat", "vector_field_real"]


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


# ---- pack/unpack complex as REAL state ----
def _pack_complex_flat(z: jnp.ndarray) -> jnp.ndarray:
    """
    Pack complex array into a real vector [Re(z), Im(z)].
    """
    zf = z.reshape(-1)
    return jnp.concatenate([jnp.real(zf), jnp.imag(zf)], axis=0)


def _unpack_complex_flat(y: jnp.ndarray, shape: tuple[int, ...], Ncomplex: int) -> jnp.ndarray:
    """
    Inverse of _pack_complex_flat.
    """
    y = y.reshape(-1)
    re = y[:Ncomplex]
    im = y[Ncomplex:2*Ncomplex]
    return (re + 1j * im).reshape(shape)


def _params_for_solver(params: dict) -> dict:
    """
    Diffrax args must be JAX-friendly and should avoid complex leaves to prevent
    complex-dtype warnings. We keep all real arrays + small scalars, but remove
    complex-valued cached items that are not required by the RHS.
    """
    # Shallow copy is fine (we don't mutate arrays).
    p = dict(params)
    # Complex-only leaves to remove from args:
    p.pop("Gk_0", None)
    p.pop("Hk_0", None)  # initial condition is passed separately as y0
    return p


@partial(jax.jit, static_argnames=("Ns", "Nl", "Nh", "Ny", "Nx", "Nz", "Ncomplex"))
def vector_field_real(t, y, params, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex):
    """
    Diffrax-compatible vector field on a REAL state vector.
    """
    Gk = _unpack_complex_flat(y, (Ns, Nl, Nh, Ny, Nx, Nz), Ncomplex)
    dGk, _phi = rhs_gk_multispecies(Gk, params, Nh=Nh, Nl=Nl)
    return _pack_complex_flat(dGk)


def _make_diag_fn(Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex, diag_config: dict):
    """
    Return a function fn(t,y,params)->dict of *REAL* diagnostics to be saved.

    Key choice: everything returned is real-valued (no complex leaves),
    which avoids Diffrax complex dtype warnings in saved output.
    """
    ky = int(diag_config.get("ky", Ny // 2))
    kx = int(diag_config.get("kx", Nx // 2))
    kz = int(diag_config.get("kz", Nz // 2))

    save_phi_line = bool(diag_config.get("save_phi_k_line", False))
    save_den_line = bool(diag_config.get("save_density_k_line", False))
    save_Em = bool(diag_config.get("save_hermite_spectrum", True))

    def fn(t, y, params):
        Gk = _unpack_complex_flat(y, (Ns, Nl, Nh, Ny, Nx, Nz), Ncomplex)
        d = cheap_diagnostics_multispecies(Gk, params)

        # Split complex probes into real/imag for safe saving
        phi_k = solve_phi_quasineutrality_multispecies(Gk, params)
        phi_probe = phi_k[ky, kx, kz]
        d["phi_probe_re"] = jnp.real(phi_probe)
        d["phi_probe_im"] = jnp.imag(phi_probe)

        Gprobe = Gk[:, 0, 0, ky, kx, kz]  # (Ns,)
        d["Gm0_probe_re_s"] = jnp.real(Gprobe)
        d["Gm0_probe_im_s"] = jnp.imag(Gprobe)

        # 1D line diagnostics
        if save_phi_line and (Ny == 1 and Nx == 1):
            line = phi_k[0, 0, :]  # (Nz,) complex
            d["phi_k_line_re"] = jnp.real(line)
            d["phi_k_line_im"] = jnp.imag(line)

        if save_den_line and (Ny == 1 and Nx == 1):
            Hk_loc = build_Hk_from_Gk_phi(Gk, phi_k, params)
            n0 = params["n0_s"]
            Jl_s = params["Jl_s"]
            Hm0 = Hk_loc[:, :, 0, ...]                    # (Ns,Nl,Ny,Nx,Nz)
            num_s = jnp.sum(Jl_s * Hm0, axis=1)           # (Ns,Ny,Nx,Nz)
            nline = (n0[:, None] * num_s[:, 0, 0, :])     # (Ns,Nz) complex
            d["n_s_k_line_re"] = jnp.real(nline)
            d["n_s_k_line_im"] = jnp.imag(nline)

        if save_Em:
            # already in dict as (Ns,Nh) real; keep explicitly
            d["E_m"] = d["E_m"]

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
        t.add_row("enable_gradB_parallel", f"{bool(params.get('enable_gradB_parallel', False))}")
        t.add_row("B_eps, B_mode", f"{float(params.get('B_eps', 0.0))}, {int(params.get('B_mode', 1))}")
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
        _p("-" * 90)
        _p(f"backend={jax.default_backend()} x64={jax.config.read('jax_enable_x64')}")
        _p(f"grid (Ny,Nx,Nz)=({Ny},{Nx},{Nz})  moments (Nl,Nh)=({Nl},{Nh})  Ns={Ns}")
        _p(f"t_max={float(params['t_max'])} timesteps={timesteps} dt={dt} adaptive={adaptive_time_step}")
        _p(f"save={save} save_every={save_every} packed_len={2*Ncomplex} bytes/state={_sizeof_fmt(bytes_per_state)}")
        _p(f"lambda_D={float(params.get('lambda_D', 0.0))}  gradB={bool(params.get('enable_gradB_parallel', False))}")
        _p("-" * 90)

    # Real ODE state (pack IC)
    y0 = _pack_complex_flat(params["Gk_0"])

    # Time grid
    ts_full = jnp.linspace(0.0, float(params["t_max"]), timesteps)
    ts = ts_full[::max(1, int(save_every))]

    stepsize_controller = PIDController(rtol=1e-7, atol=1e-9) if adaptive_time_step else ConstantStepSize()

    # IMPORTANT: pass only real-leaf params into Diffrax args
    params_solver = _params_for_solver(params)

    term = ODETerm(lambda t, y, args: vector_field_real(t, y, args, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex))

    if save == "diagnostics":
        diag_fn = _make_diag_fn(Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex, diag_config)
        saveat = SaveAt(subs=SubSaveAt(ts=ts, fn=diag_fn))
    elif save == "final":
        saveat = SaveAt(ts=jnp.array([float(params["t_max"])], dtype=jnp.float64))
    else:
        raise ValueError("save must be 'diagnostics' or 'final'.")

    # Warmup RHS eval
    t_compile0 = _time.time()
    _ = vector_field_real(0.0, y0, params_solver, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex).block_until_ready()
    _p(f"[dim]JIT compile warmup:[/dim] {_time.time()-t_compile0:.3f} s" if _USE_RICH else f"JIT compile warmup: {_time.time()-t_compile0:.3f} s")

    sol = diffeqsolve(
        term,
        solver=solver,
        t0=0.0,
        t1=float(params["t_max"]),
        dt0=float(dt),
        y0=y0,
        args=params_solver,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1_000_000,
        progress_meter=(TqdmProgressMeter() if progress else NoProgressMeter()),
    )

    out = dict(**params)  # include full params for user inspection
    out["wall_time"] = _time.time() - t_wall0
    out["time"] = sol.ts
    out["species"] = species_meta

    if save == "diagnostics":
        for k, v in sol.ys.items():
            out[k] = v

        _p("[bold green]SPECTRAX-GK multispecies: finished[/bold green]" if _USE_RICH else "SPECTRAX-GK multispecies: finished")

        # Helpful final summary lines (only scalars)
        Wf = out.get("W_free", out.get("W_total", None))
        if Wf is not None:
            _p(f"wall_time={out['wall_time']:.3f} s  final W_free={float(Wf[-1]):.6e}")
        if "phi_rms" in out:
            _p(f"final phi_rms={float(out['phi_rms'][-1]):.6e}")
        if "max_abs_phi" in out and "max_abs_G" in out:
            _p(f"final max|phi|={float(out['max_abs_phi'][-1]):.6e}  final max|G|={float(out['max_abs_G'][-1]):.6e}")

    else:
        yT = sol.ys[-1]
        out["Gk_final"] = _unpack_complex_flat(yT, (Ns, Nl, Nh, Ny, Nx, Nz), Ncomplex)
        _p("[bold green]SPECTRAX-GK multispecies: finished (final state)[/bold green]" if _USE_RICH else "SPECTRAX-GK multispecies: finished (final state)")

    return out
