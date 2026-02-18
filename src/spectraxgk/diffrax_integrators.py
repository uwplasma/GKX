"""Diffrax-based time integrators for gyrokinetic systems."""

from __future__ import annotations

from typing import Any, Callable, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams, LinearTerms, build_linear_cache
from spectraxgk.terms.assembly import assemble_rhs_cached, assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import placeholder_nonlinear_contribution

if TYPE_CHECKING:
    import diffrax as dfx
    import equinox as eqx
else:  # pragma: no cover - optional dependency loading
    try:
        import diffrax as dfx  # type: ignore[no-redef]
        import equinox as eqx  # type: ignore[no-redef]
    except Exception:
        dfx = None  # type: ignore[assignment]
        eqx = None  # type: ignore[assignment]


def _require_diffrax() -> tuple[Any, Any]:
    if dfx is None or eqx is None:
        raise ImportError("diffrax and equinox must be installed to use diffrax integrators")
    return dfx, eqx


def _solver_from_name(name: str):
    dfx, _ = _require_diffrax()
    key = name.strip().lower()
    aliases = {
        "rk4": "tsit5",
        "rk2": "heun",
        "euler": "euler",
        "heun": "heun",
        "tsit5": "tsit5",
        "dopri5": "dopri5",
        "implicit": "kvaerno5",
        "imex": "kencarp4",
        "semi-implicit": "kencarp4",
    }
    key = aliases.get(key, key)
    mapping = {
        "euler": dfx.Euler,
        "heun": dfx.Heun,
        "tsit5": dfx.Tsit5,
        "dopri5": dfx.Dopri5,
        "impliciteuler": dfx.ImplicitEuler,
        "kvaerno3": dfx.Kvaerno3,
        "kvaerno4": dfx.Kvaerno4,
        "kvaerno5": dfx.Kvaerno5,
        "kencarp3": dfx.KenCarp3,
        "kencarp4": dfx.KenCarp4,
        "kencarp5": dfx.KenCarp5,
    }
    if key not in mapping:
        raise ValueError(f"Unknown diffrax solver '{name}'")
    return mapping[key]()


def _is_imex_solver(name: str) -> bool:
    key = name.strip().lower()
    return key in {"kencarp3", "kencarp4", "kencarp5", "imex", "semi-implicit"}


def _is_implicit_solver(name: str) -> bool:
    key = name.strip().lower()
    return key in {"impliciteuler", "kvaerno3", "kvaerno4", "kvaerno5", "implicit"}


def _progress_meter(enabled: bool):
    dfx, _ = _require_diffrax()
    return dfx.TqdmProgressMeter() if enabled else dfx.NoProgressMeter()


def _stepsize_controller(adaptive: bool, rtol: float, atol: float):
    dfx, _ = _require_diffrax()
    if adaptive:
        return dfx.PIDController(rtol=rtol, atol=atol)
    return dfx.ConstantStepSize()


def _adjoint(checkpoint: bool):
    dfx, _ = _require_diffrax()
    if checkpoint:
        return dfx.RecursiveCheckpointAdjoint()
    return dfx.DirectAdjoint()


def _assemble_rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    use_custom_vjp: bool,
) -> tuple[jnp.ndarray, FieldState]:
    if use_custom_vjp:
        return assemble_rhs_cached_jit(G, cache, params, term_cfg)
    return assemble_rhs_cached(G, cache, params, terms=term_cfg, use_custom_vjp=False)


def _save_with_phi(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    use_custom_vjp: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    fields = compute_fields_cached(G, cache, params, terms=term_cfg, use_custom_vjp=use_custom_vjp)
    return G, fields.phi


def integrate_linear_diffrax(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "Heun",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    adaptive: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    max_steps: int = 4096,
    progress_bar: bool = True,
    checkpoint: bool = False,
    jit: bool | None = None,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate the linear system with diffrax."""

    dfx, eqx = _require_diffrax()
    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    if terms is None:
        terms = LinearTerms()
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    term_cfg = TermConfig(
        streaming=terms.streaming,
        mirror=terms.mirror,
        curvature=terms.curvature,
        gradb=terms.gradb,
        diamagnetic=terms.diamagnetic,
        collisions=terms.collisions,
        hypercollisions=terms.hypercollisions,
        end_damping=terms.end_damping,
        apar=terms.apar,
        bpar=terms.bpar,
        nonlinear=0.0,
    )

    use_custom_vjp = not (_is_imex_solver(method) or _is_implicit_solver(method))

    def rhs(t, G, args):
        cache_, params_, term_cfg_ = args
        dG, _fields = _assemble_rhs(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)
        return dG

    def save_fn(t, G, args):
        cache_, params_, term_cfg_ = args
        return _save_with_phi(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)

    solver = _solver_from_name(method)
    explicit_term = dfx.ODETerm(rhs)
    if _is_imex_solver(method):
        zero_term = dfx.ODETerm(lambda t, y, args: jnp.zeros_like(y))
        terms_obj = dfx.MultiTerm(zero_term, explicit_term)
    else:
        terms_obj = explicit_term

    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    num_samples = steps // sample_stride
    ts = dt_val * sample_stride * (jnp.arange(num_samples, dtype=real_dtype) + 1)

    adaptive_eff = adaptive or _is_imex_solver(method) or _is_implicit_solver(method)

    def solve():
        max_steps_eff = max(int(max_steps), int(steps))
        return dfx.diffeqsolve(
            terms_obj,
            solver,
            t0=jnp.asarray(0.0, dtype=real_dtype),
            t1=dt_val * steps,
            dt0=dt_val,
            y0=G0,
            args=(cache, params, term_cfg),
            saveat=dfx.SaveAt(ts=ts, fn=save_fn),
            stepsize_controller=_stepsize_controller(adaptive_eff, rtol, atol),
            adjoint=_adjoint(checkpoint),
            max_steps=max_steps_eff,
            progress_meter=_progress_meter(progress_bar),
        )

    if jit is None:
        jit = not progress_bar
    sol = eqx.filter_jit(solve)() if jit else solve()
    G_t, phi_t = sol.ys
    return G_t[-1], phi_t


def integrate_nonlinear_diffrax(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "KenCarp4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    adaptive: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    max_steps: int = 4096,
    progress_bar: bool = True,
    checkpoint: bool = False,
    jit: bool | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system with diffrax (placeholder nonlinear term)."""

    dfx, eqx = _require_diffrax()
    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    term_cfg = terms or TermConfig()
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    use_custom_vjp = not (_is_imex_solver(method) or _is_implicit_solver(method))

    def rhs_linear(t, G, args):
        cache_, params_, term_cfg_ = args
        dG, _fields = _assemble_rhs(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)
        return dG

    def rhs_nonlinear(t, G, args):
        _cache, _params, term_cfg_ = args
        if term_cfg_.nonlinear == 0.0:
            return jnp.zeros_like(G)
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        weight = jnp.asarray(term_cfg_.nonlinear, dtype=real_dtype)
        return placeholder_nonlinear_contribution(G, weight=weight)

    def rhs_full(t, G, args):
        return rhs_linear(t, G, args) + rhs_nonlinear(t, G, args)

    def save_fn(t, G, args):
        cache_, params_, term_cfg_ = args
        return _save_with_phi(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)

    solver = _solver_from_name(method)
    explicit_term = dfx.ODETerm(rhs_nonlinear if _is_imex_solver(method) else rhs_full)
    implicit_term = dfx.ODETerm(rhs_linear)
    if _is_imex_solver(method):
        terms_obj = dfx.MultiTerm(explicit_term, implicit_term)
    else:
        terms_obj = explicit_term

    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    ts = dt_val * (jnp.arange(steps, dtype=real_dtype) + 1)

    adaptive_eff = adaptive or _is_imex_solver(method) or _is_implicit_solver(method)

    def solve():
        max_steps_eff = max(int(max_steps), int(steps))
        return dfx.diffeqsolve(
            terms_obj,
            solver,
            t0=jnp.asarray(0.0, dtype=real_dtype),
            t1=dt_val * steps,
            dt0=dt_val,
            y0=G0,
            args=(cache, params, term_cfg),
            saveat=dfx.SaveAt(ts=ts, fn=save_fn),
            stepsize_controller=_stepsize_controller(adaptive_eff, rtol, atol),
            adjoint=_adjoint(checkpoint),
            max_steps=max_steps_eff,
            progress_meter=_progress_meter(progress_bar),
        )

    if jit is None:
        jit = not progress_bar
    sol = eqx.filter_jit(solve)() if jit else solve()
    G_t, phi_t = sol.ys
    return G_t[-1], FieldState(phi=phi_t)
