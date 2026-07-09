"""Shared Diffrax dependency, solver-policy, and state helper routines."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly import assemble_rhs_cached, assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig

if TYPE_CHECKING:  # pragma: no cover
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
        "dopri8": "dopri8",
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
        "dopri8": dfx.Dopri8,
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


def _base_complex_dtype() -> jnp.dtype:
    return jnp.complex128 if bool(getattr(jax.config, "x64_enabled", False)) else jnp.complex64


def _pack_complex_state(G: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.real(G), jnp.imag(G)], axis=-1)


def _unpack_complex_state(G_packed: jnp.ndarray) -> jnp.ndarray:
    return G_packed[..., 0] + 1j * G_packed[..., 1]


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


def _density_from_G_cached(
    G_in: jnp.ndarray,
    cache: LinearCache,
    density_species_index: int | None,
) -> jnp.ndarray:
    Jl = cache.Jl
    if G_in.ndim == 5:
        if Jl.ndim == 5:
            Jl_s = Jl[0]
        else:
            Jl_s = Jl
        return jnp.sum(Jl_s * G_in[:, 0, ...], axis=0)
    if Jl.ndim == 5:
        if density_species_index is None:
            return jnp.sum(jnp.sum(Jl * G_in[:, :, 0, ...], axis=1), axis=0)
        Jl_s = Jl[int(density_species_index)]
        return jnp.sum(Jl_s * G_in[int(density_species_index), :, 0, ...], axis=0)
    if density_species_index is None:
        return jnp.sum(jnp.sum(Jl[None, ...] * G_in[:, :, 0, ...], axis=1), axis=0)
    return jnp.sum(Jl * G_in[int(density_species_index), :, 0, ...], axis=0)

__all__ = [
    "_adjoint",
    "_assemble_rhs",
    "_base_complex_dtype",
    "_density_from_G_cached",
    "_is_imex_solver",
    "_is_implicit_solver",
    "_pack_complex_state",
    "_progress_meter",
    "_require_diffrax",
    "_save_with_phi",
    "_solver_from_name",
    "_stepsize_controller",
    "_unpack_complex_state",
]
