"""Nonlinear gyrokinetic drivers built on term-wise RHS assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    LinearTerms,
    _build_implicit_operator,
    build_linear_cache,
)
from spectraxgk.terms.assembly import assemble_rhs_cached_jit
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.integrators import integrate_nonlinear as integrate_nonlinear_scan
from spectraxgk.terms.nonlinear import placeholder_nonlinear_contribution


@dataclass(frozen=True)
class IMEXLinearOperator:
    """Reusable matrix-free linear operator for nonlinear IMEX solves."""

    state_dtype: jnp.dtype
    shape: tuple[int, ...]
    dt_val: jnp.ndarray
    precond_op: Callable[[jnp.ndarray], jnp.ndarray] | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    squeeze_species: bool


def nonlinear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
) -> Tuple[jnp.ndarray, FieldState]:
    """Compute a nonlinear RHS using linear terms plus a placeholder nonlinear term."""

    term_cfg = terms or TermConfig()
    dG, fields = assemble_rhs_cached_jit(G, cache, params, term_cfg)
    if term_cfg.nonlinear != 0.0:
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        weight = jnp.asarray(term_cfg.nonlinear, dtype=real_dtype)
        dG = dG + placeholder_nonlinear_contribution(G, weight=weight)
    return dG, fields


def integrate_nonlinear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    terms: TermConfig | None = None,
    checkpoint: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system using a cached geometry object."""

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        return integrate_nonlinear_imex_cached(
            G0,
            cache,
            params,
            dt,
            steps,
            terms=term_cfg,
            checkpoint=checkpoint,
        )

    def rhs_fn(G):
        return nonlinear_rhs_cached(G, cache, params, term_cfg)

    return integrate_nonlinear_scan(
        rhs_fn,
        G0,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
    )


def integrate_nonlinear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system using built-in cache construction."""

    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return integrate_nonlinear_cached(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
    )


def build_nonlinear_imex_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    *,
    terms: TermConfig | None = None,
    implicit_preconditioner: str | None = None,
) -> IMEXLinearOperator:
    """Build and cache the matrix-free linear operator used by nonlinear IMEX."""

    term_cfg = terms or TermConfig()
    linear_terms = LinearTerms(
        streaming=term_cfg.streaming,
        mirror=term_cfg.mirror,
        curvature=term_cfg.curvature,
        gradb=term_cfg.gradb,
        diamagnetic=term_cfg.diamagnetic,
        collisions=term_cfg.collisions,
        hypercollisions=term_cfg.hypercollisions,
        end_damping=term_cfg.end_damping,
        apar=term_cfg.apar,
        bpar=term_cfg.bpar,
    )
    G, shape, _size, dt_val, precond_op, matvec, squeeze_species = _build_implicit_operator(
        G0,
        cache,
        params,
        dt,
        linear_terms,
        implicit_preconditioner,
    )
    return IMEXLinearOperator(
        state_dtype=G.dtype,
        shape=shape,
        dt_val=dt_val,
        precond_op=precond_op,
        matvec=matvec,
        squeeze_species=squeeze_species,
    )


def integrate_nonlinear_imex_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    implicit_operator: IMEXLinearOperator | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """IMEX integrator: implicit linear operator, explicit nonlinear term."""

    term_cfg = terms or TermConfig()
    linear_cfg = TermConfig(
        streaming=term_cfg.streaming,
        mirror=term_cfg.mirror,
        curvature=term_cfg.curvature,
        gradb=term_cfg.gradb,
        diamagnetic=term_cfg.diamagnetic,
        collisions=term_cfg.collisions,
        hypercollisions=term_cfg.hypercollisions,
        end_damping=term_cfg.end_damping,
        apar=term_cfg.apar,
        bpar=term_cfg.bpar,
        nonlinear=0.0,
    )

    linear_terms = LinearTerms(
        streaming=linear_cfg.streaming,
        mirror=linear_cfg.mirror,
        curvature=linear_cfg.curvature,
        gradb=linear_cfg.gradb,
        diamagnetic=linear_cfg.diamagnetic,
        collisions=linear_cfg.collisions,
        hypercollisions=linear_cfg.hypercollisions,
        end_damping=linear_cfg.end_damping,
        apar=linear_cfg.apar,
        bpar=linear_cfg.bpar,
    )

    precond_op: Callable[[jnp.ndarray], jnp.ndarray] | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    if implicit_operator is None:
        G, shape, _size, dt_val, precond_op, matvec, squeeze_species = _build_implicit_operator(
            G0,
            cache,
            params,
            dt,
            linear_terms,
            implicit_preconditioner,
        )
    else:
        shape = implicit_operator.shape
        dt_val = implicit_operator.dt_val
        precond_op = implicit_operator.precond_op
        matvec = implicit_operator.matvec
        squeeze_species = implicit_operator.squeeze_species
        G = jnp.asarray(G0, dtype=implicit_operator.state_dtype)
        if squeeze_species and G.ndim == len(shape) - 1:
            G = G[None, ...]
        if G.shape != shape:
            raise ValueError(
                "implicit_operator shape mismatch: "
                f"expected {shape}, got {tuple(G.shape)}"
            )

    def nonlinear_term(G_in: jnp.ndarray) -> jnp.ndarray:
        if term_cfg.nonlinear == 0.0:
            return jnp.zeros_like(G_in)
        weight = jnp.asarray(term_cfg.nonlinear, dtype=jnp.real(jnp.empty((), G_in.dtype)).dtype)
        return placeholder_nonlinear_contribution(G_in, weight=weight)

    def fixed_point(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        def body(_i, g):
            dG, _fields = assemble_rhs_cached_jit(g, cache, params, linear_cfg)
            g_next = G_rhs + dt_val * dG
            return (1.0 - implicit_relax) * g + implicit_relax * g_next

        return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        G_guess = fixed_point(G_in, G_rhs)
        sol, _info = jax.scipy.sparse.linalg.gmres(
            matvec,
            G_rhs.reshape(-1),
            x0=G_guess.reshape(-1),
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            restart=implicit_restart,
            M=precond_op,
            solve_method=implicit_solve_method,
        )
        return sol.reshape(shape)

    def step(G_in, _):
        rhs = G_in + dt_val * nonlinear_term(G_in)
        G_new = solve_step(G_in, rhs)
        _dG_new, fields_new = assemble_rhs_cached_jit(G_new, cache, params, linear_cfg)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    G_out, fields_t = jax.lax.scan(step_fn, G, None, length=steps)
    G_out = G_out[0] if squeeze_species else G_out
    return G_out, fields_t
