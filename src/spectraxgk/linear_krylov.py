"""Matrix-free Krylov solvers for linear gyrokinetic operators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from spectraxgk.linear import LinearCache, LinearParams, LinearTerms
from spectraxgk.terms.assembly import assemble_rhs_cached_jit
from spectraxgk.terms.config import TermConfig


@dataclass(frozen=True)
class KrylovConfig:
    """Controls for the Krylov-based eigen solver."""

    krylov_dim: int = 24
    restarts: int = 2
    omega_cap_factor: float = 2.0


def _normalize(v: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(v)
    norm_safe = jnp.where(norm == 0.0, 1.0, norm)
    return v / norm_safe


def _apply_operator(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> jnp.ndarray:
    dG, _fields = assemble_rhs_cached_jit(v, cache, params, term_cfg)
    return dG


def _arnoldi(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    krylov_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    v0 = _normalize(v0)
    V = jnp.zeros((krylov_dim + 1,) + v0.shape, dtype=v0.dtype)
    H = jnp.zeros((krylov_dim + 1, krylov_dim), dtype=v0.dtype)
    V = V.at[0].set(v0)

    def outer(i, carry):
        V, H = carry
        w = _apply_operator(V[i], cache, params, term_cfg)

        def inner(j, inner_carry):
            w, H = inner_carry
            h = jnp.vdot(V[j], w)
            w = w - h * V[j]
            H = H.at[j, i].set(h)
            return w, H

        w, H = jax.lax.fori_loop(0, i + 1, inner, (w, H))
        h_next = jnp.linalg.norm(w)
        H = H.at[i + 1, i].set(h_next)
        v_next = jnp.where(h_next > 0.0, w / h_next, w)
        V = V.at[i + 1].set(v_next)
        return V, H

    V, H = jax.lax.fori_loop(0, krylov_dim, outer, (V, H))
    return V, H


@partial(jax.jit, static_argnames=("krylov_dim", "restarts"))
def dominant_eigenpair_cached(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    restarts: int,
    omega_cap_factor: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate the dominant eigenvalue (max real part) with restarted Arnoldi."""

    v = v0
    eig0 = jnp.asarray(0.0, dtype=v0.dtype)

    def restart_body(i, state):
        v, _eig_prev = state
        V, H = _arnoldi(v, cache, params, term_cfg, krylov_dim)
        Hk = H[:krylov_dim, :krylov_dim]
        eigvals, eigvecs = jnp.linalg.eig(Hk)
        real_part = jnp.real(eigvals)
        imag_part = jnp.imag(eigvals)
        rlt = jnp.max(jnp.abs(params.R_over_LTi)) + jnp.max(jnp.abs(params.R_over_Ln))
        omega_scale = jnp.max(jnp.abs(cache.ky)) * jnp.maximum(rlt, 1.0e-8)
        omega_cap = omega_cap_factor * omega_scale
        use_cap = omega_cap_factor > 0.0
        mask = jnp.abs(imag_part) <= omega_cap
        real_masked = jnp.where(mask, real_part, -jnp.inf)
        idx_masked = jnp.argmax(real_masked)
        idx_low_omega = jnp.argmin(jnp.abs(imag_part))
        use_mask = use_cap & jnp.any(mask)
        idx = jnp.where(use_mask, idx_masked, idx_low_omega)
        eig = eigvals[idx]
        y = eigvecs[:, idx]
        v_next = jnp.tensordot(jnp.conj(y), V[:krylov_dim], axes=1)
        v_next = _normalize(v_next)
        return v_next, eig

    v, eig = jax.lax.fori_loop(0, restarts, restart_body, (v, eig0))
    return eig, v


def dominant_eigenpair(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    krylov_dim: int = 24,
    restarts: int = 2,
    omega_cap_factor: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Python wrapper for the cached Krylov solver."""

    if terms is None:
        terms = LinearTerms()
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
    restarts = max(int(restarts), 1)
    return dominant_eigenpair_cached(
        v0,
        cache,
        params,
        term_cfg,
        krylov_dim=krylov_dim,
        restarts=restarts,
        omega_cap_factor=omega_cap_factor,
    )


def dominant_eigenvalue(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    krylov_dim: int = 24,
    restarts: int = 2,
) -> jnp.ndarray:
    eig, _vec = dominant_eigenpair(
        v0,
        cache,
        params,
        terms,
        krylov_dim=krylov_dim,
        restarts=restarts,
    )
    return eig
