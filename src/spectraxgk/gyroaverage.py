"""Gyroaveraging coefficients for Laguerre velocity space."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import bessel_jn, gammaln, i0e
import math
import numpy as np


def gamma0(b: jnp.ndarray) -> jnp.ndarray:
    """Compute Gamma_0(b) = exp(-b) I_0(b) using i0e for stability."""

    b = jnp.asarray(b)
    return i0e(b)


def bessel_j0(x: jnp.ndarray) -> jnp.ndarray:
    """Return J0(x) using jax.scipy.special.j0."""

    x = jnp.asarray(x)
    out = bessel_jn(x, v=0)[0]
    x2 = x * x
    approx = 1.0 - 0.25 * x2 + 0.015625 * x2 * x2
    out = jnp.where(jnp.abs(x) < 1.0e-3, approx, out)
    return jnp.where(jnp.isfinite(out), out, approx)


def bessel_j1(x: jnp.ndarray) -> jnp.ndarray:
    """Return J1(x) using jax.scipy.special.j1."""

    x = jnp.asarray(x)
    out = bessel_jn(x, v=1)[1]
    x2 = x * x
    approx = 0.5 * x - 0.0625 * x * x2 + (1.0 / 384.0) * x * x2 * x2
    out = jnp.where(jnp.abs(x) < 1.0e-3, approx, out)
    return jnp.where(jnp.isfinite(out), out, approx)


def J_l_all(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Gyroaveraging coefficients matching the GX Laguerre-Hermite convention."""

    if l_max < 0:
        raise ValueError("l_max must be >= 0")
    b = jnp.asarray(b)
    l = jnp.arange(l_max + 1, dtype=b.dtype)
    l_shape = (l_max + 1,) + (1,) * b.ndim
    l = l.reshape(l_shape)
    b_safe = jnp.maximum(0.5 * b, 1.0e-30)
    log_term = l * jnp.log(b_safe) - gammaln(l + 1.0)
    coef = jnp.exp(log_term)
    sign = jnp.where((l % 2) == 0, 1.0, -1.0)
    Jl = jnp.exp(-0.5 * b)[None, ...] * sign * coef
    if b.ndim > 0:
        mask = (b == 0.0)[None, ...] & (l > 0)
        Jl = jnp.where(mask, 0.0, Jl)
    return Jl


def sum_Jl2(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Truncated sum of J_l(b)^2, useful for Gamma_0 convergence checks."""

    Jl = J_l_all(b, l_max)
    return jnp.sum(Jl * Jl, axis=0)


def gx_laguerre_nj(nl: int) -> int:
    """GX default for number of Laguerre quadrature points."""

    return max(1, 3 * nl // 2 - 1)


def gx_laguerre_transform(nl: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GX-style Laguerre transform matrices and roots."""

    if nl < 1:
        raise ValueError("nl must be >= 1")
    nj = gx_laguerre_nj(nl)
    jac = np.zeros((nj, nj), dtype=float)
    for i in range(nj - 1):
        jac[i, i] = 2.0 * i + 1.0
        jac[i, i + 1] = i + 1.0
        jac[i + 1, i] = i + 1.0
    jac[nj - 1, nj - 1] = 2.0 * nj - 1.0

    evals, evecs = np.linalg.eigh(jac)
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]

    roots = evals.astype(float)
    poly = np.zeros((nl, nj), dtype=float)
    for i in range(nl):
        for j in range(i + 1):
            tmp = float(math.comb(i, j))
            tmp *= (-1.0) ** j / math.factorial(j) * ((-1.0) ** i)
            poly[i, j] = tmp

    to_grid = np.zeros((nl, nj), dtype=float)
    to_spectral = np.zeros((nj, nl), dtype=float)
    for j in range(nj):
        x_i = roots[j]
        wgt = float(evecs[0, j] ** 2)
        for ell in range(nl):
            coeffs = poly[ell, : ell + 1]
            Lmat = 0.0
            for c in coeffs[::-1]:
                Lmat = Lmat * x_i + c
            to_grid[ell, j] = Lmat
            to_spectral[j, ell] = Lmat * wgt

    return to_grid, to_spectral, roots
