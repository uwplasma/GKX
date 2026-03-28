"""Gyroaveraging coefficients for Laguerre velocity space."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln, i0e
import math
import numpy as np


def gamma0(b: jnp.ndarray) -> jnp.ndarray:
    """Compute Gamma_0(b) = exp(-b) I_0(b) using i0e for stability."""

    b = jnp.asarray(b)
    return i0e(b)


def bessel_j0(x: jnp.ndarray) -> jnp.ndarray:
    """Return J0(x) using a Cephes-style approximation (GX-compatible)."""

    x = jnp.asarray(x)
    ax = jnp.abs(x)
    y = x * x
    r = (
        57568490574.0
        + y
        * (
            -13362590354.0
            + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456)))
        )
    )
    s = (
        57568490411.0
        + y
        * (
            1029532985.0
            + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))
        )
    )
    res_small = r / s
    z = 8.0 / jnp.maximum(ax, 1.0e-30)
    y2 = z * z
    xx = ax - 0.785398164
    p = 1.0 + y2 * (
        -0.1098628627e-2
        + y2 * (0.2734510407e-4 + y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6))
    )
    q = -0.1562499995e-1 + y2 * (
        0.1430488765e-3
        + y2 * (-0.6911147651e-5 + y2 * (0.7621095161e-6 + y2 * -0.934945152e-7))
    )
    res_large = jnp.sqrt(0.636619772 / jnp.maximum(ax, 1.0e-30)) * (
        jnp.cos(xx) * p - z * jnp.sin(xx) * q
    )
    out = jnp.where(ax < 8.0, res_small, res_large)
    return jnp.where(jnp.isfinite(out), out, res_small)


def bessel_j1(x: jnp.ndarray) -> jnp.ndarray:
    """Return J1(x) using a Cephes-style approximation (GX-compatible)."""

    x = jnp.asarray(x)
    ax = jnp.abs(x)
    y = x * x
    r = (
        72362614232.0
        + y
        * (
            -7895059235.0
            + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))
        )
    )
    s = (
        144725228442.0
        + y
        * (
            2300535178.0
            + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
        )
    )
    res_small = x * (r / s)
    z = 8.0 / jnp.maximum(ax, 1.0e-30)
    y2 = z * z
    xx = ax - 2.356194491
    p = 1.0 + y2 * (
        0.183105e-2
        + y2 * (-0.3516396496e-4 + y2 * (0.2457520174e-5 + y2 * -0.240337019e-6))
    )
    q = 0.04687499995 + y2 * (
        -0.2002690873e-3
        + y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 + y2 * 0.105787412e-6))
    )
    res_large = jnp.sqrt(0.636619772 / jnp.maximum(ax, 1.0e-30)) * (
        jnp.cos(xx) * p - z * jnp.sin(xx) * q
    )
    res_large = jnp.where(x < 0.0, -res_large, res_large)
    out = jnp.where(ax < 8.0, res_small, res_large)
    return jnp.where(jnp.isfinite(out), out, res_small)


def gx_factorial(m: jnp.ndarray) -> jnp.ndarray:
    """Return GX's single-precision factorial approximation."""

    m_arr = jnp.asarray(m)
    dtype = m_arr.dtype
    exact = jnp.asarray([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0], dtype=dtype)
    m_int = m_arr.astype(jnp.int32)
    m_clamped = jnp.clip(m_int, 0, exact.shape[0] - 1)
    m_safe = jnp.where(m_arr > 0, m_arr, jnp.asarray(1.0, dtype=dtype))
    stirling = jnp.sqrt(2.0 * jnp.asarray(jnp.pi, dtype=dtype) * m_safe) * (m_safe**m_safe) * jnp.exp(
        -m_safe
    ) * (
        1.0
        + 1.0 / (12.0 * m_safe)
        + 1.0 / (288.0 * m_safe * m_safe)
    )
    return jnp.where(m_int <= 6, exact[m_clamped], stirling)


def J_l_all(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Gyroaveraging coefficients matching the GX Laguerre-Hermite convention."""

    if l_max < 0:
        raise ValueError("l_max must be >= 0")
    b = jnp.asarray(b)
    l = jnp.arange(l_max + 1, dtype=b.dtype)
    l_shape = (l_max + 1,) + (1,) * b.ndim
    l = l.reshape(l_shape)
    sign = jnp.where((l % 2) == 0, 1.0, -1.0)
    half_b = 0.5 * b
    half_b_safe = jnp.where(half_b > 0.0, half_b, 1.0)
    log_abs = l * jnp.log(half_b_safe[None, ...]) - gammaln(l + 1.0) - half_b[None, ...]
    Jl = sign * jnp.exp(log_abs)
    zero_mask = (b == 0.0)[None, ...]
    Jl = jnp.where(zero_mask & (l == 0), 1.0, Jl)
    Jl = jnp.where(zero_mask & (l > 0), 0.0, Jl)
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
