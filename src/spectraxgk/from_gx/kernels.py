"""JAX-first kernels ported from GX helper logic.

These functions are written as pure array transforms so they can be jitted and
used in differentiable geometry pipelines.
"""

from __future__ import annotations

import math

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional runtime dependency
    raise ImportError("spectraxgk.from_gx.kernels requires JAX") from exc


@jax.jit
def nperiod_mask(theta: jnp.ndarray, npol: float) -> jnp.ndarray:
    """Return a mask for entries in [-npol*pi, npol*pi]."""

    eps = 1.0e-11
    upper = npol * math.pi + eps
    lower = -npol * math.pi - eps
    return (theta <= upper) & (theta >= lower)


@jax.jit
def nperiod_contract(values: jnp.ndarray, theta: jnp.ndarray, npol: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Contract values/theta arrays to the requested npol span."""

    keep = nperiod_mask(theta, npol)
    return values[keep], theta[keep]


@jax.jit
def finite_diff_nonuniform(values: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Second-order finite difference on a non-uniform 1D grid.

    This mirrors the intent of GX's ``dermv`` helper but is expressed using
    vectorized JAX operations.
    """

    n = values.shape[0]
    if n < 3:
        return jnp.zeros_like(values)

    x0 = grid[:-2]
    x1 = grid[1:-1]
    x2 = grid[2:]
    y0 = values[:-2]
    y1 = values[1:-1]
    y2 = values[2:]

    d01 = x1 - x0
    d12 = x2 - x1
    d02 = x2 - x0

    w0 = -d12 / (d01 * d02)
    w1 = (d12 - d01) / (d01 * d12)
    w2 = d01 / (d12 * d02)

    center = w0 * y0 + w1 * y1 + w2 * y2

    # One-sided second-order edges.
    left = (
        -((2.0 * grid[0] - grid[1] - grid[2]) / ((grid[0] - grid[1]) * (grid[0] - grid[2]))) * values[0]
        -((grid[0] - grid[2]) / ((grid[1] - grid[0]) * (grid[1] - grid[2]))) * values[1]
        -((grid[0] - grid[1]) / ((grid[2] - grid[0]) * (grid[2] - grid[1]))) * values[2]
    )
    right = (
        ((grid[-2] - grid[-1]) / ((grid[-3] - grid[-2]) * (grid[-3] - grid[-1]))) * values[-3]
        +((grid[-3] - grid[-1]) / ((grid[-2] - grid[-3]) * (grid[-2] - grid[-1]))) * values[-2]
        +((2.0 * grid[-1] - grid[-2] - grid[-3]) / ((grid[-1] - grid[-3]) * (grid[-1] - grid[-2]))) * values[-1]
    )

    return jnp.concatenate([jnp.asarray([left]), center, jnp.asarray([right])])


def gx_derm(arr: jnp.ndarray, *, axis: str, parity: str = "e") -> jnp.ndarray:
    """Port of GX Miller ``derm`` finite difference helper.

    Parameters mirror the original utility:
    - axis='l' means along a surface (last index for 2D inputs)
    - axis='r' means across surfaces (first index for 2D inputs)
    - parity in {'e','o'} controls endpoint treatment for axis='l'
    """

    x = jnp.asarray(arr)
    if x.ndim == 1:
        if axis == "l":
            out = jnp.zeros_like(x)
            center = x[2:] - x[:-2]
            out = out.at[1:-1].set(center)
            if parity == "o":
                out = out.at[0].set(2.0 * (x[1] - x[0]))
                out = out.at[-1].set(2.0 * (x[-1] - x[-2]))
            return out
        if axis == "r":
            out = jnp.zeros_like(x)
            center = x[2:] - x[:-2]
            out = out.at[1:-1].set(center)
            out = out.at[0].set(2.0 * (x[1] - x[0]))
            out = out.at[-1].set(2.0 * (x[-1] - x[-2]))
            return out
        raise ValueError("axis must be 'l' or 'r'")

    if x.ndim != 2:
        raise ValueError("gx_derm expects 1D or 2D arrays")

    out = jnp.zeros_like(x)
    if axis == "r":
        out = out.at[1:-1, :].set(x[2:, :] - x[:-2, :])
        out = out.at[0, :].set(2.0 * (x[1, :] - x[0, :]))
        out = out.at[-1, :].set(2.0 * (x[-1, :] - x[-2, :]))
        return out
    if axis == "l":
        out = out.at[:, 1:-1].set(x[:, 2:] - x[:, :-2])
        if parity == "o":
            out = out.at[:, 0].set(2.0 * (x[:, 1] - x[:, 0]))
            out = out.at[:, -1].set(2.0 * (x[:, -1] - x[:, -2]))
        return out
    raise ValueError("axis must be 'l' or 'r'")


def gx_dermv(arr: jnp.ndarray, grid: jnp.ndarray, *, axis: str, parity: str = "e") -> jnp.ndarray:
    """Port of GX Miller ``dermv`` weighted finite difference helper."""

    x = jnp.asarray(arr)
    g = jnp.asarray(grid)
    if x.shape != g.shape:
        raise ValueError("arr and grid must have identical shapes")

    if x.ndim == 1:
        out = jnp.zeros_like(x)
        if axis == "l":
            if parity == "e":
                out = out.at[0].set(0.0)
                out = out.at[-1].set(0.0)
            else:
                out = out.at[0].set((4.0 * x[1] - 3.0 * x[0] - x[2]) / (2.0 * (g[1] - g[0])))
                out = out.at[-1].set((-4.0 * x[-2] + 3.0 * x[-1] + x[-3]) / (2.0 * (g[-1] - g[-2])))

            h1 = g[2:] - g[1:-1]
            h0 = g[1:-1] - g[:-2]
            center = (x[2:] / h1**2 + x[1:-1] * (1.0 / h0**2 - 1.0 / h1**2) - x[:-2] / h0**2) / (1.0 / h1 + 1.0 / h0)
            out = out.at[1:-1].set(center)
            return out

        if axis == "r":
            out = out.at[0].set((2.0 * (x[1] - x[0])) / (2.0 * (g[1] - g[0])))
            out = out.at[-1].set((2.0 * (x[-1] - x[-2])) / (2.0 * (g[-1] - g[-2])))
            h1 = g[2:] - g[1:-1]
            h0 = g[1:-1] - g[:-2]
            center = (x[2:] / h1**2 + x[1:-1] * (1.0 / h0**2 - 1.0 / h1**2) - x[:-2] / h0**2) / (1.0 / h1 + 1.0 / h0)
            out = out.at[1:-1].set(center)
            return out

        raise ValueError("axis must be 'l' or 'r'")

    if x.ndim != 2:
        raise ValueError("gx_dermv expects 1D or 2D arrays")

    if axis == "l":
        out = jnp.zeros_like(x)
        if parity == "e":
            out = out.at[:, 0].set(0.0)
            out = out.at[:, -1].set(0.0)
        else:
            out = out.at[:, 0].set((2.0 * (x[:, 1] - x[:, 0])) / (2.0 * (g[:, 1] - g[:, 0])))
            out = out.at[:, -1].set((2.0 * (x[:, -1] - x[:, -2])) / (2.0 * (g[:, -1] - g[:, -2])))

        h1 = g[:, 2:] - g[:, 1:-1]
        h0 = g[:, 1:-1] - g[:, :-2]
        center = (x[:, 2:] / h1**2 + x[:, 1:-1] * (1.0 / h0**2 - 1.0 / h1**2) - x[:, :-2] / h0**2) / (1.0 / h1 + 1.0 / h0)
        out = out.at[:, 1:-1].set(center)
        return out

    if axis == "r":
        out = jnp.zeros_like(x)
        out = out.at[0, :].set((2.0 * (x[1, :] - x[0, :])) / (2.0 * (g[1, :] - g[0, :])))
        out = out.at[-1, :].set((2.0 * (x[-1, :] - x[-2, :])) / (2.0 * (g[-1, :] - g[-2, :])))
        h1 = g[2:, :] - g[1:-1, :]
        h0 = g[1:-1, :] - g[:-2, :]
        center = (x[2:, :] / h1**2 + x[1:-1, :] * (1.0 / h0**2 - 1.0 / h1**2) - x[:-2, :] / h0**2) / (1.0 / h1 + 1.0 / h0)
        out = out.at[1:-1, :].set(center)
        return out

    raise ValueError("axis must be 'l' or 'r'")


def gx_nperiod_data_extend(arr: jnp.ndarray, nperiod: int, *, istheta: bool = False, parity: str = "e") -> jnp.ndarray:
    """Port of GX Miller ``nperiod_data_extend`` helper."""

    base = jnp.asarray(arr)
    out = base
    if nperiod <= 1:
        return out

    if istheta:
        # GX uses a fixed arr_dum (original array), shifting by 2*pi*(i+1).
        for i in range(nperiod - 1):
            app = jnp.concatenate(
                (
                    2.0 * math.pi * (i + 1) - base[::-1][1:],
                    2.0 * math.pi * (i + 1) + base[1:],
                )
            )
            out = jnp.concatenate((out, app))
    else:
        # GX computes arr_app once from the original array and repeats it.
        if parity == "e":
            app = jnp.concatenate((base[::-1][1:], base[1:]))
        else:
            app = jnp.concatenate((-base[::-1][1:], base[1:]))
        for _ in range(nperiod - 1):
            out = jnp.concatenate((out, app))
    return out


def gx_reflect_n_append(arr: jnp.ndarray, parity: str) -> jnp.ndarray:
    """Port of GX Miller ``reflect_n_append`` helper."""

    x = jnp.asarray(arr)
    if parity == "e":
        return jnp.concatenate((x[::-1][:-1], x))
    return jnp.concatenate((-x[::-1][:-1], jnp.asarray([0.0], dtype=x.dtype), x[1:]))
