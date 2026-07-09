"""JAX-first kernels for internal imported-geometry backends.

These functions are written as pure array transforms so they can be jitted and
used in differentiable geometry pipelines.
"""

from __future__ import annotations

import math

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional runtime dependency
    raise ImportError("spectraxgk.geometry.kernels requires JAX") from exc


@jax.jit
def nperiod_mask(theta: jnp.ndarray, npol: float) -> jnp.ndarray:
    """Return a mask for entries in [-npol*pi, npol*pi]."""

    eps = 1.0e-11
    upper = npol * math.pi + eps
    lower = -npol * math.pi - eps
    return (theta <= upper) & (theta >= lower)


@jax.jit
def nperiod_contract(
    values: jnp.ndarray, theta: jnp.ndarray, npol: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Contract values/theta arrays to the requested npol span."""

    keep = nperiod_mask(theta, npol)
    return values[keep], theta[keep]


@jax.jit
def finite_diff_nonuniform(values: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Second-order finite difference on a non-uniform 1D grid.

    This helper is expressed using vectorized JAX operations so it can be used
    inside differentiable geometry pipelines.
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
        -(
            (2.0 * grid[0] - grid[1] - grid[2])
            / ((grid[0] - grid[1]) * (grid[0] - grid[2]))
        )
        * values[0]
        - ((grid[0] - grid[2]) / ((grid[1] - grid[0]) * (grid[1] - grid[2])))
        * values[1]
        - ((grid[0] - grid[1]) / ((grid[2] - grid[0]) * (grid[2] - grid[1])))
        * values[2]
    )
    right = (
        ((grid[-2] - grid[-1]) / ((grid[-3] - grid[-2]) * (grid[-3] - grid[-1])))
        * values[-3]
        + ((grid[-3] - grid[-1]) / ((grid[-2] - grid[-3]) * (grid[-2] - grid[-1])))
        * values[-2]
        + (
            (2.0 * grid[-1] - grid[-2] - grid[-3])
            / ((grid[-1] - grid[-3]) * (grid[-1] - grid[-2]))
        )
        * values[-1]
    )

    return jnp.concatenate([jnp.asarray([left]), center, jnp.asarray([right])])


def centered_reflected_difference(
    arr: jnp.ndarray, *, axis: str, parity: str = "e"
) -> jnp.ndarray:
    """Centered finite difference with reflected endpoint parity.

    ``axis='l'`` means along a surface (last index for 2D inputs),
    ``axis='r'`` means across surfaces (first index for 2D inputs), and
    ``parity in {'e', 'o'}`` controls endpoint treatment for ``axis='l'``.
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
        raise ValueError("centered_reflected_difference expects 1D or 2D arrays")

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


def _weighted_center_stencil(
    x_left: jnp.ndarray,
    x_center: jnp.ndarray,
    x_right: jnp.ndarray,
    g_left: jnp.ndarray,
    g_center: jnp.ndarray,
    g_right: jnp.ndarray,
) -> jnp.ndarray:
    h1 = g_right - g_center
    h0 = g_center - g_left
    return (
        x_right / h1**2 + x_center * (1.0 / h0**2 - 1.0 / h1**2) - x_left / h0**2
    ) / (1.0 / h1 + 1.0 / h0)


def _weighted_centered_difference_1d_l(
    x: jnp.ndarray, g: jnp.ndarray, *, parity: str
) -> jnp.ndarray:
    out = jnp.zeros_like(x)
    if parity == "e":
        out = out.at[0].set(0.0)
        out = out.at[-1].set(0.0)
    else:
        out = out.at[0].set(
            (4.0 * x[1] - 3.0 * x[0] - x[2]) / (2.0 * (g[1] - g[0]))
        )
        out = out.at[-1].set(
            (-4.0 * x[-2] + 3.0 * x[-1] + x[-3]) / (2.0 * (g[-1] - g[-2]))
        )
    center = _weighted_center_stencil(
        x[:-2], x[1:-1], x[2:], g[:-2], g[1:-1], g[2:]
    )
    return out.at[1:-1].set(center)


def _weighted_centered_difference_1d_r(x: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    out = jnp.zeros_like(x)
    out = out.at[0].set((2.0 * (x[1] - x[0])) / (2.0 * (g[1] - g[0])))
    out = out.at[-1].set((2.0 * (x[-1] - x[-2])) / (2.0 * (g[-1] - g[-2])))
    center = _weighted_center_stencil(
        x[:-2], x[1:-1], x[2:], g[:-2], g[1:-1], g[2:]
    )
    return out.at[1:-1].set(center)


def _weighted_centered_difference_2d_l(
    x: jnp.ndarray, g: jnp.ndarray, *, parity: str
) -> jnp.ndarray:
    out = jnp.zeros_like(x)
    if parity == "e":
        out = out.at[:, 0].set(0.0)
        out = out.at[:, -1].set(0.0)
    else:
        out = out.at[:, 0].set(
            (2.0 * (x[:, 1] - x[:, 0])) / (2.0 * (g[:, 1] - g[:, 0]))
        )
        out = out.at[:, -1].set(
            (2.0 * (x[:, -1] - x[:, -2])) / (2.0 * (g[:, -1] - g[:, -2]))
        )
    center = _weighted_center_stencil(
        x[:, :-2],
        x[:, 1:-1],
        x[:, 2:],
        g[:, :-2],
        g[:, 1:-1],
        g[:, 2:],
    )
    return out.at[:, 1:-1].set(center)


def _weighted_centered_difference_2d_r(x: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    out = jnp.zeros_like(x)
    out = out.at[0, :].set(
        (2.0 * (x[1, :] - x[0, :])) / (2.0 * (g[1, :] - g[0, :]))
    )
    out = out.at[-1, :].set(
        (2.0 * (x[-1, :] - x[-2, :])) / (2.0 * (g[-1, :] - g[-2, :]))
    )
    center = _weighted_center_stencil(
        x[:-2, :],
        x[1:-1, :],
        x[2:, :],
        g[:-2, :],
        g[1:-1, :],
        g[2:, :],
    )
    return out.at[1:-1, :].set(center)


def weighted_centered_difference(
    arr: jnp.ndarray, grid: jnp.ndarray, *, axis: str, parity: str = "e"
) -> jnp.ndarray:
    """Weighted centered finite difference on a nonuniform coordinate grid."""

    x = jnp.asarray(arr)
    g = jnp.asarray(grid)
    if x.shape != g.shape:
        raise ValueError("arr and grid must have identical shapes")

    if x.ndim == 1:
        if axis == "l":
            return _weighted_centered_difference_1d_l(x, g, parity=parity)
        if axis == "r":
            return _weighted_centered_difference_1d_r(x, g)
        raise ValueError("axis must be 'l' or 'r'")

    if x.ndim != 2:
        raise ValueError("weighted_centered_difference expects 1D or 2D arrays")

    if axis == "l":
        return _weighted_centered_difference_2d_l(x, g, parity=parity)

    if axis == "r":
        return _weighted_centered_difference_2d_r(x, g)

    raise ValueError("axis must be 'l' or 'r'")


def extend_nperiod_data(
    arr: jnp.ndarray, nperiod: int, *, istheta: bool = False, parity: str = "e"
) -> jnp.ndarray:
    """Extend a single-period field-line profile to ``nperiod`` periods."""

    base = jnp.asarray(arr)
    out = base
    if nperiod <= 1:
        return out

    if istheta:
        # Use the original single-period array for every reflected extension.
        for i in range(nperiod - 1):
            app = jnp.concatenate(
                (
                    2.0 * math.pi * (i + 1) - base[::-1][1:],
                    2.0 * math.pi * (i + 1) + base[1:],
                )
            )
            out = jnp.concatenate((out, app))
    else:
        # Compute the reflected append block once from the original array.
        if parity == "e":
            app = jnp.concatenate((base[::-1][1:], base[1:]))
        else:
            app = jnp.concatenate((-base[::-1][1:], base[1:]))
        for _ in range(nperiod - 1):
            out = jnp.concatenate((out, app))
    return out


def reflect_and_append(arr: jnp.ndarray, parity: str) -> jnp.ndarray:
    """Append a reflected profile with even or odd endpoint parity."""

    x = jnp.asarray(arr)
    if parity == "e":
        return jnp.concatenate((x[::-1][:-1], x))
    return jnp.concatenate((-x[::-1][:-1], jnp.asarray([0.0], dtype=x.dtype), x[1:]))
