import jax.numpy as jnp
import numpy as np

from gkx.core.velocity import hermite_normed, laguerre, laguerre_transform


def hermite_check(n_max: int = 6, x_max: float = 6.0, nx: int = 8001) -> None:
    x = jnp.linspace(-x_max, x_max, nx)
    dx = x[1] - x[0]
    h = hermite_normed(x, n_max)
    w = jnp.exp(-x * x)
    gram = jnp.einsum("ix,jx,x->ij", h, h, w) * dx
    off_diag = gram - jnp.eye(n_max + 1)
    print("Hermite max off-diagonal:", jnp.max(jnp.abs(off_diag)))


def laguerre_check(l_max: int = 6, x_max: float = 40.0, nx: int = 20001) -> None:
    x = jnp.linspace(0.0, x_max, nx)
    dx = x[1] - x[0]
    lag = laguerre(x, l_max)
    w = jnp.exp(-x)
    gram = jnp.einsum("ix,jx,x->ij", lag, lag, w) * dx
    off_diag = gram - jnp.eye(l_max + 1)
    print("Laguerre max off-diagonal:", jnp.max(jnp.abs(off_diag)))


def laguerre_transform_check(nl_values: tuple[int, ...] = (4, 8, 16, 32, 64)) -> None:
    """Round-trip accuracy and conditioning of the quadrature transform pair.

    The two checks above test the analytic bases; this one tests the discrete
    transform the solver actually runs on, which is where precision was lost.
    """
    for nl in nl_values:
        to_grid, to_spectral, _ = laguerre_transform(nl)
        error = np.abs(to_grid @ to_spectral - np.eye(nl)).max()
        print(
            f"Laguerre transform nl={nl:3d}: round-trip {error:.2e}"
            f"  cond(to_grid) {np.linalg.cond(to_grid):.2e}"
        )


if __name__ == "__main__":
    hermite_check()
    laguerre_check()
    laguerre_transform_check()
