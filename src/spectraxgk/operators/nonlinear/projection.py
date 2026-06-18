"""State projection helpers for nonlinear spectral integrations."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

__all__ = [
    "_make_fixed_mode_projector",
    "_make_hermitian_projector",
    "_make_nonlinear_state_projector",
]


def _make_hermitian_projector(
    ky_vals: np.ndarray, nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Project full-ky states onto the compressed real-FFT Hermitian manifold."""

    ny_full = int(ky_vals.size)
    nyc = ny_full // 2 + 1
    use_hermitian = nyc > 2 and bool(np.any(np.asarray(ky_vals) < 0.0))
    if not use_hermitian:
        return lambda G_state: G_state

    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    if nx > 1:
        kx_neg = jnp.asarray(
            np.concatenate(([0], np.arange(nx - 1, 0, -1))), dtype=jnp.int32
        )
    else:
        kx_neg = None

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        pos = G_state[..., :nyc, :, :]
        neg = jnp.conj(pos[..., 1:neg_hi, :, :])[..., ::-1, :, :]
        if kx_neg is not None:
            neg = neg[..., kx_neg, :]
        return jnp.concatenate([pos, neg], axis=-3)

    return project


def _make_nonlinear_state_projector(
    fixed_state: jnp.ndarray | None,
    *,
    ky_vals: np.ndarray,
    nx: int,
    compressed_real_fft: bool,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compose fixed-mode and Hermitian projections for nonlinear state scans."""

    fixed_projector = _make_fixed_mode_projector(
        fixed_state,
        ky_index=fixed_mode_ky_index,
        kx_index=fixed_mode_kx_index,
    )
    hermitian_projector = (
        _make_hermitian_projector(np.asarray(ky_vals), nx=int(nx))
        if compressed_real_fft
        else (lambda G_state: G_state)
    )

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        if fixed_projector is not None:
            G_state = fixed_projector(G_state)
        return hermitian_projector(G_state)

    return project


def _make_fixed_mode_projector(
    fixed_state: jnp.ndarray | None,
    *,
    ky_index: int | None,
    kx_index: int | None,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Return a projector that keeps one Fourier mode equal to ``fixed_state``."""

    if fixed_state is None or ky_index is None or kx_index is None:
        return None
    ky_i = int(ky_index)
    kx_i = int(kx_index)
    fixed_block = jnp.asarray(fixed_state)[..., ky_i : ky_i + 1, kx_i : kx_i + 1, :]

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        return G_state.at[..., ky_i : ky_i + 1, kx_i : kx_i + 1, :].set(
            fixed_block
        )

    return project
