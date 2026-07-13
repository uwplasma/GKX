"""State projection helpers for nonlinear spectral integrations."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ShearingCoordinateUpdate",
    "_make_fixed_mode_projector",
    "_make_hermitian_projector",
    "_make_nonlinear_state_projector",
    "advance_shearing_coordinates",
]


class ShearingCoordinateUpdate(NamedTuple):
    """State and spectral coordinates after one equilibrium-flow-shear update."""

    state: jnp.ndarray
    effective_kx: jnp.ndarray
    phase: jnp.ndarray
    cumulative_mode_shift: jnp.ndarray
    incremental_mode_shift: jnp.ndarray


def _round_half_away_from_zero(value: jnp.ndarray) -> jnp.ndarray:
    """Match the C99 nearest-mode convention used at remap boundaries."""

    return jnp.sign(value) * jnp.floor(jnp.abs(value) + 0.5)


def advance_shearing_coordinates(
    state: jnp.ndarray,
    *,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    x0: float,
    shear_rate: jnp.ndarray | float,
    previous_time: jnp.ndarray | float,
    time: jnp.ndarray | float,
    dealias_mask: jnp.ndarray | None = None,
) -> ShearingCoordinateUpdate:
    r"""Advance a Fourier state in continuously shearing coordinates.

    For equilibrium :math:`E\times B` shear, each shearing wave follows

    .. math:: k_x^*(t) = k_x(0) - k_y \gamma_E t.

    The integer part of this displacement remaps Fourier amplitudes to the
    nearest radial mode. The sub-grid remainder is returned as the real-space
    phase ``exp(1j * delta_kx * x)`` and in ``effective_kx``. Integer remap
    decisions are treated as locally constant under autodiff, while the
    continuous wavenumber and phase retain their exact tangent away from the
    measure-zero crossing events.

    ``state`` uses ``(..., ky, kx, z)`` ordering. Modes shifted beyond the
    supplied two-thirds mask are discarded rather than wrapped into the
    resolved band.
    """

    value = jnp.asarray(state)
    kx_values = jnp.asarray(kx)
    ky_values = jnp.asarray(ky)
    if value.ndim < 3:
        raise ValueError("state must use (..., ky, kx, z) ordering")
    if kx_values.ndim != 1 or ky_values.ndim != 1:
        raise ValueError("kx and ky must be one-dimensional")
    if value.shape[-2] != kx_values.size or value.shape[-3] != ky_values.size:
        raise ValueError("state ky/kx axes must match the supplied grids")
    if not isinstance(x0, jax.core.Tracer) and float(np.asarray(x0)) <= 0.0:
        raise ValueError("x0 must be positive")
    if dealias_mask is not None and tuple(dealias_mask.shape) != (
        int(ky_values.size),
        int(kx_values.size),
    ):
        raise ValueError("dealias_mask must have shape (ky, kx)")

    real_dtype = jnp.real(jnp.empty((), dtype=value.dtype)).dtype
    radial_scale = jnp.asarray(x0, dtype=real_dtype)
    radial_spacing = 1.0 / radial_scale
    rate = jnp.asarray(shear_rate, dtype=real_dtype)
    old_time = jnp.asarray(previous_time, dtype=real_dtype)
    new_time = jnp.asarray(time, dtype=real_dtype)
    ky_real = jnp.asarray(ky_values, dtype=real_dtype)

    def cumulative_shift(at_time: jnp.ndarray) -> jnp.ndarray:
        continuous = -ky_real * rate * at_time / radial_spacing
        rounded = _round_half_away_from_zero(continuous).astype(jnp.int32)
        return jax.lax.stop_gradient(rounded)

    old_shift = cumulative_shift(old_time)
    new_shift = cumulative_shift(new_time)
    incremental_shift = new_shift - old_shift

    radial_modes = jnp.rint(kx_values / radial_spacing).astype(jnp.int32)
    target_modes = radial_modes[None, :, None]
    source_modes = target_modes - incremental_shift[:, None, None]
    remap = source_modes == radial_modes[None, None, :]
    remapped = jnp.einsum(
        "yts,...ysz->...ytz", remap.astype(value.dtype), value
    )
    if dealias_mask is not None:
        mask_shape = (1,) * (remapped.ndim - 3) + tuple(dealias_mask.shape) + (1,)
        remapped = remapped * jnp.reshape(
            jnp.asarray(dealias_mask, dtype=value.dtype), mask_shape
        )

    continuous_kx_shift = -ky_real * rate * new_time
    residual_kx = continuous_kx_shift - radial_spacing * new_shift
    effective_kx = kx_values[None, :] + residual_kx[:, None]
    radial_coordinate = (
        2.0
        * jnp.pi
        * radial_scale
        * jnp.arange(kx_values.size, dtype=real_dtype)
        / jnp.asarray(kx_values.size, dtype=real_dtype)
    )
    phase = jnp.exp(
        jnp.asarray(1j, dtype=jnp.result_type(value, jnp.complex64))
        * residual_kx[:, None]
        * radial_coordinate[None, :]
    )
    return ShearingCoordinateUpdate(
        state=remapped,
        effective_kx=effective_kx,
        phase=phase,
        cumulative_mode_shift=new_shift,
        incremental_mode_shift=incremental_shift,
    )


@lru_cache(maxsize=32)
def _cached_hermitian_projector(
    ky_vals: tuple[float, ...], nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    ny_full = len(ky_vals)
    nyc = ny_full // 2 + 1
    use_hermitian = nyc > 2 and any(value < 0.0 for value in ky_vals)
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


def _make_hermitian_projector(
    ky_vals: np.ndarray, nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a stable projector for one full-ky grid signature."""

    ky_key = tuple(float(value) for value in np.asarray(ky_vals, dtype=float))
    return _cached_hermitian_projector(ky_key, int(nx))


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
