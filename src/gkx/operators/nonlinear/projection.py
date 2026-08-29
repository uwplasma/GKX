"""State projection helpers for nonlinear spectral integrations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gkx.operators.nonlinear.brackets import _complete_hermitian_ky

__all__ = [
    "ShearingCoordinateUpdate",
    "_make_compressed_real_fft_projector",
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
    x0: jnp.ndarray | float,
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
    # ``remap`` is a permutation of the radial modes, so this contraction must
    # return the state's own values, only reordered. An unpinned dot is lowered to
    # TF32 on Ampere and later NVIDIA GPUs, which rounds ``value`` itself to 10
    # mantissa bits (~1e-3 relative here) even though every coefficient is 0 or 1.
    # Pinning keeps the shearing remap a reordering rather than a truncation.
    remapped = jnp.einsum(
        "yts,...ysz->...ytz",
        remap.astype(value.dtype),
        value,
        precision=jax.lax.Precision.HIGHEST,
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
    ny_full: int, two_sided: bool, nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    nyc = ny_full // 2 + 1
    use_hermitian = nyc > 2 and two_sided
    if not use_hermitian:
        return lambda G_state: G_state

    # The conjugate kx ordering stays a host array. These projectors are cached
    # and reused across traces, and a device constant materialized here would
    # belong to whichever trace happened to build it first, escaping that scope
    # the moment a second trace reuses the same grid signature. The ordering is
    # an index reversal, so numpy holds it exactly and each trace gets its own
    # constant at use.
    kx_neg = (
        np.concatenate(([0], np.arange(nx - 1, 0, -1))).astype(np.int32)
        if nx > 1
        else None
    )

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        pos = G_state[..., :nyc, :, :]
        return _complete_hermitian_ky(pos, ny_full, nx, kx_neg)

    return project


_TRACED_KY_AXIS_MESSAGE = (
    "the Hermitian projector cannot read a traced ky axis: it needs the axis "
    "layout -- length, and whether the negative-ky half is carried -- which is "
    "grid topology and has no derivative. Pass a concrete ky axis, or build "
    "the projector from the layout directly with "
    "_make_compressed_real_fft_projector, which the compressed real-FFT path "
    "uses so that a cache built inside a trace still projects."
)


def _hermitian_ky_axis_layout(ky_vals: Any) -> tuple[int, bool]:
    """Return ``(ny_full, two_sided)`` for one ky axis, read on the host.

    The projector never uses a wavenumber, only the layout of the axis it acts
    on: how many rows there are, and whether the conjugate half is among them.
    Read that off the stored array rather than a ``jnp`` copy of it -- inside a
    trace every ``jnp`` call stages out, so a round trip yields a tracer whose
    values cannot be inspected at all.
    """

    if isinstance(ky_vals, jax.core.Tracer):
        raise ValueError(_TRACED_KY_AXIS_MESSAGE)
    host = np.asarray(ky_vals, dtype=float).reshape(-1)
    return int(host.size), bool(np.any(host < 0.0))


def _make_hermitian_projector(
    ky_vals: np.ndarray, nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a stable projector for one full-ky grid signature."""

    ny_full, two_sided = _hermitian_ky_axis_layout(ky_vals)
    return _cached_hermitian_projector(ny_full, two_sided, int(nx))


def _make_compressed_real_fft_projector(
    *, ny_full: int, nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the Hermitian projector for a compressed real-FFT ky axis.

    A compressed run is on a full two-sided ky axis by construction: the
    bracket transforms the first ``ny_full // 2 + 1`` rows and rebuilds the
    rest with ``_complete_hermitian_ky``, which reads that layout off shapes
    alone. Taking the same route here keeps the projector buildable inside a
    trace. The cache's own ``ky`` is ``rho_star * grid.ky``, so it is a tracer
    whenever the cache is built inside one, and asking it for its sign pattern
    refused every compressed nonlinear gradient -- for a fact that is fixed by
    the grid and carries no derivative either way.
    """

    return _cached_hermitian_projector(int(ny_full), True, int(nx))


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
