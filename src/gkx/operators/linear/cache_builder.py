"""Geometry-dependent construction of :class:`LinearCache`."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.core_velocity import _gyro_bessel_factors, laguerre_transform
from gkx.core_grid import SpectralGrid
from gkx.operators.linear.cache_arrays import (
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_low_rank_moment_cache_arrays,
)
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.linked import (
    _build_linked_end_damping_profile,
    _build_linked_fft_maps,
)
from gkx.operators.linear.params import LinearParams, _is_tracer, _x64_enabled


def _array_namespace(*values: Any) -> Any:
    """``numpy`` while every operand is concrete, ``jax.numpy`` under a trace.

    Cache construction is a one-shot host computation. Nothing built here is
    differentiated -- the differentiable geometry route takes its derivatives
    from the implicit eigensolve VJP, not from cache assembly -- and the build
    already reads concrete geometry scalars to decide whether a twist-shift
    cache is resolvable at all. Under op-by-op dispatch, though, every ``jnp``
    primitive on this path is its own XLA compilation, so a cold build spent
    seconds compiling one kernel per multiply. ``numpy`` evaluates the same
    IEEE-754 arithmetic on the host with no compiler in the path, and
    :func:`_pack_linear_cache` moves the finished arrays to the device in one
    pass. A traced operand still selects ``jax.numpy``, so a cache built inside
    a trace stages out exactly as it did before.

    Only operations both libraries define identically are routed to the host:
    arithmetic, ``sqrt``, comparisons and casts are pinned by IEEE-754, while
    ``exp`` and ``gammaln`` are not and stay on the device.
    """

    return jnp if any(_is_tracer(value) for value in values) else np


def _as_device_dtype(xp: Any, value: Any, real_dtype: Any) -> Any:
    """``asarray`` reproducing ``jnp.asarray``'s dtype rule on the host.

    ``jnp.asarray`` gives a Python scalar JAX's default float dtype and narrows
    a wider float array when x64 is off, while ``np.asarray`` keeps float64
    throughout. The two agree in double precision and disagree in single, where
    letting the host contract a float32 metric in float64 would move the cache
    by an ulp. Pin the width here instead.
    """

    if xp is not np:
        return xp.asarray(value)
    array = np.asarray(value)
    narrow = np.dtype(real_dtype)
    if (
        np.issubdtype(array.dtype, np.floating)
        and array.dtype.itemsize > narrow.itemsize
    ):
        return array.astype(narrow)
    return array


def _fft_frequencies(xp: Any, n: int, spacing: Any, dtype: Any) -> Any:
    """FFT sample frequencies, matching :func:`jax.numpy.fft.fftfreq` exactly.

    ``numpy.fft.fftfreq`` scales the index by the reciprocal of ``n * d`` where
    JAX divides by it, which rounds differently in single precision. Spelling
    the shared definition out here keeps the host and device routes on the same
    values in both precisions.
    """

    index = xp.arange(n, dtype=dtype)
    wrapped = (index + n // 2) % n - n // 2
    return wrapped.astype(dtype) / xp.asarray(spacing * n, dtype=dtype)


@dataclass(frozen=True)
class _GridCacheArrays:
    """Grid-derived cache arrays, on the host unless the grid is traced."""

    real_dtype: Any
    dz: Any
    kz: Any
    rho_star: Any
    kx_raw: Any
    ky_raw: Any
    kx_eff: Any
    ky_eff: Any
    kx_grid: Any
    ky_grid: Any
    dealias_mask: Any
    theta: Any


@dataclass(frozen=True)
class _GeometryCacheArrays:
    """Sampled geometry profiles, on the host unless the geometry is traced.

    ``bmag_raw`` is the profile as the geometry returned it and ``bmag`` is the
    same profile cast to the cache dtype. The metric contraction below consumes
    the raw profile because :meth:`_SpectralGeometryMixin.k_perp2` does, and an
    imported geometry may carry a wider dtype than the cache stores.
    """

    geom_data: Any
    gds2: Any
    gds21: Any
    gds22: Any
    gds22_arr: Any
    bmag_raw: Any
    bmag: Any
    bgrad: Any
    jacobian: Any
    cv: Any
    gb: Any
    cv0: Any
    gb0: Any


@dataclass(frozen=True)
class _TwistShiftCachePolicy:
    boundary: str
    use_twist_shift: bool
    use_ntft: bool
    y0: float
    shat_arr: Any
    x0_eff: float
    jtwist: int
    kxfac_val: float
    kx_eff: Any
    kx_grid: Any


@dataclass(frozen=True)
class _LaguerreGyroCache:
    b: Any
    Jl: Any
    JlB: Any
    laguerre_to_grid: Any
    laguerre_to_spectral: Any
    laguerre_roots: Any
    laguerre_j0: Any
    laguerre_j1_over_alpha: Any


@dataclass(frozen=True)
class _KxLinkCache:
    kx_link_plus: Any
    kx_link_minus: Any
    kx_link_mask_plus: Any
    kx_link_mask_minus: Any
    jtwist: int


@dataclass(frozen=True)
class _LinkedFFTCache:
    linked_indices: tuple[np.ndarray, ...]
    linked_kz: tuple[np.ndarray, ...]
    linked_inverse_permutation: np.ndarray
    linked_full_cover: bool
    linked_gather_map: np.ndarray
    linked_gather_mask: np.ndarray
    linked_use_gather: bool


def _build_grid_cache_arrays(
    grid: SpectralGrid, params: LinearParams
) -> _GridCacheArrays:
    xp = _array_namespace(
        grid.z, grid.kx, grid.ky, grid.kx_grid, grid.ky_grid, params.rho_star
    )
    real_dtype = jnp.float64 if _x64_enabled() else jnp.float32
    z_raw = xp.asarray(grid.z)
    dz = xp.asarray(z_raw[1] - z_raw[0], dtype=real_dtype)
    kz = xp.asarray(
        2.0 * np.pi * _fft_frequencies(xp, int(grid.z.size), dz, real_dtype),
        dtype=real_dtype,
    )
    rho_star = xp.asarray(params.rho_star, dtype=real_dtype)
    kx_raw = xp.asarray(grid.kx, dtype=real_dtype)
    ky_raw = xp.asarray(grid.ky, dtype=real_dtype)
    return _GridCacheArrays(
        real_dtype=real_dtype,
        dz=dz,
        kz=kz,
        rho_star=rho_star,
        kx_raw=kx_raw,
        ky_raw=ky_raw,
        kx_eff=rho_star * kx_raw,
        ky_eff=rho_star * ky_raw,
        kx_grid=xp.asarray(grid.kx_grid, dtype=real_dtype) * rho_star,
        ky_grid=xp.asarray(grid.ky_grid, dtype=real_dtype) * rho_star,
        dealias_mask=xp.asarray(grid.dealias_mask, dtype=bool),
        theta=xp.asarray(z_raw, dtype=real_dtype),
    )


def _build_geometry_cache_arrays(
    geom: FluxTubeGeometryLike,
    *,
    theta: Any,
    real_dtype: Any,
) -> _GeometryCacheArrays:
    geom_data = ensure_flux_tube_geometry_data(geom, theta)
    gds2, gds21, gds22 = geom_data.metric_coeffs(theta)
    cv, gb, cv0, gb0 = geom_data.drift_coeffs(theta)
    bmag_raw = geom_data.bmag(theta)
    xp = _array_namespace(theta, gds2, cv, bmag_raw, geom_data.s_hat)

    def host(value: Any) -> Any:
        return _as_device_dtype(xp, value, real_dtype)

    gds2 = host(gds2)
    gds21 = host(gds21)
    gds22 = host(gds22)
    gds22_arr = gds22 if gds22.ndim else xp.full_like(theta, gds22)
    bmag_raw = host(bmag_raw)
    return _GeometryCacheArrays(
        geom_data=geom_data,
        gds2=gds2,
        gds21=gds21,
        gds22=gds22,
        gds22_arr=gds22_arr,
        bmag_raw=bmag_raw,
        bmag=bmag_raw.astype(real_dtype),
        bgrad=host(geom_data.bgrad(theta)).astype(real_dtype),
        jacobian=host(geom_data.jacobian(theta)).astype(real_dtype),
        cv=host(cv),
        gb=host(gb),
        cv0=host(cv0),
        gb0=host(gb0),
    )


_TRACED_SHEAR_TWIST_SHIFT_MESSAGE = (
    "differentiating with respect to magnetic shear is not supported with "
    "twist-shift boundaries: the parallel link map is the integer kx-index "
    "shift jtwist = round(2 * s_hat * gds21 / gds22), and an integer read off "
    "a traced shear has no derivative. Every other parameter differentiates "
    "normally under boundary='linked' -- keep the geometry concrete (build or "
    "sample it outside the trace) instead of routing s_hat, gds21, or gds22 "
    "through jit/grad."
)


def _default_twist_y0(grid: SpectralGrid) -> float:
    y0 = getattr(grid, "y0", None)
    if y0 is not None:
        return float(y0)
    if grid.ky.size > 1:
        return float(1.0 / float(grid.ky[1] - grid.ky[0]))
    return 1.0


def _host_edge_scalar(value: Any, *, dtype: Any = None) -> float | None:
    """Host float for a scalar, or a profile's edge sample; ``None`` if traced.

    Read the stored attribute, never a ``jnp`` copy or a ``jnp`` slice of it.
    Inside a trace every ``jnp`` call stages out, so ``jnp.asarray(0.8)`` and
    ``profile[0]`` both return tracers even when the operand is a host
    constant. Asking after such a round trip reports every concrete geometry
    as traced and refuses twist-shift caches that are perfectly resolvable;
    ``numpy`` answers from the buffer and only genuinely traced geometry
    reaches the refusal. ``dtype`` reproduces the cast the round trip used to
    apply, so the value read here is the value the cache was built from.
    """

    if _is_tracer(value):
        return None
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    return None if arr.size == 0 else float(arr[0])


def _twist_shift_geometric_factor(
    shat: float, *, gds21: Any, gds22: Any
) -> float | None:
    gds22_min = _host_edge_scalar(gds22)
    if gds22_min == 0.0:
        return 0.0
    gds21_edge = _host_edge_scalar(gds21)
    if gds22_min is None or gds21_edge is None:
        return None
    return float(2.0 * shat * gds21_edge / gds22_min)


def _jtwist_and_x0_target(
    grid: SpectralGrid,
    *,
    y0: float,
    x0_eff: float,
    twist_shift_geo_fac: float,
) -> tuple[int, float]:
    jtwist_val = getattr(grid, "jtwist", None)
    if twist_shift_geo_fac == 0.0:
        return int(jtwist_val) if jtwist_val is not None else 1, x0_eff
    jtwist = (
        int(jtwist_val)
        if jtwist_val is not None
        else int(np.round(twist_shift_geo_fac))
    )
    jtwist = 1 if jtwist == 0 else jtwist
    return jtwist, float(y0) * abs(jtwist) / abs(twist_shift_geo_fac)


def _scaled_twist_shift_kx(
    grid: SpectralGrid,
    *,
    use_ntft: bool,
    x0_eff: float,
    x0_target: float,
    kx_eff: Any,
    kx_grid: Any,
) -> tuple[Any, Any, float]:
    grid_x0 = float(getattr(grid, "x0", x0_eff))
    if use_ntft:
        if grid_x0 != 0.0:
            kx_eff = kx_eff * (grid_x0 / float(x0_eff))
        return kx_eff, kx_grid, x0_eff
    if x0_target != 0.0 and x0_target != x0_eff:
        scale = float(x0_eff) / float(x0_target)
        return kx_eff * scale, kx_grid * scale, x0_target
    return kx_eff, kx_grid, x0_eff


def _resolve_twist_shift_policy(
    grid: SpectralGrid,
    geom_data: Any,
    *,
    gds21: Any,
    gds22: Any,
    kx_eff: Any,
    kx_grid: Any,
) -> _TwistShiftCachePolicy:
    boundary = str(getattr(grid, "boundary", "periodic")).lower()
    use_twist_shift = boundary in {"linked", "fix aspect", "continuous drifts"}
    use_ntft = bool(getattr(grid, "non_twist", False))
    y0 = _default_twist_y0(grid)
    xp = _array_namespace(kx_eff, kx_grid, geom_data.s_hat)
    shat_arr = xp.asarray(geom_data.s_hat, dtype=kx_eff.dtype)
    shat_host = _host_edge_scalar(geom_data.s_hat, dtype=kx_eff.dtype)
    x0_eff = float(getattr(grid, "x0", 1.0))
    jtwist = 0
    x0_target = x0_eff
    if use_twist_shift:
        twist_shift_geo_fac = (
            None
            if shat_host is None
            else _twist_shift_geometric_factor(shat_host, gds21=gds21, gds22=gds22)
        )
        if twist_shift_geo_fac is None:
            raise ValueError(_TRACED_SHEAR_TWIST_SHIFT_MESSAGE)
        jtwist, x0_target = _jtwist_and_x0_target(
            grid, y0=y0, x0_eff=x0_eff, twist_shift_geo_fac=twist_shift_geo_fac
        )
        if use_ntft and twist_shift_geo_fac != 0.0:
            x0_eff = x0_target
        kx_eff, kx_grid, x0_eff = _scaled_twist_shift_kx(
            grid,
            use_ntft=use_ntft,
            x0_eff=x0_eff,
            x0_target=x0_target,
            kx_eff=kx_eff,
            kx_grid=kx_grid,
        )
    kxfac_val = float(getattr(grid, "kxfac", 1.0))
    return _TwistShiftCachePolicy(
        boundary=boundary,
        use_twist_shift=use_twist_shift,
        use_ntft=use_ntft,
        y0=float(y0),
        shat_arr=shat_arr,
        x0_eff=x0_eff,
        jtwist=jtwist,
        kxfac_val=kxfac_val,
        kx_eff=kx_eff,
        kx_grid=kx_grid,
    )


def _build_ntft_kperp_and_drift_arrays(
    grid: SpectralGrid,
    geom_data: Any,
    *,
    kx_eff: Any,
    ky_eff: Any,
    ky_raw: Any,
    rho_star: Any,
    gds2: Any,
    gds21: Any,
    gds22_arr: Any,
    bmag: Any,
    cv: Any,
    gb: Any,
    cv0: Any,
    gb0: Any,
    shat_arr: Any,
    x0_eff: float,
    kperp2_bmag: bool,
) -> tuple[Any, Any, Any, Any]:
    xp = _array_namespace(kx_eff, ky_eff, gds2, gds21, gds22_arr, shat_arr)
    ftwist = (geom_data.s_hat * gds21 / gds22_arr).astype(kx_eff.dtype)
    delta = xp.asarray(0.01313, dtype=kx_eff.dtype)
    ftwist_next = xp.roll(ftwist, -1)
    mid_idx = int(grid.z.size // 2)
    mid_next = (mid_idx + 1) % grid.z.size
    ftwist_mid = ftwist[mid_idx]
    ftwist_mid_next = ftwist[mid_next]
    m0 = -xp.rint(
        float(x0_eff)
        * ky_raw[:, None]
        * ((1.0 - delta) * ftwist[None, :] + delta * ftwist_next[None, :])
    ) + xp.rint(
        float(x0_eff)
        * ky_raw[:, None]
        * ((1.0 - delta) * ftwist_mid + delta * ftwist_mid_next)
    )
    m0 = m0.astype(kx_eff.dtype)
    shat_inv = 1.0 / shat_arr
    delta_kx = ky_eff[:, None] * ftwist[None, :] + (rho_star * m0 / float(x0_eff))
    term_ky = ky_eff[:, None, None] ** 2 * (
        gds2[None, None, :]
        - 2.0 * ftwist[None, None, :] * gds21[None, None, :] * shat_inv
        + (ftwist[None, None, :] ** 2) * gds22_arr[None, None, :] * shat_inv * shat_inv
    )
    term_kx = (
        (kx_eff[None, :, None] + delta_kx[:, None, :]) ** 2
        * gds22_arr[None, None, :]
        * shat_inv
        * shat_inv
    )
    kperp2 = term_ky + term_kx
    if kperp2_bmag:
        kperp2 = kperp2 * ((1.0 / bmag)[None, None, :] ** 2)
    kx_shift = kx_eff[None, :, None] + (rho_star * m0 / float(x0_eff))[:, None, :]
    cv_d = ky_eff[:, None, None] * cv[None, None, :] + (
        shat_inv * kx_shift * cv0[None, None, :]
    )
    gb_d = ky_eff[:, None, None] * gb[None, None, :] + (
        shat_inv * kx_shift * gb0[None, None, :]
    )
    return kperp2, cv_d, gb_d, cv_d + gb_d


def _build_standard_kperp_and_drift_arrays(
    geom_arrays: _GeometryCacheArrays,
    *,
    kx_eff: Any,
    ky_eff: Any,
    kperp2_bmag: bool,
) -> tuple[Any, Any, Any, Any]:
    """Contract the sampled metric with ``(kx, ky)`` on a standard flux tube.

    This is :meth:`_SpectralGeometryMixin.k_perp2` and
    :meth:`_SpectralGeometryMixin.drift_components` evaluated on the profiles
    :class:`_GeometryCacheArrays` already holds. Calling the geometry methods
    would re-broadcast the same profiles through ``jax.numpy`` and force the
    whole contraction onto the device; written here it follows the namespace of
    its operands, which is ``numpy`` for every concrete geometry.
    """

    xp = _array_namespace(kx_eff, ky_eff, geom_arrays.gds2, geom_arrays.geom_data.s_hat)
    kx0 = kx_eff[None, :, None]
    ky0 = ky_eff[:, None, None]
    s_hat = _as_device_dtype(xp, geom_arrays.geom_data.s_hat, kx_eff.dtype)
    s_hat_safe = xp.where(s_hat == 0.0, 1.0, s_hat)
    kx_hat = xp.where(s_hat == 0.0, kx0, kx0 / s_hat_safe)
    kperp2 = (
        ky0
        * (
            ky0 * geom_arrays.gds2[None, None, :]
            + 2.0 * kx_hat * geom_arrays.gds21[None, None, :]
        )
        + (kx_hat * kx_hat) * geom_arrays.gds22_arr[None, None, :]
    )
    if kperp2_bmag:
        bmag_inv = 1.0 / geom_arrays.bmag_raw[None, None, :]
        kperp2 = kperp2 * (bmag_inv * bmag_inv)
    kperp2 = kperp2.astype(kx_eff.dtype)
    cv_d = (
        ky0 * geom_arrays.cv[None, None, :] + kx_hat * geom_arrays.cv0[None, None, :]
    ).astype(kx_eff.dtype)
    gb_d = (
        ky0 * geom_arrays.gb[None, None, :] + kx_hat * geom_arrays.gb0[None, None, :]
    ).astype(kx_eff.dtype)
    omega_d = (cv_d + gb_d).astype(kx_eff.dtype)
    return kperp2, cv_d, gb_d, omega_d


def update_linear_cache_for_sheared_kx(
    cache: LinearCache,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    effective_kx_grid: jnp.ndarray,
) -> LinearCache:
    """Rebuild every continuously sheared ``kx``-dependent cache array.

    ``effective_kx_grid`` uses the same normalized units and ``(ky, kx)``
    layout as ``cache.kx_grid``. Periodic and linked standard flux tubes are
    supported. A flow-shear displacement is constant along each fixed-``ky``
    linked chain, so its precomputed twist-shift maps remain valid. Non-twist
    flux tubes use a separate, ``z``-dependent radial representation and fail
    closed here.
    """

    boundary = str(grid.boundary).lower()
    if boundary not in {"periodic", "linked"} or bool(grid.non_twist):
        raise NotImplementedError(
            "sheared-kx cache updates require a periodic or linked standard flux tube"
        )
    kx_grid = jnp.asarray(effective_kx_grid, dtype=cache.kx_grid.dtype)
    if tuple(kx_grid.shape) != tuple(cache.kx_grid.shape):
        raise ValueError("effective_kx_grid must have shape (ky, kx)")

    theta = jnp.asarray(grid.z, dtype=cache.kperp2.dtype)
    geom_data = ensure_flux_tube_geometry_data(geom, theta)
    kx0 = kx_grid[:, :, None]
    ky0 = jnp.asarray(cache.ky, dtype=cache.kperp2.dtype)[:, None, None]
    theta0 = theta[None, None, :]
    kperp2 = geom_data.k_perp2(kx0, ky0, theta0).astype(cache.kperp2.dtype)

    cv, gb, cv0, gb0 = geom_data.drift_coeffs(theta)
    shear = jnp.asarray(geom_data.s_hat, dtype=cache.kperp2.dtype)
    shear_safe = jnp.where(shear == 0.0, 1.0, shear)
    kx_hat = jnp.where(shear == 0.0, kx0, kx0 / shear_safe)
    cv_d = (ky0 * cv[None, None, :] + kx_hat * cv0[None, None, :]).astype(
        cache.cv_d.dtype
    )
    gb_d = (ky0 * gb[None, None, :] + kx_hat * gb0[None, None, :]).astype(
        cache.gb_d.dtype
    )
    mask = jnp.asarray(cache.dealias_mask, dtype=cache.kperp2.dtype)[:, :, None]
    kperp2 = kperp2 * mask
    cv_d = cv_d * mask
    gb_d = gb_d * mask
    omega_d = cv_d + gb_d

    gyro = _build_laguerre_gyro_cache(
        params,
        geom_data=geom_data,
        kperp2=kperp2,
        bmag=cache.bmag,
        Nl=int(cache.Jl.shape[1]),
        real_dtype=cache.kperp2.dtype,
    )
    return replace(
        cache,
        Jl=gyro.Jl,
        b=gyro.b.astype(cache.b.dtype),
        kperp2=kperp2,
        omega_d=omega_d,
        cv_d=cv_d,
        gb_d=gb_d,
        kx_grid=kx_grid,
        JlB=gyro.JlB.astype(cache.JlB.dtype),
        laguerre_to_grid=gyro.laguerre_to_grid,
        laguerre_to_spectral=gyro.laguerre_to_spectral,
        laguerre_roots=gyro.laguerre_roots,
        laguerre_j0=gyro.laguerre_j0,
        laguerre_j1_over_alpha=gyro.laguerre_j1_over_alpha,
    )


def _apply_dealias_to_kperp_and_drifts(
    *,
    grid: SpectralGrid,
    dealias_mask: Any,
    kperp2: Any,
    cv_d: Any,
    gb_d: Any,
    omega_d: Any,
) -> tuple[Any, Any, Any, Any]:
    apply_dealias_mask = dealias_mask is not None and int(grid.ky.size) > 1
    if not apply_dealias_mask:
        return kperp2, cv_d, gb_d, omega_d
    mask = dealias_mask[:, :, None]
    kperp2 = kperp2 * mask
    cv_d = cv_d * mask
    gb_d = gb_d * mask
    omega_d = omega_d * mask
    return kperp2, cv_d, gb_d, omega_d


def _build_kperp_and_drift_arrays(
    grid: SpectralGrid,
    *,
    grid_arrays: _GridCacheArrays,
    geom_arrays: _GeometryCacheArrays,
    twist: _TwistShiftCachePolicy,
    kperp2_bmag: bool,
) -> tuple[Any, Any, Any, Any]:
    if twist.use_ntft:
        kperp2, cv_d, gb_d, omega_d = _build_ntft_kperp_and_drift_arrays(
            grid,
            geom_arrays.geom_data,
            kx_eff=twist.kx_eff,
            ky_eff=grid_arrays.ky_eff,
            ky_raw=grid_arrays.ky_raw,
            rho_star=grid_arrays.rho_star,
            gds2=geom_arrays.gds2,
            gds21=geom_arrays.gds21,
            gds22_arr=geom_arrays.gds22_arr,
            bmag=geom_arrays.bmag,
            cv=geom_arrays.cv,
            gb=geom_arrays.gb,
            cv0=geom_arrays.cv0,
            gb0=geom_arrays.gb0,
            shat_arr=twist.shat_arr,
            x0_eff=twist.x0_eff,
            kperp2_bmag=kperp2_bmag,
        )
    else:
        kperp2, cv_d, gb_d, omega_d = _build_standard_kperp_and_drift_arrays(
            geom_arrays,
            kx_eff=twist.kx_eff,
            ky_eff=grid_arrays.ky_eff,
            kperp2_bmag=kperp2_bmag,
        )
    return _apply_dealias_to_kperp_and_drifts(
        grid=grid,
        dealias_mask=grid_arrays.dealias_mask,
        kperp2=kperp2,
        cv_d=cv_d,
        gb_d=gb_d,
        omega_d=omega_d,
    )


def _bessel_namespace(xp: Any, real_dtype: Any) -> Any:
    """Namespace for the Bessel factors: the host only in double precision.

    ``bessel_j0``/``bessel_j1`` reach ``cos`` and ``sin`` for arguments above
    eight. NumPy and XLA agree bit for bit on the float64 kernels, so the host
    reproduces the device answer exactly there; their float32 kernels differ in
    the last place, so single precision keeps the device route rather than
    quietly moving the cache by an ulp.
    """

    return xp if xp is np and np.dtype(real_dtype) == np.float64 else jnp


def _build_laguerre_gyro_cache(
    params: LinearParams,
    *,
    geom_data: Any,
    kperp2: Any,
    bmag: Any,
    Nl: int,
    real_dtype: Any,
    xp: Any = jnp,
) -> _LaguerreGyroCache:
    rho = xp.asarray(params.rho, dtype=real_dtype)
    if rho.ndim == 0:
        rho = rho[None]
    b = (rho[:, None, None, None] * rho[:, None, None, None]) * kperp2[None, ...]
    bessel_bmag_power = float(getattr(geom_data, "bessel_bmag_power", 0.0))
    if bessel_bmag_power != 0.0:
        bmag_factor = bmag[None, None, None, :] ** (-bessel_bmag_power)
        b = b * bmag_factor
    Jl, JlB = _build_gyroaverage_cache_arrays(b, Nl, real_dtype)
    lag_to_grid_np, lag_to_spec_np, lag_roots_np = laguerre_transform(Nl)
    laguerre_to_grid = xp.asarray(lag_to_grid_np, dtype=real_dtype)
    laguerre_to_spectral = xp.asarray(lag_to_spec_np, dtype=real_dtype)
    laguerre_roots = xp.asarray(lag_roots_np, dtype=real_dtype)
    alpha2 = xp.maximum(
        0.0,
        2.0 * laguerre_roots[None, :, None, None, None] * b[:, None, ...],
    )
    laguerre_j0, laguerre_j1_over_alpha = _gyro_bessel_factors(
        alpha2, xp=_bessel_namespace(xp, real_dtype)
    )
    laguerre_j0 = laguerre_j0.astype(real_dtype)
    laguerre_j1_over_alpha = laguerre_j1_over_alpha.astype(real_dtype)
    return _LaguerreGyroCache(
        b=b,
        Jl=Jl,
        JlB=JlB,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=laguerre_j0,
        laguerre_j1_over_alpha=laguerre_j1_over_alpha,
    )


def _build_kx_link_cache(
    grid: SpectralGrid,
    *,
    use_twist_shift: bool,
    y0: float,
    jtwist: int,
) -> _KxLinkCache:
    xp = _array_namespace(grid.kx, grid.ky)
    if use_twist_shift:
        iky = xp.rint(xp.asarray(grid.ky) * float(y0)).astype(jnp.int32)
        shift = xp.asarray(jtwist, dtype=jnp.int32) * iky
        kx_idx = xp.arange(grid.kx.size, dtype=jnp.int32)[None, :]
        kx_link_plus = kx_idx + shift[:, None]
        kx_link_minus = kx_idx - shift[:, None]
        kx_link_mask_plus = (kx_link_plus >= 0) & (kx_link_plus < grid.kx.size)
        kx_link_mask_minus = (kx_link_minus >= 0) & (kx_link_minus < grid.kx.size)
        return _KxLinkCache(
            kx_link_plus=xp.clip(kx_link_plus, 0, grid.kx.size - 1),
            kx_link_minus=xp.clip(kx_link_minus, 0, grid.kx.size - 1),
            kx_link_mask_plus=kx_link_mask_plus,
            kx_link_mask_minus=kx_link_mask_minus,
            jtwist=jtwist,
        )

    kx_idx = xp.arange(grid.kx.size, dtype=jnp.int32)[None, :]
    kx_link = xp.broadcast_to(kx_idx, (grid.ky.size, grid.kx.size))
    kx_mask = xp.ones((grid.ky.size, grid.kx.size), dtype=bool)
    return _KxLinkCache(
        kx_link_plus=kx_link,
        kx_link_minus=kx_link,
        kx_link_mask_plus=kx_mask,
        kx_link_mask_minus=kx_mask,
        jtwist=0,
    )


def _empty_linked_fft_cache(real_dtype: Any) -> _LinkedFFTCache:
    del real_dtype
    return _LinkedFFTCache(
        linked_indices=(),
        linked_kz=(),
        linked_inverse_permutation=np.asarray([], dtype=np.int32),
        linked_full_cover=False,
        linked_gather_map=np.asarray([], dtype=np.int32),
        linked_gather_mask=np.asarray([], dtype=bool),
        linked_use_gather=False,
    )


def _linked_fft_gather_metadata(
    linked_indices: tuple[np.ndarray, ...],
    *,
    n_modes: int,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray, bool]:
    if not linked_indices:
        return (
            np.asarray([], dtype=np.int32),
            False,
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=bool),
            False,
        )

    idx_flat = np.concatenate(
        [np.asarray(idx, dtype=np.int32).reshape(-1) for idx in linked_indices],
        axis=0,
    )
    linked_inverse_permutation = np.asarray([], dtype=np.int32)
    linked_full_cover = False
    if idx_flat.size == n_modes:
        ref = np.arange(n_modes, dtype=np.int32)
        if np.array_equal(np.sort(idx_flat), ref):
            linked_inverse_permutation = np.argsort(idx_flat).astype(np.int32)
            linked_full_cover = True

    if idx_flat.size == 0:
        return (
            linked_inverse_permutation,
            linked_full_cover,
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=bool),
            False,
        )

    gather_map = np.zeros(n_modes, dtype=np.int32)
    gather_mask = np.zeros(n_modes, dtype=bool)
    gather_map[idx_flat] = np.arange(idx_flat.size, dtype=np.int32)
    gather_mask[idx_flat] = True
    return (
        linked_inverse_permutation,
        linked_full_cover,
        gather_map,
        gather_mask,
        True,
    )


def _build_linked_fft_cache(
    grid: SpectralGrid,
    *,
    use_twist_shift: bool,
    y0: float,
    jtwist: int,
    real_dtype: Any,
) -> _LinkedFFTCache:
    if not use_twist_shift:
        return _empty_linked_fft_cache(real_dtype)

    # The chain maps are host topology, so the parallel spacing has to be read
    # off the buffer beside the wavenumbers rather than recovered from the
    # cache's own device copy, which is a tracer whenever the cache is built
    # inside one.
    z_host = np.asarray(grid.z)
    ky_mode = getattr(grid, "ky_mode", None)
    linked_indices, linked_kz = _build_linked_fft_maps(
        np.asarray(grid.kx),
        np.asarray(grid.ky),
        float(y0),
        int(jtwist),
        float(z_host[1] - z_host[0]),
        int(grid.z.size),
        real_dtype,
        None if ky_mode is None else np.asarray(ky_mode),
    )
    (
        linked_inverse_permutation,
        linked_full_cover,
        linked_gather_map,
        linked_gather_mask,
        linked_use_gather,
    ) = _linked_fft_gather_metadata(
        linked_indices,
        n_modes=int(grid.ky.size * grid.kx.size),
    )
    return _LinkedFFTCache(
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )


def _build_linked_damp_profile(
    grid: SpectralGrid,
    params: LinearParams,
    *,
    boundary: str,
    linked_indices: tuple[np.ndarray, ...],
    real_dtype: Any,
) -> np.ndarray:
    if boundary == "periodic":
        return np.asarray([], dtype=real_dtype)
    return np.asarray(
        _build_linked_end_damping_profile(
            linked_indices=linked_indices,
            ny=int(grid.ky.size),
            nx=int(grid.kx.size),
            nz=int(grid.z.size),
            widthfrac=float(params.damp_ends_widthfrac),
            ky_mode=(
                None
                if getattr(grid, "ky_mode", None) is None
                else np.asarray(grid.ky_mode, dtype=np.int32)
            ),
        ),
        dtype=real_dtype,
    )


def _build_linked_boundary_cache(
    grid: SpectralGrid,
    params: LinearParams,
    *,
    boundary: str,
    use_twist_shift: bool,
    y0: float,
    jtwist: int,
    real_dtype: Any,
) -> dict[str, Any]:
    damp_profile = _build_end_damping_profile_array(
        int(grid.z.size),
        float(params.damp_ends_widthfrac),
        boundary,
        real_dtype,
    )
    kx_links = _build_kx_link_cache(
        grid,
        use_twist_shift=use_twist_shift,
        y0=y0,
        jtwist=jtwist,
    )
    linked_fft = _build_linked_fft_cache(
        grid,
        use_twist_shift=use_twist_shift,
        y0=y0,
        jtwist=kx_links.jtwist,
        real_dtype=real_dtype,
    )
    linked_damp_profile = (
        _build_linked_damp_profile(
            grid,
            params,
            boundary=boundary,
            linked_indices=linked_fft.linked_indices,
            real_dtype=real_dtype,
        )
        if use_twist_shift
        else np.asarray([], dtype=real_dtype)
    )

    return {
        "damp_profile": damp_profile,
        "linked_damp_profile": linked_damp_profile,
        "kx_link_plus": kx_links.kx_link_plus,
        "kx_link_minus": kx_links.kx_link_minus,
        "kx_link_mask_plus": kx_links.kx_link_mask_plus,
        "kx_link_mask_minus": kx_links.kx_link_mask_minus,
        "linked_full_cover": linked_fft.linked_full_cover,
        "linked_inverse_permutation": linked_fft.linked_inverse_permutation,
        "linked_gather_map": linked_fft.linked_gather_map,
        "linked_gather_mask": linked_fft.linked_gather_mask,
        "linked_use_gather": linked_fft.linked_use_gather,
        "linked_indices": tuple(
            np.asarray(idx, dtype=np.int32) for idx in linked_fft.linked_indices
        ),
        "linked_kz": tuple(
            np.asarray(kz, dtype=real_dtype) for kz in linked_fft.linked_kz
        ),
        "jtwist": kx_links.jtwist,
    }


def _to_device_cache(cache: LinearCache) -> LinearCache:
    """Move a host-built cache onto the device in a single pass.

    The builder works in ``numpy`` so that no XLA kernel is compiled for a
    one-shot constant, but the solver consumes the cache under ``jit`` and its
    fields are declared as device arrays. Every array leaf already carries its
    final dtype, so this is a transfer and not a cast.
    """

    leaves, treedef = jax.tree_util.tree_flatten(cache)
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in leaves])


def _pack_linear_cache(
    grid: SpectralGrid,
    *,
    grid_arrays: _GridCacheArrays,
    geom_arrays: _GeometryCacheArrays,
    twist: _TwistShiftCachePolicy,
    kperp2: Any,
    cv_d: Any,
    gb_d: Any,
    omega_d: Any,
    kperp2_bmag: bool,
    gyro: _LaguerreGyroCache,
    moment_cache: dict[str, Any],
    linked_cache: dict[str, Any],
) -> LinearCache:
    xp = _array_namespace(grid.kx, grid.ky)
    ky_host = xp.asarray(grid.ky)
    kx_host = xp.asarray(grid.kx)
    mask0 = (ky_host == 0.0)[:, None, None] & (kx_host == 0.0)[None, :, None]
    real_dtype = grid_arrays.real_dtype
    return LinearCache(
        Jl=gyro.Jl,
        b=gyro.b.astype(real_dtype),
        kperp2=kperp2,
        kperp2_bmag=kperp2_bmag,
        bmag=geom_arrays.bmag,
        omega_d=omega_d,
        cv_d=cv_d,
        gb_d=gb_d,
        bgrad=geom_arrays.bgrad,
        jacobian=geom_arrays.jacobian,
        mask0=mask0,
        dz=grid_arrays.dz,
        kz=grid_arrays.kz,
        ky=grid_arrays.ky_eff.astype(real_dtype),
        kx=twist.kx_eff.astype(real_dtype),
        kx_grid=twist.kx_grid,
        ky_grid=grid_arrays.ky_grid,
        dealias_mask=grid_arrays.dealias_mask,
        kxfac=xp.asarray(twist.kxfac_val, dtype=real_dtype),
        lb_lam=moment_cache["lb_lam"],
        collision_lam=xp.asarray([], dtype=real_dtype),
        hyper_ratio=moment_cache["hyper_ratio"].astype(real_dtype),
        ratio_l=moment_cache["ratio_l"].astype(real_dtype),
        ratio_m=moment_cache["ratio_m"].astype(real_dtype),
        ratio_lm=moment_cache["ratio_lm"].astype(real_dtype),
        mask_const=moment_cache["mask_const"],
        mask_kz=moment_cache["mask_kz"],
        m_pow=moment_cache["m_pow"].astype(real_dtype),
        m_norm_kz_factor=moment_cache["m_norm_kz_factor"].astype(real_dtype),
        damp_profile=linked_cache["damp_profile"],
        linked_damp_profile=linked_cache["linked_damp_profile"],
        l=moment_cache["l"],
        m=moment_cache["m"],
        l4=moment_cache["l4"],
        sqrt_m=moment_cache["sqrt_m"].astype(real_dtype),
        sqrt_m_p1=moment_cache["sqrt_m_p1"].astype(real_dtype),
        sqrt_p=moment_cache["sqrt_p"],
        sqrt_m_ladder=moment_cache["sqrt_m_ladder"],
        JlB=gyro.JlB.astype(real_dtype),
        laguerre_to_grid=gyro.laguerre_to_grid,
        laguerre_to_spectral=gyro.laguerre_to_spectral,
        laguerre_roots=gyro.laguerre_roots,
        laguerre_j0=gyro.laguerre_j0,
        laguerre_j1_over_alpha=gyro.laguerre_j1_over_alpha,
        kx_link_plus=linked_cache["kx_link_plus"],
        kx_link_minus=linked_cache["kx_link_minus"],
        kx_link_mask_plus=linked_cache["kx_link_mask_plus"],
        kx_link_mask_minus=linked_cache["kx_link_mask_minus"],
        linked_full_cover=linked_cache["linked_full_cover"],
        linked_inverse_permutation=linked_cache["linked_inverse_permutation"],
        linked_gather_map=linked_cache["linked_gather_map"],
        linked_gather_mask=linked_cache["linked_gather_mask"],
        linked_use_gather=linked_cache["linked_use_gather"],
        linked_indices=linked_cache["linked_indices"],
        linked_kz=linked_cache["linked_kz"],
        use_twist_shift=twist.use_twist_shift,
        jtwist=int(linked_cache["jtwist"]),
    )


def build_linear_cache(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    Nl: int,
    Nm: int,
) -> LinearCache:
    """Build reusable arrays for the linear RHS."""

    grid_arrays = _build_grid_cache_arrays(grid, params)
    geom_arrays = _build_geometry_cache_arrays(
        geom,
        theta=grid_arrays.theta,
        real_dtype=grid_arrays.real_dtype,
    )
    twist = _resolve_twist_shift_policy(
        grid,
        geom_arrays.geom_data,
        gds21=geom_arrays.gds21,
        gds22=geom_arrays.gds22,
        kx_eff=grid_arrays.kx_eff,
        kx_grid=grid_arrays.kx_grid,
    )
    kperp2_bmag = bool(getattr(geom_arrays.geom_data, "kperp2_bmag", True))
    kperp2, cv_d, gb_d, omega_d = _build_kperp_and_drift_arrays(
        grid,
        grid_arrays=grid_arrays,
        geom_arrays=geom_arrays,
        twist=twist,
        kperp2_bmag=kperp2_bmag,
    )
    gyro = _build_laguerre_gyro_cache(
        params,
        geom_data=geom_arrays.geom_data,
        kperp2=kperp2,
        bmag=geom_arrays.bmag,
        Nl=Nl,
        real_dtype=grid_arrays.real_dtype,
        xp=_array_namespace(kperp2, geom_arrays.bmag, params.rho),
    )
    moment_cache = _build_low_rank_moment_cache_arrays(
        Nl, Nm, params, grid_arrays.real_dtype
    )
    linked_cache = _build_linked_boundary_cache(
        grid,
        params,
        boundary=twist.boundary,
        use_twist_shift=twist.use_twist_shift,
        y0=twist.y0,
        jtwist=twist.jtwist,
        real_dtype=grid_arrays.real_dtype,
    )
    return _to_device_cache(
        _pack_linear_cache(
            grid,
            grid_arrays=grid_arrays,
            geom_arrays=geom_arrays,
            twist=twist,
            kperp2=kperp2,
            cv_d=cv_d,
            gb_d=gb_d,
            omega_d=omega_d,
            kperp2_bmag=kperp2_bmag,
            gyro=gyro,
            moment_cache=moment_cache,
            linked_cache=linked_cache,
        )
    )
