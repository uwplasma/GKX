"""Array factories and damping factors for the linear cache."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from importlib import resources
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.core.velocity import J_l_all
from spectraxgk.operators.linear.params import LinearParams, _as_species_array

if TYPE_CHECKING:
    from spectraxgk.operators.linear.cache_model import LinearCache


_COLLISION_MATRIX_DATA = "advanced_collision_six_moment.npy"
_COLLISION_MATRIX_METADATA = "advanced_collision_six_moment.json"


@lru_cache(maxsize=1)
def _collision_matrix_bundle() -> tuple[np.ndarray, dict[str, Any]]:
    data_root = resources.files("spectraxgk").joinpath("data")
    payload = data_root.joinpath(_COLLISION_MATRIX_DATA).read_bytes()
    metadata = json.loads(
        data_root.joinpath(_COLLISION_MATRIX_METADATA).read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError("collision coefficient checksum does not match metadata")
    matrices = np.load(io.BytesIO(payload), allow_pickle=False)
    if list(matrices.shape) != metadata.get("shape"):
        raise ValueError("collision coefficient shape does not match metadata")
    return np.asarray(matrices), metadata


def load_collision_moment_matrix(model: str) -> np.ndarray:
    """Load a provenance-checked drift-kinetic collision moment matrix."""

    matrices, metadata = _collision_matrix_bundle()
    names = list(metadata["models"])
    key = model.strip().lower()
    if key not in names:
        raise ValueError(f"collision model must be one of {names}")
    return np.array(matrices[names.index(key)], copy=True)


def drift_kinetic_sugama_pair_matrices(
    mass_ratio: jnp.ndarray,
    temperature_ratio: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return normalized test/field matrices for one ordered species pair.

    This is Frei, Ernst & Ricci (2022), Appendix C, equations (C4)--(C5),
    with ``mass_ratio = m_a / m_b`` and ``temperature_ratio = T_a / T_b``.
    Multiplication by the directed collision frequency ``nu_ab`` is left to the
    caller. The matrices use the code's Hermite-major, signed-Laguerre ordering.
    """

    sigma = jnp.asarray(mass_ratio)
    tau = jnp.asarray(temperature_ratio, dtype=jnp.result_type(sigma, float))
    if sigma.ndim != 0 or tau.ndim != 0:
        raise ValueError("mass_ratio and temperature_ratio must be scalars")
    if not isinstance(sigma, jax.core.Tracer) and float(sigma) <= 0.0:
        raise ValueError("mass_ratio must be positive")
    if not isinstance(tau, jax.core.Tracer) and float(tau) <= 0.0:
        raise ValueError("temperature_ratio must be positive")
    dtype = jnp.result_type(sigma, tau)
    pi = jnp.asarray(jnp.pi, dtype=dtype)
    root_pi = jnp.sqrt(pi)
    total = sigma + tau
    test = jnp.zeros((8, 8), dtype=dtype)
    field = jnp.zeros((8, 8), dtype=dtype)
    mode_01, mode_10, mode_11, mode_20, mode_30 = 1, 2, 3, 4, 6

    test = test.at[mode_10, mode_10].set(
        -8.0 * (sigma + 1.0) / (3.0 * root_pi) * (tau / total) ** 1.5
    )
    test_10_30 = (
        4.0 / 5.0 * jnp.sqrt(6.0 / pi) * jnp.sqrt(sigma + 1.0) * tau**2 / total**2
    )
    test_10_11 = -8.0 * jnp.sqrt(sigma + 1.0) * tau**2 / (5.0 * root_pi * total**2)
    test = test.at[mode_10, mode_30].set(test_10_30)
    test = test.at[mode_30, mode_10].set(test_10_30)
    test = test.at[mode_10, mode_11].set(test_10_11)
    test = test.at[mode_11, mode_10].set(test_10_11)
    test = test.at[mode_20, mode_20].set(
        -16.0
        * jnp.sqrt(tau)
        * (5.0 * sigma**2 * (tau + 2.0) + 21.0 * sigma * tau + 6.0 * tau**2)
        / (45.0 * root_pi * total**2.5)
    )
    test_20_01 = (
        -16.0
        * jnp.sqrt(2.0 / pi)
        * jnp.sqrt(tau)
        * (-5.0 * sigma**2 * (tau - 1.0) + 3.0 * sigma * tau + 3.0 * tau**2)
        / (45.0 * total**2.5)
    )
    test = test.at[mode_20, mode_01].set(test_20_01)
    test = test.at[mode_01, mode_20].set(test_20_01)
    test = test.at[mode_01, mode_01].set(
        -16.0
        * jnp.sqrt(tau)
        * (5.0 * sigma**2 + 2.0 * (5.0 * sigma + 9.0) * sigma * tau + 3.0 * tau**2)
        / (45.0 * root_pi * total**2.5)
    )
    test = test.at[mode_30, mode_30].set(
        -4.0
        * jnp.sqrt(tau)
        * (70.0 * sigma**2 + 56.0 * sigma * tau + 31.0 * tau**2)
        / (35.0 * root_pi * total**2.5)
    )
    test_30_11 = (
        -4.0
        * jnp.sqrt(2.0 / (3.0 * pi))
        * tau**1.5
        * (28.0 * sigma + tau)
        / (35.0 * total**2.5)
    )
    test = test.at[mode_30, mode_11].set(test_30_11)
    test = test.at[mode_11, mode_30].set(test_30_11)
    test = test.at[mode_11, mode_11].set(
        -8.0
        * jnp.sqrt(tau)
        * (105.0 * sigma**2 + 98.0 * sigma * tau + 47.0 * tau**2)
        / (105.0 * root_pi * total**2.5)
    )

    field = field.at[mode_10, mode_10].set(
        8.0 * sigma * (sigma + 1.0) * tau / (3.0 * root_pi * jnp.sqrt(sigma * total**3))
    )
    field = field.at[mode_10, mode_30].set(
        -4.0
        * jnp.sqrt(6.0 / pi)
        * jnp.sqrt(sigma**3 * (sigma + 1.0))
        * tau
        / (5.0 * total**2)
    )
    field = field.at[mode_10, mode_11].set(
        8.0 * jnp.sqrt(sigma**3 * (sigma + 1.0)) * tau / (5.0 * root_pi * total**2)
    )
    field_20_01 = (
        -16.0
        * jnp.sqrt(2.0 / pi)
        * sigma
        * (sigma + 1.0)
        * tau**1.5
        / (9.0 * total**2.5)
    )
    field = field.at[mode_20, mode_20].set(
        16.0 * sigma * (sigma + 1.0) * tau**1.5 / (9.0 * root_pi * total**2.5)
    )
    field = field.at[mode_20, mode_01].set(field_20_01)
    field = field.at[mode_01, mode_20].set(field_20_01)
    field = field.at[mode_01, mode_01].set(
        32.0 * sigma * (sigma + 1.0) * tau**1.5 / (9.0 * root_pi * total**2.5)
    )
    field = field.at[mode_30, mode_10].set(
        -4.0
        * jnp.sqrt(6.0 / pi)
        * jnp.sqrt(sigma * (sigma + 1.0) * tau**3)
        / (5.0 * total**2)
    )
    field = field.at[mode_30, mode_30].set(
        36.0 * (sigma * tau) ** 1.5 / (25.0 * root_pi * total**2.5)
    )
    field_30_11 = (
        -12.0 * jnp.sqrt(6.0 / pi) * (sigma * tau) ** 1.5 / (25.0 * total**2.5)
    )
    field = field.at[mode_30, mode_11].set(field_30_11)
    field = field.at[mode_11, mode_10].set(
        8.0 * jnp.sqrt(sigma * (sigma + 1.0) * tau**3) / (5.0 * root_pi * total**2)
    )
    field = field.at[mode_11, mode_30].set(field_30_11)
    field = field.at[mode_11, mode_11].set(
        24.0 * (sigma * tau) ** 1.5 / (25.0 * root_pi * total**2.5)
    )

    laguerre_sign = jnp.asarray(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=dtype
    )
    convention = laguerre_sign[:, None] * laguerre_sign[None, :]
    return convention * test, convention * field


def interpolate_collision_moment_matrix(
    kperp_grid: jnp.ndarray,
    matrices: jnp.ndarray,
    kperp: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate collision matrices onto a scalar or spatial ``kperp`` field.

    ``matrices`` may contain one shared table with shape
    ``(kperp, modes, modes)`` or one table per species with shape
    ``(species, kperp, modes, modes)``. Values outside the tabulated interval
    use the nearest endpoint. The coefficient grid is validated on the host;
    interpolation and its derivative with respect to ``kperp`` remain in JAX.
    """

    grid = jnp.asarray(kperp_grid)
    table = jnp.asarray(matrices)
    target = jnp.asarray(kperp, dtype=jnp.result_type(grid, table))
    if grid.ndim != 1 or int(grid.size) < 2:
        raise ValueError(
            "collision kperp grid must be one-dimensional with at least two points"
        )
    if table.ndim not in {3, 4}:
        raise ValueError(
            "collision table must have shape (kperp, modes, modes) or "
            "(species, kperp, modes, modes)"
        )
    grid_axis = 0 if table.ndim == 3 else 1
    if int(table.shape[grid_axis]) != int(grid.size):
        raise ValueError("collision table kperp axis must match the coefficient grid")
    if int(table.shape[-1]) != int(table.shape[-2]):
        raise ValueError("collision table matrices must be square")
    if not isinstance(grid, jax.core.Tracer):
        host_grid = np.asarray(grid)
        if not np.all(np.isfinite(host_grid)) or not np.all(np.diff(host_grid) > 0.0):
            raise ValueError(
                "collision kperp grid must be finite and strictly increasing"
            )

    def interpolate_one(species_table: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(values, grid[0], grid[-1])
        left = jnp.clip(
            jnp.searchsorted(grid, clipped, side="right") - 1, 0, grid.size - 2
        )
        fraction = (clipped - grid[left]) / (grid[left + 1] - grid[left])
        interpolated = species_table[left] + fraction[..., None, None] * (
            species_table[left + 1] - species_table[left]
        )
        return jnp.moveaxis(interpolated, (-2, -1), (0, 1))

    if table.ndim == 3:
        return interpolate_one(table, target)
    species_count = int(table.shape[0])
    if target.ndim == 0:
        return jax.vmap(lambda species_table: interpolate_one(species_table, target))(
            table
        )
    if int(target.shape[0]) != species_count:
        raise ValueError(
            "species collision table requires scalar kperp or a species-leading kperp field"
        )
    return jax.vmap(interpolate_one)(table, target)


def apply_collision_moment_matrix(
    state: jnp.ndarray,
    matrix: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply a dense drift-kinetic matrix in Hermite-major moment ordering."""

    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    ns, nl, nm = map(int, expanded.shape[:3])
    mode_count = nl * nm
    coefficients = jnp.asarray(matrix, dtype=jnp.result_type(value, matrix))
    spatial_shape = tuple(map(int, expanded.shape[3:]))
    static_shared = (mode_count, mode_count)
    static_species = {(1,) + static_shared, (ns,) + static_shared}
    spatial_shared = static_shared + spatial_shape
    spatial_species = {(1,) + spatial_shared, (ns,) + spatial_shared}
    if coefficients.shape == static_shared or coefficients.shape == spatial_shared:
        coefficients = coefficients[None, ...]
    if coefficients.shape not in static_species | spatial_species:
        raise ValueError(
            "collision matrix must have shape for a static or state-spatial operator, "
            "with an optional leading species axis"
        )
    coefficients = jnp.broadcast_to(coefficients, (ns,) + coefficients.shape[1:])
    packed = jnp.swapaxes(expanded, 1, 2).reshape((ns, mode_count) + expanded.shape[3:])
    applied = jnp.einsum("sij...,sj...->si...", coefficients, packed)
    result = jnp.swapaxes(applied.reshape((ns, nm, nl) + expanded.shape[3:]), 1, 2)
    real_dtype = jnp.real(expanded).dtype
    frequency = jnp.asarray(nu, dtype=real_dtype).reshape(-1)
    if frequency.size == 1:
        frequency = jnp.broadcast_to(frequency, (ns,))
    if int(frequency.size) != ns:
        raise ValueError(f"nu must have length {ns} (got {frequency.size})")
    scale = frequency[(slice(None),) + (None,) * (result.ndim - 1)]
    result = jnp.asarray(weight, dtype=real_dtype) * scale * result
    return result[0] if value.ndim == 5 else result


def apply_multispecies_collision_moment_matrix(
    state: jnp.ndarray,
    matrix: jnp.ndarray,
    *,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply a target/source-species collision matrix and sum source species."""

    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    ns, nl, nm = map(int, expanded.shape[:3])
    mode_count = nl * nm
    spatial_shape = tuple(map(int, expanded.shape[3:]))
    coefficients = jnp.asarray(matrix, dtype=jnp.result_type(value, matrix))
    allowed = {
        (ns, ns, mode_count, mode_count),
        (ns, ns, mode_count, mode_count) + spatial_shape,
    }
    if coefficients.shape not in allowed:
        raise ValueError(
            "multispecies collision matrix must have target/source species and "
            "square moment axes, optionally followed by the state spatial shape"
        )
    packed = jnp.swapaxes(expanded, 1, 2).reshape((ns, mode_count) + expanded.shape[3:])
    applied = jnp.einsum("stij...,tj...->si...", coefficients, packed)
    result = jnp.swapaxes(applied.reshape((ns, nm, nl) + expanded.shape[3:]), 1, 2)
    result = jnp.asarray(weight, dtype=jnp.real(expanded).dtype) * result
    return result[0] if value.ndim == 5 else result


def _shift_axis_for_cache(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """Shift an array along one axis with zeros introduced at the boundary."""

    axis = axis % arr.ndim
    if offset == 0:
        return arr
    n = arr.shape[axis]
    if abs(offset) >= n:
        return jnp.zeros_like(arr)
    out = jnp.zeros_like(arr)
    if offset > 0:
        body = jax.lax.slice_in_dim(arr, offset, n, axis=axis)
        starts = [0] * arr.ndim
        starts[axis] = 0
        return jax.lax.dynamic_update_slice(out, body, starts)
    body = jax.lax.slice_in_dim(arr, 0, n + offset, axis=axis)
    starts = [0] * arr.ndim
    starts[axis] = -offset
    return jax.lax.dynamic_update_slice(out, body, starts)


def _numpy_dtype_for_jax(real_dtype: jnp.dtype) -> type[np.float32] | type[np.float64]:
    return np.float64 if real_dtype == jnp.float64 else np.float32


def _build_low_rank_moment_cache_arrays(
    Nl: int,
    Nm: int,
    params: LinearParams,
    real_dtype: jnp.dtype,
) -> dict[str, jnp.ndarray]:
    """Build small moment-space cache arrays without many eager JAX dispatches."""

    np_dtype: Any = _numpy_dtype_for_jax(real_dtype)
    ell: Any = np.arange(Nl, dtype=np_dtype).reshape(Nl, 1, 1, 1, 1)
    m: Any = np.arange(Nm, dtype=np_dtype).reshape(1, Nm, 1, 1, 1)
    lb_lam_np = (
        float(params.nu_laguerre) * np.arange(Nl, dtype=np_dtype)[:, None]
        + float(params.nu_hermite) * np.arange(Nm, dtype=np_dtype)[None, :]
    )
    sqrt_shape = (1, 1, Nm, 1, 1, 1)
    hermite_index: Any = np.arange(Nm, dtype=np_dtype)
    sqrt_p_np = np.sqrt(hermite_index + np_dtype(1.0)).reshape(sqrt_shape)
    sqrt_m_ladder_np = np.sqrt(hermite_index).reshape(sqrt_shape)
    l_norm = np_dtype(max(Nl - 1, 1))
    m_norm = np_dtype(max(Nm - 1, 1))
    l_norm_full = np_dtype(max(Nl, 1))
    m_norm_full = np_dtype(max(Nm, 1))
    m_norm_kz = np_dtype(max(Nm - 1, 1))
    p_hyper_l = np_dtype(params.p_hyper_l)
    p_hyper_m = np_dtype(params.p_hyper_m)
    p_hyper_lm = np_dtype(params.p_hyper_lm)
    normalized_m = m / m_norm_kz
    return {
        "lb_lam": jnp.asarray(lb_lam_np, dtype=real_dtype),
        "l": jnp.asarray(ell, dtype=real_dtype),
        "m": jnp.asarray(m, dtype=real_dtype),
        "l4": jnp.asarray(
            np.arange(Nl, dtype=np_dtype).reshape(Nl, 1, 1, 1), dtype=real_dtype
        ),
        "sqrt_m": jnp.asarray(np.sqrt(m), dtype=real_dtype),
        "sqrt_m_p1": jnp.asarray(np.sqrt(m + np_dtype(1.0)), dtype=real_dtype),
        "sqrt_p": jnp.asarray(sqrt_p_np, dtype=real_dtype),
        "sqrt_m_ladder": jnp.asarray(sqrt_m_ladder_np, dtype=real_dtype),
        "hyper_ratio": jnp.asarray(
            (ell / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper,
            dtype=real_dtype,
        ),
        "ratio_l": jnp.asarray((ell / l_norm_full) ** p_hyper_l, dtype=real_dtype),
        "ratio_m": jnp.asarray((m / m_norm_full) ** p_hyper_m, dtype=real_dtype),
        "ratio_lm": jnp.asarray(
            ((2.0 * ell + m) / (2.0 * l_norm_full + m_norm_full)) ** p_hyper_lm,
            dtype=real_dtype,
        ),
        "mask_const": jnp.asarray((m > 2.0) | (ell > 1.0), dtype=bool),
        "mask_kz": jnp.asarray(m > 2.0, dtype=bool),
        "m_pow": jnp.asarray(normalized_m**p_hyper_m, dtype=real_dtype),
        "m_norm_kz_factor": jnp.asarray(
            (p_hyper_m + 0.5) / np.sqrt(m_norm_kz), dtype=real_dtype
        ),
    }


def _build_end_damping_profile_array(
    Nz: int,
    widthfrac: float,
    boundary: str,
    real_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Build the one-dimensional end-damping profile as one host array."""

    np_dtype = np.float32
    width = max(1, int(np.floor(float(widthfrac) * int(Nz))))
    idx = np.arange(Nz, dtype=np_dtype)
    width_f = np_dtype(width)
    left_mask = idx <= width_f
    right_mask = idx >= (Nz - width_f)
    x_left = np.where(left_mask, idx / width_f, 0.0)
    x_right = np.where(right_mask, (Nz - idx) / width_f, 0.0)
    nu_left = np.where(left_mask, 1.0 - 2.0 * x_left * x_left / (1.0 + x_left**4), 0.0)
    nu_right = np.where(
        right_mask, 1.0 - 2.0 * x_right * x_right / (1.0 + x_right**4), 0.0
    )
    damp_profile_np = np.maximum(nu_left, nu_right).astype(np_dtype)
    if boundary == "periodic":
        damp_profile_np = np.zeros_like(damp_profile_np)
    return jnp.asarray(damp_profile_np, dtype=real_dtype)


def _build_gyroaverage_cache_arrays(
    b: jnp.ndarray,
    Nl: int,
    real_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build species-major gyroaverage factors without a Python-level vmap."""

    Jl = jnp.moveaxis(J_l_all(b, l_max=Nl - 1), 0, 1).astype(real_dtype)
    JlB = Jl + _shift_axis_for_cache(Jl, -1, axis=1)
    return Jl, JlB.astype(real_dtype)


def hypercollision_damping(
    cache: "LinearCache",
    params: "LinearParams",
    real_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Assemble benchmark-compatible hypercollision damping factors."""

    Nl = jnp.asarray(max(int(cache.l.shape[0]), 1), dtype=real_dtype)
    Nm = jnp.asarray(max(int(cache.m.shape[1]), 1), dtype=real_dtype)

    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    w_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    w_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)

    vth = jnp.asarray(params.vth, dtype=real_dtype)
    vth_s = vth if vth.ndim == 0 else vth[:, None, None, None, None, None]

    ratio_l = cache.ratio_l.astype(real_dtype)
    ratio_m = cache.ratio_m.astype(real_dtype)
    ratio_lm = cache.ratio_lm.astype(real_dtype)
    scaled_nu_l = Nl * nu_hyper_l
    scaled_nu_m = Nm * nu_hyper_m
    mask_const = cache.mask_const
    const_coeff = (
        vth_s * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m) + nu_hyper_lm * ratio_lm
    )

    hyper = nu_hyper * cache.hyper_ratio.astype(real_dtype)
    hyper = hyper + w_const * jnp.where(mask_const, const_coeff, 0.0)

    abs_kz = jnp.abs(cache.kz).astype(real_dtype)[None, None, None, None, None, :]
    nu_hyp_m = (
        nu_hyper_m
        * cache.m_norm_kz_factor.astype(real_dtype)
        * 2.3
        * vth_s
        * jnp.abs(jnp.asarray(params.kpar_scale, dtype=real_dtype))
    )
    kz_term = nu_hyp_m * cache.m_pow.astype(real_dtype) * abs_kz
    hyper = hyper + w_kz * jnp.where(cache.mask_kz, kz_term, 0.0)
    return hyper


def collision_damping(
    cache: "LinearCache",
    params: "LinearParams",
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool = False,
) -> jnp.ndarray:
    """Assemble collision damping from cached low-rank factors.

    Runtime caches store ``lb_lam`` as the Hermite-Laguerre Lenard-Bernstein
    diagonal only, with shape ``(Nl, Nm)``. Direct unit tests may also provide
    a pre-expanded array to exercise the damping assembly policy.
    """

    lb_lam = cache.lb_lam.astype(real_dtype)
    if lb_lam.ndim == 2:
        b = jnp.asarray(cache.b, dtype=real_dtype)
        ns = int(b.shape[0])
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        nu_s = nu[:, None, None, None, None, None]
        damping = nu_s * lb_lam[None, :, :, None, None, None]
        damping = damping + nu_s * b[:, None, None, ...]
        if squeeze_species:
            damping = damping[0]
        return damping.astype(real_dtype)

    if lb_lam.ndim == 6:
        ns = int(lb_lam.shape[0])
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam
        if squeeze_species:
            damping = damping[0]
        return damping.astype(real_dtype)

    return (jnp.asarray(params.nu, dtype=real_dtype) * lb_lam).astype(real_dtype)
