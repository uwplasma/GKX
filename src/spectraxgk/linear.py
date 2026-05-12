"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

from spectraxgk.basis import hermite_ladder_coeffs
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear_linked import (
    _build_linked_end_damping_profile,  # noqa: F401 - legacy private helper re-export
    _build_linked_fft_maps,  # noqa: F401 - legacy private helper re-export
    _signed_to_index,  # noqa: F401 - legacy private helper re-export
)
from spectraxgk.linear_cache import (
    LinearCache,
    _build_end_damping_profile_array,  # noqa: F401 - legacy private helper re-export
    _build_gyroaverage_cache_arrays,  # noqa: F401 - legacy private helper re-export
    _build_low_rank_moment_cache_arrays,  # noqa: F401 - legacy private helper re-export
    _numpy_dtype_for_jax,  # noqa: F401 - legacy private helper re-export
    build_linear_cache,
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.linear_params import (
    LinearParams,
    LinearTerms,
    Preconditioner,  # noqa: F401 - legacy public type alias re-export
    PreconditionerSpec,
    _as_species_array,
    _check_nonnegative,  # noqa: F401 - legacy private helper re-export
    _check_positive,
    _is_tracer,  # noqa: F401 - legacy private helper re-export
    _resolve_implicit_preconditioner,
    _x64_enabled,
    linear_terms_to_term_config,
    term_config_to_linear_terms,  # noqa: F401 - legacy public helper re-export
)


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)
_FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def grad_z_periodic(f: jnp.ndarray, dz: float | jnp.ndarray) -> jnp.ndarray:
    """Spectral periodic derivative along the last axis."""

    _check_positive(dz, "dz")
    n = f.shape[-1]
    dz_val = jnp.asarray(dz, dtype=jnp.real(f).dtype)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dz_val)
    f_hat = jnp.fft.fft(f, axis=-1)
    df_hat = (1j * kz) * f_hat
    return jnp.fft.ifft(df_hat, axis=-1)


def compute_b(
    grid: SpectralGrid, geom: FluxTubeGeometryLike, rho: float
) -> jnp.ndarray:
    """Compute b = rho^2 * k_perp^2(kx, ky, theta) for s-alpha geometry."""

    _check_positive(rho, "rho")
    kx0 = grid.kx[None, :, None]
    ky = grid.ky[:, None, None]
    theta = grid.z[None, None, :]
    kperp2 = geom.k_perp2(kx0, ky, theta)
    return (rho * rho) * kperp2


def lenard_bernstein_eigenvalues(
    Nl: int, Nm: int, nu_hermite: float, nu_laguerre: float
) -> jnp.ndarray:
    """Diagonal Lenard-Bernstein rates in Hermite-Laguerre space."""

    ell = jnp.arange(Nl)
    m = jnp.arange(Nm)
    return nu_laguerre * ell[:, None] + nu_hermite * m[None, :]


def apply_hermite_v(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel (ladder form)."""

    axis_m = -4
    Nm = G.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]
    G_plus = shift_axis(G, 1, axis_m)
    G_minus = shift_axis(G, -1, axis_m)
    shape = [1] * G.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    return sqrt_p * G_plus + sqrt_m * G_minus


def apply_hermite_v2(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel^2."""

    return apply_hermite_v(apply_hermite_v(G))


def apply_laguerre_x(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Laguerre coefficients by the perpendicular energy variable."""

    axis_l = -5
    Nl = G.shape[axis_l]
    ell = jnp.arange(Nl)
    G_plus = shift_axis(G, 1, axis_l)
    G_minus = shift_axis(G, -1, axis_l)
    ell_shape = [1] * G.ndim
    ell_shape[axis_l] = Nl
    ell_col = ell.reshape(ell_shape)
    return (2.0 * ell_col + 1.0) * G - (ell_col + 1.0) * G_plus - ell_col * G_minus


def shift_axis(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """Shift an array along an axis with zero padding (non-periodic)."""

    axis = axis % arr.ndim
    if offset == 0:
        return arr
    axis_len = arr.shape[axis]
    if abs(offset) >= axis_len:
        return jnp.zeros_like(arr)
    out = jnp.zeros_like(arr)
    if offset > 0:
        body = jax.lax.slice_in_dim(arr, offset, axis_len, axis=axis)
        starts = [0] * arr.ndim
        starts[axis] = 0
        return jax.lax.dynamic_update_slice(out, body, starts)
    body = jax.lax.slice_in_dim(arr, 0, axis_len + offset, axis=axis)
    starts = [0] * arr.ndim
    starts[axis] = -offset
    return jax.lax.dynamic_update_slice(out, body, starts)


def energy_operator(
    G: jnp.ndarray, coeff_const: float, coeff_par: float, coeff_perp: float
) -> jnp.ndarray:
    """Apply the energy operator (1 + v_par^2 + mu) in Hermite-Laguerre space."""

    return (
        coeff_const * G
        + coeff_par * apply_hermite_v2(G)
        + coeff_perp * apply_laguerre_x(G)
    )


def diamagnetic_drive_coeffs(
    Nl: int,
    Nm: int,
    eta_i: jnp.ndarray,
    coeff_const: float,
    coeff_par: float,
    coeff_perp: float,
) -> jnp.ndarray:
    """Return velocity-space coefficients for (1 + eta_i(E - 3/2))."""

    e00 = jnp.zeros((Nl, Nm, 1, 1, 1))
    e00 = e00.at[0, 0, 0, 0, 0].set(1.0)
    energy_e00 = energy_operator(e00, coeff_const, coeff_par, coeff_perp)
    coeffs = e00 + eta_i * (energy_e00 - 1.5 * e00)
    return coeffs[:, :, 0, 0, 0]


def quasineutrality_phi(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    tau_e: float | jnp.ndarray,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    tz: jnp.ndarray,
) -> jnp.ndarray:
    """Solve electrostatic quasineutrality for phi with optional adiabatic closure."""

    _check_nonnegative(tau_e, "tau_e")
    Gm0 = G[:, :, 0, ...]
    num = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    den = tau_e + jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * zt[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    den_safe = jnp.where(den == 0.0, jnp.inf, den)
    return num / den_safe


def build_H(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    phi: jnp.ndarray,
    tz: jnp.ndarray,
    apar: jnp.ndarray | None = None,
    vth: jnp.ndarray | None = None,
    bpar: jnp.ndarray | None = None,
    JlB: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Map G -> H for mirror/curvature/grad-B/collision terms.

    GX builds H by adding the field terms for m=0 (phi, Bpar) and the
    A_parallel term for m=1, while the streaming term applies its own
    pre-derivative field contributions. We mirror that behavior here.
    """

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    zt_arr = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
    Nm = G.shape[-4]
    m0_mask = (jnp.arange(Nm, dtype=jnp.int32) == 0).astype(G.dtype)
    m0_mask = m0_mask.reshape((1, 1, Nm, 1, 1, 1))
    phi_term = (zt_arr[:, None, None, None, None] * Jl * phi)[:, :, None, ...]
    H = G + m0_mask * phi_term
    if apar is not None:
        if vth is None:
            raise ValueError("vth must be provided when apar is supplied")
        m1_mask = (jnp.arange(Nm, dtype=jnp.int32) == 1).astype(G.dtype)
        m1_mask = m1_mask.reshape((1, 1, Nm, 1, 1, 1))
        vth_arr = jnp.asarray(vth)
        if vth_arr.ndim == 0:
            vth_arr = vth_arr[None]
        apar_term = (
            zt_arr[:, None, None, None, None]
            * vth_arr[:, None, None, None, None]
            * Jl
            * apar
        )[:, :, None, ...]
        H = H - m1_mask * apar_term
    if bpar is not None:
        if JlB is None:
            raise ValueError("JlB must be provided when bpar is supplied")
        bpar_term = (JlB * bpar)[:, :, None, ...]
        H = H + m0_mask * bpar_term
    return H[0] if squeeze_species else H


def streaming_term(
    H: jnp.ndarray, dz: float | jnp.ndarray, vth: float | jnp.ndarray
) -> jnp.ndarray:
    """Streaming term using Hermite ladder and real-space z derivative."""

    _check_positive(vth, "vth")
    dH_dz = grad_z_periodic(H, dz)
    axis_m = -4
    Nm = H.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = [(0, 0)] * H.ndim
    pad[axis_m] = (1, 1)
    dH_pad = jnp.pad(dH_dz, pad)
    slc_plus = [slice(None)] * H.ndim
    slc_minus = [slice(None)] * H.ndim
    slc_plus[axis_m] = slice(2, None)
    slc_minus[axis_m] = slice(0, -2)
    dH_plus = dH_pad[tuple(slc_plus)]
    dH_minus = dH_pad[tuple(slc_minus)]

    shape = [1] * H.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    ladder = sqrt_p * dH_plus + sqrt_m * dH_minus
    vth_arr = jnp.asarray(vth)
    if vth_arr.ndim == 0:
        vth_arr = vth_arr[None]
    v_shape = [1] * H.ndim
    v_shape[0] = vth_arr.shape[0]
    vth_arr = vth_arr.reshape(v_shape)
    return vth_arr * ladder


def linear_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS and electrostatic potential.

    Parameters
    ----------
    G : jnp.ndarray
        Laguerre-Hermite moments with shape (Nl, Nm, Ny, Nx, Nz).
    grid : SpectralGrid
        Flux-tube spectral grid.
    geom : SAlphaGeometry
        Analytic s-alpha geometry.
    params : LinearParams
        Physical and normalization parameters.
    """

    if G.ndim == 5:
        Nl, Nm = G.shape[0], G.shape[1]
    elif G.ndim == 6:
        Nl, Nm = G.shape[1], G.shape[2]
    else:
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return linear_rhs_cached(G, cache, params, terms=terms, dt=dt)


def linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry arrays."""

    from spectraxgk.terms.assembly import (
        assemble_rhs_cached,
        assemble_rhs_cached_electrostatic_jit,
        assemble_rhs_cached_jit,
    )

    term_cfg = linear_terms_to_term_config(terms)

    if use_jit:
        rhs_fn = (
            assemble_rhs_cached_electrostatic_jit
            if force_electrostatic_fields
            else assemble_rhs_cached_jit
        )
        dG, fields = rhs_fn(G, cache, params, term_cfg, dt)
    else:
        dG, fields = assemble_rhs_cached(
            G,
            cache,
            params,
            terms=term_cfg,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    return dG, fields.phi


def _is_streaming_only_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return (
        float(term_weights.streaming) == 1.0
        and float(term_weights.mirror) == 0.0
        and float(term_weights.curvature) == 0.0
        and float(term_weights.gradb) == 0.0
        and float(term_weights.diamagnetic) == 0.0
        and float(term_weights.collisions) == 0.0
        and float(term_weights.hypercollisions) == 0.0
        and float(term_weights.hyperdiffusion) == 0.0
        and float(term_weights.end_damping) == 0.0
        and float(term_weights.apar) == 0.0
        and float(term_weights.bpar) == 0.0
    )


def _is_electrostatic_slice_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return (
        float(term_weights.collisions) == 0.0
        and float(term_weights.hypercollisions) == 0.0
        and float(term_weights.hyperdiffusion) == 0.0
        and float(term_weights.end_damping) == 0.0
        and float(term_weights.apar) == 0.0
        and float(term_weights.bpar) == 0.0
    )


def _is_electrostatic_field_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return float(term_weights.apar) == 0.0 and float(term_weights.bpar) == 0.0


def _resolve_parallel_devices(
    *, num_devices: int | None = None, devices: Any | None = None
) -> list[Any]:
    """Return an explicit device list for opt-in parallel diagnostics."""

    if devices is None:
        device_list = list(jax.devices())
        if num_devices is not None:
            device_count = int(num_devices)
            if device_count < 1:
                raise ValueError("num_devices must be >= 1")
            if len(device_list) < device_count:
                raise ValueError(
                    f"requested {device_count} devices, but only {len(device_list)} are available"
                )
            device_list = device_list[:device_count]
    else:
        device_list = list(devices)
        if num_devices is not None and int(num_devices) != len(device_list):
            raise ValueError("num_devices must match the explicit devices list length")
    if not device_list:
        raise ValueError("at least one device is required")
    return device_list


def linear_rhs_streaming_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the streaming-only linear RHS with the Hermite shard-map path.

    This diagnostic route is intentionally narrower than
    :func:`linear_rhs_cached`: it covers the velocity-space streaming operator
    only and returns a zero electrostatic potential. It is used to gate the
    future production velocity decomposition before field solves, drifts,
    collisions, and nonlinear terms are exposed through the runtime path.
    """

    from spectraxgk.velocity_sharding import (
        build_velocity_sharding_plan,
        periodic_streaming_shard_map,
    )

    arr = jnp.asarray(G)
    if arr.ndim not in (5, 6):
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    dG = -periodic_streaming_shard_map(
        arr, plan, kz=cache.kz, vth=params.vth, devices=device_list
    )
    phi = jnp.zeros(arr.shape[-3:], dtype=arr.dtype)
    return dG, phi


def _streaming_electrostatic_from_phi_velocity_sharded(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    phi: jnp.ndarray,
    plan: Any,
    devices: Any,
) -> jnp.ndarray:
    """Apply electrostatic streaming with a precomputed electrostatic field."""

    from spectraxgk.terms.operators import grad_z_periodic
    from spectraxgk.velocity_sharding import periodic_streaming_shard_map

    particle_streaming = -periodic_streaming_shard_map(
        arr, plan, kz=cache.kz, vth=params.vth, devices=devices
    )
    real_dtype = jnp.real(arr).dtype
    G6 = arr[None, ...]
    tz = _as_species_array(params.tz, 1, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, 1, "vth").astype(real_dtype)
    field_rhs = _electrostatic_streaming_field_rhs(
        G6, phi=phi, Jl=cache.Jl, tz=tz, vth=vth
    )
    field_streaming = jnp.asarray(
        params.kpar_scale, dtype=real_dtype
    ) * grad_z_periodic(field_rhs, kz=cache.kz)
    return particle_streaming + field_streaming[0]


def _electrostatic_streaming_field_rhs(
    G6: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    Jl: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
) -> jnp.ndarray:
    """Build the pre-derivative electrostatic streaming field term."""

    Nm = G6.shape[2]
    m_idx = jnp.arange(Nm, dtype=jnp.int32)[None, None, :, None, None, None]
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    zt5 = zt[:, None, None, None, None]
    vth5 = vth[:, None, None, None, None]
    phi_s = phi[None, None, ...]
    drive_m1 = -zt5 * vth5 * Jl * phi_s
    return (m_idx == 1).astype(G6.dtype) * drive_m1[:, :, None, ...]


def linear_rhs_streaming_electrostatic_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
    use_custom_vjp: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute electrostatic streaming RHS with Hermite-sharded particle streaming.

    This route solves ``phi`` with the production electrostatic field solve,
    applies the Hermite velocity-sharded particle-streaming operator, and adds
    the GX-style electrostatic streaming field term. It is limited to periodic
    field-line grids and excludes electromagnetic fields by construction.
    """

    from spectraxgk.velocity_sharding import (
        build_velocity_sharding_plan,
        electrostatic_phi_shard_map,
    )

    arr = jnp.asarray(G)
    if arr.ndim != 5:
        raise NotImplementedError(
            "velocity-sharded electrostatic streaming currently supports single-species 5D states"
        )
    if bool(getattr(cache, "use_twist_shift", False)):
        raise NotImplementedError(
            "velocity-sharded electrostatic streaming currently requires a periodic z grid"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    phi = electrostatic_phi_shard_map(
        arr,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=device_list,
    )
    return _streaming_electrostatic_from_phi_velocity_sharded(
        arr, cache, params, phi=phi, plan=plan, devices=device_list
    ), phi


def _linear_rhs_electrostatic_slices_velocity_sharded_fused(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_weights: LinearTerms,
    *,
    plan: Any,
    devices: Any,
    axis_name: str = "m",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fuse the current single-species periodic electrostatic shard-map route."""

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from spectraxgk.terms.operators import grad_z_periodic, shift_axis

    dims = ("l", "m", "ky", "kx", "z")
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    if m_chunks <= 1:
        raise ValueError("fused Hermite route requires more than one Hermite chunk")
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")
    active_non_hermite = tuple(
        active_axis for active_axis in plan.active_axes if active_axis != "m"
    )
    if active_non_hermite:
        raise NotImplementedError(
            "fused electrostatic slice route currently supports only an active 'm' axis"
        )

    device_list = list(devices)
    if len(device_list) < m_chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:m_chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[m_axis] = axis_name
    state_spec = PartitionSpec(*spec_list)
    phi_spec = PartitionSpec(None, None, None)
    sharding = NamedSharding(mesh, state_spec)
    local_m = int(arr.shape[m_axis]) // m_chunks
    local_m_index = jnp.arange(local_m, dtype=jnp.int32).reshape((1, local_m, 1, 1, 1))
    prev_pairs = tuple((idx, idx + 1) for idx in range(m_chunks - 1))
    next_pairs = tuple((idx, idx - 1) for idx in range(1, m_chunks))

    real_dtype = jnp.real(arr).dtype
    jl = jnp.asarray(cache.Jl)
    if jl.ndim == 5:
        jl = jl[0]
    charge_s = jnp.asarray(params.charge_sign, dtype=real_dtype).reshape(-1)[0]
    density_s = jnp.asarray(params.density, dtype=real_dtype).reshape(-1)[0]
    tau = jnp.asarray(params.tau_e, dtype=real_dtype)
    tz_s = jnp.asarray(params.tz, dtype=real_dtype).reshape(-1)[0]
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    vth_s = jnp.asarray(params.vth, dtype=real_dtype).reshape(-1)[0]
    g0 = jnp.sum(jl * jl, axis=0)
    den_safe = jnp.where(
        tau + density_s * charge_s * zt * (1.0 - g0) == 0.0,
        jnp.inf,
        tau + density_s * charge_s * zt * (1.0 - g0),
    )
    mask0 = None if cache.mask0 is None else jnp.asarray(cache.mask0)
    ell = jnp.arange(arr.shape[0], dtype=real_dtype).reshape((arr.shape[0], 1, 1, 1, 1))
    ell_p1 = ell + 1.0
    bgrad = jnp.asarray(cache.bgrad, dtype=real_dtype).reshape(
        (1, 1, 1, 1, int(jnp.asarray(cache.bgrad).shape[-1]))
    )
    cv = jnp.asarray(cache.cv_d, dtype=real_dtype).reshape(
        (1, 1) + tuple(jnp.asarray(cache.cv_d).shape)
    )
    gb = jnp.asarray(cache.gb_d, dtype=real_dtype).reshape(
        (1, 1) + tuple(jnp.asarray(cache.gb_d).shape)
    )
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    imag = jnp.asarray(1j, dtype=arr.dtype)
    omega_star = (
        imag
        * jnp.asarray(params.omega_star_scale, dtype=real_dtype)
        * jnp.asarray(cache.ky, dtype=real_dtype)
    )
    omega_star_s = omega_star.reshape((1, omega_star.shape[0], 1, 1))
    tprim_s = jnp.asarray(params.R_over_LTi, dtype=real_dtype).reshape(-1)[0]
    fprim_s = jnp.asarray(params.R_over_Ln, dtype=real_dtype).reshape(-1)[0]
    jl_m1 = shift_axis(jl, -1, axis=0)
    jl_p1 = shift_axis(jl, 1, axis=0)
    l4 = jnp.asarray(cache.l4, dtype=real_dtype).reshape((arr.shape[0], 1, 1, 1))
    w_streaming = jnp.asarray(term_weights.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(term_weights.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(term_weights.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(term_weights.gradb, dtype=real_dtype)
    w_diamag = jnp.asarray(term_weights.diamagnetic, dtype=real_dtype)

    def shift_m(local, *, offset: int):
        depth = abs(int(offset))
        if depth == 0:
            return local
        if offset < 0:
            boundary = local[:, -depth:, ...]
            received = jax.lax.ppermute(boundary, axis_name, prev_pairs)
            return jnp.concatenate([received, local[:, :-depth, ...]], axis=1)
        boundary = local[:, :depth, ...]
        received = jax.lax.ppermute(boundary, axis_name, next_pairs)
        return jnp.concatenate([local[:, depth:, ...], received], axis=1)

    def fused(local):
        global_m = jax.lax.axis_index(axis_name) * local_m + local_m_index
        global_m_real = global_m.astype(real_dtype)
        m0 = (global_m == 0).astype(local.dtype)
        local_gm0 = jnp.sum(local * m0, axis=1)
        local_nbar = density_s * charge_s * jnp.sum(jl * local_gm0, axis=0)
        phi = jax.lax.psum(local_nbar, axis_name) / den_safe
        if mask0 is not None:
            phi = jnp.where(mask0, 0.0, phi)

        dlocal_dz = grad_z_periodic(local, kz=cache.kz)
        lower = shift_m(dlocal_dz, offset=-1)
        upper = shift_m(dlocal_dz, offset=1)
        streaming = -vth_s * (
            jnp.sqrt(global_m_real + 1.0) * upper + jnp.sqrt(global_m_real) * lower
        )
        field_drive_m1 = (global_m == 1).astype(local.dtype) * (-zt * vth_s * jl * phi)[
            :, None, ...
        ]
        streaming = streaming + kpar_scale * grad_z_periodic(
            field_drive_m1, kz=cache.kz
        )

        h = local + (global_m == 0).astype(local.dtype) * (zt * jl * phi)[:, None, ...]
        h_m_p1 = shift_m(h, offset=1)
        h_m_m1 = shift_m(h, offset=-1)
        mirror_term = (
            -jnp.sqrt(global_m_real + 1.0) * ell_p1 * h_m_p1
            - jnp.sqrt(global_m_real + 1.0) * ell * shift_axis(h_m_p1, -1, axis=0)
            + jnp.sqrt(global_m_real) * ell * h_m_m1
            + jnp.sqrt(global_m_real) * ell_p1 * shift_axis(h_m_m1, 1, axis=0)
        )
        mirror = -vth_s * bgrad * mirror_term

        h_m_p2 = shift_m(h, offset=2)
        h_m_m2 = shift_m(h, offset=-2)
        curv_term = (
            jnp.sqrt((global_m_real + 1.0) * (global_m_real + 2.0)) * h_m_p2
            + (2.0 * global_m_real + 1.0) * h
            + jnp.sqrt(global_m_real * (global_m_real - 1.0)) * h_m_m2
        )
        gradb_term = (
            (ell + 1.0) * shift_axis(h, 1, axis=0)
            + (2.0 * ell + 1.0) * h
            + ell * shift_axis(h, -1, axis=0)
        )
        curvature = -(imag * tz_s * omega_d_scale * cv) * curv_term
        gradb = -(imag * tz_s * omega_d_scale * gb) * gradb_term

        drive_m0 = (
            omega_star_s
            * phi
            * (
                jl_m1 * (l4 * tprim_s)
                + jl * (fprim_s + 2.0 * l4 * tprim_s)
                + jl_p1 * ((l4 + 1.0) * tprim_s)
            )
        )
        drive_m2 = (
            omega_star_s
            * phi
            * jl
            * (tprim_s / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)))
        )
        diamagnetic = (global_m == 0).astype(local.dtype) * drive_m0[:, None, ...]
        diamagnetic = (
            diamagnetic + (global_m == 2).astype(local.dtype) * drive_m2[:, None, ...]
        )

        rhs = (
            w_streaming * streaming
            + w_mirror * mirror
            + w_curv * curvature
            + w_gradb * gradb
        )
        rhs = rhs + w_diamag * diamagnetic
        return rhs, phi

    cache_key = (
        "electrostatic_linear_slices_fused",
        tuple(int(x) for x in arr.shape),
        str(arr.dtype),
        id(cache),
        id(params),
        float(term_weights.streaming),
        float(term_weights.mirror),
        float(term_weights.curvature),
        float(term_weights.gradb),
        float(term_weights.diamagnetic),
        tuple(str(device) for device in device_list[:m_chunks]),
        axis_name,
    )
    cached = _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE.get(cache_key)
    if cached is None:
        mapped = jax.jit(
            jax.shard_map(
                fused,
                mesh=mesh,
                in_specs=state_spec,
                out_specs=(state_spec, phi_spec),
                axis_names={axis_name},
            )
        )
        cached = (mapped, sharding)
        _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE[cache_key] = cached
    else:
        mapped, sharding = cached
    return mapped(jax.device_put(arr, sharding))


def linear_rhs_electrostatic_slices_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute gated electrostatic streaming, drift, and diamagnetic slices."""

    from spectraxgk.velocity_sharding import (
        build_velocity_sharding_plan,
        curvature_gradb_drift_shard_map,
        diamagnetic_drive_shard_map,
        electrostatic_phi_shard_map,
        mirror_drift_shard_map,
    )

    term_weights = terms if terms is not None else LinearTerms()
    if not _is_electrostatic_slice_terms(term_weights):
        raise NotImplementedError(
            "electrostatic slice route allows only electrostatic linear terms"
        )
    arr = jnp.asarray(G)
    if arr.ndim != 5:
        raise NotImplementedError(
            "velocity-sharded electrostatic slice route currently supports single-species 5D states"
        )
    if bool(getattr(cache, "use_twist_shift", False)):
        raise NotImplementedError(
            "velocity-sharded electrostatic slice route currently requires a periodic z grid"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    if len(device_list) > 1:
        return _linear_rhs_electrostatic_slices_velocity_sharded_fused(
            arr,
            cache,
            params,
            term_weights,
            plan=plan,
            devices=device_list,
        )
    real_dtype = jnp.real(arr).dtype
    phi = electrostatic_phi_shard_map(
        arr,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=device_list,
    )
    dG = jnp.zeros_like(arr)
    if float(term_weights.streaming) != 0.0:
        streaming = _streaming_electrostatic_from_phi_velocity_sharded(
            arr,
            cache,
            params,
            phi=phi,
            plan=plan,
            devices=device_list,
        )
        dG = dG + jnp.asarray(term_weights.streaming, dtype=real_dtype) * streaming
    H = build_H(arr, cache.Jl, phi, jnp.asarray([params.tz], dtype=real_dtype))
    if float(term_weights.mirror) != 0.0:
        dG = dG + mirror_drift_shard_map(
            H,
            plan,
            vth=jnp.asarray([params.vth], dtype=real_dtype),
            bgrad=cache.bgrad,
            ell=cache.l,
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            weight=jnp.asarray(term_weights.mirror, dtype=real_dtype),
            devices=device_list,
        )
    if float(term_weights.curvature) != 0.0 or float(term_weights.gradb) != 0.0:
        dG = dG + curvature_gradb_drift_shard_map(
            H,
            plan,
            tz=jnp.asarray([params.tz], dtype=real_dtype),
            omega_d_scale=params.omega_d_scale,
            cv_d=cache.cv_d,
            gb_d=cache.gb_d,
            ell=cache.l,
            m=cache.m,
            weight_curv=jnp.asarray(term_weights.curvature, dtype=real_dtype),
            weight_gradb=jnp.asarray(term_weights.gradb, dtype=real_dtype),
            devices=device_list,
        )
    if float(term_weights.diamagnetic) != 0.0:
        dG = dG + diamagnetic_drive_shard_map(
            arr,
            plan,
            phi=phi,
            Jl=cache.Jl,
            l4=cache.l4,
            tprim=params.R_over_LTi,
            fprim=params.R_over_Ln,
            omega_star_scale=params.omega_star_scale,
            ky=cache.ky,
            weight=jnp.asarray(term_weights.diamagnetic, dtype=real_dtype),
            devices=device_list,
        )
    return dG, phi


def linear_rhs_parallel_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    parallel: Any | None = None,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute linear RHS with an explicit, disabled-by-default parallel route.

    ``parallel=None`` and ``parallel.strategy="serial"`` are exact aliases for
    :func:`linear_rhs_cached`. The non-serial velocity routes are opt-in,
    Hermite-axis-only identity gates. ``backend="auto"`` selects the most
    complete currently gated electrostatic route when the term set is eligible;
    otherwise callers must request a narrower explicit backend.
    """

    if (
        parallel is None
        or str(getattr(parallel, "strategy", "serial")).lower() == "serial"
    ):
        return linear_rhs_cached(
            G,
            cache,
            params,
            terms=terms,
            use_jit=use_jit,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
        )

    strategy = str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    backend = str(getattr(parallel, "backend", "auto")).lower().replace("-", "_")
    axis = str(getattr(parallel, "axis", "hermite")).lower().replace("-", "_")
    if strategy == "velocity" and backend == "auto":
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "velocity sharding currently supports only the Hermite axis"
            )
        if _is_electrostatic_slice_terms(terms):
            backend = "electrostatic_linear_slices"
        else:
            raise NotImplementedError(
                "backend='auto' can only select gated electrostatic velocity routes; "
                "disable collision/EM/end-damping terms or request an explicit backend"
            )
    if strategy == "velocity" and backend in {
        "streaming_only",
        "linear_streaming_only",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "streaming-only velocity sharding currently supports only the Hermite axis"
            )
        if not _is_streaming_only_terms(terms):
            raise NotImplementedError(
                "velocity streaming route requires streaming-only LinearTerms"
            )
        return linear_rhs_streaming_velocity_sharded(
            G,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
        )
    if strategy == "velocity" and backend in {
        "streaming_electrostatic",
        "linear_streaming_electrostatic",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "electrostatic streaming velocity sharding currently supports only the Hermite axis"
            )
        if not _is_streaming_only_terms(terms):
            raise NotImplementedError(
                "electrostatic velocity streaming route requires streaming-only LinearTerms"
            )
        return linear_rhs_streaming_electrostatic_velocity_sharded(
            G,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
            use_custom_vjp=use_custom_vjp,
        )
    if strategy == "velocity" and backend in {
        "electrostatic_linear_slices",
        "linear_electrostatic_slices",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "electrostatic slice velocity sharding currently supports only the Hermite axis"
            )
        if not _is_electrostatic_slice_terms(terms):
            raise NotImplementedError(
                "electrostatic slice route requires collision/EM terms to be disabled"
            )
        return linear_rhs_electrostatic_slices_velocity_sharded(
            G,
            cache,
            params,
            terms=terms,
            num_devices=getattr(parallel, "num_devices", None),
        )

    raise NotImplementedError(
        "parallel linear RHS currently supports only strategy='velocity' with gated electrostatic backends"
    )


def _integrate_linear_cached_impl(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    parallel: Any | None = None,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""
    if method not in {"euler", "rk2", "rk4", "imex", "imex2", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk4', 'imex', 'imex2', 'sspx3'}"
        )
    if terms is None:
        terms = LinearTerms()

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G0.ndim == 5))
        + hyper_damp
    )
    damping = damping.astype(real_dtype)

    parallel_strategy = (
        "serial"
        if parallel is None
        else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    )

    def rhs(G_in: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if parallel_strategy == "serial":
            return linear_rhs_cached(
                G_in,
                cache,
                params,
                terms=terms,
                dt=dt_val,
                force_electrostatic_fields=force_electrostatic_fields,
            )
        return linear_rhs_parallel_cached(
            G_in, cache, params, terms=terms, parallel=parallel, dt=dt_val
        )

    def advance(G):
        dG, _phi = rhs(G)
        if method == "imex":
            dG_explicit = dG + damping * G
            return (G + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G
            G_half = (G + 0.5 * dt_val * dG_explicit) / (1.0 + 0.5 * dt_val * damping)
            dG_half, _phi = rhs(G_half)
            dG_half_exp = dG_half + damping * G_half
            return (G + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = rhs(G + 0.5 * dt_val * k1)
            return G + dt_val * k2
        if method == "sspx3":

            def _euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _ = rhs(G_state)
                return G_state + (_SSPX3_ADT * dt_val) * dG_state

            G1 = _euler_step(G)
            G2_euler = _euler_step(G1)
            G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
            G3 = _euler_step(G2)
            return (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        k1 = dG
        k2, _ = rhs(G + 0.5 * dt_val * k1)
        k3, _ = rhs(G + 0.5 * dt_val * k2)
        k4, _ = rhs(G + dt_val * k3)
        return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(G, idx):
        G_new = advance(G)
        _dG_new, phi_new = rhs(G_new)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            sim_time = (idx + 1) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi_new))
            G_new = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    0.0,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_new,
            )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    indices = jnp.arange(steps)
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, G0, indices)

    def sample_step(G, idx):
        def inner_step(i, state):
            return advance(state)

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G)
        _dG_out, phi_out = rhs(G_out)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
            sim_time = jnp.minimum((idx + 1) * sample_stride, steps) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi_out))
            G_out = jax.lax.cond(
                should_emit_progress(completed_idx, steps),
                lambda state: print_callback(
                    state,
                    completed_idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    0.0,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_out,
            )
        return G_out, phi_out

    num_samples = steps // sample_stride
    sample_indices = jnp.arange(num_samples)
    return jax.lax.scan(sample_step, G0, sample_indices)


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "method",
        "checkpoint",
        "sample_stride",
        "show_progress",
        "force_electrostatic_fields",
    ),
)
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "method",
        "checkpoint",
        "sample_stride",
        "show_progress",
        "force_electrostatic_fields",
    ),
    donate_argnums=(0,),
)
def _integrate_linear_cached_donate(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


def _build_implicit_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    terms: LinearTerms | None,
    implicit_preconditioner: PreconditionerSpec,
) -> tuple[
    jnp.ndarray,
    tuple[int, ...],
    int,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    bool,
]:
    if terms is None:
        terms = LinearTerms()
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    shape = G.shape
    size = int(np.prod(np.asarray(shape)))
    ns = shape[0]

    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=False) + hyper_damp
    )
    damping = damping.astype(real_dtype)

    ell = cache.l.astype(real_dtype)
    m = cache.m.astype(real_dtype)
    cv_d = cache.cv_d.astype(real_dtype)
    gb_d = cache.gb_d.astype(real_dtype)
    bgrad = cache.bgrad.astype(real_dtype)
    w_mirror = jnp.asarray(terms.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(terms.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(terms.gradb, dtype=real_dtype)
    diag = jnp.zeros_like(damping, dtype=state_dtype)
    imag = jnp.asarray(1j, dtype=state_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tz_b = tz[:, None, None, None, None, None]
    vth_b = vth[:, None, None, None, None, None]
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    diag = diag - imag * tz_b * omega_d_scale * (
        w_curv * cv_d[None, None, None, ...] * (2.0 * m + 1.0)
        + w_gradb * gb_d[None, None, None, ...] * (2.0 * ell + 1.0)
    )
    bgrad = bgrad[None, None, None, None, None, :]
    mirror_diag = vth_b * (2.0 * ell + 1.0) * (2.0 * m + 1.0)
    mirror_weight = 0.2
    diag = diag - w_mirror * mirror_weight * bgrad * mirror_diag

    precond_full = 1.0 / (1.0 + dt_val * damping - dt_val * diag)
    precond_full = precond_full.astype(G.dtype)
    precond_damp = (1.0 / (1.0 + dt_val * damping)).astype(G.dtype)
    kpar = params.kpar_scale * cache.kz.astype(real_dtype)
    w_stream = jnp.asarray(terms.streaming, dtype=real_dtype)
    kpar_b = kpar[None, None, None, None, None, :]
    precond_pas = 1.0 / (
        1.0
        + dt_val * damping
        - dt_val * diag
        + imag * dt_val * w_stream * vth_b * kpar_b
    )
    precond_pas = precond_pas.astype(G.dtype)
    resolved_precond = _resolve_implicit_preconditioner(implicit_preconditioner)

    sqrt_m_line = cache.sqrt_m_ladder.reshape(-1).astype(real_dtype)
    sqrt_p_line = cache.sqrt_p.reshape(-1).astype(real_dtype)

    def _solve_hermite_lines_fft(
        x: jnp.ndarray,
        *,
        kz: jnp.ndarray,
    ) -> jnp.ndarray:
        """Invert (I - dt*L_stream) approximately via FFT(z) + tridiagonal(m)."""

        x_hat = jnp.fft.fft(x, axis=-1)
        x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)  # (..., Nz, Nm)
        coeff = (
            (dt_val * w_stream * jnp.asarray(params.kpar_scale, dtype=real_dtype))
            * vth[:, None, None, None, None]
            * (imag * kz)[None, None, None, None, :]
        )
        coeff = coeff[..., None]  # (Ns, 1, 1, 1, Nz, 1)
        dl = coeff * sqrt_m_line
        du = coeff * sqrt_p_line
        du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
        d = jnp.ones_like(du)
        batch_shape = x_hat_mlast.shape
        dl = jnp.broadcast_to(dl, batch_shape)
        d = jnp.broadcast_to(d, batch_shape)
        du = jnp.broadcast_to(du, batch_shape)
        y_hat_mlast = jax.lax.linalg.tridiagonal_solve(
            dl, d, du, x_hat_mlast[..., None]
        )[..., 0]
        y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
        return jnp.fft.ifft(y_hat, axis=-1)

    def _solve_hermite_lines_linked(x: jnp.ndarray) -> jnp.ndarray:
        """Linked-FFT variant of the Hermite-line streaming preconditioner."""

        if not cache.linked_indices:
            return _solve_hermite_lines_fft(x, kz=cache.kz)

        Ny = x.shape[-3]
        Nx = x.shape[-2]
        Nz = x.shape[-1]
        lead_shape = x.shape[:-3]
        x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
        y_flat = jnp.zeros_like(x_flat)

        def _scatter_unique(
            target: jnp.ndarray, idx_flat: jnp.ndarray, updates: jnp.ndarray
        ) -> jnp.ndarray:
            idx = jnp.asarray(idx_flat, dtype=jnp.int32)
            target_t = jnp.moveaxis(target, -2, 0)
            updates_t = jnp.moveaxis(updates, -2, 0)
            idx = idx[:, None]
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=tuple(range(1, updates_t.ndim)),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            out_t = jax.lax.scatter(
                target_t,
                idx,
                updates_t,
                dnums,
                unique_indices=True,
            )
            return jnp.moveaxis(out_t, 0, -2)

        for idx_map, kz_link in zip(cache.linked_indices, cache.linked_kz):
            nChains, nLinks = idx_map.shape
            idx_flat = idx_map.reshape(-1)
            x_link = jnp.take(x_flat, idx_flat, axis=-2)
            x_link = x_link.reshape(*lead_shape, nChains, nLinks * Nz)
            x_hat = jnp.fft.fft(x_link, axis=-1)
            x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)  # (Ns, Nl, nChains, nfreq, Nm)
            coeff = (
                (dt_val * w_stream * jnp.asarray(params.kpar_scale, dtype=real_dtype))
                * vth[:, None, None, None]
                * (imag * kz_link)[None, None, None, :]
            )
            coeff = coeff[..., None]  # (Ns, 1, 1, nfreq, 1)
            dl = coeff * sqrt_m_line
            du = coeff * sqrt_p_line
            du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
            d = jnp.ones_like(du)
            batch_shape = x_hat_mlast.shape
            dl = jnp.broadcast_to(dl, batch_shape)
            d = jnp.broadcast_to(d, batch_shape)
            du = jnp.broadcast_to(du, batch_shape)
            y_hat_mlast = jax.lax.linalg.tridiagonal_solve(
                dl, d, du, x_hat_mlast[..., None]
            )[..., 0]
            y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
            y_link = jnp.fft.ifft(y_hat, axis=-1)
            y_link = y_link.reshape(*lead_shape, nChains * nLinks, Nz)
            y_flat = _scatter_unique(y_flat, idx_flat, y_link)

        return y_flat.reshape(*lead_shape, Ny, Nx, Nz)

    def apply_precond_full(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_full).reshape(size)

    def apply_precond_damp(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_damp).reshape(size)

    def apply_precond_pas(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_pas).reshape(size)

    def _project_kx_coarse(x: jnp.ndarray) -> jnp.ndarray:
        """Coarse-space projection/prolongation for twist/shift coupling.

        For periodic grids this reduces to the mean over kx. For linked grids we
        average within each linked (ky, kx) chain so the coarse correction does
        not destroy the linked coupling structure.
        """

        if not cache.use_twist_shift or not cache.linked_indices:
            x_mean = jnp.mean(x, axis=4, keepdims=True)
            return jnp.broadcast_to(x_mean, x.shape)

        Ny = x.shape[-3]
        Nx = x.shape[-2]
        Nz = x.shape[-1]
        lead_shape = x.shape[:-3]
        x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
        y_flat = jnp.zeros_like(x_flat)

        def _scatter_unique(
            target: jnp.ndarray, idx_flat: jnp.ndarray, updates: jnp.ndarray
        ) -> jnp.ndarray:
            idx = jnp.asarray(idx_flat, dtype=jnp.int32)
            target_t = jnp.moveaxis(target, -2, 0)
            updates_t = jnp.moveaxis(updates, -2, 0)
            idx = idx[:, None]
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=tuple(range(1, updates_t.ndim)),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            out_t = jax.lax.scatter(
                target_t,
                idx,
                updates_t,
                dnums,
                unique_indices=True,
            )
            return jnp.moveaxis(out_t, 0, -2)

        for idx_map in cache.linked_indices:
            nChains, nLinks = idx_map.shape
            idx_flat = idx_map.reshape(-1)
            x_link = jnp.take(x_flat, idx_flat, axis=-2)
            x_link = x_link.reshape(*lead_shape, nChains, nLinks, Nz)
            x_mean = jnp.mean(x_link, axis=-2, keepdims=True)
            x_mean = jnp.broadcast_to(x_mean, x_link.shape)
            x_updates = x_mean.reshape(*lead_shape, nChains * nLinks, Nz)
            y_flat = _scatter_unique(y_flat, idx_flat, x_updates)

        return y_flat.reshape(*lead_shape, Ny, Nx, Nz)

    def apply_precond_pas_coarse(x_flat: jnp.ndarray) -> jnp.ndarray:
        """PAS line + kx-coarse correction (additive Schur-style)."""
        x = x_flat.reshape(shape)
        x_line = x * precond_pas
        x_coarse = _project_kx_coarse(x) * precond_pas
        x_line_coarse = _project_kx_coarse(x_line)
        x_out = x_line + (x_coarse - x_line_coarse)
        return x_out.reshape(size)

    def apply_precond_hermite_line(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        x = x * precond_full
        x = (
            _solve_hermite_lines_linked(x)
            if cache.use_twist_shift
            else _solve_hermite_lines_fft(x, kz=cache.kz)
        )
        return x.reshape(size)

    def apply_precond_hermite_line_coarse(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        x_line = apply_precond_hermite_line(x.reshape(size)).reshape(shape)
        x_coarse_in = _project_kx_coarse(x)
        x_coarse_full = apply_precond_hermite_line(x_coarse_in.reshape(size)).reshape(
            shape
        )
        x_line_coarse_full = _project_kx_coarse(x_line)
        return (x_line + (x_coarse_full - x_line_coarse_full)).reshape(size)

    def apply_identity(x_flat: jnp.ndarray) -> jnp.ndarray:
        return x_flat

    precond_op: Callable[[jnp.ndarray], jnp.ndarray]
    if callable(resolved_precond):
        precond_op = resolved_precond
    else:
        key = resolved_precond or "auto"
        if key in {"auto", "diag", "diagonal", "physics", "block"}:
            precond_op = apply_precond_full
        elif key in {"damping", "collisional", "hyper"}:
            precond_op = apply_precond_damp
        elif key in {"pas", "pas-line", "pas_line"}:
            precond_op = apply_precond_pas
        elif key in {"pas-coarse", "pas_schur", "block-schur", "schur", "pas-hybrid"}:
            precond_op = apply_precond_pas_coarse
        elif key in {
            "hermite-line",
            "hermite_line",
            "hermite",
            "streaming-line",
            "streaming_line",
        }:
            precond_op = apply_precond_hermite_line
        elif key in {
            "hermite-line-coarse",
            "hermite_line_coarse",
            "hermite_coarse",
            "streaming-line-coarse",
        }:
            precond_op = apply_precond_hermite_line_coarse
        elif key in {"identity", "none", "off"}:
            precond_op = apply_identity
        else:
            raise ValueError(f"Unknown implicit_preconditioner '{resolved_precond}'")

    def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        dG, _phi = linear_rhs_cached(
            x,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
            dt=dt_val,
        )
        return (x - dt_val * dG).reshape(size)

    return G, shape, size, dt_val, precond_op, matvec, squeeze_species


def _integrate_linear_implicit_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: LinearTerms | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Implicit linear integrator using GMRES with a diagonal preconditioner."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = (
        _build_implicit_operator(G0, cache, params, dt, terms, implicit_preconditioner)
    )

    def fixed_point(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        def body(_i, g):
            dG, _phi = linear_rhs_cached(
                g,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
                dt=dt_val,
            )
            g_next = G_rhs + dt_val * dG
            return (1.0 - implicit_relax) * g + implicit_relax * g_next

        return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)

    def solve_step(G_in: jnp.ndarray) -> jnp.ndarray:
        G_guess = fixed_point(G_in, G_in)
        sol, _info = gmres(
            matvec,
            G_in.reshape(size),
            x0=G_guess.reshape(size),
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            restart=implicit_restart,
            M=precond_op,
            solve_method=implicit_solve_method,
        )
        return sol.reshape(shape)

    def step(G_in, _):
        G_new = solve_step(G_in)
        _dG_new, phi_new = linear_rhs_cached(
            G_new,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
            dt=dt_val,
        )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    if sample_stride <= 1:
        G_out, phi_t = jax.lax.scan(step_fn, G, None, length=steps)
    else:

        def sample_step(G_in, _):
            def inner_step(_i, g):
                return solve_step(g)

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG_out, phi_out = linear_rhs_cached(
                G_out_local,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
                dt=dt_val,
            )
            return G_out_local, phi_out

        num_samples = steps // sample_stride
        G_out, phi_t = jax.lax.scan(sample_step, G, None, length=num_samples)

    G_out = G_out[0] if squeeze_species else G_out
    return G_out, phi_t


def integrate_linear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    terms: LinearTerms | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    donate: bool = False,
    show_progress: bool = False,
    parallel: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    if method == "semi-implicit":
        method = "imex"
    parallel_strategy = (
        "serial"
        if parallel is None
        else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    )
    force_electrostatic_fields = _is_electrostatic_field_terms(terms)
    if method == "implicit":
        if parallel_strategy != "serial":
            raise NotImplementedError(
                "parallel linear integration currently supports only explicit fixed-step methods"
            )
        return _integrate_linear_implicit_cached(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            terms=terms,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            implicit_preconditioner=implicit_preconditioner,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
        )
    if parallel_strategy != "serial":
        if donate:
            raise NotImplementedError(
                "parallel linear integration does not currently support donated input buffers"
            )
        return _integrate_linear_cached_impl(
            G0,
            cache,
            params,
            dt,
            steps,
            method=method,
            checkpoint=checkpoint,
            terms=terms,
            sample_stride=sample_stride,
            show_progress=show_progress,
            parallel=parallel,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    integrator = _integrate_linear_cached_donate if donate else _integrate_linear_cached
    return integrator(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


def integrate_linear_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    species_index: int | None = 0,
    record_hl_energy: bool = False,
    show_progress: bool = False,
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Integrate and return (G_out, phi_t, density_t) for diagnostics."""

    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G0.ndim == 5))
        + hyper_damp
    )
    damping = damping.astype(real_dtype)

    def advance(G_in: jnp.ndarray) -> jnp.ndarray:
        dG, _phi = linear_rhs_cached(
            G_in, cache, params, terms=terms, use_jit=False, dt=dt_val
        )
        if method == "imex":
            dG_explicit = dG + damping * G_in
            return (G_in + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G_in
            G_half = (G_in + 0.5 * dt_val * dG_explicit) / (
                1.0 + 0.5 * dt_val * damping
            )
            dG_half, _phi = linear_rhs_cached(
                G_half, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            dG_half_exp = dG_half + damping * G_half
            return (G_in + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G_in + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k1,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            return G_in + dt_val * k2
        if method == "sspx3":

            def _euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _phi_state = linear_rhs_cached(
                    G_state,
                    cache,
                    params,
                    terms=terms,
                    use_jit=False,
                    dt=dt_val,
                )
                return G_state + (_SSPX3_ADT * dt_val) * dG_state

            G1 = _euler_step(G_in)
            G2_euler = _euler_step(G1)
            G2 = (1.0 - _SSPX3_W1) * G_in + (_SSPX3_W1 - 1.0) * G1 + G2_euler
            G3 = _euler_step(G2)
            return (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G_in
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        if method == "rk4":
            k1 = dG
            k2, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k1,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            k3, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k2,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            k4, _ = linear_rhs_cached(
                G_in + dt_val * k3, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            return G_in + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"Unsupported method '{method}'")

    def density_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        Jl = cache.Jl
        if G_in.ndim == 5:
            if Jl.ndim == 5:
                Jl_s = Jl[0]
            else:
                Jl_s = Jl
            return jnp.sum(Jl_s * G_in[:, 0, ...], axis=0)
        if Jl.ndim == 5:
            if species_index is None:
                return jnp.sum(jnp.sum(Jl * G_in[:, :, 0, ...], axis=1), axis=0)
            Jl_s = Jl[int(species_index)]
            return jnp.sum(Jl_s * G_in[int(species_index), :, 0, ...], axis=0)
        if species_index is None:
            return jnp.sum(jnp.sum(Jl[None, ...] * G_in[:, :, 0, ...], axis=1), axis=0)
        return jnp.sum(Jl * G_in[int(species_index), :, 0, ...], axis=0)

    def hl_energy_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        if G_in.ndim == 5:
            return jnp.sum(jnp.abs(G_in) ** 2, axis=(2, 3, 4))
        return jnp.sum(jnp.abs(G_in) ** 2, axis=(0, 3, 4, 5))

    def step(G_in, idx):
        G_out = advance(G_in)
        _dG, phi = linear_rhs_cached(
            G_out, cache, params, terms=terms, use_jit=False, dt=dt_val
        )
        density = density_from_G(G_out)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            sim_time = (idx + 1) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi))
            density_max = jnp.max(jnp.abs(density))
            G_out = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    density_max,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_out,
            )
        if record_hl_energy:
            hl_energy = hl_energy_from_G(G_out)
            return G_out, (phi, density, hl_energy)
        return G_out, (phi, density)

    if sample_stride <= 1:
        indices = jnp.arange(steps)
        G_out, outputs = jax.lax.scan(step, G0, indices)
    else:

        def sample_step(G_in, idx):
            def inner_step(_i, g):
                return advance(g)

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG, phi_out = linear_rhs_cached(
                G_out_local, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            density_out = density_from_G(G_out_local)
            if show_progress:
                from spectraxgk.utils.callbacks import (
                    print_callback,
                    should_emit_progress,
                )

                completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
                sim_time = jnp.minimum((idx + 1) * sample_stride, steps) * dt_val
                sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
                phi_max = jnp.max(jnp.abs(phi_out))
                density_max = jnp.max(jnp.abs(density_out))
                G_out_local = jax.lax.cond(
                    should_emit_progress(completed_idx, steps),
                    lambda state: print_callback(
                        state,
                        completed_idx,
                        steps,
                        0.0,
                        0.0,
                        phi_max,
                        density_max,
                        sim_time,
                        sim_total,
                        metric_labels=("|phi|_max", "|n|_max"),
                    ),
                    lambda state: state,
                    G_out_local,
                )
            if record_hl_energy:
                hl_out = hl_energy_from_G(G_out_local)
                return G_out_local, (phi_out, density_out, hl_out)
            return G_out_local, (phi_out, density_out)

        num_samples = steps // sample_stride
        sample_indices = jnp.arange(num_samples)
        G_out, outputs = jax.lax.scan(sample_step, G0, sample_indices)

    if record_hl_energy:
        phi_t, density_t, hl_t = outputs
        return G_out, phi_t, density_t, hl_t
    phi_t, density_t = outputs
    return G_out, phi_t, density_t
